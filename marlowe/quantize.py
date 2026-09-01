"""GGUF conversion, custom tensor-type mixes, and the Stage 1 converter probe.

The Stage 1 blocking risk is that ``convert_hf_to_gguf.py`` regenerates the layer layout from
``full_attention_interval`` rather than reading a non-uniform ``layer_types``. If it does,
everything downstream is dead until the converter is patched, so :func:`probe_converter`
tests it on day one by reading the produced GGUF back and checking which layers actually
carry attention tensors. That check needs no llama.cpp binaries -- there is a small GGUF
metadata reader below -- so it runs anywhere.

The Stage 0 recipes exploit the parameter budget directly. After 12 cuts the model is 62%
FFN, 26% mixers, 11% embeddings, and the two quantisation-fragile components are known: the
multiplicative output gate fused into ``q_proj`` (error compounds multiplicatively rather
than additively) and the DeltaNet projections whose recurrent state accumulates error along
the sequence -- which is why ``mamba_ssm_dtype`` is float32 in the first place. So protect
the 26% that causes the artefact and crush the 62% that does not.
"""

from __future__ import annotations

import json
import logging
import re
import struct
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from marlowe import logutil, preflight
from marlowe.arch import Layout, load_config
from marlowe.config import QuantConfig
from marlowe.eval.kl import find_binary

log = logutil.get("quantize")

GGUF_MAGIC = 0x46554747


# ---------------------------------------------------------------------------
# minimal GGUF metadata reader
# ---------------------------------------------------------------------------

_GGUF_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def _read(f: BinaryIO, fmt: str, size: int) -> Any:
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF file")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    n = _read(f, "<Q", 8)
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _GGUF_SCALARS:
        fmt, size = _GGUF_SCALARS[vtype]
        return _read(f, fmt, size)
    if vtype == _GGUF_STRING:
        return _read_string(f)
    if vtype == _GGUF_ARRAY:
        inner = _read(f, "<I", 4)
        n = _read(f, "<Q", 8)
        return [_read_value(f, inner) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {vtype}")


@dataclass
class GGUFInfo:
    path: str
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensor_names: list[str] = field(default_factory=list)
    tensor_types: dict[str, int] = field(default_factory=dict)
    file_bytes: int = 0

    @property
    def n_tensors(self) -> int:
        return len(self.tensor_names)


def read_gguf(path: str | Path, *, max_metadata: int = 100_000) -> GGUFInfo:
    """Read GGUF header, KV metadata, and tensor directory. Does not read tensor data."""
    p = Path(path)
    with p.open("rb") as f:
        if _read(f, "<I", 4) != GGUF_MAGIC:
            raise ValueError(f"{p} is not a GGUF file (bad magic)")
        version = _read(f, "<I", 4)
        n_tensors = _read(f, "<Q", 8)
        n_kv = _read(f, "<Q", 8)
        if n_kv > max_metadata or n_tensors > max_metadata:
            raise ValueError(f"{p}: implausible header ({n_tensors} tensors, {n_kv} kv)")

        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_string(f)
            vtype = _read(f, "<I", 4)
            meta[key] = _read_value(f, vtype)

        names: list[str] = []
        types: dict[str, int] = {}
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = _read(f, "<I", 4)
            for _ in range(n_dims):
                _read(f, "<Q", 8)
            types[name] = _read(f, "<I", 4)
            _read(f, "<Q", 8)  # offset
            names.append(name)

    return GGUFInfo(
        path=str(p),
        version=version,
        metadata=meta,
        tensor_names=names,
        tensor_types=types,
        file_bytes=p.stat().st_size,
    )


# ---------------------------------------------------------------------------
# GGUF layout inspection (the Stage 1 probe)
# ---------------------------------------------------------------------------

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

#: llama.cpp tensor-suffix families. Attention layers carry attn_*; recurrent layers carry
#: ssm_* or an explicit linear-attention name, depending on the converter version.
GGUF_ATTENTION_SUFFIXES = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv")
GGUF_LINEAR_SUFFIXES = ("ssm_", "linear_attn", "time_mix", "shortconv")


def gguf_layer_families(info: GGUFInfo) -> dict[int, set[str]]:
    """Map block index -> {"attention", "linear"} evidenced by tensor names."""
    out: dict[int, set[str]] = {}
    for name in info.tensor_names:
        m = BLK_RE.match(name)
        if not m:
            continue
        i = int(m.group(1))
        tail = m.group(2)
        fams = out.setdefault(i, set())
        if tail.startswith(GGUF_ATTENTION_SUFFIXES):
            fams.add("attention")
        if tail.startswith(GGUF_LINEAR_SUFFIXES):
            fams.add("linear")
    return out


@dataclass
class ProbeResult:
    ok: bool
    n_blocks: int
    expected: list[str]
    observed: list[str]
    mismatches: list[int]
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_blocks": self.n_blocks,
            "mismatches": self.mismatches,
            "detail": self.detail,
            "expected_head": self.expected[:12],
            "observed_head": self.observed[:12],
        }


def probe_converter(hf_dir: str | Path, gguf_path: str | Path) -> ProbeResult:
    """Did the converter preserve the non-uniform layout? The Stage 1 gate.

    Compares the layer families evidenced by GGUF tensor names against the ``layer_types`` in
    the source config. A converter that regenerated the layout from an interval will produce
    a clean 3:1 alternation over the *pruned* layer count -- which looks plausible and is
    wrong. That is exactly what this catches.
    """
    layout = Layout.from_config(load_config(hf_dir))
    info = read_gguf(gguf_path)
    fams = gguf_layer_families(info)

    expected = ["attention" if t != "linear_attention" else "linear" for t in layout.layer_types]
    observed: list[str] = []
    for i in range(len(expected)):
        f = fams.get(i, set())
        observed.append(sorted(f)[0] if len(f) == 1 else ("+".join(sorted(f)) or "none"))

    mismatches = [i for i, (e, o) in enumerate(zip(expected, observed)) if e != o]
    n_blocks = max(fams) + 1 if fams else 0

    if n_blocks != len(expected):
        detail = (
            f"GGUF has {n_blocks} blocks but the source config declares {len(expected)} "
            f"layers. The converter did not read layer_types."
        )
    elif mismatches:
        detail = (
            f"{len(mismatches)} layers have the wrong mixer family, first at index "
            f"{mismatches[0]} (expected {expected[mismatches[0]]}, got "
            f"{observed[mismatches[0]]}). The converter most likely regenerated the layout "
            f"from full_attention_interval instead of reading the explicit list. "
            f"Patch and vendor the converter before proceeding -- everything downstream "
            f"depends on this."
        )
    else:
        detail = f"layout preserved across all {n_blocks} blocks"

    result = ProbeResult(
        ok=not mismatches and n_blocks == len(expected),
        n_blocks=n_blocks,
        expected=expected,
        observed=observed,
        mismatches=mismatches,
        detail=detail,
    )
    logutil.event_at(
        log,
        logging.INFO if result.ok else logging.ERROR,
        "converter probe",
        ok=result.ok,
        blocks=n_blocks,
        mismatches=len(mismatches),
        detail=detail,
    )
    return result


# ---------------------------------------------------------------------------
# conversion and quantisation
# ---------------------------------------------------------------------------


def find_converter() -> Path:
    """Locate convert_hf_to_gguf.py: vendored copy first, then LLAMA_CPP_ROOT, then PATH."""
    import os
    import shutil

    vendored = Path(__file__).resolve().parent.parent / "vendor" / "convert_hf_to_gguf.py"
    if vendored.exists():
        return vendored
    root = os.environ.get("LLAMA_CPP_ROOT")
    if root:
        cand = Path(root) / "convert_hf_to_gguf.py"
        if cand.exists():
            return cand
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Set LLAMA_CPP_ROOT to a llama.cpp checkout, or "
        "place a (possibly patched) copy at vendor/convert_hf_to_gguf.py. If Stage 1 shows "
        "the stock converter mishandles non-uniform layer_types, the patched copy belongs "
        "in vendor/ so runs are reproducible."
    )


def convert_to_gguf(
    hf_dir: str | Path,
    out_path: str | Path,
    *,
    outtype: str = "bf16",
    timeout: int = 6 * 3600,
) -> Path:
    """Run convert_hf_to_gguf.py. Returns the output path."""
    import sys

    conv = find_converter()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(conv),
        str(hf_dir),
        "--outfile", str(out_path),
        "--outtype", outtype,
    ]
    with logutil.timed(log, "convert to gguf", src=str(hf_dir), outtype=outtype):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed (rc={proc.returncode}).\n"
            f"{(proc.stdout or '')[-2000:]}\n{(proc.stderr or '')[-3000:]}"
        )
    logutil.event(log, "converted", out=str(out_path), gb=round(out_path.stat().st_size / 1e9, 2))
    return out_path


def quantize(
    src_gguf: str | Path,
    out_gguf: str | Path,
    recipe: QuantConfig,
    *,
    n_threads: int | None = None,
    imatrix: str | Path | None = None,
    timeout: int = 6 * 3600,
) -> Path:
    """Run llama-quantize with a base type plus per-tensor overrides."""
    exe = find_binary("llama-quantize")
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe]
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
    for pattern, ttype in recipe.tensor_types.items():
        cmd += ["--tensor-type", f"{pattern}={ttype}"]
    cmd += [str(src_gguf), str(out_gguf), recipe.base_type]
    if n_threads:
        cmd.append(str(n_threads))

    with logutil.timed(log, "quantize", recipe=recipe.name, base=recipe.base_type):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0 or not out_gguf.exists():
        raise RuntimeError(
            f"llama-quantize failed for recipe {recipe.name!r} (rc={proc.returncode}).\n"
            f"{(proc.stderr or proc.stdout or '')[-3000:]}"
        )
    size_gb = out_gguf.stat().st_size / 1e9
    logutil.event(log, "quantized", recipe=recipe.name, gb=round(size_gb, 2))
    if size_gb > recipe.target_gb * 1.05:
        log.warning(
            "%s is %.2f GB against a %.2f GB target. The deployment budget is ~10.0 GB of "
            "weights: about 6 GB goes to KV cache, compute buffer, CUDA context and OS, and "
            "that overhead does not shrink when you prune -- it scales with hidden size "
            "(5120, unchanged) and the preserved full-attention layers.",
            recipe.name,
            size_gb,
            recipe.target_gb,
        )
    return out_gguf


def bits_per_weight(gguf_path: str | Path, n_params: int) -> float:
    """Effective bpw: file bytes x 8 / parameters. Includes metadata, so slightly generous."""
    return Path(gguf_path).stat().st_size * 8 / n_params


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------


def stage0_recipes(target_gb: float = 10.2) -> list[QuantConfig]:
    """The bit-width threshold sweep. Stage 0, decision-critical.

    Circling occurs at 2.97 bpw and not at bf16; nobody knows where in between it stops. That
    threshold decides whether Marlowe-22B at IQ3_M (3.66 bpw) is sufficient, or whether only
    18B at IQ4_XS (4.25 bpw) solves the actual problem.

    The custom mixes are the interesting arm: same total budget, spent differently. If one of
    them eliminates circling on the *unpruned* 27B at ~10 GB, the user's problem is solved
    without pruning at all, and the compression project becomes a speed and headroom play
    rather than a rescue.
    """
    gate_and_recurrent = {
        # The multiplicative output gate fused into q_proj: [12288, 5120] rather than
        # [6144, 5120]. Multiplicative error compounds; additive error averages out.
        "q_proj": "q5_K",
        # The recurrent path. Error accumulates along the sequence, which is why
        # mamba_ssm_dtype is float32 upstream.
        "linear_attn.*": "q5_K",
        # 62% of parameters and the most tolerant of them.
        "ffn_.*": "iq3_xxs",
    }
    return [
        QuantConfig(name="iq3_xxs", base_type="iq3_xxs", target_gb=target_gb),
        QuantConfig(name="iq3_s", base_type="iq3_s", target_gb=target_gb),
        QuantConfig(name="iq3_m", base_type="iq3_m", target_gb=target_gb),
        QuantConfig(name="iq4_xs", base_type="iq4_xs", target_gb=target_gb),
        QuantConfig(name="q4_k_s", base_type="q4_K_S", target_gb=target_gb),
        QuantConfig(
            name="mix_gate_q5",
            base_type="iq3_xxs",
            tensor_types=gate_and_recurrent,
            target_gb=target_gb,
        ),
        QuantConfig(
            name="mix_gate_q6",
            base_type="iq3_xxs",
            tensor_types={**gate_and_recurrent, "q_proj": "q6_K", "linear_attn.*": "q6_K"},
            target_gb=target_gb,
        ),
        QuantConfig(
            name="mix_recurrent_only",
            base_type="iq3_s",
            tensor_types={"linear_attn.*": "q5_K"},
            target_gb=target_gb,
        ),
    ]


def ship_recipes(target_gb: float = 10.2) -> list[QuantConfig]:
    """Stage 7 candidates: the nominal type plus the best custom mix from Stage 0.

    Which one ships is decided empirically on measured KL and repetition, not by bpw.
    """
    return [
        QuantConfig(name="iq3_m", base_type="iq3_m", target_gb=target_gb),
        QuantConfig(
            name="mix_gate_q5",
            base_type="iq3_xxs",
            tensor_types={"q_proj": "q5_K", "linear_attn.*": "q5_K", "ffn_.*": "iq3_xxs"},
            target_gb=target_gb,
        ),
    ]


def iter_recipe_outputs(out_dir: str | Path, recipes: list[QuantConfig]) -> Iterator[
    tuple[QuantConfig, Path]
]:
    out_dir = Path(out_dir)
    for r in recipes:
        yield r, out_dir / f"{r.name}.gguf"


def check_deployment_budget(gguf_path: str | Path, *, budget_gb: float = 10.0) -> dict[str, Any]:
    """Compare a built quant against the empirical VRAM budget.

    10.0 GB is proven, not theoretical: the user runs a 27B IQ3_XXS at 10.0 GB with a Q8 KV
    cache at 32K context, and that is at the limit.
    """
    size_gb = Path(gguf_path).stat().st_size / 1e9
    res = preflight.probe()
    result = {
        "path": str(gguf_path),
        "size_gb": round(size_gb, 2),
        "budget_gb": budget_gb,
        "fits": size_gb <= budget_gb,
        "headroom_gb": round(budget_gb - size_gb, 2),
        "vram_total_gb": round(res.vram_total / 1e9, 1),
    }
    if not result["fits"]:
        log.warning(
            "%s is %.2f GB, over the %.1f GB weight budget by %.2f GB. Fallbacks: Q4 K-cache, "
            "24K context, or partial offload.",
            Path(gguf_path).name,
            size_gb,
            budget_gb,
            size_gb - budget_gb,
        )
    return result


def write_modelfile(gguf_path: str | Path, out_path: str | Path, *, name: str) -> Path:
    """Ollama Modelfile with the thinking preset pinned.

    The preset is written into the Modelfile rather than left to a runtime default: running
    thinking mode with presence_penalty=1.5 produces repetition indistinguishable from
    quantisation damage.
    """
    from marlowe.config import THINKING

    out_path = Path(out_path)
    body = f"""# {name}
# Sampling is the Qwen3.8 THINKING preset, pinned deliberately.
# The non-thinking preset uses presence_penalty 1.5, which suppresses exactly the
# repetition this model was built to avoid -- and would mask a regression.
FROM {Path(gguf_path).resolve().as_posix()}

PARAMETER temperature {THINKING.temperature}
PARAMETER top_p {THINKING.top_p}
PARAMETER top_k {THINKING.top_k}
PARAMETER min_p {THINKING.min_p}
PARAMETER presence_penalty {THINKING.presence_penalty}
PARAMETER repeat_penalty {THINKING.repeat_penalty}
PARAMETER num_ctx 32768
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    logutil.event(log, "modelfile written", path=str(out_path))
    return out_path


def summarize_gguf(path: str | Path) -> dict[str, Any]:
    """Human-readable summary of a built GGUF, for the report table."""
    info = read_gguf(path)
    fams = gguf_layer_families(info)
    arch = info.metadata.get("general.architecture", "?")
    return {
        "path": str(path),
        "arch": arch,
        "gb": round(info.file_bytes / 1e9, 2),
        "n_tensors": info.n_tensors,
        "n_blocks": (max(fams) + 1) if fams else 0,
        "n_attention_blocks": sum(1 for f in fams.values() if "attention" in f),
        "n_linear_blocks": sum(1 for f in fams.values() if "linear" in f),
        "quant_version": info.metadata.get("general.file_type"),
    }


def dump_layout_json(path: str | Path, out: str | Path) -> Path:
    """Write the observed GGUF layout, for diffing a converter patch against the stock one."""
    info = read_gguf(path)
    fams = gguf_layer_families(info)
    payload = {
        "gguf": str(path),
        "architecture": info.metadata.get("general.architecture"),
        "blocks": {str(i): sorted(f) for i, f in sorted(fams.items())},
        "metadata_keys": sorted(info.metadata),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out
