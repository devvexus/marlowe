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
from marlowe.arch import Layout, is_mtp_tensor, load_config
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

#: Recurrent/linear-attention markers. Unambiguous: only the recurrent path emits these.
GGUF_LINEAR_SUFFIXES = ("ssm_", "linear_attn", "time_mix", "shortconv")

#: Attention markers. Deliberately excludes ``attn_qkv`` and ``attn_gate``: on this
#: architecture llama.cpp emits BOTH for the *linear* path (the DeltaNet q/k/v are fused into
#: one tensor and the output gate is separate), so treating them as attention markers labels
#: every DeltaNet layer "attention+linear". Only the separate projections are conclusive.
GGUF_ATTENTION_SUFFIXES = ("attn_q.", "attn_k.", "attn_v.", "attn_output")


def gguf_layer_families(info: GGUFInfo) -> dict[int, set[str]]:
    """Map block index -> {"attention", "linear"} evidenced by tensor names.

    A fallback. :func:`probe_converter` prefers the explicit ``recurrent_layers`` metadata,
    which is authoritative; this reads the tensor directory for converters too old to write
    that key. A block with any ``ssm_*`` tensor is recurrent regardless of what else it
    carries, because the recurrent path also emits ``attn_``-prefixed names here.
    """
    out: dict[int, set[str]] = {}
    for name in info.tensor_names:
        m = BLK_RE.match(name)
        if not m:
            continue
        i = int(m.group(1))
        tail = m.group(2)
        fams = out.setdefault(i, set())
        if tail.startswith(GGUF_LINEAR_SUFFIXES):
            fams.add("linear")
        elif tail.startswith(GGUF_ATTENTION_SUFFIXES):
            fams.add("attention")
    # ssm_* is conclusive; drop a co-occurring "attention" label from the shared attn_* names.
    return {i: ({"linear"} if "linear" in f else f) for i, f in out.items()}


def gguf_recurrent_mask(info: GGUFInfo) -> list[bool] | None:
    """The explicit per-layer recurrent mask the converter wrote, if it wrote one.

    ``<arch>.attention.recurrent_layers``. This is the authoritative record of the layout the
    loader will build, so the probe reads it in preference to inferring from tensor names.
    """
    for key, val in info.metadata.items():
        if key.endswith("attention.recurrent_layers") and isinstance(val, list):
            return [bool(v) for v in val]
    return None


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
    expected = ["attention" if t != "linear_attention" else "linear" for t in layout.layer_types]

    # Prefer the explicit mask: it is what the loader will actually use, so it answers the
    # question directly instead of inferring it from tensor naming.
    mask = gguf_recurrent_mask(info)
    if mask is not None:
        observed = ["linear" if r else "attention" for r in mask]
        n_blocks = len(mask)
        source = "explicit recurrent_layers array"
    else:
        fams = gguf_layer_families(info)
        observed = [
            (sorted(f)[0] if len(f) == 1 else ("+".join(sorted(f)) or "none"))
            for f in (fams.get(i, set()) for i in range(len(expected)))
        ]
        n_blocks = max(fams) + 1 if fams else 0
        source = "tensor names (no recurrent_layers key)"

    mismatches = [i for i, (e, o) in enumerate(zip(expected, observed)) if e != o]

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
        detail = f"layout preserved across all {n_blocks} blocks, via {source}"

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


REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_PATCH = REPO_ROOT / "vendor" / "0001-qwen35-explicit-recurrent-layers.patch"

#: Search order for a llama.cpp checkout holding convert_hf_to_gguf.py.
_CONVERTER_ROOTS = ("LLAMA_CPP_ROOT",)
_LOCAL_CHECKOUT = REPO_ROOT / ".tools" / "llama.cpp"


def find_converter() -> Path:
    """Locate convert_hf_to_gguf.py.

    Upstream has split the model classes out of the single script into a ``conversion/``
    package, so the converter is no longer one vendorable file. What gets vendored is the
    *patch* (``vendor/0001-*.patch``) plus a pinned upstream commit; this resolves the
    checkout it applies to.
    """
    import os
    import shutil

    candidates: list[Path] = []
    for env in _CONVERTER_ROOTS:
        root = os.environ.get(env)
        if root:
            candidates.append(Path(root) / "convert_hf_to_gguf.py")
    candidates.append(_LOCAL_CHECKOUT / "convert_hf_to_gguf.py")
    # A single-file copy still works for older llama.cpp revisions.
    candidates.append(REPO_ROOT / "vendor" / "convert_hf_to_gguf.py")

    for cand in candidates:
        if cand.exists():
            return cand
    found = shutil.which("convert_hf_to_gguf.py")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Set LLAMA_CPP_ROOT to a llama.cpp checkout, or "
        f"clone one into {_LOCAL_CHECKOUT}. Then apply {VENDOR_PATCH.name} -- without it the "
        "converter mis-types layers on any depth-pruned stack."
    )


def converter_writes_explicit_layout(converter: Path | None = None) -> tuple[bool, str]:
    """Does this converter emit an explicit per-layer recurrent mask? (Rule 3.4, statically)

    Answers the Stage 1 blocking question without running a conversion, downloading weights,
    or building llama.cpp: it checks whether the resolved converter can write
    ``<arch>.attention.recurrent_layers`` at all.

    The stock converter writes only ``full_attention_interval``, defaulting to 4. On a
    uniform stack that is correct and invisible. On a pruned stack the loader regenerates a
    3:1 alternation over the wrong layer count and silently mis-types layers -- including
    attention layers treated as recurrent, which drops their KV cache.
    """
    converter = converter or find_converter()
    root = converter.parent

    def read(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # The writer lives in gguf-py; the call lives in the Qwen model class (or, on older
    # single-file revisions, in the converter script itself).
    writer_files = [root / "gguf-py" / "gguf" / "gguf_writer.py", converter]
    caller_files = [converter, *sorted((root / "conversion").glob("qwen*.py"))]

    has_writer = any("def add_recurrent_layers" in read(p) for p in writer_files)
    has_call = any(
        "self.gguf_writer.add_recurrent_layers(" in read(p) for p in caller_files
    )

    if has_writer and has_call:
        return True, f"{root.name}: writes an explicit recurrent_layers array"
    if not has_writer:
        return False, (
            f"{root} has no add_recurrent_layers writer in gguf-py. It can only emit "
            f"full_attention_interval, which cannot describe a pruned hybrid stack. "
            f"Apply vendor/{VENDOR_PATCH.name}:\n"
            f"  git -C {root} apply {VENDOR_PATCH}"
        )
    return False, (
        f"{root} has the writer but the Qwen converter never calls it. "
        f"Apply vendor/{VENDOR_PATCH.name}."
    )


def checkpoint_has_mtp(hf_dir: str | Path) -> bool:
    """Does this checkpoint still carry an MTP draft head?

    The converter needs telling, because it cannot infer it. Its Qwen mixin reads
    ``mtp_num_hidden_layers`` and treats **0 as "unspecified, discover it from the tensor
    names"** -- Qwen3-Next omits the field entirely -- then asserts that discovery found
    something. A checkpoint that genuinely has zero MTP layers, which is exactly what
    ``surgery --drop-mtp`` produces, trips that assert. ``--no-mtp`` is the intended escape,
    and this is the condition for passing it.
    """
    from marlowe.surgery import load_index

    try:
        weight_map, _ = load_index(hf_dir)
    except FileNotFoundError:
        return True  # cannot tell; let the converter decide
    return any(is_mtp_tensor(name) for name in weight_map)


def convert_to_gguf(
    hf_dir: str | Path,
    out_path: str | Path,
    *,
    outtype: str = "bf16",
    no_mtp: bool | None = None,
    timeout: int = 6 * 3600,
) -> Path:
    """Run convert_hf_to_gguf.py. Returns the output path.

    ``no_mtp`` defaults to **dropping** the MTP head, for two independent reasons.

    The first is intent: ``drop_mtp`` is true for every rung, because the draft head is
    invalid once layers are removed, so no GGUF this pipeline builds should carry one. A
    parent reference that keeps it is not structurally comparable to the children it is the
    reference *for*.

    The second is that keeping it does not work. llama.cpp validates
    ``<arch>.attention.recurrent_layers`` against ``block_count``, and ``block_count``
    counts the MTP block: the 27B parent writes ``block_count=65`` with ``blk.64.nextn.*``,
    while the converter writes 64 recurrent flags from ``num_hidden_layers``. Quantisation
    then fails with "wrong array length; expected 65, got 64" -- after the full 54 GB
    conversion has been written, five minutes in. This was invisible until the first
    MTP-bearing checkpoint was converted, because every earlier conversion was of a pruned
    child that had already had its head removed.

    Auto-detection was the previous default. It could only ever pass ``--no-mtp`` for
    checkpoints that had *no* MTP tensors -- which is the case where the converter would
    otherwise assert -- and so it never fired for the one checkpoint that needed it.
    """
    import sys

    conv = find_converter()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if no_mtp is None:
        no_mtp = True
    cmd = [
        sys.executable,
        str(conv),
        str(hf_dir),
        "--outfile", str(out_path),
        "--outtype", outtype,
    ]
    if no_mtp:
        cmd.append("--no-mtp")
        logutil.event(
            log,
            "dropping the MTP head",
            src=str(hf_dir),
            had_mtp=checkpoint_has_mtp(hf_dir),
        )
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
    """Run llama-quantize with a base type plus per-tensor overrides.

    Refuses IQ-class recipes without an importance matrix rather than letting llama.cpp
    discover it. ``imatrix`` was a parameter here from the beginning and nothing ever passed
    one, so Stage 0 died on its first recipe -- after a 54 GB conversion and seven minutes --
    with "this quantization requires an importance matrix!". Failing at the call site names
    the missing input instead.
    """
    from marlowe.imatrix import recipe_needs_imatrix

    exe = find_binary("llama-quantize")
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    needs = recipe_needs_imatrix(recipe.base_type) or any(
        recipe_needs_imatrix(t) for t in recipe.tensor_types.values()
    )
    if needs and not imatrix:
        raise ValueError(
            f"recipe {recipe.name!r} quantises to {recipe.base_type}, which llama.cpp will "
            f"not produce without an importance matrix. Build one for THIS model first:\n"
            f"  from marlowe.imatrix import imatrix_for\n"
            f"  imatrix_for(<src gguf>, <calibration text>, <cache dir>)\n"
            f"An imatrix is per-model: pruning changes which weights carry activation, so a "
            f"parent's matrix does not describe a child."
        )
    if imatrix and not Path(imatrix).exists():
        raise FileNotFoundError(f"imatrix {imatrix} does not exist")

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


#: The bit-widths the ship gate is evaluated at, coarsest last.
#:
#: The deliverable is a base that *quantises well*, not a checkpoint that happens to fit one
#: card. Under-healed weights carry larger activation outliers and quantise worse, so a model
#: can pass a gate at bf16 or at one convenient bit-width and fall apart at another. Holding
#: at a single width is not evidence of a stable base; holding across the range is.
SHIP_BIT_WIDTHS: tuple[str, ...] = ("q4_K_M", "iq4_xs", "iq3_m")


def ship_recipes(target_gb: float = 10.2) -> list[QuantConfig]:
    """Stage 7 candidates: every gated bit-width, plus the custom mix from Stage 0.

    Which one is *recommended* is decided empirically on measured KL and repetition. Whether
    any of them ships is decided by the gate, which requires all of :data:`SHIP_BIT_WIDTHS`
    to pass -- see :func:`marlowe.report.ship_gate_multi`.
    """
    recipes = [
        QuantConfig(name=w, base_type=w, target_gb=target_gb) for w in SHIP_BIT_WIDTHS
    ]
    recipes.append(
        QuantConfig(
            name="mix_gate_q5",
            base_type="iq3_xxs",
            tensor_types={"q_proj": "q5_K", "linear_attn.*": "q5_K", "ffn_.*": "iq3_xxs"},
            target_gb=target_gb,
        )
    )
    return recipes


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
