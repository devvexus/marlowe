"""KL divergence and top-1 agreement against the parent, via llama.cpp.

Two-step protocol:

1. ``llama-perplexity --kl-divergence-base ref.kld`` on the parent writes reference logits
   to disk once.
2. ``llama-perplexity --kl-divergence`` on each candidate reads that file back.

The reference file is large (it stores per-token distributions) and is the reason Stage 2
runs before anything else: every later checkpoint is compared against the same bytes, so the
numbers are commensurable across the whole project.

The reference model is Q8_0 by default, not bf16 -- see :data:`REFERENCE_OUTTYPE`.

What this catches: general fidelity loss. What it cannot catch: circling. See
:mod:`marlowe.eval.repetition`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil

log = logutil.get("eval.kl")

#: Where to look for llama.cpp binaries, in order: explicit env var, then PATH.
BIN_ENV = "LLAMA_CPP_BIN"


class LlamaCppMissing(RuntimeError):
    """llama.cpp binaries are not available."""


class LlamaCppCpuOnly(RuntimeError):
    """llama.cpp is present but has no GPU backend, and this stage is decode-bound."""


#: Local build locations, tried after $LLAMA_CPP_BIN and before PATH. Mirrors
#: quantize.find_converter so a `.tools/` checkout works with no env var set.
#: GPU builds come first: a CPU-only build is roughly 10x slower to decode and every
#: generation-based metric in this project is decode-bound.
_TOOLS = Path(__file__).resolve().parents[2] / ".tools"
_LOCAL_BIN_DIRS = (
    _TOOLS / "llama-cuda",
    _TOOLS / "llama.cpp" / "build" / "bin" / "Release",
    _TOOLS / "llama.cpp" / "build" / "bin",
)


def find_binary(name: str) -> str:
    """Resolve a llama.cpp binary. Raises with actionable instructions if absent."""
    roots: list[Path] = []
    env_root = os.environ.get(BIN_ENV)
    if env_root:
        roots.append(Path(env_root))
    roots.extend(_LOCAL_BIN_DIRS)
    for root in roots:
        for cand in (root / name, root / f"{name}.exe"):
            if cand.exists():
                return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise LlamaCppMissing(
        f"{name!r} not found. Set {BIN_ENV} to the directory holding the llama.cpp "
        f"binaries, or put them on PATH. Needed binaries: llama-perplexity, "
        f"llama-quantize, llama-bench, llama-server, plus convert_hf_to_gguf.py."
    )


def have_llamacpp() -> bool:
    try:
        find_binary("llama-perplexity")
    except LlamaCppMissing:
        return False
    return True


# ---------------------------------------------------------------------------
# backend capability
# ---------------------------------------------------------------------------

#: Decode throughput on a ~10 GB quantised 27B, used for wall-clock projections.
#: CPU decode is memory-bandwidth bound, which is why the gap is an order of magnitude and
#: not a constant factor.
CPU_DECODE_TOK_S = 4.0
GPU_DECODE_TOK_S = 40.0


@dataclass(frozen=True)
class BackendInfo:
    """What llama.cpp can actually offload to."""

    devices: tuple[str, ...]
    binary: str

    @property
    def has_gpu(self) -> bool:
        return bool(self.devices)

    @property
    def decode_tok_s(self) -> float:
        return GPU_DECODE_TOK_S if self.has_gpu else CPU_DECODE_TOK_S

    def describe(self) -> str:
        if not self.devices:
            return "CPU only (no GPU backend compiled in)"
        return ", ".join(self.devices)


def detect_backend() -> BackendInfo:
    """Ask llama.cpp which devices it was built to use. Empty tuple means CPU-only."""
    exe = find_binary("llama-cli")
    try:
        proc = subprocess.run(
            [exe, "--list-devices"], capture_output=True, text=True, timeout=120, check=False
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("could not query llama.cpp devices: %s", exc)
        return BackendInfo(devices=(), binary=exe)

    devices: list[str] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        # "CUDA0: NVIDIA GeForce RTX 4080 SUPER (16375 MiB, 15061 MiB free)"
        if ":" in line and not line.lower().startswith("available"):
            tag = line.split(":", 1)[0].strip()
            if tag and not tag.lower().startswith("cpu"):
                devices.append(line)
    return BackendInfo(devices=tuple(devices), binary=exe)


def generation_hours(n_completions: int, max_tokens: int, tok_s: float) -> float:
    return n_completions * max_tokens / tok_s / 3600


def gpu_requirement_message(
    stage: str, *, n_variants: int, n_completions: int, max_tokens: int
) -> str:
    """The cost of running a generation-bound stage on CPU, stated up front."""
    tokens = n_variants * n_completions * max_tokens
    cpu_h = generation_hours(n_completions * n_variants, max_tokens, CPU_DECODE_TOK_S)
    gpu_h = generation_hours(n_completions * n_variants, max_tokens, GPU_DECODE_TOK_S)
    return (
        f"{stage} needs a GPU-enabled llama.cpp build, and this one is CPU-only.\n\n"
        f"  The repetition harness generates {n_completions} x {max_tokens} tokens per "
        f"variant x {n_variants} variant(s) = {tokens / 1000:.0f}K tokens.\n"
        f"  CPU decode (~{CPU_DECODE_TOK_S:.0f} tok/s, memory-bandwidth bound on a ~10 GB "
        f"model): ~{cpu_h:.0f} h ({cpu_h / 24:.1f} days)\n"
        f"  GPU decode (~{GPU_DECODE_TOK_S:.0f} tok/s, RTX 4080 Super):            "
        f"~{gpu_h:.0f} h ({gpu_h / 24:.1f} days)\n\n"
        f"  Fix: point {BIN_ENV} at a CUDA build, or drop one in .tools/llama-cuda/.\n"
        f"    b=b10738; curl -LO https://github.com/ggml-org/llama.cpp/releases/download/"
        f"$b/llama-$b-bin-win-cuda-13.3-x64.zip\n"
        f"    curl -LO https://github.com/ggml-org/llama.cpp/releases/download/"
        f"$b/cudart-llama-bin-win-cuda-13.3-x64.zip\n"
        f"  Or pass --allow-cpu-llamacpp to accept the wall-clock above."
    )


def require_gpu_backend(
    stage: str,
    *,
    n_variants: int = 1,
    n_completions: int = 200,
    max_tokens: int = 2048,
    allow_cpu: bool = False,
) -> BackendInfo:
    """Refuse a generation-bound stage on a CPU-only llama.cpp. Raises with the cost."""
    info = detect_backend()
    if info.has_gpu:
        return info
    if allow_cpu:
        hours = generation_hours(n_completions * n_variants, max_tokens, CPU_DECODE_TOK_S)
        log.warning(
            "%s running on a CPU-only llama.cpp build (--allow-cpu-llamacpp): expect ~%.0f h",
            stage,
            hours,
        )
        return info
    raise LlamaCppCpuOnly(
        gpu_requirement_message(
            stage, n_variants=n_variants, n_completions=n_completions, max_tokens=max_tokens
        )
    )


# ---------------------------------------------------------------------------
# output parsing
# ---------------------------------------------------------------------------

#: llama.cpp has renamed these labels more than once. Match on the stable words.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "kl_mean": re.compile(r"Mean\s+KLD\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "kl_median": re.compile(r"Median\s+KLD\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "kl_p99": re.compile(r"99\.0?%\s*KLD\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "kl_max": re.compile(r"Maximum\s+KLD\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "top1_agreement": re.compile(
        r"(?:Mean\s+)?Top-?1\s+(?:agreement|match)\s*[:=]\s*([0-9.eE+-]+)\s*%?", re.I
    ),
    "same_top_p": re.compile(r"Same\s+top\s*p\s*[:=]\s*([0-9.eE+-]+)\s*%?", re.I),
    "ppl": re.compile(r"(?:Final\s+estimate:\s*)?PPL\s*(?:=|:)\s*([0-9.eE+-]+)", re.I),
    "ppl_ratio": re.compile(r"PPL\s+ratio\s*[:=]\s*([0-9.eE+-]+)", re.I),
}


def parse_output(text: str) -> dict[str, float]:
    """Pull metrics out of llama-perplexity stdout/stderr.

    Returns only what it actually found; a missing key means llama.cpp did not print it,
    not that the value is zero.
    """
    out: dict[str, float] = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(text)
        if m:
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            # Agreement percentages are reported as 0-100; normalise to a fraction.
            if key in ("top1_agreement", "same_top_p") and val > 1.0:
                val /= 100.0
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


@dataclass
class KLResult:
    model: str
    corpus: str
    reference: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    returncode: int = 0
    stdout_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    logutil.event(log, "exec", cmd=" ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


#: The reference is built once and every checkpoint in the project is compared against the
#: same bytes, so what matters is that it is fixed, not that it is bf16. Q8_0 is the default
#: because bf16 does not fit: 56 GB against 32 GB of RAM means llama.cpp mmaps and pages from
#: disk for the entire pass. Q8_0 is ~28.6 GB and near-lossless -- its own KL to bf16 is far
#: below the deltas this project measures, and since every number is a relative comparison
#: against this same reference, the substitution costs nothing.
#:
#: This does NOT affect the bf16 repetition baseline, which comes from a hosted endpoint.
REFERENCE_OUTTYPE = "q8_0"


def build_reference(
    reference_gguf: str | Path,
    corpus: str | Path,
    out_file: str | Path,
    *,
    ctx: int = 4096,
    n_gpu_layers: int = 0,
    chunks: int | None = None,
    timeout: int = 24 * 3600,
) -> KLResult:
    """Write the reference logits file from the parent. Stage 2, step 1.

    ``reference_gguf`` should be the Q8_0 parent (see :data:`REFERENCE_OUTTYPE`); bf16 works
    too if the machine has the RAM for it.

    ``n_gpu_layers`` defaults to 0 because neither a 28.6 GB Q8_0 nor a 56 GB bf16 fits in
    16 GB of VRAM, and partial offload is slower than CPU for a one-off pass that is then
    reused forever.
    """
    exe = find_binary("llama-perplexity")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-m", str(reference_gguf),
        "-f", str(corpus),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
        "--kl-divergence-base", str(out_file),
    ]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    with logutil.timed(log, "kl reference", model=str(reference_gguf), out=str(out_file)):
        rc, text = _run(cmd, timeout)
    if rc != 0 or not out_file.exists():
        raise RuntimeError(
            f"llama-perplexity --kl-divergence-base failed (rc={rc}).\n{text[-3000:]}"
        )
    return KLResult(
        model=str(reference_gguf),
        corpus=str(corpus),
        reference=str(out_file),
        metrics=parse_output(text),
        command=cmd,
        returncode=rc,
        stdout_tail=text[-2000:],
    )


def measure(
    gguf: str | Path,
    corpus: str | Path,
    reference: str | Path,
    *,
    ctx: int = 4096,
    n_gpu_layers: int = 999,
    chunks: int | None = None,
    timeout: int = 12 * 3600,
) -> KLResult:
    """Measure a candidate against the reference logits. Stage 2, step 2."""
    exe = find_binary("llama-perplexity")
    reference = Path(reference)
    if not reference.exists():
        raise FileNotFoundError(
            f"reference logits {reference} not found. Run build_reference() on the bf16 "
            f"parent first -- every checkpoint in this project is compared against the "
            f"same reference bytes."
        )
    cmd = [
        exe,
        "-m", str(gguf),
        "-f", str(corpus),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
        "--kl-divergence",
        "--kl-divergence-base", str(reference),
    ]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    with logutil.timed(log, "kl measure", model=str(gguf)):
        rc, text = _run(cmd, timeout)
    if rc != 0:
        raise RuntimeError(f"llama-perplexity --kl-divergence failed (rc={rc}).\n{text[-3000:]}")

    metrics = parse_output(text)
    if "kl_mean" not in metrics:
        raise RuntimeError(
            "could not parse a mean KLD from llama-perplexity output. The label may have "
            f"changed upstream; update marlowe/eval/kl.py::_PATTERNS.\n{text[-2000:]}"
        )
    logutil.event(
        log,
        "kl measured",
        model=Path(gguf).name,
        **{k: round(v, 6) for k, v in metrics.items()},
    )
    return KLResult(
        model=str(gguf),
        corpus=str(corpus),
        reference=str(reference),
        metrics=metrics,
        command=cmd,
        returncode=rc,
        stdout_tail=text[-2000:],
    )


def throughput(
    gguf: str | Path,
    *,
    n_gpu_layers: int = 999,
    ctx: int = 4096,
    timeout: int = 3600,
) -> dict[str, float]:
    """Decode tok/s via llama-bench, for the throughput claim in the report table."""
    exe = find_binary("llama-bench")
    cmd = [exe, "-m", str(gguf), "-ngl", str(n_gpu_layers), "-c", str(ctx), "-o", "csv"]
    rc, text = _run(cmd, timeout)
    if rc != 0:
        raise RuntimeError(f"llama-bench failed (rc={rc}).\n{text[-2000:]}")

    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 2:
            continue
        # llama-bench csv: ..., n_prompt, n_gen, ..., avg_ts, stddev_ts
        try:
            ts = float(parts[-2])
        except ValueError:
            continue
        key = "pp_tok_s" if any(p.startswith("pp") for p in parts) else "tg_tok_s"
        out[key] = ts
    logutil.event(log, "throughput", model=Path(gguf).name, **out)
    return out
