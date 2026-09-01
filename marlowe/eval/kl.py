"""KL divergence and top-1 agreement against the bf16 parent, via llama.cpp.

Two-step protocol:

1. ``llama-perplexity --kl-divergence-base ref.kld`` on the bf16 parent writes reference
   logits to disk once.
2. ``llama-perplexity --kl-divergence`` on each candidate reads that file back.

The reference file is large (it stores per-token distributions) and is the reason Stage 2
runs before anything else: every later checkpoint is compared against the same bytes, so the
numbers are commensurable across the whole project.

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


def find_binary(name: str) -> str:
    """Resolve a llama.cpp binary. Raises with actionable instructions if absent."""
    root = os.environ.get(BIN_ENV)
    if root:
        for cand in (Path(root) / name, Path(root) / f"{name}.exe"):
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


def build_reference(
    bf16_gguf: str | Path,
    corpus: str | Path,
    out_file: str | Path,
    *,
    ctx: int = 4096,
    n_gpu_layers: int = 0,
    chunks: int | None = None,
    timeout: int = 24 * 3600,
) -> KLResult:
    """Write the reference logits file from the bf16 parent. Stage 2, step 1.

    ``n_gpu_layers`` defaults to 0: a bf16 27B does not fit in 16 GB and offloading part of
    it is slower than running on CPU for a one-off reference pass that then gets reused
    forever. Raise it only if the parent GGUF is quantised.
    """
    exe = find_binary("llama-perplexity")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-m", str(bf16_gguf),
        "-f", str(corpus),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
        "--kl-divergence-base", str(out_file),
    ]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    with logutil.timed(log, "kl reference", model=str(bf16_gguf), out=str(out_file)):
        rc, text = _run(cmd, timeout)
    if rc != 0 or not out_file.exists():
        raise RuntimeError(
            f"llama-perplexity --kl-divergence-base failed (rc={rc}).\n{text[-3000:]}"
        )
    return KLResult(
        model=str(bf16_gguf),
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
