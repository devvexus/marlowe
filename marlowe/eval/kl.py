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

import hashlib
import json
import os
import re
import shutil
import struct
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
# the reference file: how big it is, and what it certifies about itself
# ---------------------------------------------------------------------------

#: What llama-perplexity prints when it starts a pass.
#:
#: Two phrasings, and the binary contains both: ``--kl-divergence-base`` runs the perplexity
#: path and prints "calculating perplexity over N chunks", while ``--kl-divergence`` prints
#: "computing over N chunks". Matching only the second one rejected a perfectly good 77 GB
#: reference after a 19-minute run -- a guard that fails closed still has to be right about
#: what it is guarding.
_RUN_SHAPE = re.compile(
    r"(?:calculating perplexity|computing)\s+over\s+(\d+)\s+chunks,\s*n_ctx\s*=\s*(\d+)",
    re.I,
)

#: ``_logits_`` magic, then n_ctx, n_vocab and n_chunk as little-endian uint32.
_KLD_MAGIC = b"_logits_"


def read_kld_header(path: str | Path) -> dict[str, int]:
    """Read what a .kld file says about itself.

    Better provenance than parsing stdout: llama.cpp writes these three numbers *into the
    artefact*, so they survive a lost log, and they describe the file rather than the run
    that was supposed to produce it. stdout is still parsed, as a cross-check.
    """
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != _KLD_MAGIC:
            raise RuntimeError(
                f"{path} does not start with {_KLD_MAGIC!r}; it is not a .kld reference."
            )
        n_ctx, n_vocab, n_chunk = struct.unpack("<III", f.read(12))
    return {"ctx": n_ctx, "n_vocab": n_vocab, "chunks": n_chunk}


def kld_tokens_per_chunk(ctx: int) -> int:
    """Tokens whose logits get written, per chunk.

    llama.cpp scores only the second half of each window -- the first ``n_ctx//2`` tokens
    are context for the rest and are never predicted -- so this is ``ctx - 1 - ctx//2``,
    not ``ctx``.

    Both counts are real and they differ by 2x, which is the whole trap: at ``-c 8192`` the
    scored count is 4096, numerically identical to the old default context, so a slip that
    uses the wrong one looks like a plausible number arrived at correctly.
    """
    return ctx - 1 - ctx // 2


def kld_chunks_for_corpus(corpus_tokens: int, ctx: int) -> int:
    """Chunks a corpus of ``corpus_tokens`` yields: each chunk *consumes* ``ctx`` tokens.

    Note the asymmetry with :func:`kld_tokens_per_chunk`. A chunk eats ``ctx`` tokens of
    corpus and scores half of them. Dividing a corpus budget by the scored count doubles
    the chunk estimate and doubles the projected disk.
    """
    return corpus_tokens // ctx


def kld_row_width(n_vocab: int) -> int:
    """uint16 values written per scored token: ``2*((n_vocab+1)//2) + 4``.

    The +4 carries the per-token scalars llama.cpp packs beside the quantised log-probs.
    This is read off the file format, so treat it as a prediction: ``build_reference``
    checks it against the bytes that actually land and fails if they disagree.
    """
    return 2 * ((n_vocab + 1) // 2) + 4


#: ``_logits_`` magic plus n_ctx, n_vocab, n_chunk.
KLD_HEADER_BYTES = 8 + 3 * 4


def kld_expected_bytes(n_vocab: int, ctx: int, chunks: int) -> int:
    """Predicted size of a .kld reference file, in bytes."""
    return KLD_HEADER_BYTES + chunks * kld_tokens_per_chunk(ctx) * kld_row_width(n_vocab) * 2


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sidecar_path(kld: str | Path) -> Path:
    """Sidecar beside the reference recording what it actually is."""
    return Path(str(kld) + ".json")


def read_sidecar(kld: str | Path) -> dict[str, Any] | None:
    sc = sidecar_path(kld)
    if not sc.exists():
        return None
    try:
        return json.loads(sc.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


#: Headroom to leave beyond the reference file itself. The pass runs for hours next to
#: quantisation writing 10-25 GB files; filling the volume kills both.
DISK_MARGIN_BYTES = 40 * 1000**3

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


def gguf_vocab_size(gguf: str | Path) -> int:
    """Read n_vocab from GGUF metadata. Needed to predict the reference file's size."""
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf))
    tokens = reader.fields.get("tokenizer.ggml.tokens")
    if tokens is None:
        raise RuntimeError(f"{gguf} has no tokenizer.ggml.tokens; cannot determine n_vocab")
    return len(tokens.data)


def count_corpus_tokens(gguf: str | Path, corpus: str | Path, *, timeout: int = 3600) -> int:
    """Exact token count of the corpus under this model's tokenizer, via llama-tokenize.

    Exact rather than estimated because it feeds the disk guard: a 2x error in the token
    count is a 2x error in the projected reference size, and the volume is 95% full.
    """
    exe = find_binary("llama-tokenize")
    rc, text = _run([exe, "-m", str(gguf), "-f", str(corpus), "--ids"], timeout)
    if rc != 0:
        raise RuntimeError(f"llama-tokenize failed (rc={rc}).\n{text[-2000:]}")
    body = text[text.find("[") : text.rfind("]") + 1]
    ids = re.findall(r"-?\d+", body)
    if not ids:
        raise RuntimeError(
            f"could not parse token ids from llama-tokenize output.\n{text[-2000:]}"
        )
    return len(ids)


def _parse_run_shape(text: str) -> tuple[int | None, int | None]:
    """(chunks, n_ctx) as llama-perplexity reported them, or (None, None)."""
    m = _RUN_SHAPE.search(text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def certify_reference(
    out_file: str | Path,
    reference_gguf: str | Path,
    corpus: str | Path,
    *,
    expected_ctx: int,
    corpus_tokens: int | None = None,
    stdout_text: str = "",
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a finished .kld against what was asked for, and write its sidecar.

    Split out of :func:`build_reference` so a reference that completed can be certified
    without re-running the pass. A 77 GB write costs ~19 minutes; a guard that can only be
    satisfied by redoing that is a guard that will get switched off.

    The .kld header is the authority here, not stdout. llama.cpp writes n_ctx, n_vocab and
    n_chunk *into the artefact*, so they describe the file itself and survive a lost log.
    stdout is a cross-check when available: if it disagrees with the header, something is
    wrong that neither number alone would reveal.
    """
    out_file = Path(out_file)
    header = read_kld_header(out_file)
    n_vocab, actual_ctx, actual_chunks = header["n_vocab"], header["ctx"], header["chunks"]

    printed_chunks, printed_ctx = _parse_run_shape(stdout_text)
    if printed_ctx is not None and (printed_ctx, printed_chunks) != (actual_ctx, actual_chunks):
        raise RuntimeError(
            f"{out_file} header says {actual_chunks} chunks at n_ctx={actual_ctx}, but "
            f"llama-perplexity printed {printed_chunks} at n_ctx={printed_ctx}. The file and "
            f"the run that made it disagree; do not trust either."
        )

    if actual_ctx != expected_ctx:
        raise RuntimeError(
            f"asked for -c {expected_ctx} but {out_file} was built at n_ctx={actual_ctx}. "
            f"llama.cpp adopts a base file's context silently, so every measurement against "
            f"this file would run at {actual_ctx} without saying so. Delete it."
        )

    if corpus_tokens is None:
        corpus_tokens = count_corpus_tokens(reference_gguf, corpus)
    planned = kld_chunks_for_corpus(corpus_tokens, actual_ctx)
    # The stop condition is "what it produced disagrees with what we computed", not a list of
    # values known to be wrong. `planned` is the exact llama-tokenize count divided by ctx,
    # so any disagreement means the tokenizer, the context or the chunking is not what the
    # disk projection and the corpus manifest were built on -- whatever the number happens to
    # be. Enumerating specific wrong values only catches the mistakes already imagined.
    if actual_chunks != planned:
        raise RuntimeError(
            f"{out_file} covers {actual_chunks} chunks; {planned} were predicted from an "
            f"exact llama-tokenize count of {corpus_tokens} tokens divided by -c "
            f"{actual_ctx}. The reference does not cover the corpus the manifest describes; "
            f"delete it and resolve the disagreement before rebuilding."
        )

    size = out_file.stat().st_size
    predicted = kld_expected_bytes(n_vocab, actual_ctx, actual_chunks)
    if abs(size - predicted) > max(1 << 20, predicted // 100):
        raise RuntimeError(
            f"{out_file} is {size / 1e9:.3f} GB but the file format predicts "
            f"{predicted / 1e9:.3f} GB for {actual_chunks} chunks at n_ctx={actual_ctx}, "
            f"n_vocab={n_vocab}. One of those is wrong, and the disk projection that sized "
            f"this run rests on the same arithmetic."
        )

    sidecar = {
        "reference": str(out_file),
        "model": str(reference_gguf),
        "corpus": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "corpus_tokens": corpus_tokens,
        "ctx": actual_ctx,
        "ctx_requested": expected_ctx,
        "chunks": actual_chunks,
        "chunks_predicted": planned,
        "scored_tokens": actual_chunks * kld_tokens_per_chunk(actual_ctx),
        "n_vocab": n_vocab,
        "bytes": size,
        "command": command or [],
        "provenance": "ctx, n_vocab and chunks read from the .kld header; stdout cross-checked",
        "ppl": parse_output(stdout_text).get("ppl"),
    }
    sidecar_path(out_file).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    logutil.event(
        log,
        "kl reference certified",
        ctx=actual_ctx,
        chunks=actual_chunks,
        scored_tokens=sidecar["scored_tokens"],
        gb=round(size / 1e9, 2),
    )
    return sidecar


def build_reference(
    reference_gguf: str | Path,
    corpus: str | Path,
    out_file: str | Path,
    *,
    ctx: int = 8192,
    n_gpu_layers: int = 0,
    chunks: int | None = None,
    n_vocab: int | None = None,
    timeout: int = 24 * 3600,
) -> KLResult:
    """Write the reference logits file from the parent. Stage 2, step 1.

    ``reference_gguf`` should be the Q8_0 parent (see :data:`REFERENCE_OUTTYPE`); bf16 works
    too if the machine has the RAM for it.

    ``n_gpu_layers`` defaults to 0 because neither a 28.6 GB Q8_0 nor a 56 GB bf16 fits in
    16 GB of VRAM, and partial offload is slower than CPU for a one-off pass that is then
    reused forever.

    Three things happen around the run that are not optional:

    * **Disk is checked first.** The file costs ~2.03 GB per chunk at ``-c 8192`` for this
      project's 248,320-token vocabulary. Discovering that at 100% disk, hours in, destroys
      the run and whatever else is writing at the time.
    * **The context is read back from llama-perplexity's own output**, not from ``ctx``.
      llama.cpp does not error when a reference is built at a context other than intended,
      and a 4096 file is byte-indistinguishable from an 8192 one afterwards.
    * **A sidecar is written** recording what the file actually is, so :func:`measure` can
      refuse a mismatched pairing instead of quietly producing commensurable-looking
      numbers that are not commensurable.
    """
    exe = find_binary("llama-perplexity")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if n_vocab is None:
        n_vocab = gguf_vocab_size(reference_gguf)

    corpus_tokens = count_corpus_tokens(reference_gguf, corpus)
    available = kld_chunks_for_corpus(corpus_tokens, ctx)
    planned = min(chunks, available) if chunks else available
    if planned < 1:
        raise ValueError(
            f"{corpus} tokenises to {corpus_tokens} tokens, fewer than one {ctx}-token "
            f"chunk. The reference would be empty."
        )
    per_chunk = kld_expected_bytes(n_vocab, ctx, 1) - KLD_HEADER_BYTES
    projected = kld_expected_bytes(n_vocab, ctx, planned)
    free = shutil.disk_usage(out_file.parent).free
    logutil.event(
        log,
        "kl reference plan",
        corpus_tokens=corpus_tokens,
        chunks=planned,
        ctx=ctx,
        scored_tokens=planned * kld_tokens_per_chunk(ctx),
        gb_per_chunk=round(per_chunk / 1e9, 2),
        projected_gb=round(projected / 1e9, 1),
        free_gb=round(free / 1e9, 1),
    )
    if free < projected + DISK_MARGIN_BYTES:
        raise ResourceWarning(
            f"the reference needs ~{projected / 1e9:.1f} GB ({planned} chunks x "
            f"{per_chunk / 1e9:.2f} GB) plus a {DISK_MARGIN_BYTES / 1e9:.0f} GB working "
            f"margin, but only {free / 1e9:.1f} GB is free on {out_file.parent}. Shrink the "
            f"corpus or pass chunks=N: at -c {ctx} each chunk consumes {ctx} corpus tokens "
            f"and costs {per_chunk / 1e9:.2f} GB."
        )

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

    certify_reference(
        out_file, reference_gguf, corpus,
        expected_ctx=ctx, corpus_tokens=corpus_tokens, stdout_text=text, command=cmd,
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
    ctx: int | None = None,
    n_gpu_layers: int = 999,
    chunks: int | None = None,
    timeout: int = 12 * 3600,
) -> KLResult:
    """Measure a candidate against the reference logits. Stage 2, step 2.

    ``ctx`` defaults to whatever the reference was actually built at, read from its sidecar.
    Passing a different value is an error rather than an override: llama.cpp adopts the base
    file's context without saying so, so a mismatch yields numbers that look fine and are
    comparable to nothing.
    """
    exe = find_binary("llama-perplexity")
    reference = Path(reference)
    if not reference.exists():
        raise FileNotFoundError(
            f"reference logits {reference} not found. Run build_reference() on the bf16 "
            f"parent first -- every checkpoint in this project is compared against the "
            f"same reference bytes."
        )

    meta = read_sidecar(reference)
    if meta is None:
        raise RuntimeError(
            f"{reference} has no sidecar ({sidecar_path(reference).name}), so the context "
            f"and corpus behind it are unknown. llama.cpp adopts the base file's context "
            f"silently, which makes a mismatch invisible in the result. Rebuild the "
            f"reference with build_reference()."
        )
    ref_ctx = int(meta["ctx"])
    if ctx is not None and ctx != ref_ctx:
        raise ValueError(
            f"reference {reference.name} was built at n_ctx={ref_ctx}, but ctx={ctx} was "
            f"requested. llama.cpp would quietly use {ref_ctx} and report a KL that is not "
            f"the one asked for. Pass ctx={ref_ctx}, or rebuild the reference."
        )
    ctx = ref_ctx
    if meta.get("corpus_sha256") and sha256_file(corpus) != meta["corpus_sha256"]:
        raise ValueError(
            f"{corpus} is not the corpus the reference was built from "
            f"({Path(meta['corpus']).name}). KL is only defined over the same tokens."
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
