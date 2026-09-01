"""Importance matrices for IQ-class quantisation.

llama.cpp refuses IQ2/IQ3 quantisation outright without one -- "this quantization requires
an importance matrix!" -- which is where Stage 0 died: ``stage0_recipes()`` leads with
``iq3_xxs``, and ``SHIP_BIT_WIDTHS`` includes ``iq3_m``, so the ship gate needs one too.
``quantize()` had an ``imatrix`` parameter from the beginning and nothing ever passed it.

**One per model, not one per project.** An importance matrix is a record of which weights
carried activation magnitude while the model ran on the calibration text. Prune 12 layers and
the surviving layers carry different activations, so the parent's matrix does not describe a
child. The parent's matrix is for Stage 0's control curve; each healed child gets its own for
Stage 7. Reusing one across models would produce a quantisation optimised for a model that no
longer exists -- which loads, runs, and is quietly worse.

Cached by a hash of the source GGUF, so a rebuilt or re-healed model gets a fresh matrix
automatically rather than silently inheriting a stale one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from marlowe import logutil
from marlowe.eval.kl import find_binary

log = logutil.get("imatrix")

#: Context length for the imatrix pass.
#:
#: llama.cpp defaults to 512. That is wrong for this architecture: 16 of the 52 layers are
#: full attention, and their importance is context-dependent in a way a 512-token window
#: cannot show -- a short window under-weights exactly the layers whose contribution is
#: long-range. The recurrent layers have the same problem in a different form, since their
#: state accumulates along the sequence. The pass is cheap either way.
IMATRIX_CTX = 4096

#: Quantisation types llama.cpp will not produce without an importance matrix.
#:
#: Kept as a prefix set rather than an exact list: llama.cpp gates on the IQ family, and a
#: new IQ type appearing upstream should fail closed here rather than be attempted and
#: rejected after a full quantisation pass.
IMATRIX_REQUIRED_PREFIXES: tuple[str, ...] = ("iq1", "iq2", "iq3")

#: Chunks below which a matrix is treated as under-sampled and says so.
#:
#: llama.cpp's own examples land around 100-300 chunks, so this is the low end of normal
#: rather than a hard floor. It exists because an interrupted pass leaves a file that is
#: structurally perfect -- every tensor present, every block covered -- and silently
#: under-sampled: a kernel panic stopped one at 120 of ~420 chunks and nothing about the
#: artifact said so. Stage 0 is the control curve every later "healing or quantisation?"
#: judgement is measured against, and that is the worst place to accept an uncontrolled
#: variable.
MIN_GOOD_CHUNKS = 200


def recipe_needs_imatrix(base_type: str) -> bool:
    """Does this quantisation type require an importance matrix?"""
    t = base_type.lower().replace("-", "_")
    return any(t.startswith(p) for p in IMATRIX_REQUIRED_PREFIXES)


def gguf_fingerprint(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Cheap, stable identity for a GGUF: size plus head and tail bytes.

    Hashing 50 GB to decide whether to reuse a cached matrix would cost more than the pass
    it saves. Size plus both ends distinguishes every model this pipeline produces -- rungs
    differ in size, and two checkpoints of the same shape differ in their tensor data, which
    the tail captures.
    """
    p = Path(path)
    h = hashlib.sha256()
    size = p.stat().st_size
    h.update(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def fitting_gpu_layers(src_gguf: str | Path, n_blocks: int, *, headroom_gb: float = 2.0) -> int:
    """How many layers of ``src_gguf`` fit in free VRAM.

    ``-ngl 999`` is the usual incantation and it is wrong here. The imatrix source is a Q8_0
    of the 27B parent -- ~28.6 GB against 16 GB of VRAM -- so asking for every layer fails.
    Offloading what fits and leaving the rest on CPU is the difference between a pass that
    takes an hour and one that does not run.
    """
    import torch

    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info(0)
    budget = free / 1e9 - headroom_gb
    per_layer = (Path(src_gguf).stat().st_size / 1e9) / max(n_blocks, 1)
    return max(0, min(n_blocks, int(budget / per_layer)))


def build_imatrix(
    src_gguf: str | Path,
    corpus_txt: str | Path,
    out_path: str | Path,
    *,
    ctx: int = IMATRIX_CTX,
    n_gpu_layers: int = 0,
    timeout: int = 6 * 3600,
    merge_from: str | Path | None = None,
) -> Path:
    """Run llama-imatrix over ``corpus_txt``. Returns the matrix path.

    ``merge_from`` folds an existing matrix in via ``--in-file``, so segments accumulate.

    ``corpus_txt`` is plain text, not JSONL -- llama-imatrix reads raw text and chunks it
    internally, so the 32K-token sequence packing Stage 3 needs is irrelevant here. The same
    calibration corpus serves both; only the serialisation differs.
    """
    exe = find_binary("llama-imatrix")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-m", str(src_gguf),
        "-f", str(corpus_txt),
        "-o", str(out_path),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
    ]
    if merge_from is not None:
        cmd += ["--in-file", str(merge_from)]
    # Stream to a log rather than capturing.
    #
    # capture_output holds everything until the process exits, so a pass that takes hours
    # shows nothing at all while it runs -- and if it is killed (this machine kernel-panicked
    # during one), the output is lost with it. llama-imatrix prints per-chunk progress, which
    # is the only way to know whether a long pass is advancing or wedged.
    progress = out_path.with_suffix(".log")
    with (
        logutil.timed(log, "imatrix", src=str(src_gguf), ctx=ctx, progress=str(progress)),
        progress.open("w", encoding="utf-8", errors="replace") as fh,
    ):
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
    if proc.returncode != 0 or not out_path.exists():
        tail = ""
        with contextlib.suppress(OSError):
            tail = progress.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"llama-imatrix failed (rc={proc.returncode}).\n{tail}")
    logutil.event(
        log, "imatrix built", out=str(out_path),
        mb=round(out_path.stat().st_size / 1e6, 1), ctx=ctx,
    )
    return out_path


def split_corpus(corpus_txt: str | Path, out_dir: str | Path, segments: int) -> list[Path]:
    """Split a text corpus into ``segments`` roughly equal parts, on line boundaries.

    Splitting is what makes an interrupted pass recoverable. ``--chunks`` cannot help: it
    caps how many chunks are read *from the start of the file*, so a second run with it
    re-reads the same opening text rather than advancing. Separate files advance.
    """
    corpus_txt = Path(corpus_txt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = corpus_txt.read_text(encoding="utf-8", errors="replace")
    target = len(data) // max(segments, 1)
    parts: list[Path] = []
    start = 0
    for i in range(segments):
        if i == segments - 1:
            end = len(data)
        else:
            end = data.find("\n", start + target)
            end = len(data) if end == -1 else end + 1
        p = out_dir / f"{corpus_txt.stem}.part{i:02d}.txt"
        p.write_text(data[start:end], encoding="utf-8")
        parts.append(p)
        start = end
        if start >= len(data):
            break
    return parts


def build_imatrix_segmented(
    src_gguf: str | Path,
    corpus_txt: str | Path,
    out_path: str | Path,
    *,
    segments: int = 4,
    ctx: int = IMATRIX_CTX,
    n_gpu_layers: int = 0,
    timeout: int = 6 * 3600,
) -> Path:
    """Build an imatrix in resumable segments, merging each into the last.

    A single pass over the full corpus is one long bet: this machine kernel-panicked
    (bugcheck 0x139) partway through one, and the ~40 minutes it had done were only
    salvageable because llama-imatrix happens to checkpoint. Segmenting makes that structural
    -- each finished segment is banked on disk, and a rerun skips what is already done.

    Statistics merge additively, so chaining ``--in-file`` across segments gives the same
    relative importance as one pass over the concatenation.
    """
    src_gguf, out_path = Path(src_gguf), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f"{out_path.stem}.segments"
    parts = split_corpus(corpus_txt, work, segments)

    previous: Path | None = None
    for i, part in enumerate(parts):
        seg_out = work / f"seg{i:02d}.gguf"
        if seg_out.exists():
            info = describe_imatrix(seg_out)
            logutil.event(
                log, "imatrix segment cached", segment=i, of=len(parts),
                chunks=info.get("imatrix.chunk_count"),
            )
            previous = seg_out
            continue
        logutil.event(log, "imatrix segment", segment=i, of=len(parts), corpus=str(part))
        build_imatrix(
            src_gguf, part, seg_out, ctx=ctx, n_gpu_layers=n_gpu_layers,
            timeout=timeout, merge_from=previous,
        )
        previous = seg_out

    assert previous is not None
    # The last segment carries every earlier one merged into it.
    out_path.write_bytes(previous.read_bytes())
    info = describe_imatrix(out_path)
    logutil.event(
        log, "imatrix complete", segments=len(parts),
        chunks=info.get("imatrix.chunk_count"), tokens_seen=info.get("tokens_seen"),
    )
    return out_path


def describe_imatrix(path: str | Path) -> dict[str, Any]:
    """Read an imatrix's own provenance: how many chunks it actually saw, and of what.

    Derived from the file, not from a sidecar. llama-imatrix writes a complete, valid GGUF at
    every periodic save, so an interrupted run leaves a usable matrix that is indistinguishable
    from a finished one by size or structure -- only ``imatrix.chunk_count`` says how much of
    the corpus it covers. A sidecar written after the fact cannot describe a run that never
    reached its own last line: this project's first imatrix was interrupted by a kernel panic
    at 120 of ~420 chunks and left no sidecar at all.
    """
    import sys as _sys

    _sys.path.insert(0, ".tools/llama.cpp/gguf-py")
    try:
        from gguf import GGUFReader

        r = GGUFReader(str(path))
    except Exception as exc:  # noqa: BLE001 - unreadable provenance is a fact to record
        # Say so rather than raise. Provenance we cannot read is not the same as provenance
        # that is fine, and it is also not a reason to abort a quantisation -- but it must
        # never be reported as a clean matrix.
        return {"path": str(path), "unreadable": f"{type(exc).__name__}: {str(exc)[:120]}"}
    out: dict[str, Any] = {"path": str(path), "tensors": len(r.tensors)}
    for f in r.fields.values():
        if f.name.startswith("imatrix."):
            v = f.contents()
            out[f.name] = list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
    chunks = out.get("imatrix.chunk_count")
    size = out.get("imatrix.chunk_size")
    if isinstance(chunks, int) and isinstance(size, int):
        out["tokens_seen"] = chunks * size
    return out


def warn_if_undersampled(info: dict[str, Any], *, min_chunks: int = MIN_GOOD_CHUNKS) -> bool:
    """Say so when a matrix covers less of its corpus than it should. Returns True if fine."""
    chunks = info.get("imatrix.chunk_count")
    if info.get("unreadable"):
        log.warning(
            "imatrix provenance is unreadable (%s); cannot confirm how much of the corpus "
            "it covers", info["unreadable"],
        )
        return False
    if not isinstance(chunks, int):
        log.warning("imatrix reports no chunk_count; cannot confirm its coverage")
        return False
    if chunks < min_chunks:
        log.warning(
            "imatrix saw only %d chunks (%s tokens), below the %d that reads as fully "
            "sampled. It is structurally complete -- every tensor and block is present -- so "
            "nothing will fail; the quantisation is simply tuned on thinner statistics. "
            "Rebuild it if this feeds Stage 0's control curve, which every later "
            "healing-vs-quantisation judgement is measured against.",
            chunks, f"{info.get('tokens_seen', 0):,}", min_chunks,
        )
        return False
    return True


def imatrix_for(
    src_gguf: str | Path,
    corpus_txt: str | Path,
    cache_dir: str | Path,
    **kw: Any,
) -> Path:
    """The importance matrix for exactly this model, building it if absent.

    Keyed on the source GGUF's fingerprint, so a re-converted or re-healed model never
    inherits the previous one's matrix.
    """
    cache_dir = Path(cache_dir)
    fp = gguf_fingerprint(src_gguf)
    out = cache_dir / f"imatrix-{fp}.dat"
    meta = cache_dir / f"imatrix-{fp}.json"
    if out.exists():
        # Report what it actually covers, not merely that it exists. An interrupted run
        # leaves a complete, valid, PARTIAL matrix, and nothing about the file's size or
        # structure distinguishes it from a finished one.
        info = describe_imatrix(out)
        logutil.event(
            log, "imatrix cache hit", path=str(out), fingerprint=fp,
            chunks=info.get("imatrix.chunk_count"), tokens_seen=info.get("tokens_seen"),
        )
        warn_if_undersampled(info)
        _write_meta(meta, src_gguf, fp, corpus_txt, kw, info)
        return out
    build_imatrix(src_gguf, corpus_txt, out, **kw)
    _write_meta(meta, src_gguf, fp, corpus_txt, kw, describe_imatrix(out))
    return out


def _write_meta(
    meta: Path, src_gguf: str | Path, fp: str, corpus: str | Path,
    kw: dict[str, Any], info: dict[str, Any],
) -> None:
    meta.write_text(
        json.dumps(
            {
                "source_gguf": str(src_gguf),
                "fingerprint": fp,
                "corpus": str(corpus),
                "ctx": kw.get("ctx", IMATRIX_CTX),
                "measured": info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
