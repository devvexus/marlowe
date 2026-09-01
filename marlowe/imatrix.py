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
) -> Path:
    """Run llama-imatrix over ``corpus_txt``. Returns the matrix path.

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
    with logutil.timed(log, "imatrix", src=str(src_gguf), ctx=ctx):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"llama-imatrix failed (rc={proc.returncode}).\n"
            f"{(proc.stderr or proc.stdout or '')[-3000:]}"
        )
    logutil.event(
        log, "imatrix built", out=str(out_path),
        mb=round(out_path.stat().st_size / 1e6, 1), ctx=ctx,
    )
    return out_path


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
        logutil.event(log, "imatrix cache hit", path=str(out), fingerprint=fp)
        return out
    build_imatrix(src_gguf, corpus_txt, out, **kw)
    meta.write_text(
        json.dumps(
            {
                "source_gguf": str(src_gguf),
                "fingerprint": fp,
                "corpus": str(corpus_txt),
                "ctx": kw.get("ctx", IMATRIX_CTX),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
