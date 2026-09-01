"""Stage 5: offline top-K logprob cache from the 4-bit teacher.

Teacher and student cannot both be resident on a 16 GB card, so distillation is done against
a cache written ahead of time rather than a live teacher. Forward-only at roughly 2500 tok/s,
50M tokens, top-K=16 -- about 5 GB on disk.

Storage is 96 bytes per token: 16 uint32 indices plus 16 float16 logprobs. Input ids are
stored alongside so the healing loop never re-tokenises and can never drift out of alignment
with the distribution it is matching.

Logprobs are stored *unnormalised over the top-K* -- they are true logprobs from the full
softmax. That preserves the tail mass ``1 - sum(exp(topk))``, which the healing loss reports
as a diagnostic: if the teacher's top-16 captures only a thin slice of the distribution, a
top-K objective is a poor proxy and the K should go up.

A 4-bit teacher inherits the quantisation error of its own weights. That is a real cost and
it is recorded in the manifest; it beats no teacher by a wide margin.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil, preflight
from marlowe.config import TeacherConfig

log = logutil.get("teacher")

BYTES_PER_TOKEN_PER_K = 6  # uint32 index + float16 logprob


@dataclass
class ShardMeta:
    index: int
    path: str
    n_sequences: int
    n_tokens: int
    seq_len: int
    top_k: int
    mean_captured_mass: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CacheIndex:
    """The manifest for a teacher cache directory. Resumption reads this."""

    teacher: str
    corpus: str
    seq_len: int
    top_k: int
    quantization: str
    shards: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return sum(int(s["n_tokens"]) for s in self.shards)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> CacheIndex | None:
        p = Path(path)
        if not p.exists():
            return None
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def estimate_size(tokens: int, top_k: int) -> int:
    """Bytes on disk for a cache of this size, including the stored input ids."""
    return tokens * (top_k * BYTES_PER_TOKEN_PER_K + 4)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def iter_sequences(
    tokenizer: Any, corpus_path: str | Path, seq_len: int, max_tokens: int
) -> Iterator[list[int]]:
    """Pack a JSONL corpus into fixed-length token sequences.

    In-domain data where available. At this compression ratio there is no obligation to
    recover the parent's general capability, only the part actually used.
    """
    emitted = 0
    buf: list[int] = []
    with Path(corpus_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"] if isinstance(obj, dict) else str(obj)
            buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
            while len(buf) >= seq_len:
                yield buf[:seq_len]
                buf = buf[seq_len:]
                emitted += seq_len
                if emitted >= max_tokens:
                    return


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def build_cache(
    teacher_path: str,
    out_dir: str | Path,
    cfg: TeacherConfig,
    *,
    max_gpu_gb: float | None = None,
    resume: bool = True,
) -> CacheIndex:
    """Write the top-K cache. Resumable at shard granularity."""
    import numpy as np
    import torch

    from marlowe.arch import Layout, load_config
    from marlowe.score import _hidden_from, build_harness, load_4bit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"

    need_bytes = estimate_size(cfg.tokens, cfg.top_k)
    preflight.require(
        disk_gb=need_bytes / 1e9 * 1.1, vram_gb=12.0, path=out_dir, what="stage5-teacher"
    )

    existing = CacheIndex.load(index_path) if resume else None
    if existing is not None:
        stale = [s for s in existing.shards if not (out_dir / Path(s["path"]).name).exists()]
        if stale:
            log.warning("dropping %d cache shards missing from disk", len(stale))
            existing.shards = [s for s in existing.shards if s not in stale]
        if existing.top_k != cfg.top_k or existing.seq_len != cfg.seq_len:
            raise ValueError(
                f"existing cache at {out_dir} has top_k={existing.top_k} seq_len="
                f"{existing.seq_len}, config wants {cfg.top_k}/{cfg.seq_len}. Point at a "
                f"fresh directory or delete the old cache."
            )
        if existing.n_tokens >= cfg.tokens:
            logutil.event(log, "cache already complete", tokens=existing.n_tokens)
            return existing
        logutil.event(log, "resuming cache", have=existing.n_tokens, want=cfg.tokens)

    idx = existing or CacheIndex(
        teacher=teacher_path,
        corpus=str(cfg.corpus_path),
        seq_len=cfg.seq_len,
        top_k=cfg.top_k,
        quantization="nf4-double",
    )

    layout = Layout.from_config(load_config(teacher_path))
    with logutil.timed(log, "load 4-bit teacher", path=teacher_path):
        model, tok = load_4bit(teacher_path, max_gpu_gb=max_gpu_gb)
    h = build_harness(model, layout.n_layers, cfg.seq_len)

    already = idx.n_tokens
    shard_seqs = max(1, cfg.shard_tokens // cfg.seq_len)
    buf_ids: list[list[int]] = []
    buf_topk: list[Any] = []
    buf_lp: list[Any] = []
    masses: list[float] = []
    shard_i = len(idx.shards)

    def flush() -> None:
        nonlocal shard_i, buf_ids, buf_topk, buf_lp, masses
        if not buf_ids:
            return
        name = f"shard-{shard_i:05d}.npz"
        np.savez(
            out_dir / name,
            input_ids=np.asarray(buf_ids, dtype=np.int32),
            topk_idx=np.stack(buf_topk).astype(np.uint32),
            topk_logprob=np.stack(buf_lp).astype(np.float16),
        )
        meta = ShardMeta(
            index=shard_i,
            path=name,
            n_sequences=len(buf_ids),
            n_tokens=len(buf_ids) * cfg.seq_len,
            seq_len=cfg.seq_len,
            top_k=cfg.top_k,
            mean_captured_mass=float(sum(masses) / max(len(masses), 1)),
        )
        idx.shards.append(meta.as_dict())
        idx.save(index_path)  # after every shard: a crash loses at most one shard
        logutil.event(
            log,
            "shard written",
            shard=shard_i,
            seqs=len(buf_ids),
            tokens=meta.n_tokens,
            captured_mass=round(meta.mean_captured_mass, 4),
            total_tokens=idx.n_tokens,
        )
        shard_i += 1
        buf_ids, buf_topk, buf_lp, masses = [], [], [], []

    skip_tokens = already
    produced = already
    with logutil.timed(log, "teacher cache", target_tokens=cfg.tokens):
        for seq in iter_sequences(tok, cfg.corpus_path, cfg.seq_len, cfg.tokens):
            # Fast-forward past sequences already cached, without running the model.
            if skip_tokens >= cfg.seq_len:
                skip_tokens -= cfg.seq_len
                continue
            ids = torch.tensor([seq], device=h.device)
            with torch.no_grad():
                hidden = _hidden_from(h.decoder(input_ids=ids, use_cache=False))
                logits = h.lm_head(hidden).float()
                lp = torch.log_softmax(logits, dim=-1)[0]
                top = torch.topk(lp, cfg.top_k, dim=-1)
                masses.append(float(top.values.exp().sum(-1).mean().item()))
                buf_ids.append(seq)
                buf_topk.append(top.indices.cpu().numpy())
                buf_lp.append(top.values.cpu().numpy())
                del hidden, logits, lp, top

            produced += cfg.seq_len
            if len(buf_ids) >= shard_seqs:
                flush()
            if produced >= cfg.tokens:
                break
        flush()

    del model
    torch.cuda.empty_cache()
    logutil.event(
        log,
        "teacher cache complete",
        tokens=idx.n_tokens,
        shards=len(idx.shards),
        gb=round(sum((out_dir / s["path"]).stat().st_size for s in idx.shards) / 1e9, 2),
    )
    return idx


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def iter_cache(
    cache_dir: str | Path, *, shuffle_seed: int | None = None
) -> Iterator[tuple[Any, Any, Any]]:
    """Yield (input_ids, topk_idx, topk_logprob) arrays, one shard at a time.

    Shard-level shuffling only: within-shard order is preserved so a resumed run can seek by
    (shard, offset) without holding the whole cache in memory.
    """
    import numpy as np

    cache_dir = Path(cache_dir)
    idx = CacheIndex.load(cache_dir / "index.json")
    if idx is None:
        raise FileNotFoundError(f"no teacher cache index at {cache_dir}/index.json")

    order: Sequence[dict[str, Any]] = idx.shards
    if shuffle_seed is not None:
        import random

        order = list(idx.shards)
        random.Random(shuffle_seed).shuffle(order)

    for shard in order:
        with np.load(cache_dir / shard["path"]) as z:
            yield z["input_ids"], z["topk_idx"], z["topk_logprob"]
