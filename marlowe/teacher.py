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
import os
import time
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
    memory_plan_path: str | Path | None = None,
) -> CacheIndex:
    """Write the top-K cache. Resumable at shard granularity.

    ``memory_plan_path`` must name a plan carrying a **soak-confirmed** selection. The cache
    is built at one fixed sequence length whose distributions are position-aligned, so the
    length has to be right before the first shard is written, not discovered afterwards.

    The memory search alone is not enough to authorise that. It probes 8 steps, and
    rank32-seq768 measured 0.09 GB paged over 8 steps against 11.5 GB over 500 -- growth
    torch could not see at all, reporting flat fragmentation and 0.53 GB of headroom
    throughout. A cache built from a candidate is hours spent on a number nobody stood
    behind, and it is discovered at the start of Stage 6, when it is most expensive.
    """
    if memory_plan_path is not None:
        from marlowe.heal import require_selected

        selected = require_selected(memory_plan_path)
        logutil.event(log, "cache building from soak-selected rung", **selected)
    import numpy as np
    import torch

    from marlowe.arch import Layout, load_config
    from marlowe.score import _hidden_from, build_harness, load_4bit


    def _load_with_retry(path: str, *, max_gpu_gb=None, attempts: int = 3):
        """Load the teacher, retrying a hard crash.

        Measured on this machine: three identical fresh-process loads of the 27B parent gave
        one segfault at tensor 754 of 851 and two clean completions. The fault is in
        transformers' _materialize_copy -- the mmap read of a safetensors slice -- and it
        moves between runs, so it is environmental rather than a corrupt shard. A contributing
        cause was found and removed (a background process leaking 4.9M handles, 93% of the
        machine's total), but the load is not proven reliable, and a segfault at hour thirty
        of a cache build is not something to discover without a retry in place.

        Load-only. Once past it the forward path is stable.
        """
        import json as _json

        # A checkpoint that is already NF4 on disk must be loaded WITHOUT handing
        # from_pretrained another BitsAndBytesConfig. Passing one makes transformers treat it
        # as on-the-fly quantisation again -- the exact path that segfaults reading a 45 GB
        # bf16 mmap -- so the pre-quantisation would buy nothing while appearing to.
        cfg_path = Path(path) / "config.json"
        pre_quantized = False
        if cfg_path.exists():
            try:
                pre_quantized = bool(
                    _json.loads(cfg_path.read_text(encoding="utf-8")).get("quantization_config")
                )
            except (OSError, ValueError):
                pre_quantized = False

        def _embed_name() -> str | None:
            """The embedding module path, taken from the model's own module tree.

            Deriving it from tensor names is wrong: this checkpoint stores
            ``model.language_model.embed_tokens.weight`` while the built module is
            ``model.embed_tokens`` -- transformers strips the wrapper. A device_map key that
            matches no submodule is not an error, it is a warning, so the offload silently
            does not happen and the model loads entirely onto the card. Measured: 17.17 GB of
            a 17.17 GB card, leaving nothing for the forward pass.

            Built on the meta device, so this costs no memory and touches no GPU.
            """
            import torch as _torch
            from transformers import AutoConfig, AutoModelForCausalLM

            try:
                cfg_ = AutoConfig.from_pretrained(path, trust_remote_code=True)
                with _torch.device("meta"):
                    skeleton = AutoModelForCausalLM.from_config(cfg_, trust_remote_code=True)
                for name, _ in skeleton.named_modules():
                    if name.endswith("embed_tokens"):
                        return name
            except Exception as exc:  # noqa: BLE001
                log.warning("could not determine the embedding module name: %s", exc)
            return None

        def _load():
            if not pre_quantized:
                return load_4bit(path, max_gpu_gb=max_gpu_gb)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # The embedding is bf16 even in an NF4 checkpoint -- it is not a Linear, so
            # nothing quantises it -- and at 248320 x 5120 that is 2.54 GB. Putting the whole
            # checkpoint on the card OOMs; load_4bit has always offloaded this and the
            # pre-quantized path has to as well.
            embed = _embed_name()
            device_map: dict[str, Any] = {"": 0}
            if embed:
                device_map[embed] = "cpu"
            kw: dict[str, Any] = {"trust_remote_code": True, "device_map": device_map}
            if max_gpu_gb:
                kw["device_map"] = "auto"
                kw["max_memory"] = {0: f"{max_gpu_gb:.1f}GiB", "cpu": "24GiB"}
            logutil.event(log, "pre-quantized load", embed_offloaded=embed)
            m = AutoModelForCausalLM.from_pretrained(path, **kw)

            # Naming the embedding in device_map attaches accelerate's AlignDevicesHook, and
            # that hook copies the WHOLE 2.54 GB table onto the device before every forward.
            # The offload then saves nothing: the memory is paid anyway, as a per-forward
            # transient. Measured on this teacher: 2.764 GB of driver spill above the idle
            # floor, against a 2.543 GB table -- and ~0.21 s of PCIe per 1.60 s sequence,
            # about 13% of throughput.
            #
            # CpuGatherEmbedding does the lookup on the host and moves only
            # [batch, seq, hidden] -- 7.9 MB at seq 768 instead of 2540 MB. This fix was
            # applied to heal.load_student this morning and not here, because this load path
            # was written afterwards.
            # OFF by default. The gather removes accelerate's 2.54 GB per-forward staging,
            # which is real, but it also forces a GPU->CPU->GPU round trip at the start of
            # every forward and measured 337.5 tok/s against the hook path's 479 -- a 30%
            # loss. The staging is evidently overlapped with compute; the gather's sync is
            # not. Set MARLOWE_TEACHER_CPU_GATHER=1 to re-enable.
            if embed and os.environ.get("MARLOWE_TEACHER_CPU_GATHER", "0") == "1":
                from marlowe.heal import use_cpu_gather_embedding

                info = use_cpu_gather_embedding(m, embed)
                logutil.event(log, "teacher cpu-gather embedding", **info)
                # Imported here, not taken from the enclosing scope: the except block below
                # does `import torch`, which makes `torch` a local of _load_with_retry, so a
                # free-variable read from this nested function raises NameError before it is
                # bound. That failed every first attempt and left the dead model on the card.
                import torch as _t

                _t.cuda.empty_cache()
            return m, AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        logutil.event(log, "teacher load", path=str(path), pre_quantized=pre_quantized)
        # The failure is recorded as text, not as the exception object. A retained exception
        # holds its __traceback__, which holds the frame of _load, which holds the partially
        # loaded model -- so the next attempt runs with the previous attempt's teacher still
        # resident. Measured: two teachers on a 16.4 GB card and 15.3 GB spilled to host.
        last_msg: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                loaded = _load()
                if attempt > 1:
                    logutil.event(log, "teacher loaded after retry", attempt=attempt)
                return loaded
            except BaseException as exc:  # noqa: BLE001 - a hard crash is the case being handled
                last_msg = f"{type(exc).__name__}: {exc}"
                exc.__traceback__ = None  # drop the frame chain holding the failed model
                del exc
                log.warning(
                    "teacher load attempt %d/%d failed: %s", attempt, attempts, last_msg,
                )
                import gc

                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(20)  # let driver host backing drain before trying again
        raise RuntimeError(
            f"teacher load failed {attempts} times. The last error was "
            f"{last_msg}. A SIGSEGV kills the process outright and never "
            f"reaches this handler -- if the run died silently, check the exit code (139) and "
            f"see marlowe.supervisor, which retries at the process level."
        )

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
        model, tok = _load_with_retry(teacher_path, max_gpu_gb=max_gpu_gb)
    h = build_harness(model, layout.n_layers, cfg.seq_len)

    already = idx.n_tokens
    shard_seqs = max(1, cfg.shard_tokens // cfg.seq_len)
    # Preallocated, in the ON-DISK dtypes. The previous version accumulated Python lists of
    # int64 indices and float32 logprobs and cast them at flush() -- twice the width in
    # memory, and then np.stack plus .astype each allocated another full copy while the
    # lists were still live. Measured on a 5M-token shard: ~4 GB of buffers with a ~4 GB
    # spike at the boundary, on a 31 GB machine that was already at 0.41 GB available.
    #
    # Writing straight into a preallocated array in the final dtype removes both: the
    # footprint is constant, known before the run starts, and flush() is a slice.
    buf_ids = np.empty((shard_seqs, cfg.seq_len), dtype=np.int32)
    buf_topk = np.empty((shard_seqs, cfg.seq_len, cfg.top_k), dtype=np.uint32)
    buf_lp = np.empty((shard_seqs, cfg.seq_len, cfg.top_k), dtype=np.float16)
    n_buf = 0
    masses: list[float] = []
    shard_i = len(idx.shards)
    logutil.event(
        log,
        "shard buffers preallocated",
        shard_seqs=shard_seqs,
        bytes=int(buf_ids.nbytes + buf_topk.nbytes + buf_lp.nbytes),
        gb=round((buf_ids.nbytes + buf_topk.nbytes + buf_lp.nbytes) / 1e9, 3),
    )

    def flush() -> None:
        nonlocal shard_i, n_buf, masses
        if n_buf == 0:
            return
        name = f"shard-{shard_i:05d}.npz"
        # Slices of the preallocated arrays: no stack, no cast, no second copy.
        np.savez(
            out_dir / name,
            input_ids=buf_ids[:n_buf],
            topk_idx=buf_topk[:n_buf],
            topk_logprob=buf_lp[:n_buf],
        )
        meta = ShardMeta(
            index=shard_i,
            path=name,
            n_sequences=n_buf,
            n_tokens=n_buf * cfg.seq_len,
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
            seqs=n_buf,
            tokens=meta.n_tokens,
            captured_mass=round(meta.mean_captured_mass, 4),
            total_tokens=idx.n_tokens,
        )
        shard_i += 1
        n_buf = 0
        masses = []

    skip_tokens = already
    produced = already
    _t_start = time.time()
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
                # Cast on the GPU, before the transfer, straight into the preallocated row.
                # int32 is safe: the vocabulary is 248,320, far under 2**31.
                buf_ids[n_buf] = seq
                buf_topk[n_buf] = top.indices.to(torch.int32).cpu().numpy()
                buf_lp[n_buf] = top.values.to(torch.float16).cpu().numpy()
                n_buf += 1
                del hidden, logits, lp, top

            produced += cfg.seq_len
            # Progress every 256 sequences, not every shard. A shard is 5M tokens -- roughly
            # half an hour -- so shard-granular logging leaves a multi-hour job with no
            # observable rate until it is already committed, and no way to answer "will this
            # finish in time" before it matters.
            if (produced // cfg.seq_len) % 256 == 0:
                el = max(time.time() - _t_start, 1e-9)
                done = produced - already
                rate = done / el
                remaining = max(cfg.tokens - produced, 0)
                logutil.event(
                    log,
                    "cache progress",
                    tokens=produced,
                    target=cfg.tokens,
                    pct=round(100 * produced / max(cfg.tokens, 1), 2),
                    tok_s=round(rate, 1),
                    elapsed_h=round(el / 3600, 2),
                    eta_h=round(remaining / rate / 3600, 2) if rate > 0 else None,
                )
            if n_buf >= shard_seqs:
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
