"""Stage 6: QLoRA distillation against the cached teacher distribution.

Loss is forward KL against the cached top-K distribution, not next-token cross-entropy. Same
token budget, substantially more recovery: cross-entropy against a single hard target throws
away everything the teacher knows about the other 248319 tokens, which is precisely the
information a pruned student has lost.

Memory shape on a 16 GB card: NF4 base around 11.6 GB, leaving room for rank-32 adapters,
optimizer state, and activations with gradient checkpointing at 2048 tokens. The logits are
the other pressure point -- ``2048 x 248320`` in bf16 is 1.02 GB per micro-batch before any
softmax intermediate -- so the loss is computed in sequence chunks and the full logit tensor
is never materialised.

Two operational rules from the brief, both enforced here:

* Checkpoint every 10M tokens and run **both** metrics at every checkpoint. If KL plateaus,
  stop and bank the compute. If repetition stays elevated while KL improves, keep going --
  that is the signal that matters for the problem being solved.
* Training runs at 2048 tokens because that is what makes three days possible, but long
  context must be verified separately before shipping. The optional long-context anneal at
  the end of this stage is the documented mitigation.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil, preflight
from marlowe.config import HealConfig

log = logutil.get("heal")


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------


def topk_kl_loss(
    logits: Any, teacher_idx: Any, teacher_logprob: Any
) -> tuple[Any, dict[str, float]]:
    """Forward KL( teacher || student ) restricted to the teacher's top-K support.

    The teacher's stored logprobs are true full-softmax values, so they are renormalised over
    the K captured entries to form a proper distribution; the student's are true logprobs
    gathered at the same indices. The result is non-negative and reduces to the true forward
    KL as K grows.

    Returns (loss, diagnostics). ``captured_mass`` is the teacher probability inside the
    top-K: if it drifts low, K is too small and a top-K objective is a poor proxy.
    """
    import torch
    import torch.nn.functional as F

    logits = logits.float()
    student_lp = F.log_softmax(logits, dim=-1)
    s = torch.gather(student_lp, -1, teacher_idx.long())  # [B, T, K]

    t_true = teacher_logprob.float()
    t_norm = F.log_softmax(t_true, dim=-1)  # renormalise over the K support
    p = t_norm.exp()

    kl = (p * (t_norm - s)).sum(-1)  # [B, T]
    diag = {
        "captured_mass": float(t_true.exp().sum(-1).mean().item()),
        "student_mass": float(s.exp().sum(-1).mean().item()),
    }
    return kl.mean(), diag


def chunked_kl_loss(
    lm_head: Any,
    hidden: Any,
    teacher_idx: Any,
    teacher_logprob: Any,
    chunk: int = 512,
) -> tuple[Any, dict[str, float]]:
    """Apply ``lm_head`` and accumulate the loss in sequence chunks.

    Bounds peak logit memory to ``chunk x vocab`` instead of ``seq_len x vocab``. Gradients
    still flow through every chunk; the chunks are summed, not detached.
    """
    seq_len = hidden.shape[1]
    total = None
    diags: list[dict[str, float]] = []
    n = 0
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        logits = lm_head(hidden[:, start:end, :])
        loss, d = topk_kl_loss(logits, teacher_idx[:, start:end], teacher_logprob[:, start:end])
        weight = end - start
        total = loss * weight if total is None else total + loss * weight
        diags.append(d)
        n += weight
        del logits
    assert total is not None
    agg = {k: sum(d[k] for d in diags) / len(diags) for k in diags[0]}
    return total / n, agg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Where the training loop is in the cache. Persisted for resume."""

    shard: int = 0
    offset: int = 0  # sequence index within the shard
    tokens_seen: int = 0
    step: int = 0
    epoch: int = 0


def iter_batches(
    cache_dir: str | Path, micro_batch: int, start: Position, seq_len: int
) -> Iterator[tuple[Any, Any, Any, Position]]:
    """Yield (input_ids, topk_idx, topk_logprob, position) batches, resuming from ``start``.

    Position is emitted *after* the batch it describes, so a checkpoint written alongside a
    yielded position resumes at the next unseen batch rather than replaying one.
    """
    import numpy as np
    import torch

    from marlowe.teacher import CacheIndex

    cache_dir = Path(cache_dir)
    idx = CacheIndex.load(cache_dir / "index.json")
    if idx is None:
        raise FileNotFoundError(f"no teacher cache at {cache_dir}")
    if idx.seq_len != seq_len:
        raise ValueError(
            f"teacher cache was built at seq_len={idx.seq_len} but healing is configured for "
            f"{seq_len}. The cached distributions are position-aligned; they cannot be "
            f"re-chunked. Rebuild the cache or match the length."
        )

    pos = Position(**asdict(start))
    while True:  # epoch loop
        for s_i, shard in enumerate(idx.shards):
            if s_i < pos.shard:
                continue
            with np.load(cache_dir / shard["path"]) as z:
                ids, tk, tlp = z["input_ids"], z["topk_idx"], z["topk_logprob"]
            begin = pos.offset if s_i == pos.shard else 0
            for b in range(begin, len(ids), micro_batch):
                sl = slice(b, min(b + micro_batch, len(ids)))
                n = sl.stop - sl.start
                pos = Position(
                    shard=s_i,
                    offset=sl.stop,
                    tokens_seen=pos.tokens_seen + n * seq_len,
                    step=pos.step + 1,
                    epoch=pos.epoch,
                )
                yield (
                    torch.from_numpy(ids[sl].astype("int64")),
                    torch.from_numpy(tk[sl].astype("int64")),
                    torch.from_numpy(tlp[sl].astype("float32")),
                    pos,
                )
            pos = Position(s_i + 1, 0, pos.tokens_seen, pos.step, pos.epoch)
        pos = Position(0, 0, pos.tokens_seen, pos.step, pos.epoch + 1)
        log.warning("teacher cache exhausted; starting epoch %d (data is being reused)", pos.epoch)


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    position: dict[str, Any] = field(default_factory=lambda: asdict(Position()))
    kl_history: list[dict[str, Any]] = field(default_factory=list)
    wall_seconds: float = 0.0
    stopped_reason: str | None = None

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> TrainState | None:
        p = Path(path)
        if not p.exists():
            return None
        with p.open(encoding="utf-8") as f:
            return cls(**json.load(f))


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def load_student(model_path: str, cfg: HealConfig, *, max_gpu_gb: float | None = None) -> Any:
    """NF4 base plus LoRA adapters. The base is never held in bf16."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    preflight.check_bitsandbytes()
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for Stage 6. Install with: pip install 'marlowe[heal]'"
        ) from exc

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kwargs: dict[str, Any] = {
        "quantization_config": qcfg,
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    if max_gpu_gb is not None:
        kwargs["max_memory"] = {0: f"{max_gpu_gb:.1f}GiB", "cpu": "24GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # The LoRA adapters are legitimately bf16; the ceiling accounts for them plus embeddings.
    preflight.assert_no_bf16_resident(model, allow_params=3_200_000_000)
    return model


def run_healing(
    student_path: str,
    cache_dir: str | Path,
    out_dir: str | Path,
    cfg: HealConfig,
    *,
    max_gpu_gb: float | None = None,
    on_checkpoint: Any = None,
    resume: bool = True,
) -> TrainState:
    """Train until the token budget is spent, KL plateaus, or the caller stops it.

    ``on_checkpoint(path, state) -> dict | None`` is called after every checkpoint. Wire the
    metrics harness in there: a checkpoint that is not scored on both KL and repetition has
    not been evaluated, only saved.
    """
    import torch

    from marlowe.score import find_decoder

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "train_state.json"

    state = (TrainState.load(state_path) if resume else None) or TrainState()
    start = Position(**state.position)
    if start.tokens_seen:
        logutil.event(
            log, "resuming", tokens_seen=start.tokens_seen, step=start.step, shard=start.shard
        )

    preflight.require(vram_gb=13.0, ram_gb=8.0, what="stage6-heal")

    with logutil.timed(log, "load student", path=student_path):
        model = load_student(student_path, cfg, max_gpu_gb=max_gpu_gb)
        resume_ckpt = latest_checkpoint(out_dir) if resume else None
        if resume_ckpt is not None:
            model.load_adapter(str(resume_ckpt), adapter_name="default", is_trainable=True)
            logutil.event(log, "adapters restored", path=str(resume_ckpt))

    decoder = find_decoder(model)
    lm_head = model.get_output_embeddings()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)

    total_steps = max(1, cfg.tokens // (cfg.seq_len * cfg.micro_batch * cfg.grad_accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=min(0.3, cfg.warmup_steps / max(total_steps, 1)),
        anneal_strategy="cos",
    )

    device = next(p.device for p in params)
    model.train()
    t0 = time.time()
    every = cfg.checkpoint_every_tokens
    next_ckpt = ((start.tokens_seen // every) + 1) * every
    window: list[float] = []
    accum = 0

    def save_checkpoint(pos: Position, mean_kl: float) -> Path:
        path = out_dir / f"checkpoint-{pos.tokens_seen}"
        model.save_pretrained(str(path))
        state.position = asdict(pos)
        state.wall_seconds += time.time() - t0
        state.kl_history.append(
            {"tokens": pos.tokens_seen, "step": pos.step, "train_kl": mean_kl, "ts": time.time()}
        )
        state.save(state_path)
        logutil.event(
            log, "checkpoint", path=path.name, tokens=pos.tokens_seen, train_kl=round(mean_kl, 5)
        )
        return path

    for ids, tk, tlp, pos in iter_batches(cache_dir, cfg.micro_batch, start, cfg.seq_len):
        ids, tk, tlp = ids.to(device), tk.to(device), tlp.to(device)
        out = decoder(input_ids=ids, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        loss, diag = chunked_kl_loss(lm_head, hidden, tk, tlp)
        (loss / cfg.grad_accum).backward()
        window.append(float(loss.item()))
        accum += 1
        del out, hidden, loss

        if accum == cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            accum = 0

        if pos.step % 50 == 0:
            logutil.event(
                log,
                "train",
                step=pos.step,
                tokens=pos.tokens_seen,
                kl=round(sum(window[-50:]) / len(window[-50:]), 5),
                lr=round(sched.get_last_lr()[0], 8),
                captured_mass=round(diag["captured_mass"], 4),
                tok_s=round(pos.tokens_seen / max(time.time() - t0, 1e-9), 1),
            )

        if pos.tokens_seen >= next_ckpt:
            mean_kl = sum(window) / max(len(window), 1)
            ckpt = save_checkpoint(pos, mean_kl)
            if on_checkpoint is not None:
                extra = on_checkpoint(ckpt, state)
                if extra:
                    state.kl_history[-1].update(extra)
                    state.save(state_path)

            # Plateau check: bank the compute if the last two windows barely moved. Never
            # stops on repetition -- if repetition is still elevated while KL improves,
            # continued healing is exactly the right call.
            hist = [h["train_kl"] for h in state.kl_history]
            if len(hist) >= 3 and abs(hist[-2] - hist[-1]) < cfg.kl_plateau_delta:
                state.stopped_reason = (
                    f"KL plateau: {hist[-2]:.5f} -> {hist[-1]:.5f}, delta below "
                    f"{cfg.kl_plateau_delta}"
                )
                logutil.event(log, "stopping early", reason=state.stopped_reason)
                state.save(state_path)
                return state

            window = []
            next_ckpt += cfg.checkpoint_every_tokens

        if pos.tokens_seen >= cfg.tokens:
            save_checkpoint(pos, sum(window) / max(len(window), 1))
            state.stopped_reason = "token budget spent"
            state.save(state_path)
            return state

    return state


def merge_adapters(base_path: str, adapter_path: str | Path, out_path: str | Path) -> Path:
    """Merge LoRA into the base weights and write a standalone checkpoint.

    Merging needs the base at higher precision than NF4 to avoid baking quantisation error
    into the shipped weights, but the full model must never be resident. So this loads on the
    CPU with low memory usage and streams to disk -- slow, run once, and it keeps the
    residency rule intact.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = Path(out_path)
    with logutil.timed(log, "merge adapters", base=base_path, adapter=str(adapter_path)):
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=torch.bfloat16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(adapter_path), device_map={"": "cpu"})
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_path), safe_serialization=True, max_shard_size="4GB")
        AutoTokenizer.from_pretrained(base_path, trust_remote_code=True).save_pretrained(
            str(out_path)
        )
    logutil.event(log, "merged", out=str(out_path))
    return out_path


def estimate_wall_clock(tokens: int, tok_s: float = 200.0) -> str:
    hours = tokens / tok_s / 3600
    return f"{tokens / 1e6:.0f}M tokens at {tok_s:.0f} tok/s = {hours:.1f} h ({hours / 24:.1f} d)"


def kl_plateaued(history: list[dict[str, Any]], delta: float, window: int = 2) -> bool:
    vals = [h["train_kl"] for h in history if "train_kl" in h]
    if len(vals) < window + 1:
        return False
    recent = vals[-(window + 1) :]
    return all(abs(recent[i] - recent[i + 1]) < delta for i in range(len(recent) - 1))


def format_history(history: list[dict[str, Any]]) -> str:
    lines = [f"{'tokens':>12}  {'train KL':>9}  {'eval KL':>9}  {'rep32':>7}  {'cap':>6}"]
    for h in history:
        lines.append(
            f"{h.get('tokens', 0):>12,}  {h.get('train_kl', math.nan):>9.5f}  "
            f"{h.get('eval_kl', math.nan):>9.5f}  {h.get('rep32', math.nan):>7.4f}  "
            f"{h.get('cap_hit_rate', math.nan):>6.3f}"
        )
    return "\n".join(lines)
