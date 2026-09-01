"""Stage 6: QLoRA distillation against the cached teacher distribution.

Loss is forward KL against the cached top-K distribution, not next-token cross-entropy. Same
token budget, substantially more recovery: cross-entropy against a single hard target throws
away everything the teacher knows about the other 248319 tokens, which is precisely the
information a pruned student has lost.

Memory on a 16 GB card, for the 22.3B student. This is plain peft plus bitsandbytes, not
Unsloth, so nothing here inherits Unsloth's savings and every megabyte is accounted for.

Two savings are near-free and are in every configuration:

* ``embed_tokens`` on CPU -- it is not a Linear, so NF4 cannot touch it, and it is 2.5 GB.
  An embedding lookup on CPU is cheap. :func:`assert_layers_on_gpu` checks that accelerate
  spilled the embeddings and not a decoder layer, which would train at a fraction of speed
  and read as a hung job.
* A chunked loss -- ``2048 x 248320`` in bf16 is 1.02 GB before any softmax intermediate, so
  the full logit tensor is never materialised.

Two cost quality, and are taken only if needed, in this order (see :data:`MEMORY_CANDIDATES`):

* **fp32 Adam moments** (1 GB). Optimizer state never enters the forward pass, so giving it
  up is the smaller concession.
* **NF4 lm_head** (2.5 GB). The loss is top-K KL against teacher *logits*, so quantising the
  head injects noise into precisely the quantity being matched -- and the adapter then learns
  to compensate for an error that disappears when :func:`merge_adapters` loads the base in
  bf16. bitsandbytes skips lm_head by default for this reason. Last resort, and it brings
  :data:`LM_HEAD_MITIGATION` with it: the head gets LoRA and the final norm trains in full.

:func:`search_memory_plan` measures each candidate rather than trusting arithmetic -- sizing
by hand gets the order of magnitude right and the last gigabyte wrong, and the last gigabyte
is the entire question. Stage 6 runs it before spending the budget.

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
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil, preflight
from marlowe.config import HealConfig

log = logutil.get("heal")

#: Required device-level VRAM headroom for a configuration to count as fitting.
#:
#: 1.0 GB, not the 0.3 GB this started at. Half a gigabyte is not a margin to stake a three-
#: to-five-day unattended run on: expandable_segments reduces fragmentation but does not
#: eliminate it, and a day lost to an OOM at hour 30 costs far more than the throughput gap
#: to the next candidate down. Rejecting a configuration that would *probably* have held is
#: the cheap error here.
#:
#: Measured device-level (total minus free from the driver), not from torch accounting --
#: see :func:`probe_training` for why that distinction is load-bearing.
FIT_MARGIN_GB = 1.0

#: Bytes resident on the device before this process loads anything: CUDA context, driver
#: allocations, any other process. Measured ONCE, on first use, while torch holds nothing.
#:
#: Measuring it per probe was wrong as soon as the search began caching the base model
#: across candidates: with a model already resident, (total - free) includes it, and
#: max_memory_reserved() includes it too, so the two were summed and the model counted twice.
#: That reported 41 GB on a 17 GB card. Two individually-correct changes, made together.
_CUDA_CONTEXT_BYTES: int | None = None


def cuda_context_bytes() -> int:
    """Device bytes not attributable to this process's allocator. Cached after first call."""
    global _CUDA_CONTEXT_BYTES
    import torch

    if _CUDA_CONTEXT_BYTES is None:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        # Anything torch has already reserved is ours, not context; subtract it back out so
        # this is measurable even when called late.
        _CUDA_CONTEXT_BYTES = max(0, (total - free) - torch.cuda.memory_reserved(0))
        logutil.event(
            log, "cuda context measured", gb=round(_CUDA_CONTEXT_BYTES / 1e9, 2)
        )
    return _CUDA_CONTEXT_BYTES


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
            # Read the length from the shard, not from the caller's argument. They agree in
            # a uniform cache, but tokens_seen drives checkpointing, the token budget and
            # resume -- so it must count what was actually consumed, not what was requested.
            shard_seq = int(shard.get("seq_len", seq_len))
            begin = pos.offset if s_i == pos.shard else 0
            for b in range(begin, len(ids), micro_batch):
                sl = slice(b, min(b + micro_batch, len(ids)))
                n = sl.stop - sl.start
                pos = Position(
                    shard=s_i,
                    offset=sl.stop,
                    tokens_seen=pos.tokens_seen + n * shard_seq,
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


#: GPU cap for the student. Chosen so accelerate spills the bf16 embedding table to CPU
#: while every decoder layer stays resident -- see :func:`assert_layers_on_gpu`. An embedding
#: lookup on CPU costs little; a decoder layer on CPU costs everything.
DEFAULT_MAX_GPU_GB = 11.5


def embedding_module_name(model_path: str | Path) -> str:
    """Module path of the input embedding table, derived from the tensors present.

    Derived rather than hardcoded: the wrapper namespace differs between a multimodal
    checkpoint and the text-only one surgery produces.
    """
    from marlowe.surgery import load_index

    weight_map, _ = load_index(model_path)
    for name in weight_map:
        if name.endswith("embed_tokens.weight"):
            return name[: -len(".weight")]
    raise ValueError(f"no embed_tokens tensor found in {model_path}")


def assert_layers_on_gpu(model: Any) -> dict[str, Any]:
    """Verify accelerate offloaded the embeddings and nothing else.

    ``max_memory`` constrains *how much* goes to the GPU, not *what*. If it picks a decoder
    layer to spill instead of the embedding table, training still runs and is 50x slower --
    which reads as a hung job, not a misconfiguration. So the placement is checked rather
    than assumed.
    """
    dmap = getattr(model, "hf_device_map", None)
    if not dmap:
        return {"device_map": None}

    offloaded = {k: v for k, v in dmap.items() if str(v) in ("cpu", "disk")}
    bad = [k for k in offloaded if ".layers." in k]
    if bad:
        raise preflight.ResourceError(
            f"accelerate offloaded {len(bad)} decoder layer(s) to CPU/disk: {bad[:4]}.\n"
            f"  Training would run at a small fraction of GPU speed and look like a hung job. "
            f"Raise --max-gpu-gb, lower lora_rank, or reduce seq_len."
        )
    logutil.event(
        log,
        "device placement",
        offloaded=sorted(offloaded),
        n_offloaded=len(offloaded),
        note="embeddings on CPU is intended; decoder layers must not be",
    )
    return {"offloaded": sorted(offloaded)}


#: Exact module path of the final RMSNorm on a text-only qwen3_5 stack.
#:
#: Exact, not a suffix: peft matches ``modules_to_save`` by name suffix, and a bare "norm"
#: also matches ``input_layernorm``, ``post_attention_layernorm`` and ``linear_attn.norm`` --
#: three per layer, verified against a real model.
FINAL_NORM_MODULE = "model.norm"

#: Applied whenever lm_head ends up quantised. Gives the head a trainable low-rank correction
#: and trains the final norm in full. The norm is 5120 parameters and it rescales the residual
#: stream entering the head, which is precisely what deleting upstream layers miscalibrates.
LM_HEAD_MITIGATION = ("lm_head",)


def effective_lora_targets(cfg: HealConfig) -> tuple[list[str], list[str]]:
    """Return (LoRA target modules, modules trained in full).

    When the head is quantised, it also gets LoRA and the final norm is trained fully. That
    does not undo the quantisation noise, but it puts trainable capacity exactly where the
    noise enters, which is the cheapest available mitigation.
    """
    targets = list(cfg.target_modules)
    save_full = list(cfg.modules_to_save)
    if cfg.quantize_lm_head:
        targets += [m for m in LM_HEAD_MITIGATION if m not in targets]
        if FINAL_NORM_MODULE not in save_full:
            save_full.append(FINAL_NORM_MODULE)

    bare = [m for m in save_full if m == "norm"]
    if bare:
        raise ValueError(
            f"modules_to_save contains a bare {bare[0]!r}, which peft matches by suffix -- it "
            f"would also select input_layernorm, post_attention_layernorm and "
            f"linear_attn.norm, three per layer. Use the exact path "
            f"{FINAL_NORM_MODULE!r}."
        )
    return targets, save_full


def attach_lora(base: Any, cfg: HealConfig) -> Any:
    """Wrap a prepared NF4 base in LoRA adapters. Cheap; the base load is what is not."""
    from peft import LoraConfig, get_peft_model

    targets, save_full = effective_lora_targets(cfg)
    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        modules_to_save=save_full or None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    # LoRA adapters are legitimately bf16; the ceiling accounts for them plus embeddings.
    preflight.assert_no_bf16_resident(model, allow_params=3_200_000_000)
    return model


def build_optimizer(params: list[Any], cfg: HealConfig) -> Any:
    """AdamW with 8-bit moments where available.

    The LoRA state here is ~176M parameters, so fp32 Adam moments cost 1.4 GB against 0.35 GB
    at 8 bits. On a card with ~4 GB left after weights and activations, that is the margin.
    Falls back to torch AdamW with a warning rather than failing.
    """
    import torch

    if cfg.optimizer_8bit:
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(params, lr=cfg.learning_rate, weight_decay=0.0)
            logutil.event(log, "optimizer", kind="bnb.AdamW8bit", n_tensors=len(params))
            return opt
        except (ImportError, AttributeError) as exc:
            log.warning("AdamW8bit unavailable (%s); falling back to fp32 AdamW", exc)
    logutil.event(log, "optimizer", kind="torch.AdamW", n_tensors=len(params))
    return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)


def load_student(
    model_path: str,
    cfg: HealConfig,
    *,
    max_gpu_gb: float | None = None,
    base_only: bool = False,
) -> Any:
    """NF4 base plus LoRA adapters. The base is never held in bf16.

    Two memory decisions matter on a 16 GB card, and both are the difference between fitting
    and not:

    * **lm_head is quantised.** bitsandbytes skips it by default, and at 248320 x 5120 that
      is 2.5 GB of fp16 for one tensor.
    * **embed_tokens is offloaded to CPU.** It is not a Linear, so NF4 cannot touch it, and
      it is another 2.5 GB. A lookup on CPU is cheap.

    Without both, the 22B student needs ~18 GB before optimizer state.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    preflight.check_bitsandbytes()
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for Stage 6. Install with: pip install 'marlowe[heal]'"
        ) from exc

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bitsandbytes skips lm_head by default and that default is right here: the loss is
        # top-K KL against teacher logits, so quantising the head adds noise to exactly what
        # is being matched. Only overridden when the memory search cannot fit fp16.
        llm_int8_skip_modules=[] if cfg.quantize_lm_head else ["lm_head"],
        # Required for the CPU embedding offload. Without it bitsandbytes refuses outright:
        # "Some modules are dispatched on the CPU or the disk." The flag name says int8 but
        # it gates 4-bit offload too. The offloaded module is embed_tokens, which is not a
        # Linear and was never going to be quantised anyway.
        llm_int8_enable_fp32_cpu_offload=True,
    )
    # Explicit placement, not `device_map="auto"` plus a max_memory cap.
    #
    # Letting accelerate choose what to spill picks Linear4bit modules, and attaching its
    # execution hooks to an offloaded 4-bit module reads `quant_state.offset.item()` on a
    # meta tensor and dies. Naming the one module to offload -- the embedding table, which is
    # not a Linear and was never going to be quantised -- avoids that entirely and is what we
    # wanted anyway.
    embed = embedding_module_name(model_path)
    device_map: dict[str, Any] = {"": 0, embed: "cpu"}
    kwargs: dict[str, Any] = {
        "quantization_config": qcfg,
        "device_map": device_map,
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    assert_layers_on_gpu(model)
    if base_only:
        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    return attach_lora(
        prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        ),
        cfg,
    )


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
    optim = build_optimizer(params, cfg)

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
        loss, diag = chunked_kl_loss(lm_head, hidden, tk, tlp, chunk=cfg.loss_chunk)
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


@dataclass
class ProbeResult:
    """What a short training probe actually measured. No projections in here."""

    tok_s: float
    peak_vram_gb: float
    total_vram_gb: float
    step_s: float
    seq_len: int
    n_steps: int
    lora_params: int
    offloaded: list[str] = field(default_factory=list)
    optimizer: str = ""
    lm_head_precision: str = ""
    modules_trained_in_full: list[str] = field(default_factory=list)
    #: What torch itself accounted for. Lower than peak_vram_gb by the CUDA context and
    #: allocator fragmentation; kept for diagnosis, never for the fit decision.
    torch_allocated_gb: float = 0.0
    torch_reserved_gb: float = 0.0

    def headroom_gb(self) -> float:
        return self.total_vram_gb - self.peak_vram_gb

    def projected_hours(self, tokens: int) -> float:
        return tokens / self.tok_s / 3600

    def render(self, tokens: int) -> str:
        h = self.projected_hours(tokens)
        return (
            f"measured over {self.n_steps} steps at seq_len {self.seq_len}:\n"
            f"  throughput     {self.tok_s:8.1f} tok/s  ({self.step_s * 1000:.0f} ms/step)\n"
            f"  peak VRAM      {self.peak_vram_gb:8.2f} GB of {self.total_vram_gb:.1f} "
            f"({self.headroom_gb():.2f} GB headroom, device-level)\n"
            f"  torch acct     {self.torch_allocated_gb:8.2f} GB allocated / "
            f"{self.torch_reserved_gb:.2f} GB reserved\n"
            f"  LoRA params    {self.lora_params / 1e6:8.1f} M    optimizer {self.optimizer}\n"
            f"  lm_head        {self.lm_head_precision:>8}    full-trained "
            f"{', '.join(self.modules_trained_in_full) or '(none)'}\n"
            f"  offloaded      {', '.join(self.offloaded) or '(nothing)'}\n"
            f"  projection     {tokens / 1e6:.0f}M tokens = {h:.1f} h ({h / 24:.2f} d)"
        )


@dataclass(frozen=True)
class MemoryCandidate:
    """One point in the memory/quality trade-off, with the reason it sits where it does."""

    name: str
    quantize_lm_head: bool
    optimizer_8bit: bool
    rationale: str
    #: Override the configured sequence length. None keeps it.
    seq_len: int | None = None
    #: Override the LoRA rank. Halving it roughly halves adapter parameters and their
    #: optimizer moments, without touching the loss target.
    lora_rank: int | None = None
    #: Requires an explicit human decision before it may be selected. Set on every candidate
    #: that quantises the head: that is a quality cliff, not another notch on a dial.
    requires_approval: bool = False

    def apply(self, cfg: HealConfig) -> HealConfig:
        import copy

        out = copy.deepcopy(cfg)
        out.quantize_lm_head = self.quantize_lm_head
        out.optimizer_8bit = self.optimizer_8bit
        if self.seq_len is not None:
            out.seq_len = self.seq_len
        if self.lora_rank is not None:
            out.lora_rank = self.lora_rank
        return out


#: Ordered best-quality-first. The search takes the first that fits with margin.
#:
#: CPU embeddings and a chunked loss are in *every* candidate: they are near-free -- an
#: embedding lookup on CPU is cheap, and chunking the loss costs a little throughput and no
#: quality at all -- so there is never a reason to give them up.
#:
#: The ordering is by quality cost, not by memory saved. An NF4 lm_head saves the most (2.5
#: GB) and costs the most: it injects quantisation noise into the very logits the top-K KL
#: loss is matching, and the adapter then learns to compensate for an error that vanishes
#: when merge_adapters loads the base in bf16 -- strictly worse than a wash, since the
#: correction does not survive to inference.
#:
#: Halving the sequence sits *above* that: it changes how much long-range structure each step
#: sees, which is a real cost, but it leaves the loss target itself intact. fp32 Adam moments
#: come last because optimizer state never enters the forward pass at all.
MEMORY_CANDIDATES: tuple[MemoryCandidate, ...] = (
    MemoryCandidate(
        name="fp16-head-2048",
        quantize_lm_head=False,
        optimizer_8bit=True,
        rationale="fp16 lm_head: no quantisation noise in the logits the loss matches",
    ),
    MemoryCandidate(
        name="fp16-head-1536",
        quantize_lm_head=False,
        optimizer_8bit=True,
        seq_len=1536,
        rationale="50% more long-range state exercise than 1024, at no other cost",
    ),
    MemoryCandidate(
        name="fp16-head-1024",
        quantize_lm_head=False,
        optimizer_8bit=True,
        seq_len=1024,
        rationale="halve the sequence rather than corrupt the loss target",
    ),
    MemoryCandidate(
        name="fp16-head-1024-rank16",
        quantize_lm_head=False,
        optimizer_8bit=True,
        seq_len=1024,
        lora_rank=16,
        rationale="half the adapter capacity, but the loss target stays intact",
    ),
    # --- everything below quantises the head and needs an explicit decision -------------
    MemoryCandidate(
        name="nf4-head-adamw32",
        quantize_lm_head=True,
        optimizer_8bit=False,
        requires_approval=True,
        rationale="NF4 head, but fp32 optimizer moments; head gets LoRA + trained final norm",
    ),
    MemoryCandidate(
        name="nf4-head-adamw8",
        quantize_lm_head=True,
        optimizer_8bit=True,
        requires_approval=True,
        rationale="most aggressive; both savings taken",
    ),
)


@dataclass
class SearchResult:
    chosen: MemoryCandidate | None
    config: HealConfig | None
    probe: ProbeResult | None
    attempts: list[tuple[str, str]] = field(default_factory=list)
    #: Candidates skipped because they need an explicit decision, not because they failed.
    blocked: list[tuple[str, str]] = field(default_factory=list)

    @property
    def exhausted_without_approval(self) -> bool:
        return self.chosen is None and bool(self.blocked)

    def render(self, tokens: int) -> str:
        lines = ["memory configuration search (best quality first):"]
        for name, outcome in self.attempts:
            lines.append(f"  {name:<24} {outcome}")
        if self.chosen is None or self.probe is None:
            if self.blocked:
                lines += [
                    "",
                    "  THE fp16 LADDER IS EXHAUSTED. Stopping rather than selecting a "
                    "quantised head.",
                    "",
                    *(f"  not taken  {n:<24} {w}" for n, w in self.blocked),
                    "",
                    "  Quantising lm_head injects noise into the logits the top-K KL loss is "
                    "matching, and merge_adapters loads the base in bf16 -- so the adapter "
                    "learns to cancel an error that never reaches inference.",
                    "  That is a decision to take knowingly. Pass --allow-quantized-head to "
                    "proceed, or lower heal.lora_rank / heal.loss_chunk further.",
                ]
            else:
                lines.append("\n  NOTHING FIT. Lower seq_len, lora_rank, or loss_chunk.")
            return "\n".join(lines)
        lines += [
            f"\nselected: {self.chosen.name} -- {self.chosen.rationale}",
            self.probe.render(tokens),
        ]
        return "\n".join(lines)


def search_memory_plan(
    model_path: str,
    cfg: HealConfig,
    *,
    margin_gb: float = FIT_MARGIN_GB,
    n_steps: int = 8,
    max_gpu_gb: float | None = None,
    candidates: tuple[MemoryCandidate, ...] = MEMORY_CANDIDATES,
    probe_all: bool = False,
    allow_quantized_head: bool = False,
) -> SearchResult:
    """Measure candidates in quality order and take the first that fits with margin.

    Sizing by arithmetic gets the order of magnitude right and the last gigabyte wrong, and
    the last gigabyte is the whole question here. So each candidate is actually run.
    """
    result = SearchResult(chosen=None, config=None, probe=None)
    # Quantisation is deterministic and the frozen base does not change between candidates --
    # only sequence length, LoRA rank and the optimizer do. Reloading 45 GB of safetensors per
    # candidate dominated the search's wall clock, and Stage 8 runs the whole search again on
    # the 18B. The base is cached per head precision, which is the only thing that alters it.
    bases: dict[bool, Any] = {}

    def base_for(trial: HealConfig) -> Any:
        key = trial.quantize_lm_head
        if key not in bases:
            bases[key] = load_student(model_path, trial, max_gpu_gb=max_gpu_gb, base_only=True)
            logutil.event(log, "base cached", quantize_lm_head=key)
        return bases[key]

    for cand in candidates:
        if result.chosen is not None and not probe_all:
            break
        if cand.requires_approval and not allow_quantized_head:
            # Deliberately not "the next thing to try". Quantising the head corrupts the
            # logits the loss is matching, and merge_adapters loads the base in bf16 -- so
            # the adapter's compensation never reaches inference. That is a decision to be
            # taken knowingly, not a step the search takes on its own.
            result.blocked.append((cand.name, cand.rationale))
            continue
        trial = cand.apply(cfg)
        try:
            wrapped = attach_lora(base_for(trial), trial)
            probe = probe_training(
                model_path, trial, n_steps=n_steps, max_gpu_gb=max_gpu_gb, base=wrapped
            )
            # Strip the adapters so the next candidate wraps a clean base. The frozen NF4
            # weights are untouched by training, so the base is reusable as-is.
            bases[trial.quantize_lm_head] = wrapped.unload()
        except Exception as exc:
            if not _is_oom(exc):
                raise
            result.attempts.append((cand.name, f"OOM -- {type(exc).__name__}"))
            logutil.event(log, "candidate OOM", candidate=cand.name)
            # A candidate that OOMed may have left the base in an unknown state; drop it.
            bases.pop(trial.quantize_lm_head, None)
            _free_cuda()
            continue

        headroom = probe.headroom_gb()
        fits = headroom >= margin_gb
        result.attempts.append(
            (
                cand.name,
                f"{'FITS' if fits else 'too tight'}: peak {probe.peak_vram_gb:.2f} GB, "
                f"headroom {headroom:.2f} GB, {probe.tok_s:.0f} tok/s",
            )
        )
        _free_cuda()
        if fits and result.chosen is None:
            result.chosen, result.config, result.probe = cand, trial, probe
            logutil.event(
                log,
                "memory plan selected",
                candidate=cand.name,
                headroom_gb=round(headroom, 2),
                tok_s=round(probe.tok_s, 1),
            )
            if not probe_all:
                bases.clear()
                _free_cuda()
                return result
    bases.clear()
    _free_cuda()
    return result


MEMORY_PLAN_FILE = "memory_plan.json"

#: HealConfig fields a MemoryCandidate is allowed to override, derived from the dataclass
#: rather than hand-listed.
#:
#: Persistence must cover exactly this set. Hand-listing it is how lora_rank got left out
#: when the rank-16 candidate was added: the plan would have restored seq_len and the head
#: precision but silently reverted the rank that made the configuration fit, and Stage 6
#: would have OOM'd on a plan that had been measured as fitting.
def _candidate_override_fields() -> tuple[str, ...]:
    import dataclasses

    fixed = {"name", "rationale", "requires_approval"}
    return tuple(
        f.name for f in dataclasses.fields(MemoryCandidate) if f.name not in fixed
    )


def save_memory_plan(path: str | Path, search: SearchResult) -> Path:
    """Persist the selected trade-offs so later stages use what was measured.

    This matters more than it looks. The teacher cache is built at a fixed sequence length
    and its distributions are position-aligned, so :func:`iter_batches` refuses a cache whose
    length does not match training. If the search runs inside Stage 6 and picks the 1024
    candidate, a 2048 cache built six hours earlier is already worthless. So the plan is
    decided before Stage 5 and both stages read it from here.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = search.config
    payload: dict[str, Any] = {
        "candidate": search.chosen.name if search.chosen else None,
        "rationale": search.chosen.rationale if search.chosen else None,
        "attempts": search.attempts,
        "blocked": search.blocked,
        # Every field a candidate can override, plus loss_chunk which the search never
        # varies but Stage 6 must not silently change either.
        **{k: (getattr(cfg, k) if cfg else None) for k in _candidate_override_fields()},
        "loss_chunk": cfg.loss_chunk if cfg else None,
        "measured": (
            {
                "tok_s": round(search.probe.tok_s, 1),
                "peak_vram_gb": round(search.probe.peak_vram_gb, 2),
                "headroom_gb": round(search.probe.headroom_gb(), 2),
                "lm_head_precision": search.probe.lm_head_precision,
            }
            if search.probe
            else None
        ),
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return p


def load_memory_plan(path: str | Path, cfg: HealConfig) -> tuple[HealConfig, dict[str, Any]] | None:
    """Apply a persisted plan onto ``cfg``. Returns None when there is no plan on disk."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("candidate") is None:
        return None

    import copy

    out = copy.deepcopy(cfg)
    restored: list[str] = []
    for key in (*_candidate_override_fields(), "loss_chunk"):
        if payload.get(key) is not None:
            setattr(out, key, payload[key])
            restored.append(key)
    missing = [k for k in _candidate_override_fields() if k not in payload]
    if missing:
        raise ValueError(
            f"{p} predates fields {missing} and would restore a configuration that was "
            f"never measured. Delete it and re-run the Stage 5 memory search."
        )
    logutil.event(
        log,
        "memory plan loaded",
        candidate=payload["candidate"],
        restored=restored,
        seq_len=out.seq_len,
        lora_rank=out.lora_rank,
    )
    return out, payload


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text


def _free_cuda() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def probe_training(
    model_path: str,
    cfg: HealConfig,
    *,
    n_steps: int = 12,
    max_gpu_gb: float | None = None,
    base: Any = None,
) -> ProbeResult:
    """Run a few real training steps and measure throughput and peak VRAM.

    The ~200 tok/s figure in the project brief assumed Unsloth's memory savings. This stack
    is plain peft plus bitsandbytes, so that number is not transferable, and a three-day
    commitment should not rest on an estimate. Two minutes of real steps settles both
    questions -- does it fit, and how long will it take -- before the budget is spent.

    Teacher data is synthetic here: the shapes and the compute are identical to training, and
    the loss value is meaningless but never used.
    """
    import torch

    from marlowe.score import find_decoder

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Everything already on the device before this model loads: the CUDA context, the
    # driver's own allocations, any other process. Measured once, with torch holding
    # nothing, because it cannot be recovered from torch's own counters.
    baseline_used = cuda_context_bytes()

    model = base if base is not None else load_student(
        model_path, cfg, max_gpu_gb=max_gpu_gb
    )
    decoder = find_decoder(model)
    lm_head = model.get_output_embeddings()
    params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in params)
    optim = build_optimizer(params, cfg)
    vocab = int(model.config.vocab_size if hasattr(model.config, "vocab_size") else 0) or 248320

    model.train()
    device = next(p.device for p in params)
    seq, k = cfg.seq_len, 16
    ids = torch.randint(0, vocab, (cfg.micro_batch, seq), device=device)
    t_idx = torch.randint(0, vocab, (cfg.micro_batch, seq, k), device=device)
    t_lp = torch.log_softmax(torch.randn(cfg.micro_batch, seq, k, device=device), dim=-1)

    timings: list[float] = []
    for i in range(n_steps):
        t0 = time.time()
        out = decoder(input_ids=ids, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        loss, _ = chunked_kl_loss(lm_head, hidden, t_idx, t_lp, chunk=cfg.loss_chunk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        del out, hidden, loss
        # Discard the first two steps: allocator warm-up and cuBLAS autotuning.
        if i >= 2:
            timings.append(time.time() - t0)

    # The footprint that actually constrains the run: the CUDA context plus the allocator's
    # high-water reservation. The context is measured once and globally (see
    # cuda_context_bytes) precisely so a cached base model is not counted twice.
    #
    # Neither obvious counter works alone. max_memory_allocated() counts live tensor bytes
    # and misses the CUDA context and fragmentation -- it under-reports by ~1 GB. And
    # (total - free) from the driver includes the caching allocator's *reserved* pool, which
    # under expandable_segments grows opportunistically and is never handed back, so it
    # saturates at total: every candidate measured 17.17 GB with 0.00 GB headroom, which is
    # not a measurement at all. max_memory_reserved() is the allocator's peak claim on CUDA,
    # which is the part that cannot be given back.
    peak_device_used = baseline_used + torch.cuda.max_memory_reserved()

    step_s = sum(timings) / max(len(timings), 1)
    dmap = getattr(model, "hf_device_map", {}) or {}
    result = ProbeResult(
        tok_s=cfg.micro_batch * seq / step_s,
        peak_vram_gb=peak_device_used / 1e9,
        torch_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        torch_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        total_vram_gb=total_vram,
        step_s=step_s,
        seq_len=seq,
        n_steps=len(timings),
        lora_params=n_lora,
        offloaded=sorted(k for k, v in dmap.items() if str(v) in ("cpu", "disk")),
        optimizer=type(optim).__name__,
        lm_head_precision="nf4" if cfg.quantize_lm_head else "fp16",
        modules_trained_in_full=effective_lora_targets(cfg)[1],
    )
    del optim
    if base is None:
        del model
    torch.cuda.empty_cache()
    logutil.event(
        log,
        "training probe",
        tok_s=round(result.tok_s, 1),
        peak_vram_gb=round(result.peak_vram_gb, 2),
        headroom_gb=round(result.headroom_gb(), 2),
        lora_m=round(n_lora / 1e6, 1),
    )
    if result.headroom_gb() < 0.5:
        log.warning(
            "only %.2f GB of VRAM headroom at seq_len %d. A long-context anneal or a "
            "fragmentation spike will OOM. Lower seq_len, lora_rank, or loss_chunk.",
            result.headroom_gb(),
            seq,
        )
    return result


#: Measured on the 22.3B student, plain peft + bitsandbytes, fp16 head, AdamW8bit,
#: micro_batch 1, gradient checkpointing on, loss chunked at 256. RTX 4080 Super.
#:
#: The project brief assumed ~200 tok/s, which came from an Unsloth-based estimate. This
#: stack is not Unsloth, and the gap is a factor of 3-4. Schedules must use these.
MEASURED_TOK_S: dict[str, float] = {
    "22b-seq2048-rank32": 44.3,
    "22b-seq1024-rank32": 73.5,
    "22b-seq2048-rank32-nf4head-adamw32": 9.8,
}


def scale_tok_s(tok_s: float, from_layers: int, to_layers: int) -> float:
    """Scale throughput between rungs by decoder depth.

    Per-step cost is dominated by the decoder stack, which is linear in layer count. The
    embedding and head are constant, so this slightly *under*-estimates the smaller model's
    speed -- an error in the conservative direction for scheduling.
    """
    return tok_s * from_layers / to_layers


def schedule_rows(
    tok_s: float,
    budgets: Sequence[int],
    *,
    layers: int = 52,
    child_layers: int = 41,
) -> list[dict[str, float]]:
    """Wall-clock for a heal budget on each rung, and both together."""
    child_tok_s = scale_tok_s(tok_s, layers, child_layers)
    rows: list[dict[str, float]] = []
    for tokens in budgets:
        h1 = tokens / tok_s / 3600
        h2 = tokens / child_tok_s / 3600
        rows.append(
            {
                "tokens": float(tokens),
                "rung1_h": h1,
                "rung1_d": h1 / 24,
                "rung2_h": h2,
                "rung2_d": h2 / 24,
                "both_d": (h1 + h2) / 24,
            }
        )
    return rows


#: Forward-only work costs roughly a third of a training step (no backward, no optimizer,
#: no stored activations). ESTIMATE, not measured -- Stage 5 logs its real rate.
FORWARD_ONLY_SPEEDUP = 3.0


def teacher_cache_days(tok_s: float, tokens: int, *, student_layers: int = 52,
                       teacher_layers: int = 64) -> float:
    """Rough Stage 5 wall-clock, scaled from the measured training rate.

    The brief budgeted 6 hours from an assumed 2500 tok/s. That came from the same source as
    the 200 tok/s training estimate, which measured 73.5. Treat this as the order of
    magnitude and let Stage 5's own logs replace it.
    """
    fwd = scale_tok_s(tok_s * FORWARD_ONLY_SPEEDUP, student_layers, teacher_layers)
    return tokens / fwd / 3600 / 24


def render_schedule(tok_s: float, budgets: Sequence[int], **kw: Any) -> str:
    rows = schedule_rows(tok_s, budgets, **kw)
    child = scale_tok_s(tok_s, kw.get("layers", 52), kw.get("child_layers", 41))
    out = [
        f"healing wall-clock at {tok_s:.1f} tok/s measured on the 22B "
        f"({child:.0f} tok/s estimated for the 18B, scaled by depth)",
        "",
        f"{'tokens':>8}  {'22B':>10}  {'18B':>10}  {'both rungs':>12}",
        f"{'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 12}",
    ]
    for r in rows:
        out.append(
            f"{r['tokens'] / 1e6:>7.0f}M  {r['rung1_d']:>8.1f} d  "
            f"{r['rung2_d']:>8.1f} d  {r['both_d']:>10.1f} d"
        )
    out += [
        "",
        "healing only. Stage 5 builds a teacher cache per rung, forward-only on the 27B:",
        "",
        f"{'tokens':>8}  {'cache/rung':>12}  {'both caches':>13}   ESTIMATE, not measured",
        f"{'-' * 8}  {'-' * 12}  {'-' * 13}",
    ]
    for r in rows:
        d = teacher_cache_days(tok_s, int(r["tokens"]))
        out.append(f"{r['tokens'] / 1e6:>7.0f}M  {d:>10.1f} d  {2 * d:>11.1f} d")
    return "\n".join(out)


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
