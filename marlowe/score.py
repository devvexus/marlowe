"""Stage 3: ablation-KL layer scoring on a 4-bit model.

Score every eligible ``linear_attention`` layer by the damage its removal does, measured as
forward KL( full || ablated ) over sampled logit positions on long sequences.

Three deliberate departures from the standard depth-pruning recipe:

* **Ablation KL, not angular distance.** Angular-distance metrics were derived for softmax
  attention residual streams. A DeltaNet layer's per-token output delta can be small while
  its contribution to state maintenance across 100K tokens is large, so angular distance
  systematically nominates exactly the layers that must not be cut.
* **Long sequences, late positions.** Calibrate at >= 8192 (32768 preferred) and sample
  scored positions from the last three quarters. Recurrent-state damage accumulates and is
  invisible near position zero.
* **4-bit throughout.** Whole-layer ablation damage is orders of magnitude larger than NF4
  quantisation noise, so the *ranking* survives 4-bit while the full model never has to be
  resident. Absolute KL values from this stage are not comparable to Stage 2 KL numbers and
  are not used as such.

Memory notes. Materialising full logits would cost ``seq_len x 248320 x 2`` bytes -- 16 GB
at 32768, on a card that is already holding 13 GB of weights. So this runs the decoder for
hidden states and applies ``lm_head`` only at the sampled positions, and prefills long
sequences in chunks so activation memory is bounded by the chunk rather than the sequence.
"""

from __future__ import annotations

import gc
import json
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from marlowe import logutil, preflight
from marlowe.arch import Layout, positional_selection
from marlowe.config import ScoreConfig

log = logutil.get("score")


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------


def make_identity(returns_tuple: bool) -> Any:
    """A residual passthrough standing in for a decoder block.

    Decoder ``forward`` signatures differ across transformers versions -- some return a bare
    tensor, some a tuple. Mirror whatever the real layer returns so the caller does not care.
    An ablated layer simply never writes its cache slot, which is harmless: nothing reads it.
    """
    import torch.nn as nn

    class _Identity(nn.Module):  # type: ignore[misc]  # nn.Module is Any to mypy
        def __init__(self) -> None:
            super().__init__()
            self.returns_tuple = returns_tuple
            self.layer_idx = -1  # never used, but present so renumbering walks do not trip

        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            return (hidden_states,) if self.returns_tuple else hidden_states

    return _Identity()


def detect_return_style(layers: Any, hidden_size: int, device: Any, dtype: Any) -> bool:
    """Does a decoder block return a tuple, or a bare tensor?

    Read from the layer's own return annotation first, and only probe if that is absent.
    The probe alone was not enough and failed silently in the wrong direction: it called the
    layer with hidden states only, but ``Qwen3_5DecoderLayer.forward`` takes
    ``position_embeddings`` as a required positional argument, so the call always raised and
    the handler assumed ``True``.

    That guess is wrong for this architecture -- the signature says ``-> torch.FloatTensor``
    and the caller does ``hidden_states = decoder_layer(...)`` -- so every ablated layer
    returned ``(hidden,)`` and the *next* real layer called ``input_layernorm`` on a tuple.
    Stage 3 ran the whole reference pass, then died on its first ablated candidate with
    "'tuple' object has no attribute 'float'", eight minutes in.

    An annotation cannot be wrong about its own return type in the way a probe can be wrong
    about why it raised.
    """
    import inspect

    import torch

    try:
        ann = inspect.signature(type(layers[0]).forward).return_annotation
        text = str(ann)
        if ann is not inspect.Signature.empty and text != "None":
            is_tuple = "tuple" in text.lower()
            logutil.event(
                log, "return style from annotation", annotation=text[:60], tuple=is_tuple
            )
            return is_tuple
    except (ValueError, TypeError, AttributeError):
        pass

    with torch.no_grad():
        try:
            out = layers[0](torch.zeros(1, 4, hidden_size, dtype=dtype, device=device))
        except Exception:  # noqa: BLE001 - probing; fall through to the documented default
            logutil.event_at(
                log, logging.WARNING,
                "return-style probe failed and there is no annotation; assuming a bare "
                "tensor, which is what current transformers decoder layers return",
            )
            return False
    return isinstance(out, tuple)


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------


def find_decoder(model: Any) -> Any:
    """Return the module whose output is the final (normed) hidden state.

    Qwen3_5ForConditionalGeneration nests the text model under a multimodal wrapper and the
    exact path has moved between transformers versions, so resolve it by interface first and
    fall back to a structural search.
    """
    for getter in ("get_decoder", "get_text_decoder"):
        fn = getattr(model, getter, None)
        if callable(fn):
            try:
                dec = fn()
            except Exception:  # noqa: BLE001
                continue
            if dec is not None and hasattr(dec, "layers"):
                return dec
    for path in ("model.language_model", "model.text_model", "model", "language_model"):
        mod: Any = model
        try:
            for part in path.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if hasattr(mod, "layers"):
            return mod
    raise RuntimeError(
        "could not locate the text decoder. Inspect model.named_modules() and pass the "
        "path explicitly."
    )


def find_decoder_layers(model: Any, expected_len: int) -> tuple[Any, str, Any]:
    """Return (parent, attr_name, ModuleList) for the text decoder stack.

    Guards on both length and hidden size so the 27-layer vision tower is never selected
    even if its depth were to match.
    """
    import torch.nn as nn

    hits: list[tuple[str, Any]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == expected_len:
            hits.append((name, mod))
    if not hits:
        raise RuntimeError(
            f"no ModuleList of length {expected_len} found. Inspect the model structure "
            f"and pass the path explicitly."
        )
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous decoder stack, candidates: {[h[0] for h in hits]}")
    name, layers = hits[0]
    parent_path, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return parent, attr, layers


@dataclass
class Harness:
    """Everything the scoring loop needs, resolved once."""

    model: Any
    decoder: Any
    lm_head: Any
    layers: Any
    device: Any
    dtype: Any
    returns_tuple: bool
    chunk_size: int


def build_harness(model: Any, n_layers: int, chunk_size: int) -> Harness:
    import torch

    decoder = find_decoder(model)
    _, _, layers = find_decoder_layers(model, n_layers)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None; cannot compute logits")

    param = next(p for p in layers.parameters())
    device = param.device
    dtype = torch.bfloat16 if param.dtype not in (torch.float16, torch.bfloat16) else param.dtype
    returns_tuple = detect_return_style(layers, decoder.config.hidden_size, device, dtype)
    return Harness(model, decoder, lm_head, layers, device, dtype, returns_tuple, chunk_size)


def _hidden_from(out: Any) -> Any:
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out[0] if isinstance(out, tuple) else out


def forward_logprobs_at(h: Harness, input_ids: Any, positions: Sequence[int]) -> Any:
    """Log-softmax at ``positions`` only, in fp32 on CPU.

    Prefills in chunks with the cache on, so peak activation memory tracks ``chunk_size``
    rather than sequence length. The recurrent state and KV cache carry across chunks, which
    is exactly the behaviour being measured.
    """
    import torch
    import torch.nn.functional as F

    seq_len = input_ids.shape[1]
    pos = list(positions)
    wanted = set(pos)
    collected: dict[int, Any] = {}

    with torch.no_grad():
        if seq_len <= h.chunk_size:
            hidden = _hidden_from(h.decoder(input_ids=input_ids, use_cache=False))
            sel = hidden[:, pos, :]
            logits = h.lm_head(sel).float()
            out = F.log_softmax(logits, dim=-1)[0].cpu()
            del hidden, sel, logits
            return out

        past: Any = None
        for start in range(0, seq_len, h.chunk_size):
            end = min(start + h.chunk_size, seq_len)
            chunk = input_ids[:, start:end]
            res = h.decoder(input_ids=chunk, use_cache=True, past_key_values=past)
            past = getattr(res, "past_key_values", None)
            hidden = _hidden_from(res)
            local = [p - start for p in pos if start <= p < end]
            if local:
                sel = hidden[:, local, :]
                lp = F.log_softmax(h.lm_head(sel).float(), dim=-1)[0].cpu()
                for j, p in enumerate([p for p in pos if start <= p < end]):
                    collected[p] = lp[j]
                del sel, lp
            del hidden, res
        del past
        gc.collect()
        torch.cuda.empty_cache()

    missing = wanted - set(collected)
    if missing:
        raise RuntimeError(f"chunked prefill missed positions {sorted(missing)[:8]}")
    return torch.stack([collected[p] for p in pos])


def kl_against(ref_logprobs: Any, got_logprobs: Any) -> tuple[float, int]:
    """Sum of forward KL( ref || got ) and the number of positions, both on CPU."""
    import torch.nn.functional as F

    kl = F.kl_div(got_logprobs, ref_logprobs, log_target=True, reduction="none").sum(-1)
    return float(kl.sum().item()), int(kl.numel())


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class LayerScore:
    index: int
    layer_type: str
    period: int
    kl: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_candidates(
    h: Harness,
    candidates: Iterable[int],
    batches: Sequence[Any],
    positions: Sequence[Sequence[int]],
    ref_logprobs: Sequence[Any],
    *,
    already_ablated: Sequence[int] = (),
    checkpoint: str | Path | None = None,
) -> dict[int, float]:
    """Mean forward-KL( full || ablated ) per candidate, in nats.

    ``already_ablated`` layers are held out for the whole sweep -- used by greedy mode to
    score the next cut on top of the cuts already taken.

    ``checkpoint`` makes the sweep resumable. Stage 3 is ~8 hours and wrote nothing until it
    returned, so a crash at hour seven cost seven hours -- and this machine kernel-panicked
    under sustained GPU load. Each candidate is independent, so scoring one is a natural
    commit point: roughly fifteen minutes of work, written as it completes.
    """
    import gc as _gc
    import json as _json

    import torch

    cands = list(candidates)
    scores: dict[int, float] = {}

    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt is not None and ckpt.exists():
        # Keys are layer indices; JSON stringifies them.
        done = {int(k): float(v) for k, v in _json.loads(ckpt.read_text()).items()}
        # Only trust scores for layers this sweep is actually asking about. A checkpoint from
        # a different ablation prefix describes a different question.
        scores.update({k: v for k, v in done.items() if k in set(cands)})
        if scores:
            logutil.event(
                log, "resuming candidate sweep", have=len(scores), of=len(cands),
                path=str(ckpt),
            )

    saved_prefix = {i: h.layers[i] for i in already_ablated}
    for i in already_ablated:
        h.layers[i] = make_identity(h.returns_tuple)
    try:
        for n, idx in enumerate(cands):
            if idx in scores:
                continue
            original = h.layers[idx]
            h.layers[idx] = make_identity(h.returns_tuple)
            total, count = 0.0, 0
            for batch, pos, ref in zip(batches, positions, ref_logprobs):
                got = forward_logprobs_at(h, batch, pos)
                s, c = kl_against(ref, got)
                total += s
                count += c
                del got
            h.layers[idx] = original
            scores[idx] = total / max(count, 1)
            logutil.event(
                log,
                "candidate scored",
                n=n + 1,
                of=len(cands),
                layer=idx,
                kl=round(scores[idx], 6),
            )
            if ckpt is not None:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
                tmp.write_text(_json.dumps(scores, indent=2), encoding="utf-8")
                tmp.replace(ckpt)  # atomic: a crash mid-write must not corrupt the resume
            _gc.collect()
            torch.cuda.empty_cache()
    finally:
        for i, mod in saved_prefix.items():
            h.layers[i] = mod
    return scores


def oneshot_select(scores: dict[int, float], layout: Layout, n_remove: int) -> list[int]:
    """Take the ``n_remove`` cheapest layers subject to the per-period constraints.

    One pass over an already-computed ranking: ~172 forward passes total versus ~2200 for
    greedy-with-rescoring, for a selection that differs only where two individually cheap
    adjacent layers turn out to be jointly expensive.
    """
    chosen: list[int] = []
    per_period: dict[int, int] = {}
    for idx in sorted(scores, key=lambda i: scores[i]):
        if len(chosen) == n_remove:
            break
        p = layout.period_of(idx)
        if per_period.get(p, 0) >= layout.period_capacity(p):
            continue
        chosen.append(idx)
        per_period[p] = per_period.get(p, 0) + 1
    if len(chosen) < n_remove:
        raise ValueError(
            f"constraints admit only {len(chosen)} of {n_remove} requested cuts "
            f"(budget {layout.budget()}). Raise max_per_period -- which degrades the "
            f"3:1 alternation -- or accept a larger model."
        )
    return sorted(chosen)


def greedy_select(
    h: Harness,
    layout: Layout,
    batches: Sequence[Any],
    positions: Sequence[Sequence[int]],
    ref_logprobs: Sequence[Any],
    n_remove: int,
    rescore_every: int,
) -> tuple[list[int], dict[int, float]]:
    """Remove the least-damaging layer repeatedly, re-scoring every ``rescore_every`` cuts.

    Catches interaction that one-shot misses, at roughly 12x the forward passes.
    """
    removed: list[int] = []
    per_period: dict[int, int] = {}
    scores: dict[int, float] = {}
    last_full: dict[int, float] = {}
    stale = True

    def admissible(i: int) -> bool:
        if i in removed:
            return False
        p = layout.period_of(i)
        return per_period.get(p, 0) < layout.period_capacity(p)

    for step in range(n_remove):
        if stale:
            pool = [i for i in layout.candidates() if admissible(i)]
            if not pool:
                raise RuntimeError(
                    f"exhausted candidates after {step} removals; budget is {layout.budget()}"
                )
            logutil.event(log, "greedy rescore", step=step + 1, of=n_remove, pool=len(pool))
            scores = score_candidates(
                h, pool, batches, positions, ref_logprobs, already_ablated=removed
            )
            if not last_full:
                last_full = dict(scores)
            stale = False

        eligible = {i: s for i, s in scores.items() if admissible(i)}
        if not eligible:
            stale = True
            continue
        pick = min(eligible, key=lambda i: eligible[i])
        removed.append(pick)
        per_period[layout.period_of(pick)] = per_period.get(layout.period_of(pick), 0) + 1
        logutil.event(
            log,
            "greedy cut",
            layer=pick,
            kl=round(scores[pick], 6),
            taken=len(removed),
            of=n_remove,
        )
        scores.pop(pick, None)
        if (step + 1) % rescore_every == 0:
            stale = True

    return sorted(removed), last_full


# ---------------------------------------------------------------------------
# calibration data
# ---------------------------------------------------------------------------


def read_jsonl_text(path: str | Path) -> list[str]:
    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else str(obj))
    if not texts:
        raise ValueError(f"no usable rows in {path}")
    return texts


def build_batches(
    tokenizer: Any,
    path: str | Path,
    seq_len: int,
    n_seqs: int,
    n_positions: int,
    seed: int,
    start_frac: float = 0.25,
) -> tuple[list[Any], list[list[int]]]:
    """Pack calibration text into fixed-length sequences and sample scored positions.

    Positions come from the last ``1 - start_frac`` of each sequence: recurrent-state damage
    accumulates along the sequence and is invisible near the start, so scoring uniformly
    would under-weight exactly the layers that must not be cut.
    """
    import torch

    rng = random.Random(seed)
    texts = read_jsonl_text(path)

    ids: list[list[int]] = []
    buf: list[int] = []
    for t in texts:
        buf.extend(tokenizer(t, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            ids.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(ids) >= n_seqs:
            break
    if len(ids) < n_seqs:
        raise ValueError(
            f"calibration data yields only {len(ids)} sequences of {seq_len} tokens; need "
            f"{n_seqs}. Use longer documents -- short snippets systematically under-weight "
            f"the DeltaNet layers."
        )

    batches: list[Any] = []
    positions: list[list[int]] = []
    lo = int(seq_len * start_frac)
    for seq in ids[:n_seqs]:
        batches.append(torch.tensor([seq]))
        pool = list(range(lo, seq_len))
        positions.append(sorted(rng.sample(pool, min(n_positions, len(pool)))))
    return batches, positions


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def load_4bit(
    model_path: str,
    *,
    max_gpu_gb: float | None = None,
    quantize_lm_head: bool = True,
) -> tuple[Any, Any]:
    """Load the model NF4 with double quantisation. Never loads bf16 resident.

    ``lm_head`` is quantised by default: bitsandbytes skips it out of the box, and at
    248320 x 5120 that alone is 2.5 GB of fp16 on a 16 GB card. Embeddings stay in bf16
    (they are not Linear layers) and are offloaded to CPU when ``max_gpu_gb`` forces it --
    an embedding lookup is cheap on CPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from marlowe.heal import embedding_module_name

    preflight.check_bitsandbytes()

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=[] if quantize_lm_head else ["lm_head"],
        # Required whenever accelerate spills anything to CPU, which device_map="auto" does
        # as soon as the model does not fit outright: the 27B parent at NF4 is ~15.2 GB
        # against ~15.8 GB usable, so a single spilled module is enough. Without this,
        # bitsandbytes refuses the load entirely -- "Some modules are dispatched on the CPU
        # or the disk" -- 26 seconds in, which is how Stage 3 died on its first attempt.
        #
        # The flag name says int8; it gates 4-bit offload too. heal.load_student has carried
        # this since it was written, and this loader did not: the same wrong assumption in
        # two places, fixed in one.
        llm_int8_enable_fp32_cpu_offload=True,
    )
    kwargs: dict[str, Any] = {
        "quantization_config": qcfg,
        # Explicit placement, not device_map="auto".
        #
        # heal.load_student learned this and this loader had not: letting accelerate choose
        # what to spill picks Linear4bit modules, and attaching its execution hooks to an
        # offloaded 4-bit module reads quant_state.offset.item() on a meta tensor. On this
        # stack that is not an exception -- it is a segmentation fault during load, with the
        # log ending mid-sentence and no traceback to read.
        #
        # Naming the one module to offload avoids it entirely. embed_tokens is not a Linear,
        # so NF4 was never going to touch it, and at 248320 x 5120 it is ~2.5 GB -- which is
        # also what brings the 27B parent from ~15.2 GB (against ~15.8 GB usable) down to a
        # comfortable fit rather than a knife-edge one.
        "device_map": {"": 0, embedding_module_name(model_path): "cpu"},
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    if max_gpu_gb is not None:
        kwargs["max_memory"] = {0: f"{max_gpu_gb:.1f}GiB", "cpu": "24GiB"}

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()

    # Section 1 rule 1, asserted rather than assumed.
    preflight.assert_no_bf16_resident(model, allow_params=2_800_000_000)
    return model, tok


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def write_profile(
    path: str | Path,
    scores: dict[int, float],
    layout: Layout,
    selected: list[int],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Write the per-layer damage profile as JSON.

    This artefact has standalone value independent of the pruning run: no published layer
    redundancy measurements exist for a hybrid linear/full-attention stack.
    """
    rows = [
        LayerScore(
            index=i,
            layer_type=layout.layer_types[i],
            period=layout.period_of(i),
            kl=scores[i],
        ).as_dict()
        for i in sorted(scores)
    ]
    ranked = sorted(rows, key=lambda r: float(r["kl"]))
    for rank, row in enumerate(ranked):
        row["rank"] = rank
    profile = {
        "layout": {
            "n_layers": layout.n_layers,
            "type_counts": layout.type_counts,
            "n_attention": layout.n_attention,
            "layer_types": list(layout.layer_types),
            "periods": layout.periods(),
        },
        "constraints": {
            "max_per_period": layout.max_per_period,
            "protect_first_periods": layout.protect_first_periods,
            "protect_last_periods": layout.protect_last_periods,
            "budget": layout.budget(),
        },
        "scores": rows,
        "ranked_least_damaging_first": [r["index"] for r in ranked],
        "selected": selected,
        **extra,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return profile


def plot_profile(profile: dict[str, Any], path: str | Path) -> bool:
    """Damage-vs-depth plot. Returns False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot (pip install 'marlowe[plot]')")
        return False

    rows = profile["scores"]
    idx = [r["index"] for r in rows]
    kl = [r["kl"] for r in rows]
    sel = set(profile["selected"])
    colors = ["#c0392b" if i in sel else "#2c7fb8" for i in idx]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(idx, kl, color=colors, width=0.8)
    for p in profile["layout"]["periods"]:
        ax.axvline(p[-1] + 0.5, color="#bbbbbb", lw=0.6, ls=":")
    ax.set_xlabel("layer index (dotted lines = period boundaries at full_attention layers)")
    ax.set_ylabel("ablation KL (nats)")
    ax.set_title(
        f"Qwen3.8 layer redundancy: damage from removing each linear_attention layer\n"
        f"red = selected for removal ({len(sel)} cuts)"
    )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def run_scoring(
    model_path: str,
    out_dir: str | Path,
    cfg: ScoreConfig,
    n_remove: int,
    *,
    max_per_period: int = 2,
    protect_first: int = 1,
    protect_last: int = 1,
    chunk_size: int = 4096,
    max_gpu_gb: float | None = None,
) -> dict[str, Any]:
    """Score all eligible layers and select ``n_remove``. Returns the profile dict."""
    import torch

    from marlowe.arch import ArchDims, load_config

    out_dir = Path(out_dir)
    raw_cfg = load_config(model_path)
    layout = Layout.from_config(
        raw_cfg,
        max_per_period=max_per_period,
        protect_first_periods=protect_first,
        protect_last_periods=protect_last,
    )
    dims = ArchDims.from_config(raw_cfg)

    logutil.event(log, "layout", layout=layout.describe())
    if n_remove > layout.budget():
        raise ValueError(
            f"{n_remove} cuts exceeds the constraint budget of {layout.budget()}. Raising "
            f"max_per_period past {max_per_period} would leave a period with no DeltaNet "
            f"layer."
        )
    if cfg.seq_len < 8192:
        raise ValueError(
            f"seq_len={cfg.seq_len} is below the 8192 floor. Short calibration nominates "
            f"exactly the DeltaNet layers that must not be cut (rule 3.2)."
        )
    if cfg.seq_len < 16384:
        log.warning(
            "seq_len=%d: short calibration under-weights DeltaNet state maintenance. "
            "32768 strongly preferred.",
            cfg.seq_len,
        )

    preflight.require(vram_gb=12.0, ram_gb=6.0, what="stage3-score")

    with logutil.timed(log, "load 4-bit model", path=model_path):
        model, tok = load_4bit(model_path, max_gpu_gb=max_gpu_gb, quantize_lm_head=True)

    h = build_harness(model, layout.n_layers, chunk_size)
    logutil.event(
        log, "harness", device=str(h.device), returns_tuple=h.returns_tuple, chunk=chunk_size
    )

    with logutil.timed(log, "build calibration set", seq_len=cfg.seq_len, n_seqs=cfg.n_seqs):
        batches, positions = build_batches(
            tok,
            cfg.calib_path,
            cfg.seq_len,
            cfg.n_seqs,
            cfg.n_positions,
            cfg.seed,
            cfg.position_start_frac,
        )
        batches = [b.to(h.device) for b in batches]

    with logutil.timed(log, "reference logprobs"):
        ref_logprobs = [forward_logprobs_at(h, b, p) for b, p in zip(batches, positions)]
    gc.collect()
    torch.cuda.empty_cache()

    candidates = layout.candidates()
    n_passes = cfg.n_seqs * (1 + len(candidates))
    logutil.event(
        log, "scoring", mode=cfg.mode, candidates=len(candidates), forward_passes=n_passes
    )

    if cfg.mode == "greedy":
        selected, scores = greedy_select(
            h, layout, batches, positions, ref_logprobs, n_remove, cfg.rescore_every
        )
    else:
        scores = score_candidates(
            h, candidates, batches, positions, ref_logprobs,
            # ~8 hours of forwards; commit each candidate as it lands.
            checkpoint=Path(out_dir) / "candidate_scores.partial.json",
        )
        selected = oneshot_select(scores, layout, n_remove)

    control = positional_selection(layout, n_remove) if cfg.run_positional_control else []
    if control:
        # Not a KL comparison -- that needs two surgeries and is done in Stage 4. This is the
        # cheap prior: how much measured damage does the positional arm accept?
        logutil.event(
            log,
            "positional control",
            layers=control,
            summed_kl=round(sum(scores.get(i, float("nan")) for i in control), 6),
            selected_summed_kl=round(sum(scores[i] for i in selected), 6),
            overlap=len(set(control) & set(selected)),
        )

    child, _ = layout.apply(selected)
    profile = write_profile(
        out_dir / "layer_profile.json",
        scores,
        layout,
        selected,
        {
            "model": model_path,
            "mode": cfg.mode,
            "seq_len": cfg.seq_len,
            "n_seqs": cfg.n_seqs,
            "n_positions": cfg.n_positions,
            "position_start_frac": cfg.position_start_frac,
            "seed": cfg.seed,
            "quantization": "nf4-double",
            "forward_passes": n_passes,
            "positional_control": control,
            "child_layout": list(child.layer_types),
            "projected_params": dims.total_params(child.layer_types),
        },
    )
    plot_profile(profile, out_dir / "layer_profile.png")

    logutil.event(
        log,
        "selection",
        removed=selected,
        n=len(selected),
        projected_b=round(dims.total_params(child.layer_types) / 1e9, 4),
        child=child.describe(),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return profile
