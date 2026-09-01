#!/usr/bin/env python3
"""
Structure-aware depth pruning for Qwen3.8-27B (Qwen3_5ForConditionalGeneration).

The model is a hybrid stack: layer_types is an explicit 64-entry list of
  3x linear_attention (Gated DeltaNet) -> 1x full_attention (gated GQA), x16.

Only 16 layers carry a KV cache and do exact token-to-token retrieval. Contiguous
block removal (Gromov / ShortGPT) would delete a proportional share of those and
destroy long-context recall. This script only ever removes linear_attention blocks,
scores them by measured ablation damage on LONG sequences, and keeps at least one
DeltaNet layer per period so the alternation survives.

  27.74B  64 layers (48 linear + 16 full)
  18.16B  39 layers (23 linear + 16 full)   <- default target, 25 removals

Usage:
  python prune_qwen38.py --calib calib.jsonl --remove 25 --out ./Qwen3.8-18B
  python prune_qwen38.py --calib calib.jsonl --target-params 18e9 --dry-run

calib.jsonl: one JSON object per line with a "text" field. Use in-domain data,
and use LONG documents -- see --seq-len.
"""

import argparse, copy, gc, json, math, os, random, sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# locating the decoder stack
# ----------------------------------------------------------------------------

def find_decoder_layers(model, expected_len):
    """Return (parent_module, attr_name, ModuleList) for the text decoder stack.

    Qwen3_5ForConditionalGeneration nests the text model under a multimodal
    wrapper and the exact path has moved between transformers versions, so
    locate it by shape instead of hardcoding model.model.language_model.layers.
    """
    hits = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == expected_len:
            # vision tower has its own stack (depth 27); guard on hidden size
            hits.append((name, mod))
    if not hits:
        raise RuntimeError(
            f"no ModuleList of length {expected_len} found. Inspect the model "
            f"structure manually and pass the path explicitly."
        )
    if len(hits) > 1:
        names = [h[0] for h in hits]
        raise RuntimeError(f"ambiguous decoder stack, candidates: {names}")
    name, layers = hits[0]
    parent_path, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return parent, attr, layers


def text_config(config):
    return getattr(config, "text_config", config)


# ----------------------------------------------------------------------------
# ablation harness
# ----------------------------------------------------------------------------

class Identity(nn.Module):
    """Residual passthrough standing in for a decoder block.

    Decoder layer forward signatures differ across transformers versions --
    some return a bare tensor, some a tuple. Mirror whatever the real layer
    returns so the loop upstream does not care.
    """

    def __init__(self, returns_tuple):
        super().__init__()
        self.returns_tuple = returns_tuple

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,) if self.returns_tuple else hidden_states


def detect_return_style(layers, sample_out):
    return isinstance(sample_out, tuple)


@torch.no_grad()
def forward_logits(model, batch, positions):
    """Run the model, return logits at `positions` only (fp32, on CPU).

    Keeping full logits would be 8192 x 248320 x 4B = 8 GB per sequence, so
    subsample positions up front.
    """
    out = model(input_ids=batch, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    sel = logits[:, positions, :].float()
    del out, logits
    return sel.cpu()


@torch.no_grad()
def score_blocks(model, layers, candidates, batches, positions_per_batch,
                 ref_logprobs, returns_tuple, verbose=True):
    """Mean forward-KL( full || ablated ) for each candidate block, in nats."""
    scores = {}
    for n, idx in enumerate(candidates):
        original = layers[idx]
        layers[idx] = Identity(returns_tuple)
        total, count = 0.0, 0
        for batch, positions, ref in zip(batches, positions_per_batch, ref_logprobs):
            got = forward_logits(model, batch, positions)
            lp = F.log_softmax(got, dim=-1)
            kl = F.kl_div(lp, ref, log_target=True, reduction="none").sum(-1)
            total += kl.sum().item()
            count += kl.numel()
            del got, lp, kl
        layers[idx] = original
        scores[idx] = total / max(count, 1)
        if verbose:
            print(f"    [{n+1:>2}/{len(candidates)}] layer {idx:>2}  KL={scores[idx]:.5f}",
                  flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return scores


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------

@dataclass
class Layout:
    layer_types: list
    protect_first_periods: int = 1
    protect_last_periods: int = 1
    max_per_period: int = 2

    def periods(self):
        """Group layer indices into full_attention-terminated periods."""
        out, cur = [], []
        for i, t in enumerate(self.layer_types):
            cur.append(i)
            if t == "full_attention":
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out

    def candidates(self):
        """Linear-attention layers eligible for removal."""
        pers = self.periods()
        lo = self.protect_first_periods
        hi = len(pers) - self.protect_last_periods
        elig = []
        for p_i, p in enumerate(pers):
            if not (lo <= p_i < hi):
                continue
            elig += [i for i in p if self.layer_types[i] == "linear_attention"]
        return elig

    def period_of(self, idx):
        for p_i, p in enumerate(self.periods()):
            if idx in p:
                return p_i
        raise KeyError(idx)

    def budget(self):
        pers = self.periods()
        lo, hi = self.protect_first_periods, len(pers) - self.protect_last_periods
        return (hi - lo) * self.max_per_period


def greedy_select(model, layers, layout, batches, positions, ref_logprobs,
                  returns_tuple, n_remove, rescore_every):
    """Iteratively remove the least-damaging block, re-scoring periodically.

    One-shot scoring misses interaction: two individually-cheap adjacent blocks
    can be jointly expensive. Re-scoring every k removals catches most of that
    without paying for a full greedy sweep.
    """
    removed, per_period = [], {}
    scores, stale = None, True
    live = nn.ModuleList([l for l in layers])  # working reference

    for step in range(n_remove):
        if stale or scores is None:
            pool = [i for i in layout.candidates()
                    if i not in removed
                    and per_period.get(layout.period_of(i), 0) < layout.max_per_period]
            if not pool:
                raise RuntimeError(
                    f"exhausted candidates after {step} removals; raise "
                    f"--max-per-period or lower --remove")
            print(f"  scoring {len(pool)} candidates (step {step+1}/{n_remove})...")
            # ablate everything already removed, then score the pool on top
            saved = {i: layers[i] for i in removed}
            for i in removed:
                layers[i] = Identity(returns_tuple)
            scores = score_blocks(model, layers, pool, batches, positions,
                                  ref_logprobs, returns_tuple)
            for i, m in saved.items():
                layers[i] = m
            stale = False

        eligible = {i: s for i, s in scores.items()
                    if i not in removed
                    and per_period.get(layout.period_of(i), 0) < layout.max_per_period}
        if not eligible:
            stale = True
            continue
        pick = min(eligible, key=eligible.get)
        removed.append(pick)
        per_period[layout.period_of(pick)] = per_period.get(layout.period_of(pick), 0) + 1
        print(f"  -> remove layer {pick:>2} (KL={scores[pick]:.5f}); "
              f"{len(removed)}/{n_remove}")
        del scores[pick]
        if (step + 1) % rescore_every == 0:
            stale = True

    return sorted(removed)


# ----------------------------------------------------------------------------
# surgery
# ----------------------------------------------------------------------------

def renumber(layers):
    """Reset every layer_idx in the stack to its new position.

    Hybrid caches map layer_idx -> cache slot for BOTH the KV cache (full
    attention) and the recurrent conv/state cache (DeltaNet). Skip this and the
    model runs and emits fluent garbage.
    """
    fixed = 0
    for new_idx, block in enumerate(layers):
        for mod in [block] + list(block.modules()):
            if hasattr(mod, "layer_idx"):
                mod.layer_idx = new_idx
                fixed += 1
    return fixed


def apply_surgery(model, parent, attr, layers, removed, cfg):
    keep = [i for i in range(len(layers)) if i not in set(removed)]
    new_layers = nn.ModuleList([layers[i] for i in keep])
    setattr(parent, attr, new_layers)

    tcfg = text_config(cfg)
    old_types = list(tcfg.layer_types)
    new_types = [old_types[i] for i in keep]
    tcfg.layer_types = new_types
    tcfg.num_hidden_layers = len(new_types)

    # The layout is no longer uniform, so a fixed interval cannot describe it.
    # Some runtimes REGENERATE layer_types from this field instead of reading
    # the explicit list -- leaving a stale value here is a silent corruption.
    if hasattr(tcfg, "full_attention_interval"):
        tcfg.full_attention_interval = None

    n_fixed = renumber(new_layers)
    return new_types, keep, n_fixed


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# calibration data
# ----------------------------------------------------------------------------

def build_batches(tokenizer, path, seq_len, n_seqs, n_positions, seed):
    rng = random.Random(seed)
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else str(obj))
    if not texts:
        raise ValueError(f"no usable rows in {path}")

    ids = []
    buf = []
    for t in texts:
        buf.extend(tokenizer(t, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            ids.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(ids) >= n_seqs:
            break
    if len(ids) < n_seqs:
        raise ValueError(
            f"calibration data yields only {len(ids)} sequences of {seq_len} "
            f"tokens; need {n_seqs}. Use longer documents -- short snippets "
            f"systematically under-weight the DeltaNet layers.")

    batches, positions = [], []
    for seq in ids[:n_seqs]:
        batches.append(torch.tensor([seq]))
        # bias sampling toward later positions: recurrent-state damage
        # accumulates along the sequence and is invisible near the start
        pool = list(range(seq_len // 4, seq_len))
        positions.append(sorted(rng.sample(pool, min(n_positions, len(pool)))))
    return batches, positions


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--calib", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--out", default="./Qwen3.8-18B")
    ap.add_argument("--remove", type=int, default=None,
                    help="number of linear_attention blocks to drop (default: solve for --target-params)")
    ap.add_argument("--target-params", type=float, default=18.0e9)
    ap.add_argument("--seq-len", type=int, default=8192,
                    help="calibration sequence length; 32768 strongly preferred")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--n-positions", type=int, default=512,
                    help="logit positions sampled per sequence for the KL")
    ap.add_argument("--max-per-period", type=int, default=2,
                    help="max linear layers removable from each 4-layer period")
    ap.add_argument("--protect-first", type=int, default=1)
    ap.add_argument("--protect-last", type=int, default=1)
    ap.add_argument("--rescore-every", type=int, default=4)
    ap.add_argument("--drop-mtp", action="store_true",
                    help="delete the MTP draft head (-0.38B; it is invalid after pruning anyway)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
    try:
        from transformers import AutoModelForMultimodalLM as AutoModel_
    except ImportError:
        AutoModel_ = AutoModelForCausalLM

    print(f"loading {args.model} ...")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tcfg = text_config(cfg)
    n_layers = tcfg.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel_.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    base_params = count_params(model)
    print(f"loaded: {base_params/1e9:.3f}B params, {n_layers} layers")

    parent, attr, layers = find_decoder_layers(model, n_layers)
    layout = Layout(list(tcfg.layer_types),
                    protect_first_periods=args.protect_first,
                    protect_last_periods=args.protect_last,
                    max_per_period=args.max_per_period)

    n_lin = sum(1 for t in layout.layer_types if t == "linear_attention")
    n_full = sum(1 for t in layout.layer_types if t == "full_attention")
    print(f"layout: {n_lin} linear_attention, {n_full} full_attention, "
          f"{len(layout.periods())} periods")
    print(f"eligible for removal: {len(layout.candidates())}, "
          f"budget under constraints: {layout.budget()}")

    # measure one block's cost empirically rather than trusting the table
    probe = layers[layout.candidates()[0]]
    block_params = sum(p.numel() for p in probe.parameters())
    print(f"linear_attention block: {block_params/1e6:.1f}M params")

    if args.remove is None:
        n_remove = max(0, math.ceil((base_params - args.target_params) / block_params))
        print(f"solving for {args.target_params/1e9:.2f}B -> remove {n_remove} blocks")
    else:
        n_remove = args.remove
    projected = (base_params - n_remove * block_params) / 1e9
    print(f"projected: {projected:.2f}B, {n_layers - n_remove} layers")
    if n_remove > layout.budget():
        sys.exit(f"ERROR: {n_remove} exceeds budget {layout.budget()}. "
                 f"Raise --max-per-period (degrades the alternation) or accept a larger model.")

    print(f"\nbuilding calibration set ({args.n_seqs} x {args.seq_len} tokens)...")
    if args.seq_len < 16384:
        print("  WARNING: short calibration under-weights DeltaNet layers. Their "
              "per-token output delta looks small while their contribution to "
              "state maintenance across 100K tokens is large. Prefer 32768.")
    batches, positions = build_batches(tok, args.calib, args.seq_len,
                                       args.n_seqs, args.n_positions, args.seed)
    dev = next(model.parameters()).device
    batches = [b.to(dev) for b in batches]

    print("computing reference logprobs...")
    ref_logprobs = []
    for b, p in zip(batches, positions):
        ref_logprobs.append(F.log_softmax(forward_logits(model, b, p), dim=-1))
    gc.collect(); torch.cuda.empty_cache()

    # probe the decoder return signature once
    with torch.no_grad():
        sample = layers[0](torch.zeros(1, 4, tcfg.hidden_size,
                                       dtype=torch.bfloat16, device=dev))
    returns_tuple = isinstance(sample, tuple)
    del sample

    print(f"\ngreedy selection ({n_remove} removals, rescore every "
          f"{args.rescore_every})...")
    removed = greedy_select(model, layers, layout, batches, positions,
                            ref_logprobs, returns_tuple, n_remove,
                            args.rescore_every)

    print(f"\nremoving layers: {removed}")
    per_period = {}
    for i in removed:
        per_period.setdefault(layout.period_of(i), []).append(i)
    for p in sorted(per_period):
        print(f"  period {p:>2}: dropped {per_period[p]}")

    report = {
        "base_model": args.model,
        "removed_layers": removed,
        "base_params": base_params,
        "seq_len": args.seq_len,
        "n_seqs": args.n_seqs,
        "constraints": {
            "max_per_period": args.max_per_period,
            "protect_first": args.protect_first,
            "protect_last": args.protect_last,
        },
    }

    if args.dry_run:
        print("\n--dry-run: not writing weights")
        print(json.dumps(report, indent=2))
        return

    new_types, keep, n_fixed = apply_surgery(model, parent, attr, layers, removed, cfg)
    print(f"renumbered layer_idx on {n_fixed} modules")

    if args.drop_mtp:
        for name, _ in list(model.named_children()):
            if "mtp" in name.lower():
                delattr(model, name)
                print(f"dropped MTP head: {name}")
        if hasattr(tcfg, "mtp_num_hidden_layers"):
            tcfg.mtp_num_hidden_layers = 0

    final = count_params(model)
    print(f"\nfinal: {final/1e9:.3f}B params, {len(new_types)} layers "
          f"({sum(1 for t in new_types if t=='linear_attention')} linear, "
          f"{sum(1 for t in new_types if t=='full_attention')} full)")

    os.makedirs(args.out, exist_ok=True)
    print(f"saving to {args.out} ...")
    model.config = cfg
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    except Exception as e:
        print(f"  (processor not saved: {e}; copy preprocessor configs by hand "
              f"or vision input will break)")

    report["final_params"] = final
    report["new_layer_types"] = new_types
    with open(os.path.join(args.out, "pruning_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\ndone. This checkpoint is NOT usable yet -- it needs healing.")


if __name__ == "__main__":
    main()
