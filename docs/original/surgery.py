#!/usr/bin/env python3
"""
Streaming depth surgery for Qwen3.8-27B. No GPU, no full model load.

Reads the safetensors shards one tensor at a time, drops the tensors belonging
to removed layers, renumbers the survivors, and writes new shards. Peak memory
is one tensor (~250MB), so this runs on a laptop. You need ~60GB of free disk
for the source checkpoint plus ~40GB for the output.

Only linear_attention layers are removable. All 16 full_attention layers are
preserved -- they carry the KV cache and all exact retrieval, and losing any of
them wrecks long-context recall in a way short benchmarks will not show you.

  python surgery.py --src ./Qwen3.8-27B --out ./Qwen3.8-22B \
      --remove 4,6,9,13,17,22,25,29,33,38,41,45 --drop-vision --drop-mtp

Get the --remove list from the scoring pass. If you have not scored yet, use
--auto to take an evenly spaced default, but understand that is a placeholder:
it removes layers by position, not by measured damage.
"""

import argparse, json, os, re, shutil, sys
from collections import defaultdict


def load_index(src):
    """Return (shard_map, is_sharded). shard_map: tensor_name -> shard filename."""
    idx_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)["weight_map"], True
    single = os.path.join(src, "model.safetensors")
    if not os.path.exists(single):
        sys.exit(f"no safetensors found in {src}")
    from safetensors import safe_open
    with safe_open(single, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}, False


LAYER_RE = re.compile(r"^(.*?\.layers\.)(\d+)(\..*)$")


def detect_stack(weight_map, expect_layers):
    """Find the text decoder prefix.

    The vision tower has its own `.layers.` namespace (depth 27), so pick the
    prefix whose highest index matches the text stack, not the first match.
    """
    maxima = defaultdict(int)
    for name in weight_map:
        m = LAYER_RE.match(name)
        if m:
            maxima[m.group(1)] = max(maxima[m.group(1)], int(m.group(2)))
    hits = [p for p, mx in maxima.items() if mx == expect_layers - 1]
    if not hits:
        sys.exit(f"no prefix with {expect_layers} layers. Found: "
                 + ", ".join(f"{p}->{mx+1}" for p, mx in maxima.items()))
    if len(hits) > 1:
        sys.exit(f"ambiguous text stack: {hits}")
    return hits[0], dict(maxima)


def plan(weight_map, prefix, removed, drop_vision, drop_mtp):
    """Build old_name -> new_name, with None meaning drop."""
    removed = set(removed)
    kept = [i for i in range(64) if i not in removed]
    remap = {old: new for new, old in enumerate(kept)}

    mapping, dropped_bytes_keys = {}, []
    for name in weight_map:
        if drop_vision and (name.startswith("visual.") or ".visual." in name
                            or name.startswith("model.visual.")):
            dropped_bytes_keys.append(name)
            continue
        if drop_mtp and "mtp" in name.lower():
            dropped_bytes_keys.append(name)
            continue
        m = LAYER_RE.match(name)
        if m and m.group(1) == prefix:
            idx = int(m.group(2))
            if idx in removed:
                dropped_bytes_keys.append(name)
                continue
            mapping[name] = f"{m.group(1)}{remap[idx]}{m.group(3)}"
        else:
            mapping[name] = name
    return mapping, dropped_bytes_keys, kept, remap


def rewrite_config(src, out, kept, drop_vision, drop_mtp):
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    t = cfg.get("text_config", cfg)

    old_types = t["layer_types"]
    new_types = [old_types[i] for i in kept]
    t["layer_types"] = new_types
    t["num_hidden_layers"] = len(new_types)

    # The layout is no longer uniform. Some runtimes REGENERATE layer_types from
    # this field instead of reading the list, which would silently rebuild the
    # original 3:1 pattern over the wrong number of layers.
    t.pop("full_attention_interval", None)

    if drop_mtp:
        t["mtp_num_hidden_layers"] = 0
        t.pop("mtp_use_dedicated_embeddings", None)
    if drop_vision:
        cfg.pop("vision_config", None)
        cfg["language_model_only"] = True
        # architecture must change or the loader looks for a vision tower
        cfg["architectures"] = ["Qwen3_5ForCausalLM"]
        cfg["model_type"] = t.get("model_type", "qwen3_5_text")

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return new_types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unpacked Qwen3.8-27B safetensors dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove", default=None,
                    help="comma-separated layer indices from the scoring pass")
    ap.add_argument("--auto", type=int, default=None,
                    help="PLACEHOLDER: evenly spaced removals, ignores measured damage")
    ap.add_argument("--drop-vision", action="store_true")
    ap.add_argument("--drop-mtp", action="store_true")
    ap.add_argument("--shard-size", type=float, default=4.5, help="GB per output shard")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    t = cfg.get("text_config", cfg)
    n_layers = t["num_hidden_layers"]
    layer_types = t["layer_types"]

    weight_map, sharded = load_index(args.src)
    prefix, maxima = detect_stack(weight_map, n_layers)
    print(f"text stack: {prefix}*  ({n_layers} layers)")
    for p, mx in maxima.items():
        if p != prefix:
            print(f"  (other stack, untouched: {p}* -> {mx+1} layers)")

    linear = [i for i, ty in enumerate(layer_types) if ty == "linear_attention"]
    if args.remove:
        removed = sorted(int(x) for x in args.remove.split(","))
    elif args.auto:
        # protect first and last period, spread the rest
        pool = [i for i in linear if 4 <= i < n_layers - 4]
        step = len(pool) / args.auto
        removed = sorted({pool[int(k * step)] for k in range(args.auto)})
        print("WARNING: --auto ignores measured damage. Score first.")
    else:
        sys.exit("pass --remove or --auto")

    bad = [i for i in removed if layer_types[i] != "linear_attention"]
    if bad:
        sys.exit(f"refusing: layers {bad} are full_attention. Removing them "
                 f"destroys long-context retrieval.")

    mapping, dropped, kept, remap = plan(
        weight_map, prefix, removed, args.drop_vision, args.drop_mtp)
    new_types = [layer_types[i] for i in kept]

    print(f"\nremoving {len(removed)} layers: {removed}")
    print(f"result: {len(kept)} layers "
          f"({new_types.count('linear_attention')} linear, "
          f"{new_types.count('full_attention')} full)")
    print(f"tensors: {len(mapping)} kept, {len(dropped)} dropped")

    # verify every period retains at least one linear layer
    period, held = [], []
    for i in range(n_layers):
        period.append(i)
        if layer_types[i] == "full_attention":
            held.append(period); period = []
    for pi, blk in enumerate(held):
        surv = [i for i in blk if i in kept and layer_types[i] == "linear_attention"]
        if not surv:
            print(f"  WARNING: period {pi} has no DeltaNet layer left")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(args.out, exist_ok=True)

    # group source reads by shard so each file is opened once
    by_shard = defaultdict(list)
    for old in mapping:
        by_shard[weight_map[old]].append(old)

    limit = int(args.shard_size * 1e9)
    buf, buf_bytes, shard_i, new_map = {}, 0, 0, {}
    total_bytes = 0

    def flush():
        nonlocal buf, buf_bytes, shard_i
        if not buf:
            return
        shard_i += 1
        fn = f"model-{shard_i:05d}.safetensors"
        save_file(buf, os.path.join(args.out, fn), metadata={"format": "pt"})
        for k in buf:
            new_map[k] = fn
        print(f"  wrote {fn}  ({buf_bytes/1e9:.2f} GB, {len(buf)} tensors)")
        buf, buf_bytes = {}, 0

    print()
    for shard in sorted(by_shard):
        with safe_open(os.path.join(args.src, shard), framework="pt") as f:
            for old in by_shard[shard]:
                tensor = f.get_tensor(old)
                nbytes = tensor.numel() * tensor.element_size()
                buf[mapping[old]] = tensor
                buf_bytes += nbytes
                total_bytes += nbytes
                if buf_bytes >= limit:
                    flush()
    flush()

    # rename to the standard N-of-M convention
    final = {}
    for i in range(1, shard_i + 1):
        src_fn = f"model-{i:05d}.safetensors"
        dst_fn = f"model-{i:05d}-of-{shard_i:05d}.safetensors"
        os.rename(os.path.join(args.out, src_fn), os.path.join(args.out, dst_fn))
        final[src_fn] = dst_fn
    new_map = {k: final[v] for k, v in new_map.items()}

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes},
                   "weight_map": new_map}, f, indent=2)

    rewrite_config(args.src, args.out, kept, args.drop_vision, args.drop_mtp)

    for fn in os.listdir(args.src):
        if fn.startswith(("tokenizer", "vocab", "merges", "chat_template",
                          "generation_config", "special_tokens")):
            shutil.copy2(os.path.join(args.src, fn), os.path.join(args.out, fn))
        if not args.drop_vision and fn.startswith(("preprocessor", "video_preprocessor")):
            shutil.copy2(os.path.join(args.src, fn), os.path.join(args.out, fn))

    with open(os.path.join(args.out, "pruning_report.json"), "w") as f:
        json.dump({"src": args.src, "removed_layers": removed,
                   "kept_layers": kept, "new_layer_types": new_types,
                   "dropped_vision": args.drop_vision,
                   "dropped_mtp": args.drop_mtp,
                   "total_bytes": total_bytes}, f, indent=2)

    print(f"\n{total_bytes/1e9:.2f} GB written to {args.out}")
    print(f"params ~{total_bytes/2e9:.2f}B (bf16)")
    print("\nThis checkpoint loads but is NOT trained. Heal before judging it.")


if __name__ == "__main__":
    main()
