"""Captured mass per POSITION, from a written cache shard.

``ShardMeta.mean_captured_mass`` is a mean of per-sequence means, and it hides the thing that
matters. Captured mass is ``sum(exp(topk_logprob))`` -- how much of the teacher's true
distribution the stored top-K accounts for. Where the teacher is confident it is ~1.0; where
the teacher is genuinely uncertain it falls, and those are the reasoning positions, the ones
a top-K objective approximates worst. Averaging over 768 positions buries them.

Nothing needs re-running to get this: the shard already stores the top-K logprobs per
position, so the full distribution is recoverable offline, on CPU.

Read it as: if p10 is high, K=64 covers even the uncertain positions and the top-K objective
is a good proxy. If p10 is low while the mean looks fine, the objective is well-specified on
the easy tokens and vague on exactly the hard ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def analyse(path: Path) -> dict:
    import numpy as np

    # Streamed in sequence-chunks and kept in float32. A whole shard's logprobs are 640 MB
    # as float16; .astype(np.float64) would make that 2.56 GB, which is not a reasonable
    # thing to allocate on a machine whose available memory is the binding constraint.
    with np.load(path) as z:
        lp_all = z["topk_logprob"]                     # (seqs, seq_len, K) float16
        n_seq = lp_all.shape[0]
        chunk = max(1, min(256, n_seq))
        parts = []
        for i in range(0, n_seq, chunk):
            block = lp_all[i : i + chunk].astype(np.float32)
            parts.append(np.exp(block).sum(axis=-1))    # (chunk, seq_len)
            del block
        mass = np.concatenate(parts, axis=0)
        del parts
        top_k = int(lp_all.shape[-1])
        ids = z["input_ids"]
    flat = mass.reshape(-1)
    q = np.percentile(flat, [1, 5, 10, 25, 50, 75, 90])
    # Per-sequence means: what the shard metadata records, for comparison.
    per_seq = mass.mean(axis=1)
    return {
        "shard": path.name,
        "sequences": int(mass.shape[0]),
        "seq_len": int(mass.shape[1]),
        "K": top_k,
        "positions": int(flat.size),
        "mean": float(flat.mean()),
        "p1": float(q[0]), "p5": float(q[1]), "p10": float(q[2]),
        "p25": float(q[3]), "median": float(q[4]), "p75": float(q[5]), "p90": float(q[6]),
        "min": float(flat.min()), "max": float(flat.max()),
        "frac_below_0_90": float((flat < 0.90).mean()),
        "frac_below_0_75": float((flat < 0.75).mean()),
        "frac_below_0_50": float((flat < 0.50).mean()),
        "mean_of_per_sequence_means": float(per_seq.mean()),
        "p10_of_per_sequence_means": float(np.percentile(per_seq, 10)),
        "tokens_checked": int(ids.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/marlowe-22b/teacher-cache")
    ap.add_argument("--out", default="runs/marlowe-22b/metrics/captured_mass.json")
    ap.add_argument("--shards", type=int, default=1, help="how many shards to analyse")
    args = ap.parse_args()

    shards = sorted(Path(args.cache).glob("shard-*.npz"))
    if not shards:
        print(f"no shards in {args.cache} yet", flush=True)
        return 1
    results = [analyse(p) for p in shards[: args.shards]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for r in results:
        print(f"\n{r['shard']}: {r['sequences']} seqs x {r['seq_len']} = "
              f"{r['positions']:,} positions, K={r['K']}", flush=True)
        print(f"  mean   {r['mean']:.4f}      <- what ShardMeta records", flush=True)
        print(f"  p10    {r['p10']:.4f}      <- the reasoning positions", flush=True)
        print(f"  p1 {r['p1']:.4f}  p5 {r['p5']:.4f}  p25 {r['p25']:.4f}  "
              f"median {r['median']:.4f}  p90 {r['p90']:.4f}", flush=True)
        print(f"  min {r['min']:.4f}  max {r['max']:.4f}", flush=True)
        print(f"  positions below 0.90: {r['frac_below_0_90']:.2%}   "
              f"below 0.75: {r['frac_below_0_75']:.2%}   "
              f"below 0.50: {r['frac_below_0_50']:.2%}", flush=True)
        print(f"  (per-sequence means: mean {r['mean_of_per_sequence_means']:.4f}, "
              f"p10 {r['p10_of_per_sequence_means']:.4f} -- averaging hides the tail)",
              flush=True)
    print("\nCAPTURED_MASS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
