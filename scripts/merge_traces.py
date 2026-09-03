"""Build the healing corpus at exactly the token budget: all traces, documents subsampled.

Both kinds of trace go in. The healing loss is per-token KL against the teacher's
distribution, so a truncated reasoning block is still valid reasoning tokens -- there is
simply no stop signal at the end of it, and KL never asks for one.

**Phase B must exclude the truncated ones.** SFT trains on hard labels, and a hard label on a
reasoning block that stops mid-thought teaches the model not to conclude. That is the
opposite of the uplift Phase B exists to produce, so ``meta.complete`` is written on every row
and is not optional metadata.

**The budget is met here, not downstream.** Emitting a 40M-token file and letting something
later truncate to 35M would be a silent disaster: truncation is positional, traces are
appended last, so the traces -- the entire point of the exercise -- would be the first thing
cut. Documents are subsampled instead, seeded and recorded, with the proof-pile-2 /
fineweb-edu ratio preserved inside the subsample so the mix is not quietly re-weighted.

Why so many traces are truncated: generation ran at a 4096-token cap, set as a guard against
the parent circling at depth rather than sized against the distribution it was meant to
bound. It landed in the middle of the mass -- p50 = p90 = max = 4096 -- so 86% hit it and 76%
contain a thinking block with no answer. A probe at 24576 measured completed traces from
3,996 to 15,137 tokens, with math prompts exceeding 24,576 outright.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_traces(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    return rows


def to_corpus_row(r: dict[str, Any]) -> dict[str, Any]:
    """One corpus document: the chat-templated exchange with the thinking block intact.

    The thinking block IS the behaviour being transferred, so it is preserved verbatim and
    marked, never merged into the answer or stripped.
    """
    think = (r.get("reasoning") or "").strip()
    answer = (r.get("content") or "").strip()
    cap_hit = bool((r.get("metrics") or {}).get("cap_hit", False))
    parts = [f"<|im_start|>user\n{r['prompt']}<|im_end|>", "<|im_start|>assistant"]
    if think:
        parts.append(f"<think>\n{think}\n</think>")
    if answer:
        parts.append(answer)
    complete = bool(answer) and not cap_hit
    if complete:
        parts.append("<|im_end|>")
    return {
        "text": "\n".join(parts),
        "meta": {
            "source": "phase-a-trace",
            "kind": r.get("kind"),
            "id": r.get("id"),
            "tokens": r.get("completion_tokens", 0),
            # Load-bearing. Phase B filters on this; see the module docstring.
            "complete": complete,
            "has_answer": bool(answer),
            "cap_hit": cap_hit,
            "rep8": (r.get("metrics") or {}).get("rep8"),
        },
    }


def load_documents(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def subsample_documents(
    docs: list[dict[str, Any]], budget_tokens: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take documents up to ``budget_tokens``, preserving each source's token share.

    Proportional per source rather than a flat random draw: a flat draw preserves the ratio
    only in expectation, and the whole point of the healing mix -- ~65% proof-pile-2, ~35%
    fineweb-edu -- is that it was chosen deliberately.
    """
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for d in docs:
        by_source[(d.get("meta") or {}).get("source", "unknown")].append(d)

    source_tokens = {
        s: sum((d.get("meta") or {}).get("tokens", 0) for d in v) for s, v in by_source.items()
    }
    total = sum(source_tokens.values())
    if total <= budget_tokens:
        return docs, {
            "subsampled": False,
            "reason": f"corpus is {total} tokens, at or under the {budget_tokens} budget",
            "source_tokens": source_tokens,
        }

    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    kept_tokens: dict[str, int] = {}
    for source, pool in sorted(by_source.items()):
        share = source_tokens[source] / total
        target = int(budget_tokens * share)
        rng.shuffle(pool)
        got = 0
        for d in pool:
            t = (d.get("meta") or {}).get("tokens", 0)
            if got + t > target and got > 0:
                continue
            kept.append(d)
            got += t
            if got >= target:
                break
        kept_tokens[source] = got

    rng.shuffle(kept)
    return kept, {
        "subsampled": True,
        "seed": seed,
        "source_tokens_before": source_tokens,
        "source_tokens_after": kept_tokens,
        "ratio_before": {s: round(v / total, 4) for s, v in source_tokens.items()},
        "ratio_after": {
            s: round(v / max(sum(kept_tokens.values()), 1), 4) for s, v in kept_tokens.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces_raw.jsonl")
    ap.add_argument("--heal", default="data/heal_corpus.jsonl")
    ap.add_argument("--out", default="data/heal_corpus_phase_a.jsonl")
    ap.add_argument("--reference", default="data/kl_reference.jsonl")
    ap.add_argument("--manifest", default="runs/marlowe-22b/manifests/heal_corpus_phase_a.json")
    ap.add_argument("--budget-tokens", type=int, default=35_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traces = [to_corpus_row(t) for t in load_traces(Path(args.traces))]
    complete = [r for r in traces if r["meta"]["complete"]]
    truncated = [r for r in traces if not r["meta"]["complete"]]
    tok_complete = sum(r["meta"]["tokens"] for r in complete)
    tok_truncated = sum(r["meta"]["tokens"] for r in truncated)
    trace_tokens = tok_complete + tok_truncated

    doc_budget = max(0, args.budget_tokens - trace_tokens)
    docs = load_documents(Path(args.heal))
    kept, sub = subsample_documents(docs, doc_budget, args.seed)
    doc_tokens = sum((d.get("meta") or {}).get("tokens", 0) for d in kept)

    rows = kept + traces
    random.Random(args.seed).shuffle(rows)

    # Disjointness on the FINAL file, not on the inputs: the merge is where a collision would
    # actually enter the corpus the ship gate is scored against.
    ref_hashes = set()
    with Path(args.reference).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ref_hashes.add(_hash(json.loads(line)["text"]))
    clash = [r for r in rows if _hash(r["text"]) in ref_hashes]
    if clash:
        raise RuntimeError(
            f"{len(clash)} document(s) in the final corpus collide with the held-out KL "
            f"reference. The ship gate would be scoring its own training data."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    total = doc_tokens + trace_tokens
    manifest = {
        "out": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "budget_tokens": args.budget_tokens,
        "total_tokens": total,
        "documents": len(kept),
        "document_tokens": doc_tokens,
        "traces": len(traces),
        "trace_tokens": trace_tokens,
        "trace_fraction": round(trace_tokens / max(total, 1), 4),
        "complete_traces": len(complete),
        "complete_tokens": tok_complete,
        "truncated_traces": len(truncated),
        "truncated_tokens": tok_truncated,
        "subsample": sub,
        "phase_b_eligible_documents": len(complete),
        "phase_b_note": (
            "Phase B (SFT, hard labels) must filter meta.complete == true. A hard label on a "
            "reasoning block truncated mid-thought teaches the model not to conclude."
        ),
        "budget_note": (
            "Built AT the budget by subsampling documents, never traces. A larger file with "
            "downstream truncation would cut positionally, and traces are the last thing "
            "appended -- they would be the first thing lost."
        ),
        "cap_note": (
            "Traces generated at a 4096-token cap, set as a circling guard and not sized "
            "against the distribution it was meant to bound: p50 = p90 = max = 4096, 86% "
            "cap-hit, 76% with no answer. A 24576 probe measured completed traces from 3,996 "
            "to 15,137 tokens, math exceeding 24,576. Round 2 budgets by tokens, not prompts."
        ),
        "disjoint_from_kl_reference": True,
        "seed": args.seed,
    }
    mp = Path(args.manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
