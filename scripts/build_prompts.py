"""Build the Phase A trace prompts, and prove they are safe to generate from.

Sources, per docs/TRANSFER_PLAN.md, in priority order: arXiv abstracts turned into PI-shaped
questions, math and logic problems, and synthetic research-programme tasks. About 10% are
tool-use shaped.

Two contamination rules, and they are different questions:

* **Against the KL reference.** A prompt that appears in `kl_reference.jsonl` would put trace
  text into the file the ship gate scores against, which is the same defect as a healing
  corpus overlapping the reference -- caught once already, at 112 documents. Checked here,
  and fatal.
* **Against public evaluation sets** (GPQA, HLE, AIME 2024+, MMLU-Pro, LiveCodeBench,
  SciCode). Those sets are not on this machine, so a hash intersection cannot be computed.
  What *can* be said is stated instead of assumed: every prompt is either synthesised from a
  template or derived from a named proof-pile-2 shard, and the derivation is recorded, so the
  claim "no prompt came from an eval set" rests on construction rather than on a check that
  was never run. If those sets are obtained later, `assert_no_eval_overlap` runs against them
  without rebuilding anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

#: Turned into questions a research assistant would be asked about a paper.
ARXIV_TEMPLATES = (
    "Explain the method described below and identify its limitations.\n\n{text}",
    "Identify the weakest assumption in the following work and say why it matters.\n\n{text}",
    "Propose a follow-up experiment that would test the central claim below.\n\n{text}",
    "Summarise the contribution below, then state what evidence would falsify it.\n\n{text}",
    "What would you need to measure to reproduce the result below? Be specific.\n\n{text}",
)

#: Synthetic PI tasks. No source text, so no provenance question at all.
PI_TASKS = (
    "Design an experiment to test whether {x}. State the control, the measurement, and the "
    "result that would falsify the hypothesis.",
    "Critique this study design: {x}. Identify the largest threat to validity and how you "
    "would address it.",
    "Plan a six-month research programme investigating {x}. Give milestones and the decision "
    "point at which you would abandon the approach.",
    "You have a result showing {x}, but it does not replicate. Enumerate the possible causes "
    "in order of prior probability and say how you would distinguish them.",
    "Two measurements disagree about {x}. Neither instrument is obviously wrong. How do you "
    "proceed?",
)

PI_SUBJECTS = (
    "a pruned language model recovers its parent's behaviour through distillation",
    "quantisation error concentrates in the token-mixing projections rather than the FFN",
    "long-range state in a recurrent layer degrades faster than attention under pruning",
    "importance-matrix calibration length changes which layers are judged removable",
    "repetition collapse is a sampling attractor rather than a capability loss",
    "a model's throughput is a reliable proxy for whether it fits in memory",
    "held-out corpus construction affects measured KL more than model size does",
    "chain-of-thought length correlates with answer correctness on multi-step problems",
    "an evaluation harness's cap length changes the ranking of two models",
    "distillation from a quantised teacher bounds the student below the teacher's precision",
)

#: proof-pile-2 OpenWebMath, per the plan's second priority source.
MATH_TEMPLATES = (
    "Work through the following problem. Show each step and state where the argument could "
    "fail.\n\n{text}",
    "Is the reasoning below sound? Identify any gap and repair it.\n\n{text}",
    "Solve the problem below, then give an independent check of your answer.\n\n{text}",
)

#: ~10% of the set. Shaped like tool use without requiring a live tool.
TOOL_TASKS = (
    "You have access to a shell, a Python interpreter, and a file system. Determine {x}. "
    "State each command you would run and what you would conclude from each outcome.",
    "You can query a database and plot results. Investigate {x}. Say what you would query "
    "first and why that ordering matters.",
    "Given a profiler and a debugger, diagnose {x}. Describe the measurement you would take "
    "before changing anything.",
)

TOOL_SUBJECTS = (
    "why a training run's memory use grows over hours while its allocator reports a flat pool",
    "whether a slow process is blocked on I/O, on the GPU, or on a lock",
    "which of forty configuration changes since Friday caused a regression",
    "whether a numerical discrepancy comes from precision or from a logic error",
    "why a job that fits on one machine pages on an identically-specified one",
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _abstract(text: str, max_chars: int = 1400) -> str | None:
    """First substantive paragraph of an arXiv document, as a stand-in for the abstract."""
    body = re.sub(r"\s+", " ", text[:8000]).strip()
    if len(body) < 400:
        return None
    return body[:max_chars]


def load_corpus_texts(path: Path, source: str, limit: int) -> list[str]:
    out: list[str] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # The corpus carries source under "meta", not at the top level. The mismatch was
            # caught by the shortfall guard below rather than by silently yielding an
            # all-synthetic prompt set, which is what an unchecked filter would have done.
            meta = row.get("meta") or {}
            if (meta.get("source") or row.get("source")) != source:
                continue
            a = _abstract(row.get("text", ""))
            if a:
                out.append(a)
            if len(out) >= limit:
                break
    return out


def assert_disjoint_from_reference(prompts: list[dict[str, Any]], reference: Path) -> int:
    """No prompt may reproduce a document in the held-out KL reference.

    Fatal, not advisory. The reference is what the ship gate scores against; trace text
    leaking into it would make the gate partly measure its own training data.
    """
    if not reference.exists():
        raise FileNotFoundError(
            f"{reference} not found, so disjointness against the KL reference cannot be "
            f"checked. Refusing to generate prompts whose safety is merely assumed."
        )
    ref_hashes = set()
    with reference.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ref_hashes.add(_hash(json.loads(line).get("text", "")))
    clash = [p for p in prompts if _hash(p["source_text"] or p["prompt"]) in ref_hashes]
    if clash:
        raise RuntimeError(
            f"{len(clash)} prompt(s) reproduce a document in the held-out KL reference. "
            f"Trace text derived from them would contaminate the file the ship gate scores "
            f"against."
        )
    return len(ref_hashes)


def assert_no_eval_overlap(prompts: list[dict[str, Any]], eval_hash_file: Path | None) -> str:
    """Check against public eval sets when they are available, and say so when they are not."""
    if eval_hash_file is None or not eval_hash_file.exists():
        return (
            "NOT CHECKED -- no local copy of GPQA/HLE/AIME/MMLU-Pro/LiveCodeBench/SciCode. "
            "Every prompt is synthesised from a template or derived from a named "
            "proof-pile-2 shard; the claim rests on construction, not on a hash intersection."
        )
    hashes = {line.strip() for line in eval_hash_file.read_text(encoding="utf-8").splitlines()}
    clash = [p for p in prompts if _hash(p["prompt"]) in hashes]
    if clash:
        raise RuntimeError(f"{len(clash)} prompt(s) appear in an evaluation set.")
    return f"checked against {len(hashes)} evaluation-set hashes; no overlap"


def build(n: int, heal_path: Path, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    n_tool = max(1, round(n * 0.10))
    n_pi = round((n - n_tool) * 0.35)
    n_arxiv = n - n_tool - n_pi

    prompts: list[dict[str, Any]] = []
    # One paper supports several genuinely different questions -- "what is the weakest
    # assumption" and "propose a follow-up" are different tasks over the same text -- so the
    # supply is documents x templates, not documents. There are only 611 arXiv documents in
    # the healing corpus, which the shortfall guard caught rather than letting the mix drift.
    n_math = round(n_arxiv * 0.30)
    n_arxiv -= n_math
    texts = load_corpus_texts(heal_path, "arxiv", 20_000)
    pairs = [(t, tm) for t in texts for tm in ARXIV_TEMPLATES]
    if len(pairs) < n_arxiv:
        raise RuntimeError(
            f"only {len(texts)} arXiv documents x {len(ARXIV_TEMPLATES)} templates = "
            f"{len(pairs)} distinct prompts in {heal_path}, need {n_arxiv}. Silently "
            f"shrinking the arXiv share would change the prompt mix without saying so."
        )
    rng.shuffle(pairs)
    for i in range(n_arxiv):
        text, tmpl = pairs[i]
        prompts.append({
            "id": f"arxiv-{i:04d}", "kind": "arxiv",
            "prompt": tmpl.format(text=text),
            "source_text": text,
            "provenance": "proof-pile-2 arXiv via data/heal_corpus.jsonl",
        })
    mtexts = load_corpus_texts(heal_path, "open-web-math", 20_000)
    mpairs = [(t, tm) for t in mtexts for tm in MATH_TEMPLATES]
    if len(mpairs) < n_math:
        raise RuntimeError(
            f"only {len(mpairs)} distinct OpenWebMath prompts available, need {n_math}."
        )
    rng.shuffle(mpairs)
    for i in range(n_math):
        text, tmpl = mpairs[i]
        prompts.append({
            "id": f"math-{i:04d}", "kind": "math",
            "prompt": tmpl.format(text=text),
            "source_text": text,
            "provenance": "proof-pile-2 OpenWebMath via data/heal_corpus.jsonl",
        })
    for i in range(n_pi):
        tmpl = PI_TASKS[i % len(PI_TASKS)]
        prompts.append({
            "id": f"pi-{i:04d}", "kind": "pi",
            "prompt": tmpl.format(x=PI_SUBJECTS[i % len(PI_SUBJECTS)]),
            "source_text": None, "provenance": "synthetic template",
        })
    for i in range(n_tool):
        tmpl = TOOL_TASKS[i % len(TOOL_TASKS)]
        prompts.append({
            "id": f"tool-{i:04d}", "kind": "tool",
            "prompt": tmpl.format(x=TOOL_SUBJECTS[i % len(TOOL_SUBJECTS)]),
            "source_text": None, "provenance": "synthetic template",
        })
    rng.shuffle(prompts)
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--heal", default="data/heal_corpus.jsonl")
    ap.add_argument("--reference", default="data/kl_reference.jsonl")
    ap.add_argument("--eval-hashes", default=None)
    ap.add_argument("--out", default="data/trace_prompts.jsonl")
    ap.add_argument("--manifest", default="runs/marlowe-22b/manifests/trace_prompts.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompts = build(args.n, Path(args.heal), args.seed)
    n_ref = assert_disjoint_from_reference(prompts, Path(args.reference))
    eval_note = assert_no_eval_overlap(
        prompts, Path(args.eval_hashes) if args.eval_hashes else None
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({k: v for k, v in p.items() if k != "source_text"}) + "\n")

    kinds: dict[str, int] = {}
    for p in prompts:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    manifest = {
        "n": len(prompts), "kinds": kinds, "seed": args.seed,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "disjoint_from_kl_reference": True,
        "kl_reference_documents_checked": n_ref,
        "eval_set_contamination": eval_note,
        "sources": {"arxiv": "proof-pile-2 via heal_corpus.jsonl",
                    "pi": "synthetic templates", "tool": "synthetic templates"},
    }
    mp = Path(args.manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
