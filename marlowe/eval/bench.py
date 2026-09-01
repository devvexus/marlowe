"""GPQA Diamond and needle-in-a-haystack.

Two capability checks with narrow, specific jobs:

* **GPQA Diamond (198 questions)** -- reasoning, at milestones only. Expensive in thinking
  mode, and MCQA understates degradation on this project's real workload: 58% of AAII weight
  is agentic plus coding, long generative traces where errors compound. Treat a flat GPQA as
  weak evidence, not proof.
* **Needle @128K** -- the DeltaNet check. Healing runs at 2048 tokens because that is what
  makes three days possible, and the training loss never shows whether long-context state
  maintenance survived. The 16 preserved full_attention layers carry exact retrieval, but
  the recurrent layers around them were disturbed. This is the pre-ship gate for that risk.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil
from marlowe.config import SamplingPreset, preset
from marlowe.eval.repetition import Backend

log = logutil.get("eval.bench")

LETTERS = "ABCD"


# ---------------------------------------------------------------------------
# GPQA Diamond
# ---------------------------------------------------------------------------


@dataclass
class MCQuestion:
    id: str
    question: str
    correct: str
    incorrect: list[str]


def load_gpqa(path: str | Path) -> list[MCQuestion]:
    """Load GPQA Diamond from a local CSV or JSONL export.

    The dataset is gated on HuggingFace, so this reads a file the user has already obtained
    rather than downloading. Accepts the official CSV column names or a simplified JSONL.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"GPQA Diamond not found at {p}. The dataset is gated: accept the terms at "
            f"huggingface.co/datasets/Idavidrein/gpqa and export gpqa_diamond.csv here."
        )

    rows: list[MCQuestion] = []
    if p.suffix.lower() == ".csv":
        import csv

        with p.open(encoding="utf-8", newline="") as f:
            for i, r in enumerate(csv.DictReader(f)):
                rows.append(
                    MCQuestion(
                        id=str(r.get("Record ID", i)),
                        question=r["Question"],
                        correct=r["Correct Answer"],
                        incorrect=[
                            r["Incorrect Answer 1"],
                            r["Incorrect Answer 2"],
                            r["Incorrect Answer 3"],
                        ],
                    )
                )
    else:
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                rows.append(
                    MCQuestion(
                        id=str(o.get("id", i)),
                        question=o["question"],
                        correct=o["correct"],
                        incorrect=list(o["incorrect"]),
                    )
                )
    if len(rows) != 198:
        log.warning("expected 198 GPQA Diamond questions, loaded %d", len(rows))
    return rows


def format_mcq(q: MCQuestion, rng: random.Random) -> tuple[str, str]:
    """Render the question with shuffled choices. Returns (prompt, correct letter).

    Choices are shuffled per question with a seeded RNG: an unshuffled set lets a model
    score above chance from answer-position bias alone.
    """
    options = [*q.incorrect, q.correct]
    rng.shuffle(options)
    correct_letter = LETTERS[options.index(q.correct)]
    body = "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    prompt = (
        f"{q.question}\n\n{body}\n\n"
        f"Think it through, then end your reply with exactly one line of the form:\n"
        f"Answer: <letter>"
    )
    return prompt, correct_letter


_ANSWER_PATTERNS = [
    re.compile(r"Answer\s*[:\-]\s*\(?([A-D])\)?", re.I),
    re.compile(r"\b(?:final\s+)?answer\s+is\s+\(?([A-D])\)?", re.I),
    re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}"),
]


def extract_letter(text: str) -> str | None:
    """Pull the answer letter out of a completion.

    Searches from the end: in thinking mode the model often names several options while
    reasoning, and only the last statement is its answer.
    """
    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(text)
        if matches:
            return str(matches[-1]).upper()
    tail = text.strip()[-200:]
    loose = re.findall(r"\b([A-D])\b", tail)
    return loose[-1].upper() if loose else None


@dataclass
class GPQAReport:
    label: str
    backend: str
    n: int
    n_answered: int
    accuracy: float
    #: Accuracy counting unparseable replies as wrong -- the honest number, since a model
    #: that reasons forever and never states an answer has failed the task.
    accuracy_strict: float
    unparsed_rate: float
    mean_tokens: float
    cap_hit_rate: float
    preset: str = "thinking"
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_gpqa(
    backend: Backend,
    questions: Sequence[MCQuestion],
    *,
    label: str,
    max_tokens: int = 4096,
    sampling_preset: str = "thinking",
    seed: int = 0,
) -> GPQAReport:
    sampling: SamplingPreset = preset(sampling_preset)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for i, q in enumerate(questions):
        prompt, correct = format_mcq(q, rng)
        c = backend.generate(prompt, max_tokens, sampling, seed + i)
        if c.stop_reason == "error":
            rows.append({"id": q.id, "error": c.error, "correct": correct, "got": None})
            continue
        got = extract_letter(c.text)
        rows.append(
            {
                "id": q.id,
                "correct": correct,
                "got": got,
                "ok": got == correct,
                "n_tokens": c.n_tokens,
                "hit_cap": c.hit_cap,
            }
        )
        if (i + 1) % 25 == 0:
            done = [r for r in rows if r.get("got")]
            logutil.event(
                log,
                "gpqa progress",
                done=i + 1,
                of=len(questions),
                acc=round(statistics.fmean([r["ok"] for r in done]), 4) if done else None,
            )

    scored = [r for r in rows if "ok" in r]
    answered = [r for r in scored if r["got"] is not None]
    report = GPQAReport(
        label=label,
        backend=backend.name,
        n=len(rows),
        n_answered=len(answered),
        accuracy=statistics.fmean([r["ok"] for r in answered]) if answered else 0.0,
        accuracy_strict=statistics.fmean([bool(r.get("ok")) for r in rows]) if rows else 0.0,
        unparsed_rate=1.0 - (len(answered) / len(rows)) if rows else 0.0,
        mean_tokens=statistics.fmean([r.get("n_tokens", 0) for r in scored]) if scored else 0.0,
        cap_hit_rate=statistics.fmean([bool(r.get("hit_cap")) for r in scored]) if scored else 0.0,
        preset=sampling.name,
        per_question=rows,
    )
    logutil.event(
        log,
        "gpqa report",
        label=label,
        accuracy=round(report.accuracy, 4),
        strict=round(report.accuracy_strict, 4),
        unparsed=round(report.unparsed_rate, 4),
    )
    return report


# ---------------------------------------------------------------------------
# needle in a haystack
# ---------------------------------------------------------------------------

DEFAULT_NEEDLE = (
    "The maintenance access code for the Thornfield relay station is {code}, "
    "and it must be entered before the third chime."
)
DEFAULT_QUESTION = (
    "What is the maintenance access code for the Thornfield relay station? "
    "Answer with the code only."
)


def build_haystack(
    filler: str, target_tokens: int, tokenize: Any, needle: str, depth_frac: float
) -> str:
    """Repeat ``filler`` to ``target_tokens`` and insert ``needle`` at ``depth_frac``."""
    unit = tokenize(filler)
    if not unit:
        raise ValueError("filler text tokenizes to nothing")
    reps = max(1, target_tokens // len(unit) + 1)
    body = (filler + "\n") * reps
    words = body.split("\n")
    at = max(0, min(len(words) - 1, int(len(words) * depth_frac)))
    words.insert(at, needle)
    return "\n".join(words)


@dataclass
class NeedleReport:
    label: str
    backend: str
    context_tokens: int
    depths: list[float]
    #: depth fraction -> recall rate at that depth
    recall_by_depth: dict[str, float] = field(default_factory=dict)
    recall: float = 0.0
    n_trials: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_needle(
    backend: Backend,
    *,
    label: str,
    filler_path: str | Path,
    tokenizer_path: str | None,
    context_tokens: int = 131072,
    depths: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    trials_per_depth: int = 3,
    max_tokens: int = 128,
    sampling_preset: str = "thinking",
    seed: int = 0,
) -> NeedleReport:
    """Retrieval at long context. The pre-ship gate for 2K-healing damage.

    Run before shipping any checkpoint healed at short sequence length. A failure here with
    healthy short-context KL is the specific signature of unrepaired DeltaNet state
    maintenance, and the documented fallback is a final anneal at 16K.
    """
    from marlowe.eval.repetition import make_tokenizer

    sampling = preset(sampling_preset)
    tokenize, _ = make_tokenizer(tokenizer_path)
    filler = Path(filler_path).read_text(encoding="utf-8")
    rng = random.Random(seed)

    by_depth: dict[float, list[bool]] = {d: [] for d in depths}
    errors = 0
    for d in depths:
        for t in range(trials_per_depth):
            code = f"{rng.randint(100000, 999999)}"
            needle = DEFAULT_NEEDLE.format(code=code)
            haystack = build_haystack(filler, context_tokens, tokenize, needle, d)
            prompt = f"{haystack}\n\n{DEFAULT_QUESTION}"
            c = backend.generate(prompt, max_tokens, sampling, seed + int(d * 1000) + t)
            if c.stop_reason == "error":
                errors += 1
                log.warning("needle error at depth %.2f: %s", d, c.error)
                continue
            by_depth[d].append(code in c.text)
            logutil.event(
                log, "needle trial", depth=d, trial=t, found=code in c.text, ctx=context_tokens
            )

    flat = [ok for v in by_depth.values() for ok in v]
    report = NeedleReport(
        label=label,
        backend=backend.name,
        context_tokens=context_tokens,
        depths=list(depths),
        recall_by_depth={
            f"{d:.2f}": (statistics.fmean(v) if v else 0.0) for d, v in by_depth.items()
        },
        recall=statistics.fmean(flat) if flat else 0.0,
        n_trials=len(flat),
        errors=errors,
    )
    logutil.event(
        log,
        "needle report",
        label=label,
        ctx=context_tokens,
        recall=round(report.recall, 4),
        by_depth=report.recall_by_depth,
    )
    return report
