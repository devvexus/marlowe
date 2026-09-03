"""Held-out evaluation during Stage 6, so a flat run is visible in minutes not days.

The training loss tells you the student is fitting the cache. It cannot tell you the student
is *generalising*, and Stage 6 runs for days. So a small held-out cache is built alongside the
training one, from `kl_reference.jsonl` -- the corpus that is disjoint from healing by
construction -- and the student is forwarded over it periodically with no gradient.

**This is not the ship gate, and the numbers are not comparable to it.** The gate is full-
distribution KL against a Q8_0 parent, measured through llama.cpp. This is top-K KL against
the NF4 teacher's cached distribution, measured in torch. Different reference, different
precision, different tail treatment. It is a trend signal: is the number falling, and is it
still falling. Reading a gate verdict off it would be comparing two things that share a name.

The step-0 row exists for the same reason. Without a measurement in these units before any
training, "0.28 at 10M tokens" is unanchored -- there is nothing to say whether that is
progress or where it started.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil

log = logutil.get("heldout")

#: Tokens in the held-out cache. Small on purpose: it is forwarded every 500K training tokens,
#: so its cost is paid ~70 times over a 35M-token run. 100K keeps each pass under a minute
#: while giving a stable enough mean to see a trend.
HELDOUT_TOKENS = 100_000

#: How often the held-out pass runs, in training tokens.
EVAL_EVERY_TOKENS = 500_000

#: Checkpoint interval. 2.5M rather than 10M: at ~190 tok/s a 10M interval is nearly four
#: hours of unrecoverable work if the machine dies, and this one has kernel-panicked under
#: sustained GPU load.
CHECKPOINT_EVERY_TOKENS = 2_500_000

#: Keep the last four checkpoints plus every fifth, so a long run keeps a coarse history
#: without filling the disk.
KEEP_LAST = 4
KEEP_EVERY = 5


@dataclass
class HeldoutResult:
    """One held-out measurement, in top-K-vs-NF4-teacher units."""

    step: int
    tokens: int
    kl_mean: float
    top1_agreement: float
    n_sequences: int
    elapsed_s: float
    wall: str = ""

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "heldout"
        # Stamped on every row so a file read six weeks from now cannot be mistaken for gate
        # numbers, whatever the surrounding context has been lost.
        d["units"] = "topk-kl-vs-nf4-teacher"
        d["not_the_ship_gate"] = True
        return d


@dataclass
class ProgressWriter:
    """Append-only JSONL the operator can tail while the run is in flight.

    Append-only and flushed per row: a reader may open it at any moment, and a rewrite would
    hand them a truncated file. One row per event, so `marlowe watch` can tail rather than
    re-parse.
    """

    path: Path
    _fh: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_progress(path: str | Path) -> list[dict[str, Any]]:
    """Read a progress file, tolerating a torn final line.

    The writer flushes per row, but a reader can still catch a partial write. A truncated last
    line is normal and must not read as corruption.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def smoothed(rows: list[dict[str, Any]], key: str, window: int = 100) -> float | None:
    """Mean of the last ``window`` values of ``key``. None when there are none."""
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    if not vals:
        return None
    tail = vals[-window:]
    return sum(tail) / len(tail)


def step_zero(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The baseline row. Its absence is a hard error at the start of Stage 6, not a warning."""
    for r in rows:
        if r.get("kind") == "heldout" and r.get("step") == 0:
            return r
    return None


class MissingStepZero(RuntimeError):
    """Stage 6 was asked to train without a held-out baseline in the same units."""


def require_step_zero(progress_path: str | Path) -> dict[str, Any]:
    """Hard precondition for Stage 6.

    Without a step-0 row every later held-out number is unanchored: the 10M decision rule is
    "below 0.25 and falling", and "falling" needs something to fall from. Measuring the
    baseline after training has begun is not the same measurement.
    """
    row = step_zero(read_progress(progress_path))
    if row is None:
        raise MissingStepZero(
            f"no step-0 held-out row in {progress_path}. Stage 6 will not start without a "
            f"baseline measured in the same units as the periodic evaluation -- top-K KL "
            f"against the NF4 teacher, over the held-out cache, before any optimizer step. "
            f"Run the step-0 evaluation first."
        )
    return row


def checkpoints_to_keep(indices: list[int]) -> list[int]:
    """The last :data:`KEEP_LAST`, plus every :data:`KEEP_EVERY`-th. Sorted."""
    if not indices:
        return []
    ordered = sorted(indices)
    keep = set(ordered[-KEEP_LAST:])
    keep.update(i for n, i in enumerate(ordered, start=1) if n % KEEP_EVERY == 0)
    return sorted(keep)


def decision(
    heldout_rows: list[dict[str, Any]], *, low: float = 0.25, high: float = 0.30
) -> tuple[str, str]:
    """The 10M rule, evaluated on the held-out number rather than the training loss.

    Returns (verdict, reason). Verdicts: ``continue``, ``stop``, ``undecided``.
    """
    if len(heldout_rows) < 2:
        return "undecided", "fewer than two held-out measurements"
    latest = heldout_rows[-1]["kl_mean"]
    previous = heldout_rows[-2]["kl_mean"]
    falling = latest < previous
    if latest < low and falling:
        return "continue", f"held-out KL {latest:.4f} < {low} and falling"
    if latest > high and not falling:
        return "stop", (
            f"held-out KL {latest:.4f} > {high} and not falling "
            f"(previous {previous:.4f}). The token budget is insufficient; whether to extend "
            f"it is the operator's call, not the run's."
        )
    return "undecided", f"held-out KL {latest:.4f}, {'falling' if falling else 'flat or rising'}"


def flat_start_warning(
    rows: list[dict[str, Any]], *, tokens: int = 500_000, tol: float = 1e-3
) -> str | None:
    """Report a flat training loss over the first ``tokens`` immediately, not at 10M.

    A curve that has not moved in the first half-million tokens is not going to be rescued by
    the next thirty-four and a half million, and finding out at 10M costs most of a day.
    """
    early = [r for r in rows if r.get("kind") == "train" and r.get("tokens", 0) <= tokens]
    if len(early) < 10:
        return None
    first = sum(r["kl"] for r in early[:5]) / 5
    last = sum(r["kl"] for r in early[-5:]) / 5
    if abs(first - last) < tol:
        return (
            f"training KL is flat over the first {tokens:,} tokens "
            f"({first:.5f} -> {last:.5f}, change {abs(first - last):.2e}). Report this now "
            f"rather than at the 10M checkpoint: a curve that has not moved here will not be "
            f"rescued by the remaining budget, and waiting costs most of a day."
        )
    return None
