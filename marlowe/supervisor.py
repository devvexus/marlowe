"""Restart supervision for multi-day training runs.

Why this exists
---------------

Stage 6 is two days on the 22B and longer on the ladder, unattended, on a desktop machine
that has kernel-panicked under sustained GPU load (bugcheck 0x139) partway through an
imatrix pass. Three failure modes are known to occur here and none of them announce
themselves usefully:

* **The machine dies.** Nothing in-process can catch that. Only a checkpoint on disk and
  something that notices the process is gone can recover it.
* **Trapped fragmentation grows.** The allocator's reserved pool creeps above what live
  tensors occupy, and a run that fit at step 500 OOMs at step 30,000. The soak measures the
  slope; the supervisor is what turns a measured slope into a bounded restart interval.
* **The run silently stops making progress.** A hung CUDA call, a driver reset that leaves
  the process alive, a dataloader blocked on I/O. Wall-clock alone cannot tell this from slow
  progress, so the supervisor watches *steps completed*, not liveness.

What it deliberately does not do
--------------------------------

It does not restart on a rising loss, and it does not tune anything. Automatic recovery is
for mechanical failures with unambiguous signatures. A training curve that turns over is a
result to look at, not a fault to retry -- and an automated system that reacts to it will
paper over exactly the evidence Stage 6's 10M-token decision point depends on.

It also does not decide *what* to restart from. The trainer owns checkpointing; this owns
noticing and relaunching.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil

log = logutil.get("supervisor")

#: Steps of no progress before the run is declared hung.
#:
#: Not a wall-clock timeout. A slow step and a dead process look identical on a clock, and
#: this machine's steps vary by a factor of two between a warm cache and a cold one.
DEFAULT_STALL_STEPS = 0

#: Seconds without the progress file advancing before declaring a stall. Generous, because
#: a false restart on a healthy run costs a checkpoint interval of work.
DEFAULT_STALL_SECONDS = 1800

#: How many times to relaunch before giving up and leaving the wreckage for a human.
#:
#: Finite on purpose. A crash loop that restarts forever burns the compute window and
#: produces a log nobody reads; three attempts distinguishes "the machine hiccuped" from
#: "this configuration does not work", which is the distinction that matters at 3 a.m.
DEFAULT_MAX_RESTARTS = 3


@dataclass
class Progress:
    """What the trainer publishes so the supervisor can tell progress from liveness."""

    step: int = 0
    tokens: int = 0
    updated_at: float = 0.0
    checkpoint: str | None = None
    trapped_bytes: int = 0

    @classmethod
    def read(cls, path: str | Path) -> Progress | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            return cls(**json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            # A half-written file is normal: the trainer may be mid-write. Treat it as "no
            # news", never as a stall -- a torn read must not trigger a restart.
            return None

    def write(self, path: str | Path) -> None:
        """Atomic, because the supervisor reads this file concurrently."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self)), encoding="utf-8")
        os.replace(tmp, p)


@dataclass
class RestartRecord:
    attempt: int
    reason: str
    at: float
    last_step: int
    returncode: int | None = None


@dataclass
class SupervisionResult:
    completed: bool
    attempts: int
    restarts: list[RestartRecord] = field(default_factory=list)
    final_step: int = 0
    gave_up_reason: str | None = None

    def render(self) -> str:
        head = "COMPLETED" if self.completed else f"GAVE UP -- {self.gave_up_reason}"
        lines = [f"supervisor: {head} after {self.attempts} attempt(s), "
                 f"last step {self.final_step}"]
        for r in self.restarts:
            lines.append(f"  restart {r.attempt}: {r.reason} at step {r.last_step} "
                         f"(rc={r.returncode})")
        return "\n".join(lines)


def supervise(
    command: list[str],
    progress_path: str | Path,
    *,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    stall_seconds: float = DEFAULT_STALL_SECONDS,
    poll_seconds: float = 30.0,
    trapped_limit_bytes: int | None = None,
    _spawn: Any = None,
    _now: Any = None,
) -> SupervisionResult:
    """Run ``command`` until it completes, restarting it on crash or stall.

    ``progress_path`` is the file the trainer updates with :class:`Progress`. The supervisor
    reads it to distinguish three states a process cannot report about itself: making
    progress, alive but stuck, and gone.

    A restart is only ever *relaunching the same command*. The trainer resumes from its own
    checkpoint; nothing here reaches into training state.

    ``trapped_limit_bytes`` restarts pre-emptively when the trainer reports allocator-trapped
    memory above a threshold. That is the measured-fragmentation case: a soak that shows a
    slope sets a restart interval, and restarting on a schedule you chose beats OOMing on one
    you did not.

    ``_spawn`` and ``_now`` are injection points for tests. Supervision logic that can only
    be exercised by crashing a real two-day job would never be exercised.
    """
    spawn = _spawn or (lambda cmd: subprocess.Popen(cmd))
    now = _now or time.monotonic

    result = SupervisionResult(completed=False, attempts=0)
    last_step = 0
    last_progress_at = now()

    for attempt in range(1, max_restarts + 2):
        result.attempts = attempt
        logutil.event(log, "launching", attempt=attempt, cmd=" ".join(map(str, command[:4])))
        proc = spawn(command)
        reason: str | None = None

        while True:
            rc = proc.poll()
            if rc is not None:
                if rc == 0:
                    result.completed = True
                    result.final_step = last_step
                    logutil.event(log, "run completed", attempt=attempt, step=last_step)
                    return result
                reason = f"exited rc={rc}"
                break

            progress = Progress.read(progress_path)
            if progress is not None and progress.step > last_step:
                last_step = progress.step
                last_progress_at = now()
                if (
                    trapped_limit_bytes is not None
                    and progress.trapped_bytes > trapped_limit_bytes
                ):
                    reason = (
                        f"trapped {progress.trapped_bytes / 1e6:.0f} MB exceeds the "
                        f"{trapped_limit_bytes / 1e6:.0f} MB restart threshold"
                    )
                    _terminate(proc)
                    break

            if now() - last_progress_at > stall_seconds:
                reason = f"no progress for {stall_seconds:.0f}s at step {last_step}"
                _terminate(proc)
                break

            time.sleep(poll_seconds) if _now is None else None

        result.final_step = last_step
        result.restarts.append(
            RestartRecord(attempt=attempt, reason=reason or "unknown", at=now(),
                          last_step=last_step, returncode=proc.poll())
        )
        logutil.event(log, "restarting", attempt=attempt, reason=reason, step=last_step)
        if attempt > max_restarts:
            result.gave_up_reason = (
                f"{max_restarts} restarts exhausted; last failure: {reason}"
            )
            log.error(
                "supervisor giving up after %d restarts. The last failure was %r at step %d. "
                "A crash loop is not recovery -- look at the run rather than restarting it "
                "again.",
                max_restarts, reason, last_step,
            )
            return result
    return result


def _terminate(proc: Any) -> None:
    """Ask, then insist. A training process holding 15 GB of VRAM should get the chance to
    release it, but not an unbounded one."""
    try:
        proc.terminate()
    except (OSError, AttributeError):
        return
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        proc.kill()
    except (OSError, AttributeError):
        pass


def restart_interval_from_slope(
    slope_bytes_per_step: float, headroom_bytes: int, safety: float = 0.5
) -> int | None:
    """Steps before trapped fragmentation eats the headroom, halved for safety.

    Returns None when the slope is flat or falling, which means no scheduled restart is
    needed. This is the number the soak exists to produce: a measured slope becomes a
    restart interval rather than a worry.
    """
    if slope_bytes_per_step <= 0:
        return None
    return max(1, int(safety * headroom_bytes / slope_bytes_per_step))
