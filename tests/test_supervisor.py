"""The supervisor is exercised here or it is not exercised at all.

Its whole purpose is to handle what happens at hour thirty of an unattended run on a machine
that has kernel-panicked under sustained GPU load. Waiting for a real two-day job to fail in
order to find out whether the recovery works is not a test strategy, so ``supervise`` takes
injection points for spawning and for the clock.
"""

from __future__ import annotations

import json

import pytest

from marlowe.supervisor import (
    Progress,
    restart_interval_from_slope,
    supervise,
)


class _FakeProc:
    """A process whose exit is scripted."""

    def __init__(self, codes: list[int | None]) -> None:
        self._codes = list(codes)
        self.terminated = False
        self.killed = False
        self._last: int | None = None

    def poll(self) -> int | None:
        self._last = self._codes.pop(0) if self._codes else self._last
        return self._last

    def terminate(self) -> None:
        self.terminated = True
        self._codes = [143]

    def kill(self) -> None:
        self.killed = True
        self._codes = [137]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def test_a_clean_exit_is_not_restarted(tmp_path) -> None:
    launches: list[list[str]] = []

    def spawn(cmd):
        launches.append(cmd)
        return _FakeProc([0])

    res = supervise(["train"], tmp_path / "p.json", _spawn=spawn, _now=_Clock())
    assert res.completed
    assert len(launches) == 1
    assert res.restarts == []


def test_a_crash_is_restarted_and_recorded(tmp_path) -> None:
    """The machine dying mid-run is the case this exists for."""
    outcomes = [[1], [0]]

    def spawn(cmd):
        return _FakeProc(outcomes.pop(0))

    res = supervise(["train"], tmp_path / "p.json", _spawn=spawn, _now=_Clock())
    assert res.completed
    assert len(res.restarts) == 1
    assert "rc=1" in res.restarts[0].reason


def test_a_crash_loop_gives_up_instead_of_restarting_forever(tmp_path) -> None:
    """Three attempts distinguishes a hiccup from a configuration that does not work.

    Restarting forever burns the compute window and produces a log nobody reads.
    """

    def spawn(cmd):
        return _FakeProc([1])

    res = supervise(["train"], tmp_path / "p.json", max_restarts=3,
                    _spawn=spawn, _now=_Clock())
    assert not res.completed
    assert res.attempts == 4, "three restarts after the first attempt"
    assert res.gave_up_reason is not None and "exhausted" in res.gave_up_reason


def test_a_live_but_stalled_process_is_restarted(tmp_path) -> None:
    """Alive is not the same as progressing: a hung CUDA call keeps the process up."""
    progress = tmp_path / "p.json"
    Progress(step=10, updated_at=0.0).write(progress)
    procs: list[_FakeProc] = []

    def spawn(cmd):
        p = _FakeProc([None] * 50 + [0])
        procs.append(p)
        return p

    res = supervise(["train"], progress, stall_seconds=5, _spawn=spawn, _now=_Clock())
    assert procs[0].terminated, "a stalled process must be terminated, not merely abandoned"
    assert res.restarts and "no progress" in res.restarts[0].reason


def test_progress_prevents_a_stall_restart(tmp_path) -> None:
    """Steps, not wall-clock. A slow step and a dead process look identical on a clock."""
    progress = tmp_path / "p.json"
    Progress(step=1).write(progress)
    state = {"step": 1}

    class _Advancing(_FakeProc):
        def poll(self):
            state["step"] += 5
            Progress(step=state["step"]).write(progress)
            return super().poll()

    def spawn(cmd):
        return _Advancing([None] * 20 + [0])

    res = supervise(["train"], progress, stall_seconds=3, _spawn=spawn, _now=_Clock())
    assert res.completed
    assert res.restarts == [], "a run that is progressing must never be restarted"
    assert res.final_step > 1


def test_a_torn_progress_file_is_not_a_stall(tmp_path) -> None:
    """A half-written file is normal -- the trainer may be mid-write.

    Treating a torn read as a stall would restart a healthy run, losing a checkpoint
    interval of work for a race that resolves itself in milliseconds.
    """
    progress = tmp_path / "p.json"
    progress.write_text('{"step": 12, "tok', encoding="utf-8")
    assert Progress.read(progress) is None
    assert Progress.read(tmp_path / "absent.json") is None


def test_progress_writes_are_atomic(tmp_path) -> None:
    """The supervisor reads this file concurrently; a partial write must never be visible."""
    progress = tmp_path / "p.json"
    Progress(step=7, tokens=1234, checkpoint="ckpt-7").write(progress)
    assert json.loads(progress.read_text(encoding="utf-8"))["step"] == 7
    assert not (tmp_path / "p.json.tmp").exists(), "the temp file must be renamed, not left"


def test_growing_fragmentation_triggers_a_planned_restart(tmp_path) -> None:
    """The measured-slope case: restart on a schedule you chose, not an OOM you did not."""
    progress = tmp_path / "p.json"
    Progress(step=1, trapped_bytes=10_000_000).write(progress)
    state = {"step": 1}

    class _Leaking(_FakeProc):
        def poll(self):
            state["step"] += 1
            Progress(step=state["step"],
                     trapped_bytes=10_000_000 * state["step"]).write(progress)
            return super().poll()

    outcomes = [[None] * 40 + [0], [0]]

    def spawn(cmd):
        return _Leaking(outcomes.pop(0))

    res = supervise(["train"], progress, trapped_limit_bytes=50_000_000,
                    _spawn=spawn, _now=_Clock())
    assert res.restarts, "a trapped-memory breach must restart before it OOMs"
    assert "trapped" in res.restarts[0].reason


def test_a_flat_slope_needs_no_scheduled_restart() -> None:
    assert restart_interval_from_slope(0.0, headroom_bytes=300_000_000) is None
    assert restart_interval_from_slope(-5.0, headroom_bytes=300_000_000) is None


def test_a_rising_slope_becomes_a_restart_interval() -> None:
    """50 MB per 1000 steps against 300 MB of headroom, halved for safety."""
    per_step = 50_000_000 / 1000
    interval = restart_interval_from_slope(per_step, headroom_bytes=300_000_000)
    assert interval == 3000


class TestASearchIsACandidateNotASelection:
    """8 steps cannot see per-step accumulation, so a search verdict cannot select a rung.

    rank32-seq768 measured 0.09 GB paged over 8 steps and 11.5 GB over 500 -- about 23 MB a
    step, with torch reporting flat fragmentation and 0.53 GB of headroom the whole time. An
    idle control put desktop noise at 33 MB per three minutes, so the growth was real.
    """

    def _soak(self, **kw):
        from marlowe.heal import SoakResult

        base = dict(candidate="rank32-seq768", steps=500, paged_bytes=0,
                    paging_verdict=False, trapped_slope_bytes_per_step=0.0, tok_s=190.0)
        base.update(kw)
        return SoakResult(**base)

    def test_the_cache_refuses_a_plan_with_no_soak(self, tmp_path) -> None:
        import json

        from marlowe.heal import NotSelected, require_selected

        plan = tmp_path / "memory_plan.json"
        plan.write_text(json.dumps({"config": {"seq_len": 768}}), encoding="utf-8")
        with pytest.raises(NotSelected, match="CANDIDATE"):
            require_selected(plan)

    def test_a_paging_soak_cannot_promote_a_candidate(self, tmp_path) -> None:
        from marlowe.heal import NotSelected, select_from_soak

        soak = self._soak(paged_bytes=11_547_000_000, paging_verdict=True)
        assert not soak.passed
        assert "11.55 GB" in (soak.why_not() or "")
        with pytest.raises(NotSelected, match="did not pass its soak"):
            select_from_soak(tmp_path / "memory_plan.json", soak)

    def test_growing_fragmentation_cannot_promote_a_candidate(self, tmp_path) -> None:
        from marlowe.heal import NotSelected, select_from_soak

        soak = self._soak(trapped_slope_bytes_per_step=80e6 / 1000)
        assert not soak.passed
        assert "per 1000 steps" in (soak.why_not() or "")
        with pytest.raises(NotSelected):
            select_from_soak(tmp_path / "memory_plan.json", soak)

    def test_a_passed_soak_promotes_and_the_cache_then_accepts(self, tmp_path) -> None:
        from marlowe.heal import require_selected, select_from_soak

        plan = tmp_path / "memory_plan.json"
        select_from_soak(plan, self._soak())
        selected = require_selected(plan)
        assert selected["candidate"] == "rank32-seq768"
        assert selected["steps"] == 500

    def test_a_search_result_never_reads_as_final(self) -> None:
        from marlowe.heal import MEMORY_CANDIDATES, SearchResult

        res = SearchResult(chosen=MEMORY_CANDIDATES[0], config=None, probe=None)
        assert res.is_provisional is True
        assert res.candidate is MEMORY_CANDIDATES[0]
