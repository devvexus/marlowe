"""The soak gate: does trapped fragmentation plateau, or grow?

A multi-day run fails by fragmentation accumulating until an allocation cannot be served --
at hour 30, not at step 8. Every fit measured in this project is an 8-step probe, which
cannot see that. The soak answers the different question, and the answer is a *slope*, not a
survival: a flat trace after warm-up means the allocator reached a steady block pattern and
40,000 steps is credible; a positive slope sets the restart interval instead of failing the
configuration outright.

The arithmetic is pinned here against known inputs because a slope that is quietly wrong
would authorise a three-day run, which is exactly the class of error this codebase produces.
"""

from __future__ import annotations

from marlowe.heal import ProbeResult


def _result(trace):
    return ProbeResult(
        tok_s=100.0, peak_vram_gb=15.0, total_vram_gb=17.17, step_s=0.1,
        seq_len=1024, n_steps=len(trace), lora_params=0, trace=trace,
    )


class TestFragmentationSlope:
    def test_a_flat_trace_has_zero_slope(self) -> None:
        r = _result([(s, 12.0, 0.44) for s in range(0, 500, 10)])
        slope = r.fragmentation_slope_gb_per_1k()
        assert slope is not None and abs(slope) < 1e-9

    def test_a_known_ramp_is_recovered(self) -> None:
        """0.001 GB per step is exactly 1.0 GB per 1000 steps."""
        r = _result([(s, 12.0, 0.4 + 0.001 * s) for s in range(0, 500, 10)])
        slope = r.fragmentation_slope_gb_per_1k()
        assert slope is not None and abs(slope - 1.0) < 1e-6

    def test_warmup_is_excluded_from_the_fit(self) -> None:
        """A steep warm-up followed by a plateau must read as a plateau.

        The allocator carves its pool in the first steps. Fitting through that reports a
        steep slope on a configuration that is actually stable -- and would send a good
        configuration back for rework.
        """
        trace = [(s, 12.0, 0.1 + 0.02 * s) for s in range(0, 50, 10)]  # warm-up ramp
        trace += [(s, 12.0, 1.0) for s in range(50, 500, 10)]  # plateau
        r = _result(trace)
        slope = r.fragmentation_slope_gb_per_1k(skip=50)
        assert slope is not None and abs(slope) < 1e-9
        assert r.steps_to_exhaust(1.0, skip=50) is None, "a plateau never exhausts"

    def test_too_few_samples_returns_none_not_zero(self) -> None:
        """Absent evidence must not read as evidence of stability."""
        assert _result([(0, 12.0, 0.4), (10, 12.0, 0.4)]).fragmentation_slope_gb_per_1k() is None
        assert _result([]).fragmentation_slope_gb_per_1k() is None

    def test_steps_to_exhaust_matches_the_slope(self) -> None:
        r = _result([(s, 12.0, 0.4 + 0.0005 * s) for s in range(0, 500, 10)])
        # 0.5 GB per 1000 steps against a 1.0 GB margin -> 2000 steps.
        got = r.steps_to_exhaust(1.0)
        assert got is not None and abs(got - 2000) < 1.0

    def test_a_shrinking_trace_never_exhausts(self) -> None:
        r = _result([(s, 12.0, 1.0 - 0.0005 * s) for s in range(0, 500, 10)])
        assert r.steps_to_exhaust(1.0) is None

    def test_short_fit_probes_carry_no_trace(self) -> None:
        """log_every defaults off, so a normal fitcheck is unchanged."""
        assert _result([]).trace == []
