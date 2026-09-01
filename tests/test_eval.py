"""Repetition metrics, the preset guard, and llama.cpp output parsing."""

from __future__ import annotations

from typing import ClassVar

import pytest

from marlowe.config import NON_THINKING, THINKING, preset
from marlowe.eval.bench import MCQuestion, extract_letter, format_mcq
from marlowe.eval.kl import parse_output
from marlowe.eval.repetition import (
    Completion,
    Prompt,
    RepetitionReport,
    longest_repeated_run,
    ngram_repetition,
    run_repetition,
    tail_cycle_period,
)


class TestNgramRepetition:
    def test_no_repeats(self) -> None:
        assert ngram_repetition(list(range(100)), 8) == 0.0

    def test_total_repeat(self) -> None:
        """A pure cycle: almost every n-gram is a duplicate."""
        toks = [1, 2, 3, 4] * 50
        assert ngram_repetition(toks, 8) > 0.9

    def test_shorter_than_n(self) -> None:
        assert ngram_repetition([1, 2, 3], 8) == 0.0

    def test_n32_ignores_short_churn(self) -> None:
        """n=8 sees phrase-level recycling; n=32 must not fire on it.

        A 16-token stock phrase repeated between stretches of fresh tokens. Nine 8-grams per
        block sit wholly inside the phrase and repeat; no 32-gram fits inside a 16-token
        phrase, so every 32-gram straddles fresh tokens and is unique. That separation is the
        reason both sizes are reported.
        """
        phrase = list(range(1, 17))
        toks: list[int] = []
        for i in range(30):
            toks += phrase + [1000 + i * 20 + j for j in range(20)]
        assert ngram_repetition(toks, 8) > 0.2
        assert ngram_repetition(toks, 32) == 0.0

    def test_n32_fires_on_paragraph_loops(self) -> None:
        block = list(range(64))
        assert ngram_repetition(block * 6, 32) > 0.7


class TestTailCycle:
    def test_detects_the_period(self) -> None:
        assert tail_cycle_period([9, 9, 9] + [1, 2, 3, 4] * 200, window=512) == 4

    def test_period_one(self) -> None:
        assert tail_cycle_period([7] * 600) == 1

    def test_no_cycle_on_varied_text(self) -> None:
        assert tail_cycle_period(list(range(600))) == 0

    def test_ignores_a_clean_prefix(self) -> None:
        """Circling is a tail phenomenon; a good opening must not mask it."""
        toks = list(range(2000)) + [5, 6, 7] * 200
        assert tail_cycle_period(toks, window=512) == 3

    def test_too_short(self) -> None:
        assert tail_cycle_period([1, 2]) == 0


class TestLongestRepeatedRun:
    def test_finds_the_block(self) -> None:
        block = list(range(50))
        got = longest_repeated_run([*block, 999, *block])
        assert got >= 50

    def test_none_when_unique(self) -> None:
        assert longest_repeated_run(list(range(200))) == 0


class TestPresetGuard:
    def test_thinking_values_are_pinned(self) -> None:
        assert (THINKING.temperature, THINKING.top_p, THINKING.top_k) == (1.0, 0.95, 20)
        assert THINKING.presence_penalty == 0.0

    def test_non_thinking_differs_where_it_matters(self) -> None:
        assert NON_THINKING.presence_penalty == 1.5

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError):
            preset("creative")

    def test_harness_refuses_non_thinking(self) -> None:
        """presence_penalty=1.5 suppresses exactly what the harness measures."""

        class Dummy:
            name = "dummy"

            def generate(self, *a: object, **k: object) -> Completion:
                return Completion("p", "text", 4, "stop")

        with pytest.raises(ValueError, match="refusing to run"):
            run_repetition(
                Dummy(), [Prompt("p", "hi")], label="x", sampling_preset="non_thinking"
            )

    def test_llamacpp_args_are_complete(self) -> None:
        args = THINKING.as_llamacpp_args()
        for flag in ("--temp", "--top-p", "--top-k", "--min-p", "--presence-penalty"):
            assert flag in args


class FakeBackend:
    """Deterministic backend: returns a looping completion for prompts tagged as triggers."""

    name = "fake"
    extra_body: ClassVar[dict[str, object]] = {}

    def generate(self, prompt: str, max_tokens: int, sampling: object, seed: int) -> Completion:
        if "LOOP" in prompt:
            text = " ".join(["the same clause again and again and again"] * 60)
            return Completion("", text, max_tokens, "length")
        return Completion("", " ".join(f"w{i}" for i in range(300)), 300, "stop")


class TestHarness:
    def test_separates_loopers_from_clean(self) -> None:
        prompts = [
            Prompt("clean", "answer briefly"),
            Prompt("looper", "LOOP please", known_trigger=True),
        ]
        report = run_repetition(FakeBackend(), prompts, label="t", n_completions=20)
        assert isinstance(report, RepetitionReport)
        assert report.n_completions == 20
        assert 0.0 < report.cap_hit_rate < 1.0
        assert report.repetition["rep32"] > 0.0
        assert report.tokenization == "whitespace-words"

    def test_trigger_subset_is_reported_separately(self) -> None:
        prompts = [Prompt("clean", "hi"), Prompt("looper", "LOOP", known_trigger=True)]
        report = run_repetition(FakeBackend(), prompts, label="t", n_completions=20)
        assert report.trigger_subset["cap_hit_rate"] == 1.0
        assert report.trigger_subset["rep32"] > report.repetition["rep32"]

    def test_loop_rate_needs_both_cap_and_cycle(self) -> None:
        report = run_repetition(
            FakeBackend(), [Prompt("clean", "hi")], label="t", n_completions=8
        )
        assert report.cap_hit_rate == 0.0
        assert report.loop_rate == 0.0

    def test_errors_are_counted_not_silently_dropped(self) -> None:
        class Broken:
            name = "broken"

            def generate(self, *a: object, **k: object) -> Completion:
                return Completion("", "", 0, "error", error="connection refused")

        with pytest.raises(RuntimeError, match="every generation failed"):
            run_repetition(Broken(), [Prompt("p", "hi")], label="t", n_completions=4)

    def test_per_prompt_breakdown(self) -> None:
        prompts = [Prompt("a", "hi"), Prompt("b", "LOOP")]
        report = run_repetition(FakeBackend(), prompts, label="t", n_completions=10)
        assert {r["prompt_id"] for r in report.by_prompt} == {"a", "b"}


class TestKLParsing:
    def test_parses_a_realistic_block(self) -> None:
        text = """
        ====== Perplexity statistics ======
        Mean PPL(Q)                   :   6.123456 +/- 0.03
        ====== KL divergence statistics ======
        Mean    KLD:   0.031415 +/-   0.000200
        Maximum KLD:  12.500000
        99.0%   KLD:   0.250000
        Median  KLD:   0.010000
        Mean Top-1 agreement:  92.15 %
        """
        got = parse_output(text)
        assert got["kl_mean"] == pytest.approx(0.031415)
        assert got["kl_median"] == pytest.approx(0.010)
        assert got["kl_max"] == pytest.approx(12.5)
        assert got["top1_agreement"] == pytest.approx(0.9215)

    def test_normalises_percentages(self) -> None:
        assert parse_output("Top-1 agreement: 88.0 %")["top1_agreement"] == pytest.approx(0.88)

    def test_leaves_fractions_alone(self) -> None:
        assert parse_output("Top-1 agreement: 0.88")["top1_agreement"] == pytest.approx(0.88)

    def test_missing_keys_are_absent_not_zero(self) -> None:
        assert "kl_mean" not in parse_output("nothing useful here")


class TestGPQA:
    def test_extracts_the_last_answer(self) -> None:
        text = "Maybe A, or possibly C. Actually reconsider B.\n\nAnswer: D"
        assert extract_letter(text) == "D"

    def test_handles_boxed(self) -> None:
        assert extract_letter(r"so \boxed{C}") == "C"

    def test_handles_prose(self) -> None:
        assert extract_letter("Therefore the final answer is B.") == "B"

    def test_returns_none_when_absent(self) -> None:
        assert extract_letter("I could not decide.") is None

    def test_shuffles_choices(self) -> None:
        import random

        q = MCQuestion("1", "Q?", "right", ["w1", "w2", "w3"])
        letters = {format_mcq(q, random.Random(s))[1] for s in range(30)}
        assert len(letters) > 1, "choices must be shuffled or position bias inflates accuracy"

    def test_correct_letter_tracks_the_shuffle(self) -> None:
        import random

        q = MCQuestion("1", "Q?", "right", ["w1", "w2", "w3"])
        prompt, letter = format_mcq(q, random.Random(7))
        line = [ln for ln in prompt.splitlines() if ln.startswith(f"({letter})")]
        assert line and line[0].endswith("right")
