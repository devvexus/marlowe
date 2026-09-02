"""A custom mix whose patterns match nothing is silently its own base type.

llama-quantize matches ``--tensor-type NAME=TYPE`` with a case-insensitive regex *search*
against GGUF tensor names, and ignores a pattern that matches nothing. Two failure modes
follow, and both produce a file that looks right:

* **under-matching** -- the shipped recipes used HF safetensors names (``q_proj``,
  ``linear_attn.*``) against a GGUF that calls those ``attn_q`` / ``attn_qkv`` / ``ssm_*``.
  All three of Stage 0's custom mixes were plain iq3_xxs or plain iq3_s under names claiming
  otherwise, and would have been reported as three distinct points on the curve.
* **over-matching** -- because it is a search and not a full match, bare ``attn_q`` also
  selects ``attn_qkv.weight``, a different tensor in a different set of layers. Confirmed
  against the real binary with ``llama-quantize --dry-run``.
"""

from __future__ import annotations

import re

import pytest

from marlowe.quantize import (
    FFN_WEIGHTS,
    MIXER_WEIGHTS,
    assert_patterns_match,
    mixer_protected_types,
)

# The real hybrid layout: 16 full-attention layers, 36 linear-attention, 52 blocks total.
NAMES = (
    [f"blk.{i}.attn_q.weight" for i in range(16)]
    + [f"blk.{i}.attn_k.weight" for i in range(16)]
    + [f"blk.{i}.attn_v.weight" for i in range(16)]
    + [f"blk.{i}.attn_output.weight" for i in range(16)]
    + [f"blk.{i}.attn_q_norm.weight" for i in range(16)]
    + [f"blk.{i}.attn_qkv.weight" for i in range(36)]
    + [f"blk.{i}.attn_gate.weight" for i in range(36)]
    + [f"blk.{i}.ssm_out.weight" for i in range(36)]
    + [f"blk.{i}.ssm_alpha.weight" for i in range(36)]
    + [f"blk.{i}.ssm_beta.weight" for i in range(36)]
    + [f"blk.{i}.ffn_down.weight" for i in range(52)]
    + [f"blk.{i}.ffn_gate.weight" for i in range(52)]
    + [f"blk.{i}.ffn_up.weight" for i in range(52)]
    + ["token_embd.weight", "output.weight", "output_norm.weight"]
)


@pytest.fixture
def gguf(monkeypatch):
    import marlowe.quantize as q

    monkeypatch.setattr(q, "gguf_tensor_names", lambda _p: NAMES)
    return "fake.gguf"


def test_the_shipped_patterns_matched_nothing(gguf) -> None:
    """The actual defect: HF names against a GGUF."""
    with pytest.raises(ValueError, match=r"match no tensor"):
        assert_patterns_match(gguf, {"q_proj": "q5_K", "linear_attn.*": "q5_K"})


def test_the_error_names_the_tensors_that_do_exist(gguf) -> None:
    """Whoever hits this needs the real names, not just the news that theirs are wrong."""
    with pytest.raises(ValueError) as exc:
        assert_patterns_match(gguf, {"q_proj": "q5_K"})
    text = str(exc.value)
    assert "attn_q.weight" in text and "ssm_out.weight" in text


def test_one_bad_pattern_among_good_ones_still_fails(gguf) -> None:
    """A mix that is 90% right is not 90% applied -- the missing rung is simply absent."""
    types = mixer_protected_types()
    types["linear_attn.*"] = "q5_K"
    with pytest.raises(ValueError, match=r"linear_attn"):
        assert_patterns_match(gguf, types)


def test_anchoring_stops_attn_q_from_selecting_attn_qkv(gguf) -> None:
    """Verified against llama-quantize --dry-run: bare attn_q overrides attn_qkv too."""
    unanchored = sum(1 for n in NAMES if re.search("attn_q", n, re.I))
    anchored = sum(1 for n in NAMES if re.search(r"attn_q\.weight", n, re.I))
    assert unanchored == 68, "attn_q, attn_qkv and attn_q_norm all contain 'attn_q'"
    assert anchored == 16, "anchoring selects only the full-attention query projection"


def test_the_mixer_mix_covers_both_attention_families(gguf) -> None:
    """The hybrid trap: naming only one family silently leaves a quarter unprotected."""
    counts = assert_patterns_match(gguf, mixer_protected_types())
    full = sum(counts[p] for p in MIXER_WEIGHTS if "qkv" not in p and "ssm" not in p
               and "gate" not in p)
    linear = sum(counts[p] for p in MIXER_WEIGHTS if "qkv" in p or "ssm" in p
                 or "attn_gate" in p)
    assert full == 64, "16 layers x q/k/v/output"
    assert linear == 180, "36 layers x qkv/gate/ssm_out/ssm_alpha/ssm_beta"
    assert sum(counts[p] for p in FFN_WEIGHTS) == 156, "52 layers x down/gate/up"


def test_the_mix_spends_on_mixers_and_saves_on_ffn(gguf) -> None:
    types = mixer_protected_types()
    assert {types[p] for p in MIXER_WEIGHTS} == {"q5_K"}
    assert {types[p] for p in FFN_WEIGHTS} == {"iq3_xxs"}
    assert types[r"token_embd\.weight"] == "q4_K"


def test_a_fully_matching_mix_passes_and_reports_counts(gguf) -> None:
    counts = assert_patterns_match(gguf, mixer_protected_types())
    assert all(v > 0 for v in counts.values())
    assert sum(counts.values()) == 401


class TestTheGateIsPhaseAOnly:
    """Phase B raises KL to the parent on purpose; the gate must not read that as failure.

    Phase A heals toward the 27B, so "KL below the iq3_xxs baseline" is the objective. Phase B
    trains the merged checkpoint on traces from stronger teachers, and the student then
    reasons in ways the 27B does not -- KL rises, and that is the goal succeeding. A gate
    applied there rejects the better model. See docs/TRANSFER_PLAN.md.
    """

    def _args(self):
        from marlowe.report import CheckpointRecord

        return {
            "candidates": {},
            "required_widths": (),
            "iq3_xxs_baseline": CheckpointRecord("iq3_xxs"),
            "bf16_baseline": CheckpointRecord("bf16"),
        }

    def test_phase_b_is_refused_rather_than_failed(self) -> None:
        from marlowe.report import ship_gate_multi

        with pytest.raises(ValueError, match=r"Phase A only|defined on Phase A"):
            ship_gate_multi(**self._args(), phase="B")

    def test_the_refusal_says_what_to_measure_instead(self) -> None:
        from marlowe.report import ship_gate_multi

        with pytest.raises(ValueError) as exc:
            ship_gate_multi(**self._args(), phase="B")
        text = str(exc.value)
        assert "RISES BY DESIGN" in text
        assert "benchmarks" in text and "TRANSFER_PLAN" in text

    def test_phase_a_is_the_default_and_evaluates_rather_than_raising(self) -> None:
        """It returns a verdict. `passed` is False here because no width was built, which is
        the gate's own rule -- a width never built is a failure, not an absence."""
        from marlowe.report import ship_gate_multi

        result = ship_gate_multi(**self._args())
        assert result.passed is False
        assert result.per_width == {}
