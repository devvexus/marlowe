"""Parameter accounting and layout rules.

The accounting tests check against the published Qwen3.8-27B figures. They are exact, not
approximate: if a shape assumption in ArchDims drifts, these fail rather than silently
producing a model of the wrong size.
"""

from __future__ import annotations

import pytest

from marlowe.arch import (
    FULL,
    LINEAR,
    ArchDims,
    Layout,
    assert_layer_types_match_tensors,
    detect_text_stack_prefix,
    is_attention_class,
    is_removable,
    layer_signatures,
    positional_selection,
)
from tests.conftest import (
    MTP_PREFIX,
    TEXT_PREFIX,
    VISION_PREFIX,
    full_layer_tensors,
    layer_types_for,
    linear_layer_tensors,
    shared_layer_tensors,
    tensor_names,
)

M = 1e6
B = 1e9


# ---------------------------------------------------------------------------
# parameter accounting -- published values
# ---------------------------------------------------------------------------


class TestParamAccounting:
    def test_ffn(self) -> None:
        assert ArchDims().ffn_params == 267_386_880
        assert ArchDims().ffn_params / M == pytest.approx(267.4, abs=0.05)

    def test_linear_mixer(self) -> None:
        assert ArchDims().linear_mixer_params / M == pytest.approx(115.9, abs=0.05)

    def test_full_mixer(self) -> None:
        assert ArchDims().full_mixer_params / M == pytest.approx(104.9, abs=0.05)

    def test_q_proj_is_doubled_by_the_output_gate(self) -> None:
        """attn_output_gate fuses a gate into q_proj: [12288, 5120], not [6144, 5120]."""
        gated = ArchDims()
        plain = ArchDims(attn_output_gate=False)
        d = gated.full_mixer_params - plain.full_mixer_params
        assert d == 5120 * 24 * 256  # exactly one extra q_proj-sized projection

    def test_block_sizes(self) -> None:
        d = ArchDims()
        assert d.linear_block_params / M == pytest.approx(383.3, abs=0.05)
        assert d.full_block_params / M == pytest.approx(372.2, abs=0.05)

    def test_embeddings_untied_counted_twice(self) -> None:
        d = ArchDims()
        assert d.embedding_params / B == pytest.approx(1.271, abs=0.001)
        assert d.embedding_total == 2 * d.embedding_params
        assert ArchDims(tie_word_embeddings=True).embedding_total == d.embedding_params

    def test_base_after_dropping_vision_and_mtp(self) -> None:
        base = ArchDims().total_params(Layout.qwen38_27b().layer_types)
        assert base / B == pytest.approx(26.895, abs=0.002)

    def test_full_model_reconstructs_published_total(self) -> None:
        """26.895 B text + 0.46 B vision + 0.38 B MTP = 27.74 B, vs HF's reported 28 B."""
        base = ArchDims().total_params(Layout.qwen38_27b().layer_types)
        assert (base + 0.46e9 + 0.38e9) / B == pytest.approx(27.74, abs=0.01)

    @pytest.mark.parametrize(
        "n_cuts,params,layers",
        [(12, 22.30, 52), (23, 18.08, 41)],
    )
    def test_ladder_targets(self, n_cuts: int, params: float, layers: int) -> None:
        d = ArchDims()
        base = d.total_params(Layout.qwen38_27b().layer_types)
        assert (base - n_cuts * d.linear_block_params) / B == pytest.approx(params, abs=0.005)
        assert 64 - n_cuts == layers

    def test_documented_linear_formula_matches(self) -> None:
        """params(n) = 26.895 - 0.3833 x n, to the precision the brief states it."""
        d = ArchDims()
        base = d.total_params(Layout.qwen38_27b().layer_types)
        for n in range(0, 25):
            doc = 26.895 - 0.3833 * n
            assert (base - n * d.linear_block_params) / B == pytest.approx(doc, abs=0.005)

    def test_from_config_overrides_defaults(self, mini_checkpoint) -> None:
        from marlowe.arch import load_config

        d = ArchDims.from_config(load_config(mini_checkpoint))
        assert d.hidden_size == 8
        assert d.vocab_size == 64

    def test_unknown_attention_type_approximates_and_warns(self, caplog) -> None:
        d = ArchDims()
        types = [LINEAR, LINEAR, LINEAR, "sparse_attention"] * 2
        with caplog.at_level("WARNING"):
            got = d.total_params(types)
        assert "no parameter formula" in caplog.text
        assert got > 0
        with pytest.raises(ValueError, match="no parameter formula"):
            d.total_params(types, strict=True)

    def test_unknown_attention_type_accepts_override(self) -> None:
        d = ArchDims()
        types = [LINEAR, "sparse_attention"]
        exact = d.total_params(types, {"sparse_attention": 12_345})
        assert exact == d.linear_block_params + 12_345 + d.ffn_params + d.embedding_total + 5120


# ---------------------------------------------------------------------------
# removability whitelist
# ---------------------------------------------------------------------------


class TestRemovabilityWhitelist:
    def test_only_linear_attention_is_removable(self) -> None:
        assert is_removable(LINEAR)
        assert not is_removable(FULL)
        assert not is_removable("sparse_attention")
        assert not is_removable("something_invented_next_year")

    def test_everything_else_is_attention_class(self) -> None:
        for t in (FULL, "sparse_attention", "swa", "who_knows"):
            assert is_attention_class(t)

    def test_refuses_full_attention(self) -> None:
        layout = Layout.qwen38_27b()
        with pytest.raises(ValueError, match="refusing to remove attention-class"):
            layout.validate_removal([3])

    def test_refuses_unknown_type_and_says_why(self) -> None:
        """The Flash-Next case: an unrecognised type must fail closed, loudly."""
        layout = Layout.repeating([LINEAR, LINEAR, LINEAR, "qwen_sparse_attention"], 4)
        with pytest.raises(ValueError) as exc:
            layout.validate_removal([3])
        assert "not recognised" in str(exc.value)
        assert "qwen_sparse_attention" in str(exc.value)

    def test_unknown_type_never_appears_as_a_candidate(self) -> None:
        layout = Layout.repeating([LINEAR, LINEAR, LINEAR, "brand_new_attention"], 6)
        for i in layout.candidates():
            assert layout.layer_types[i] == LINEAR

    def test_unknown_type_warns_at_construction(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            Layout.repeating([LINEAR, "brand_new_attention"], 3)
        assert "NEVER removable" in caplog.text

    def test_known_type_does_not_warn(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            Layout.qwen38_27b()
        assert "NEVER removable" not in caplog.text


# ---------------------------------------------------------------------------
# derived period structure
# ---------------------------------------------------------------------------


class TestPeriods:
    def test_qwen38_periods(self) -> None:
        layout = Layout.qwen38_27b()
        periods = layout.periods()
        assert len(periods) == 16
        assert all(len(p) == 4 for p in periods)
        assert periods[0] == [0, 1, 2, 3]
        assert layout.n_attention == 16
        assert layout.n_removable_type == 48

    def test_flash_next_shape_derives_12_periods(self) -> None:
        """48 layers as 12 x (3 GDN + 1 QSA). Same code, no constant 4 anywhere."""
        layout = Layout.repeating([LINEAR, LINEAR, LINEAR, "sparse_attention"], 12)
        assert layout.n_layers == 48
        assert len(layout.periods()) == 12
        assert layout.n_attention == 12
        assert layout.budget() == (12 - 2) * 2

    def test_irregular_periods_are_read_off_the_list(self) -> None:
        types = [LINEAR, FULL, LINEAR, LINEAR, LINEAR, FULL, LINEAR, FULL]
        layout = Layout(types, protect_first_periods=0, protect_last_periods=0)
        assert layout.periods() == [[0, 1], [2, 3, 4, 5], [6, 7]]
        # period 0 has 1 linear -> capacity 0 (must keep one); period 1 has 3 -> capacity 2
        assert [layout.period_capacity(i) for i in range(3)] == [0, 2, 0]

    def test_period_length_5(self) -> None:
        layout = Layout.repeating([LINEAR] * 4 + [FULL], 5, protect_first_periods=0,
                                  protect_last_periods=0)
        assert len(layout.periods()) == 5
        assert all(len(p) == 5 for p in layout.periods())
        assert layout.budget() == 5 * 2

    def test_trailing_layers_without_a_closing_attention_layer(self) -> None:
        layout = Layout([LINEAR, LINEAR, FULL, LINEAR, LINEAR])
        assert layout.periods() == [[0, 1, 2], [3, 4]]

    def test_protection_window(self) -> None:
        layout = Layout.qwen38_27b()
        cands = layout.candidates()
        assert min(cands) >= 4  # first period protected
        assert max(cands) < 60  # last period protected
        assert len(cands) == 42


# ---------------------------------------------------------------------------
# budget and selection
# ---------------------------------------------------------------------------


class TestBudget:
    def test_27b_budget(self) -> None:
        assert Layout.qwen38_27b().budget() == 28

    def test_ladder_is_feasible(self) -> None:
        """After 12 cuts there must still be room for 11 more."""
        parent = Layout.qwen38_27b()
        child, _ = parent.apply(positional_selection(parent, 12))
        assert child.n_layers == 52
        assert child.budget() >= 11

    def test_budget_respects_one_linear_per_period(self) -> None:
        """A period down to a single linear layer contributes zero, not max_per_period."""
        layout = Layout([LINEAR, FULL] * 4, protect_first_periods=0, protect_last_periods=0)
        assert layout.budget() == 0

    def test_validate_rejects_emptying_a_period(self) -> None:
        layout = Layout(
            [LINEAR, LINEAR, FULL] * 3,
            protect_first_periods=0,
            protect_last_periods=0,
            max_per_period=2,
        )
        with pytest.raises(ValueError, match="would retain no"):
            layout.validate_removal([0, 1])

    def test_validate_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            Layout.qwen38_27b().validate_removal([4, 4])

    def test_validate_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            Layout.qwen38_27b().validate_removal([999])

    def test_validate_rejects_protected_period(self) -> None:
        with pytest.raises(ValueError, match="outside the removable window"):
            Layout.qwen38_27b().validate_removal([0])

    def test_validate_rejects_too_many_per_period(self) -> None:
        with pytest.raises(ValueError, match="max_per_period"):
            Layout.qwen38_27b().validate_removal([4, 5, 6])

    def test_apply_records_provenance(self) -> None:
        parent = Layout.qwen38_27b()
        child, kept = parent.apply([4, 9])
        assert child.provenance == [4, 9]
        assert len(kept) == 62
        # Child indices are a different space from the parent's: after two cuts the
        # attention layers have moved, so the second rung must select from the child's own
        # candidates rather than reusing parent indices. positional_selection also honours
        # the per-period capacity, which candidates() alone does not.
        second = positional_selection(child, 2)
        grandchild, _ = child.apply(second)
        assert grandchild.provenance == [4, 9, *sorted(second)]
        assert grandchild.n_attention == 16  # never lost one along the way


class TestPositionalSelection:
    def test_respects_constraints(self) -> None:
        layout = Layout.qwen38_27b()
        picks = positional_selection(layout, 12)
        assert len(picks) == 12
        layout.validate_removal(picks)  # must not raise

    def test_spreads_across_periods(self) -> None:
        layout = Layout.qwen38_27b()
        periods = {layout.period_of(i) for i in positional_selection(layout, 12)}
        assert len(periods) >= 10

    def test_works_on_a_pruned_parent(self) -> None:
        parent = Layout.qwen38_27b()
        child, _ = parent.apply(positional_selection(parent, 12))
        picks = positional_selection(child, 11)
        child.validate_removal(picks)

    def test_raises_when_over_budget(self) -> None:
        with pytest.raises(ValueError):
            positional_selection(Layout.qwen38_27b(), 99)


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_reads_explicit_list(self, mini_checkpoint) -> None:
        from marlowe.arch import load_config

        layout = Layout.from_config(load_config(mini_checkpoint))
        assert layout.n_layers == 12
        assert layout.n_attention == 3

    def test_refuses_to_reconstruct_from_interval(self) -> None:
        cfg = {"text_config": {"num_hidden_layers": 64, "full_attention_interval": 4}}
        with pytest.raises(ValueError, match="no explicit layer_types"):
            Layout.from_config(cfg)

    def test_handles_flattened_config(self) -> None:
        cfg = {"num_hidden_layers": 4, "layer_types": [LINEAR, LINEAR, LINEAR, FULL]}
        assert Layout.from_config(cfg).n_layers == 4


# ---------------------------------------------------------------------------
# tensor prefix detection -- the vision-tower decoy
# ---------------------------------------------------------------------------


class TestPrefixDetection:
    def test_picks_the_text_stack_not_the_vision_tower(self, mini_checkpoint) -> None:
        from marlowe.surgery import load_index

        weight_map, _ = load_index(mini_checkpoint)
        prefix, counts = detect_text_stack_prefix(weight_map, 12)
        assert prefix == TEXT_PREFIX
        assert counts[VISION_PREFIX] == 27
        assert counts[MTP_PREFIX] == 1

    def test_layer_count_is_what_discriminates(self, mini_checkpoint) -> None:
        """The decoys are distinguished by depth, not by name or position.

        Asking for 27 layers selects the vision tower, which proves the text stack was not
        picked by luck of ordering or by a name heuristic.
        """
        from marlowe.surgery import load_index

        weight_map, _ = load_index(mini_checkpoint)
        namespaces = {n.rsplit(".layers.", 1)[0] + ".layers." for n in weight_map
                      if ".layers." in n}
        assert len(namespaces) == 3, "fixture must carry the vision and MTP decoys"

        assert detect_text_stack_prefix(weight_map, 12)[0] == TEXT_PREFIX
        assert detect_text_stack_prefix(weight_map, 27)[0] == VISION_PREFIX
        assert detect_text_stack_prefix(weight_map, 1)[0] == MTP_PREFIX

    def test_raises_when_no_stack_matches(self) -> None:
        with pytest.raises(ValueError, match="no prefix with 99 layers"):
            detect_text_stack_prefix(["a.layers.0.w", "a.layers.1.w"], 99)

    def test_raises_on_ambiguity(self) -> None:
        names = ["a.layers.0.w", "a.layers.1.w", "b.layers.0.w", "b.layers.1.w"]
        with pytest.raises(ValueError, match="ambiguous"):
            detect_text_stack_prefix(names, 2)


# ---------------------------------------------------------------------------
# rule 3.4 -- layer_types vs tensors
# ---------------------------------------------------------------------------


class TestLayerTypeAssertion:
    def test_accepts_a_consistent_checkpoint(self, mini_layer_types, mini_names) -> None:
        assert_layer_types_match_tensors(mini_layer_types, mini_names, TEXT_PREFIX)

    def test_catches_a_swapped_entry(self, mini_layer_types, mini_names) -> None:
        bad = list(mini_layer_types)
        bad[3], bad[4] = bad[4], bad[3]  # full <-> linear
        with pytest.raises(AssertionError, match="does not match"):
            assert_layer_types_match_tensors(bad, mini_names, TEXT_PREFIX)

    def test_catches_an_off_by_one_shift(self, mini_layer_types, mini_names) -> None:
        """The classic renumbering bug: everything present, everything misaligned."""
        shifted = [mini_layer_types[-1], *mini_layer_types[:-1]]
        with pytest.raises(AssertionError):
            assert_layer_types_match_tensors(shifted, mini_names, TEXT_PREFIX)

    def test_catches_a_regenerated_uniform_layout(self, mini_names) -> None:
        """What a converter reading full_attention_interval would produce after a cut."""
        regenerated = layer_types_for(12)
        regenerated[7], regenerated[8] = regenerated[8], regenerated[7]
        with pytest.raises(AssertionError):
            assert_layer_types_match_tensors(regenerated, mini_names, TEXT_PREFIX)

    def test_catches_missing_tensors(self, mini_layer_types, mini_names) -> None:
        pruned = [n for n in mini_names if not n.startswith(f"{TEXT_PREFIX}5.")]
        with pytest.raises(AssertionError, match="have no tensors"):
            assert_layer_types_match_tensors(mini_layer_types, pruned, TEXT_PREFIX)

    def test_catches_orphan_tensors(self, mini_layer_types, mini_names) -> None:
        with pytest.raises(AssertionError, match="only 11 entries"):
            assert_layer_types_match_tensors(mini_layer_types[:-1], mini_names, TEXT_PREFIX)

    def test_generic_case_unknown_mixer_names(self) -> None:
        """A stack whose mixers this module has never heard of must still be checked.

        Neither ``qsa`` nor ``gdn_v2`` is in the recognised name sets, so the family
        refinement cannot fire. The partition check alone has to catch the misalignment.
        """
        types = ["linear_attention", "linear_attention", "qwen_sparse_attention"] * 3
        names: list[str] = []
        for i, t in enumerate(types):
            mixer = "gdn_v2" if t == "linear_attention" else "qsa"
            names += [
                f"x.layers.{i}.{mixer}.in_proj.weight",
                f"x.layers.{i}.{mixer}.out_proj.weight",
                f"x.layers.{i}.mlp.gate_proj.weight",
                f"x.layers.{i}.input_layernorm.weight",
            ]
        assert_layer_types_match_tensors(types, names, "x.layers.")

        bad = list(types)
        bad[0], bad[2] = bad[2], bad[0]
        with pytest.raises(AssertionError, match="spans 2 different tensor signatures"):
            assert_layer_types_match_tensors(bad, names, "x.layers.")

    def test_catches_types_that_share_a_signature(self) -> None:
        """Two declared types with identical tensors: a distinction the weights do not have."""
        types = ["linear_attention", "full_attention"]
        names = [
            "x.layers.0.mixer.in_proj.weight",
            "x.layers.0.mlp.gate_proj.weight",
            "x.layers.1.mixer.in_proj.weight",
            "x.layers.1.mlp.gate_proj.weight",
        ]
        with pytest.raises(AssertionError, match="share one tensor signature"):
            assert_layer_types_match_tensors(types, names, "x.layers.")

    def test_signature_ignores_ffn_and_norms(self) -> None:
        names = shared_layer_tensors(0) + linear_layer_tensors(0)
        sigs = layer_signatures(names, TEXT_PREFIX)
        from marlowe.arch import mixer_components

        assert mixer_components(sigs[0]) == {"linear_attn"}

    def test_ignores_other_namespaces(self, mini_layer_types, mini_names) -> None:
        noisy = [*mini_names, *tensor_names(layer_types_for(27), VISION_PREFIX)]
        assert_layer_types_match_tensors(mini_layer_types, noisy, TEXT_PREFIX)

    def test_full_attention_signature_is_distinct(self) -> None:
        lin = layer_signatures(linear_layer_tensors(0), TEXT_PREFIX)[0]
        full = layer_signatures(full_layer_tensors(1), TEXT_PREFIX)[1]
        assert lin != full
