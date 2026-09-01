"""End-to-end streaming surgery against the synthetic checkpoint."""

from __future__ import annotations

import json

import pytest

from marlowe.arch import Layout, load_config, positional_selection
from marlowe.surgery import (
    load_index,
    plan_surgery,
    rewrite_config,
    run_surgery,
    verify_checkpoint,
)
from tests.conftest import TEXT_PREFIX, layer_types_for, write_checkpoint


class TestPlan:
    def test_renumbers_survivors_contiguously(self, mini_checkpoint, mini_layer_types) -> None:
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(load_config(mini_checkpoint))
        plan = plan_surgery(
            weight_map, layout, [5, 6], drop_vision=False, drop_mtp=False,
            vision_mode="keep-wrapper",
        )
        new_indices = sorted(
            {int(n.split(".layers.")[1].split(".")[0])
             for n in plan.mapping.values() if n.startswith(TEXT_PREFIX)}
        )
        assert new_indices == list(range(10)), "renumbering must leave no gaps"

    def test_drops_only_the_removed_layers(self, mini_checkpoint) -> None:
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(load_config(mini_checkpoint))
        plan = plan_surgery(
            weight_map, layout, [5], drop_vision=False, drop_mtp=False,
            vision_mode="keep-wrapper",
        )
        assert all(f"{TEXT_PREFIX}5." in n for n in plan.dropped)

    def test_drops_vision_and_mtp(self, mini_checkpoint) -> None:
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(load_config(mini_checkpoint))
        plan = plan_surgery(weight_map, layout, [5], drop_vision=True, drop_mtp=True)
        assert not any("visual" in n for n in plan.mapping)
        assert not any("mtp" in n.lower() for n in plan.mapping)
        assert any("visual" in n for n in plan.dropped)

    def test_extract_text_collapses_the_wrapper_namespace(self, mini_checkpoint) -> None:
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(load_config(mini_checkpoint))
        plan = plan_surgery(
            weight_map, layout, [5], drop_vision=True, drop_mtp=True, vision_mode="extract-text"
        )
        assert plan.new_root == "model."
        assert any(n.startswith("model.layers.0.") for n in plan.mapping.values())
        assert not any(".language_model." in n for n in plan.mapping.values())
        assert "lm_head.weight" in plan.mapping.values()

    def test_keep_wrapper_leaves_names_alone(self, mini_checkpoint) -> None:
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(load_config(mini_checkpoint))
        plan = plan_surgery(
            weight_map, layout, [5], drop_vision=True, drop_mtp=True, vision_mode="keep-wrapper"
        )
        assert plan.new_root == plan.text_root
        assert any(n.startswith(TEXT_PREFIX) for n in plan.mapping.values())

    def test_refuses_attention_layers(self, mini_checkpoint) -> None:
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(load_config(mini_checkpoint))
        with pytest.raises(ValueError, match="refusing to remove attention-class"):
            plan_surgery(weight_map, layout, [3], drop_vision=True, drop_mtp=True)


class TestConfigRewrite:
    def test_removes_full_attention_interval(self, mini_checkpoint) -> None:
        """Rule 3.4. A stale interval silently rebuilds the wrong stack."""
        cfg = load_config(mini_checkpoint)
        assert cfg["text_config"]["full_attention_interval"] == 4
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(cfg)
        plan = plan_surgery(weight_map, layout, [5], drop_vision=True, drop_mtp=True)
        new = rewrite_config(cfg, plan, drop_vision=True, drop_mtp=True)
        assert "full_attention_interval" not in new
        assert "full_attention_interval" not in new.get("text_config", {})

    def test_does_not_mutate_the_input(self, mini_checkpoint) -> None:
        cfg = load_config(mini_checkpoint)
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(cfg)
        plan = plan_surgery(weight_map, layout, [5], drop_vision=True, drop_mtp=True)
        rewrite_config(cfg, plan, drop_vision=True, drop_mtp=True)
        assert cfg["text_config"]["full_attention_interval"] == 4
        assert cfg["text_config"]["num_hidden_layers"] == 12

    def test_flattens_when_extracting_text(self, mini_checkpoint) -> None:
        cfg = load_config(mini_checkpoint)
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(cfg)
        plan = plan_surgery(weight_map, layout, [5], drop_vision=True, drop_mtp=True)
        new = rewrite_config(cfg, plan, drop_vision=True, drop_mtp=True)
        assert "text_config" not in new
        assert "vision_config" not in new
        assert new["architectures"] == ["Qwen3_5ForCausalLM"]
        assert new["num_hidden_layers"] == 11
        assert len(new["layer_types"]) == 11
        assert new["mtp_num_hidden_layers"] == 0

    def test_keeps_nesting_in_wrapper_mode(self, mini_checkpoint) -> None:
        cfg = load_config(mini_checkpoint)
        weight_map, _ = load_index(mini_checkpoint)
        layout = Layout.from_config(cfg)
        plan = plan_surgery(
            weight_map, layout, [5], drop_vision=True, drop_mtp=True, vision_mode="keep-wrapper"
        )
        new = rewrite_config(cfg, plan, drop_vision=True, drop_mtp=True,
                             vision_mode="keep-wrapper")
        assert "text_config" in new
        assert new["text_config"]["num_hidden_layers"] == 11


class TestEndToEnd:
    def test_writes_a_verifiable_checkpoint(self, mini_checkpoint, tmp_path) -> None:
        out = tmp_path / "child"
        report = run_surgery(mini_checkpoint, out, [5, 6], shard_size_gb=0.000001)
        assert report["actual_params"] > 0
        result = verify_checkpoint(out)
        assert result["n_layers"] == 10
        assert result["n_attention"] == 3  # never lose an attention layer
        assert (out / "model.safetensors.index.json").exists()
        assert (out / "pruning_report.json").exists()
        assert (out / "tokenizer_config.json").exists()

    def test_shards_are_named_n_of_m(self, mini_checkpoint, tmp_path) -> None:
        out = tmp_path / "child"
        run_surgery(mini_checkpoint, out, [5], shard_size_gb=0.000001)
        shards = sorted(p.name for p in out.glob("*.safetensors"))
        assert len(shards) > 1
        total = len(shards)
        for i, name in enumerate(shards, start=1):
            assert name == f"model-{i:05d}-of-{total:05d}.safetensors"

    def test_index_total_size_counts_only_kept_tensors(self, mini_checkpoint, tmp_path) -> None:
        out = tmp_path / "child"
        run_surgery(mini_checkpoint, out, [5, 6])
        with (out / "model.safetensors.index.json").open(encoding="utf-8") as f:
            idx = json.load(f)
        actual = sum(p.stat().st_size for p in out.glob("*.safetensors"))
        # Safetensors files carry a header, so the recorded payload must be under the file
        # size but the same order of magnitude.
        assert 0 < idx["metadata"]["total_size"] <= actual

    def test_dry_run_writes_nothing(self, mini_checkpoint, tmp_path) -> None:
        out = tmp_path / "child"
        report = run_surgery(mini_checkpoint, out, [5], dry_run=True)
        assert report["dry_run"] is True
        assert not out.exists()

    def test_unsharded_source(self, tmp_path) -> None:
        src = write_checkpoint(tmp_path / "flat", layer_types_for(12), sharded=False)
        out = tmp_path / "child"
        run_surgery(src, out, [5])
        assert verify_checkpoint(out)["n_layers"] == 11

    def test_verify_rejects_a_stale_interval(self, mini_checkpoint, tmp_path) -> None:
        out = tmp_path / "child"
        run_surgery(mini_checkpoint, out, [5])
        cfg = load_config(out)
        cfg["full_attention_interval"] = 4  # simulate a rewrite that forgot rule 3.4
        with (out / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f)
        with pytest.raises(AssertionError, match="full_attention_interval survived"):
            verify_checkpoint(out)

    def test_verify_rejects_a_corrupted_layer_types(self, mini_checkpoint, tmp_path) -> None:
        out = tmp_path / "child"
        run_surgery(mini_checkpoint, out, [5])
        cfg = load_config(out)
        cfg["layer_types"][0], cfg["layer_types"][3] = (
            cfg["layer_types"][3],
            cfg["layer_types"][0],
        )
        with (out / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f)
        with pytest.raises(AssertionError):
            verify_checkpoint(out)


class TestLadder:
    """The 22B -> 18B step, which is where an index-space bug would bite.

    The original script hardcoded ``range(64)`` when building the kept-index list, so on any
    parent that was not 64 layers deep the remap silently included indices that do not exist.
    That is exactly the second rung of this ladder.
    """

    def test_second_cut_on_a_pruned_parent(self, tmp_path) -> None:
        parent = write_checkpoint(tmp_path / "p", layer_types_for(16))
        first = tmp_path / "rung1"
        run_surgery(parent, first, [5, 6], drop_vision=True, drop_mtp=True)
        assert verify_checkpoint(first)["n_layers"] == 14

        second = tmp_path / "rung2"
        layout = Layout.from_config(load_config(first))
        # positional_selection honours per-period capacity; candidates() alone does not, and
        # period 1 of this child is already down to a single linear layer.
        picks = positional_selection(layout, 1)
        run_surgery(first, second, picks, drop_vision=True, drop_mtp=True)
        result = verify_checkpoint(second)
        assert result["n_layers"] == 13
        assert result["n_attention"] == 4

    def test_kept_indices_come_from_the_actual_depth(self, tmp_path) -> None:
        """Regression: a hardcoded 64 would produce indices past the end of a 16-layer stack."""
        parent = write_checkpoint(tmp_path / "p", layer_types_for(16))
        weight_map, _ = load_index(parent)
        layout = Layout.from_config(load_config(parent))
        plan = plan_surgery(weight_map, layout, [5], drop_vision=True, drop_mtp=True)
        assert plan.kept == [i for i in range(16) if i != 5]
        assert max(plan.kept) == 15

    def test_provenance_accumulates_across_rungs(self, tmp_path) -> None:
        parent = write_checkpoint(tmp_path / "p", layer_types_for(16))
        first = tmp_path / "rung1"
        r1 = run_surgery(parent, first, [5, 6], drop_vision=True, drop_mtp=True)
        assert r1["provenance"] == [5, 6]
        layout = Layout.from_config(load_config(first))
        r2 = run_surgery(first, tmp_path / "rung2", positional_selection(layout, 1),
                         drop_vision=True, drop_mtp=True)
        assert len(r2["provenance"]) == 3


class TestRenumberLiveModules:
    def test_resets_layer_idx_everywhere(self) -> None:
        """Rule 3.3: both cache families key off this attribute."""
        import torch.nn as nn

        from marlowe.surgery import renumber_live_modules

        class Block(nn.Module):
            def __init__(self, i: int) -> None:
                super().__init__()
                self.layer_idx = i
                self.mixer = nn.Module()
                self.mixer.layer_idx = i

        layers = nn.ModuleList([Block(i) for i in (0, 1, 4, 7)])
        fixed = renumber_live_modules(layers)
        assert fixed == 8  # block + mixer, for each of 4 layers
        assert [b.layer_idx for b in layers] == [0, 1, 2, 3]
        assert [b.mixer.layer_idx for b in layers] == [0, 1, 2, 3]
