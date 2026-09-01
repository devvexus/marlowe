"""Regression tests for defects found in the original scripts and during the build.

Each test here corresponds to a specific bug that was real, that earlier checks did not
catch, and that produces a model which loads and runs. They are collected in one place
because that shared property is what makes them worth pinning: none of them raise on their
own.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from marlowe.arch import Layout, load_config, positional_selection
from marlowe.quantize import converter_writes_explicit_layout
from marlowe.surgery import load_index, plan_surgery, run_surgery, verify_checkpoint
from tests.conftest import layer_types_for, write_checkpoint


class TestHardcodedLayerCount:
    """`plan()` used `range(64)` to build the kept-index list.

    Invisible on the 64-layer parent and on every earlier check. It only bites on the
    ladder's second rung, where the parent is 52 layers, and it produces a checkpoint whose
    tensors are renumbered against a 64-entry index space.
    """

    @pytest.mark.parametrize("n_layers", [12, 16, 20, 52, 64])
    def test_kept_indices_match_actual_depth(self, tmp_path, n_layers: int) -> None:
        src = write_checkpoint(tmp_path / f"p{n_layers}", layer_types_for(n_layers))
        weight_map, _ = load_index(src)
        layout = Layout.from_config(load_config(src))
        removed = positional_selection(layout, 1)
        plan = plan_surgery(weight_map, layout, removed, drop_vision=True, drop_mtp=True)

        assert plan.kept == [i for i in range(n_layers) if i not in removed]
        assert max(plan.kept) == n_layers - 1
        # The specific corruption: an index space larger than the model.
        assert all(i < n_layers for i in plan.kept)

    def test_remap_is_dense_and_ordered(self, tmp_path) -> None:
        src = write_checkpoint(tmp_path / "p", layer_types_for(52))
        weight_map, _ = load_index(src)
        layout = Layout.from_config(load_config(src))
        plan = plan_surgery(
            weight_map, layout, positional_selection(layout, 11),
            drop_vision=True, drop_mtp=True,
        )
        new_idx = sorted(
            {int(n.split(".layers.")[1].split(".")[0])
             for n in plan.mapping.values() if ".layers." in n}
        )
        assert new_idx == list(range(41)), "renumbering must be dense from zero"

    def test_full_ladder_preserves_every_attention_layer(self, tmp_path) -> None:
        """27B -> 22B -> 18B at real depths. The end-to-end guard."""
        src = write_checkpoint(tmp_path / "p64", layer_types_for(64))
        r1, r2 = tmp_path / "r1", tmp_path / "r2"

        l0 = Layout.from_config(load_config(src))
        run_surgery(src, r1, positional_selection(l0, 12), drop_vision=True, drop_mtp=True)
        v1 = verify_checkpoint(r1)
        assert v1["n_layers"] == 52
        assert v1["n_attention"] == 16

        l1 = Layout.from_config(load_config(r1))
        run_surgery(r1, r2, positional_selection(l1, 11), drop_vision=True, drop_mtp=True)
        v2 = verify_checkpoint(r2)
        assert v2["n_layers"] == 41
        assert v2["n_attention"] == 16, "the ladder must never cost an attention layer"


class TestRenumberCounting:
    """`renumber_live_modules` prepended the block to `block.modules()`, which already
    yields it. The assignment was idempotent, but the returned count -- the only evidence in
    the log that renumbering ran at all -- was inflated by 50%.
    """

    def test_count_is_exact(self) -> None:
        import torch.nn as nn

        from marlowe.surgery import renumber_live_modules

        class Block(nn.Module):
            def __init__(self, i: int) -> None:
                super().__init__()
                self.layer_idx = i
                self.mixer = nn.Module()
                self.mixer.layer_idx = i

        layers = nn.ModuleList([Block(i) for i in (0, 1, 4, 7)])
        assert renumber_live_modules(layers) == 8  # 2 carriers x 4 layers, counted once each

    def test_modules_without_layer_idx_are_not_counted(self) -> None:
        import torch.nn as nn

        from marlowe.surgery import renumber_live_modules

        class Block(nn.Module):
            def __init__(self, i: int) -> None:
                super().__init__()
                self.layer_idx = i
                self.mlp = nn.Linear(2, 2)  # no layer_idx

        assert renumber_live_modules(nn.ModuleList([Block(0), Block(1)])) == 2


class TestProvenancePersistence:
    """Provenance lived in memory only, so a second `run_surgery` on a child reported it as
    a single cut from an unremarkable stack rather than as 27B -> 22B -> 18B.
    """

    def test_lineage_survives_to_disk(self, tmp_path) -> None:
        import json

        src = write_checkpoint(tmp_path / "p", layer_types_for(16))
        r1, r2 = tmp_path / "r1", tmp_path / "r2"
        run_surgery(src, r1, [5, 6], drop_vision=True, drop_mtp=True)

        l1 = Layout.from_config(load_config(r1))
        run_surgery(r1, r2, positional_selection(l1, 1), drop_vision=True, drop_mtp=True)

        prov = json.loads((r2 / "pruning_report.json").read_text(encoding="utf-8"))["provenance"]
        assert len(prov) == 3, "second rung must carry the first rung's cuts"
        assert prov[:2] == [5, 6]

    def test_original_model_has_empty_provenance(self, tmp_path) -> None:
        from marlowe.surgery import read_provenance

        src = write_checkpoint(tmp_path / "p", layer_types_for(12))
        assert read_provenance(src) == []


class TestStageRegistration:
    """A helper inserted between `@register(...)` and its function silently registered the
    wrong callable. The stage still "existed" and would have failed only when run.
    """

    def test_every_stage_maps_to_its_own_function(self) -> None:
        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        for name, stage in REGISTRY.items():
            # "stage3-score" -> the function must belong to stage3
            prefix = name.split("-")[0]
            assert stage.fn.__name__.startswith(prefix), (
                f"{name} is registered to {stage.fn.__name__}, which does not belong to it -- "
                f"a decorator is attached to the wrong function"
            )
            assert not stage.fn.__name__.startswith("_"), (
                f"{name} is registered to the private helper {stage.fn.__name__}"
            )

    def test_build_order_is_complete_and_ordered(self) -> None:
        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import BUILD_ORDER, REGISTRY

        assert set(BUILD_ORDER) <= set(REGISTRY)
        # Stage 1 before Stage 0: if the converter cannot handle a non-uniform layout,
        # Stage 0's results are irrelevant until that is fixed.
        assert BUILD_ORDER[0] == "stage1-smoke"
        assert BUILD_ORDER.index("stage1-smoke") < BUILD_ORDER.index("stage0-bitwidth")

    def test_declared_requirements_exist(self) -> None:
        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        for name, stage in REGISTRY.items():
            for req in stage.requires:
                assert req in REGISTRY, f"{name} requires unknown stage {req!r}"


class TestExtraThreading:
    """`run_all` dropped `extra`, so `marlowe run all` lost the hosted bf16 endpoint and
    Stage 2 would fail closed even when it was correctly configured.
    """

    def test_run_all_forwards_extra(self) -> None:
        import inspect

        from marlowe.pipeline import run_all

        assert "extra" in inspect.signature(run_all).parameters

    def test_cli_passes_extra_to_run_all(self) -> None:
        import inspect

        from marlowe import cli

        src = inspect.getsource(cli.cmd_run)
        assert "run_all(cfg, run_dir=args.run_dir, force=args.force, extra=extra)" in src


class TestConverterLayoutSupport:
    """The Stage 1 blocker, checked statically.

    The stock converter writes only `full_attention_interval`. The loader
    (`src/models/qwen35.cpp`) prefers an explicit `attention.recurrent_layers` array and
    falls back to the interval, so a pruned stack gets a regenerated 3:1 alternation over
    the wrong layer count.
    """

    def _fake_checkout(self, tmp_path, *, patched: bool):
        root = tmp_path / "llama.cpp"
        (root / "gguf-py" / "gguf").mkdir(parents=True)
        (root / "conversion").mkdir(parents=True)
        (root / "convert_hf_to_gguf.py").write_text("# entry point\n", encoding="utf-8")

        writer = "def add_full_attention_interval(self, interval): ...\n"
        caller = "        self.gguf_writer.add_full_attention_interval(4)\n"
        if patched:
            writer += "    def add_recurrent_layers(self, value): ...\n"
            caller += "        self.gguf_writer.add_recurrent_layers([True])\n"
        (root / "gguf-py" / "gguf" / "gguf_writer.py").write_text(writer, encoding="utf-8")
        (root / "conversion" / "qwen.py").write_text(caller, encoding="utf-8")
        return root / "convert_hf_to_gguf.py"

    def test_detects_the_stock_converter(self, tmp_path) -> None:
        ok, detail = converter_writes_explicit_layout(self._fake_checkout(tmp_path, patched=False))
        assert not ok
        assert "no add_recurrent_layers writer" in detail
        assert "0001-qwen35-explicit-recurrent-layers.patch" in detail

    def test_detects_a_patched_converter(self, tmp_path) -> None:
        ok, detail = converter_writes_explicit_layout(self._fake_checkout(tmp_path, patched=True))
        assert ok, detail

    def test_writer_without_a_caller_is_still_broken(self, tmp_path) -> None:
        """Half-applied patch: gguf-py updated, the Qwen class not."""
        root = self._fake_checkout(tmp_path, patched=False).parent
        (root / "gguf-py" / "gguf" / "gguf_writer.py").write_text(
            "    def add_recurrent_layers(self, value): ...\n", encoding="utf-8"
        )
        ok, detail = converter_writes_explicit_layout(root / "convert_hf_to_gguf.py")
        assert not ok
        assert "never calls it" in detail

    def test_vendored_patch_is_present_and_pins_upstream(self) -> None:
        from marlowe.quantize import VENDOR_PATCH

        assert VENDOR_PATCH.exists(), "the converter patch must be tracked for reproducibility"
        text = VENDOR_PATCH.read_text(encoding="utf-8")
        assert "Upstream commit this applies to:" in text
        assert "add_recurrent_layers" in text


class TestPrunedLayoutIsNotAnInterval:
    """The premise of the whole converter problem: after pruning, no single interval
    describes the stack. Pinned so the claim stays true if constraints change.
    """

    def test_no_interval_describes_a_pruned_stack(self) -> None:
        parent = Layout.qwen38_27b()
        child, _ = parent.apply(positional_selection(parent, 12))
        want = [t == "linear_attention" for t in child.layer_types]

        for interval in range(2, child.n_layers + 1):
            regenerated = [(i + 1) % interval != 0 for i in range(child.n_layers)]
            assert regenerated != want, (
                f"interval {interval} reproduces the pruned layout; the converter's fallback "
                f"would happen to be correct and this test's premise is wrong"
            )

    def test_the_default_interval_mistypes_attention_layers(self) -> None:
        """Not just 'different' -- it drops KV caches, which is the silent failure."""
        parent = Layout.qwen38_27b()
        child, _ = parent.apply(positional_selection(parent, 12))
        regenerated = [(i + 1) % 4 != 0 for i in range(child.n_layers)]

        lost = [
            i
            for i, t in enumerate(child.layer_types)
            if t == "full_attention" and regenerated[i]
        ]
        assert lost, "expected attention layers to be mis-typed as recurrent"


class TestBf16BaselineIsFatal:
    """Stage 2 used to note-and-continue without the bf16 repetition baseline.

    That baseline is the definition of success -- circling appears at 2.97 bpw and vanishes
    at bf16 -- so its absence turns Stage 7 into "repetition improved", a bar the unhealed
    checkpoint might clear on its own. It now fails at stage entry, before the multi-hour
    GGUF conversion, and before the ~80 hours of scoring/caching/healing that follow.
    """

    def _ctx(self, tmp_path, **extra):
        from marlowe.config import load_run_config
        from marlowe.manifest import Manifest
        from marlowe.pipeline import StageContext

        cfg = load_run_config("configs/marlowe-22b.yaml")
        return StageContext(
            name="stage2-baselines",
            cfg=cfg,
            run_dir=tmp_path,
            manifest=Manifest(stage="stage2-baselines"),
            extra=extra,
        )

    def test_missing_endpoint_raises(self, tmp_path) -> None:
        from marlowe.pipeline import MissingBaseline
        from marlowe.stages import _require_bf16_backend

        with pytest.raises(MissingBaseline) as exc:
            _require_bf16_backend(self._ctx(tmp_path))
        msg = str(exc.value)
        assert "--allow-missing-bf16-baseline" in msg
        assert "--hosted-base-url" in msg, "the error must say how to fix it"

    def test_escape_hatch_returns_none(self, tmp_path) -> None:
        from marlowe.stages import _require_bf16_backend

        ctx = self._ctx(tmp_path, allow_missing_bf16_baseline=True)
        assert _require_bf16_backend(ctx) is None
        assert any("PROCEEDING WITHOUT" in n for n in ctx.manifest.notes)
        assert any("fail closed" in n for n in ctx.manifest.notes)

    def test_unreachable_endpoint_raises(self, tmp_path, monkeypatch) -> None:
        """Configured-but-wrong fails exactly as expensively as never-configured."""
        from marlowe.eval.repetition import Completion, OpenAICompatBackend
        from marlowe.pipeline import MissingBaseline
        from marlowe.stages import _require_bf16_backend

        monkeypatch.setattr(
            OpenAICompatBackend,
            "generate",
            lambda self, *a, **k: Completion("", "", 0, "error", error="connection refused"),
        )
        ctx = self._ctx(tmp_path, hosted_base_url="http://127.0.0.1:1", hosted_model="m")
        with pytest.raises(MissingBaseline, match="did not answer a one-token probe"):
            _require_bf16_backend(ctx)

    def test_unreachable_endpoint_is_survivable_with_the_flag(self, tmp_path, monkeypatch) -> None:
        from marlowe.eval.repetition import Completion, OpenAICompatBackend
        from marlowe.stages import _require_bf16_backend

        monkeypatch.setattr(
            OpenAICompatBackend,
            "generate",
            lambda self, *a, **k: Completion("", "", 0, "error", error="refused"),
        )
        ctx = self._ctx(
            tmp_path,
            hosted_base_url="http://127.0.0.1:1",
            hosted_model="m",
            allow_missing_bf16_baseline=True,
        )
        assert _require_bf16_backend(ctx) is None

    def test_live_endpoint_returns_a_backend(self, tmp_path, monkeypatch) -> None:
        from marlowe.eval.repetition import Completion, OpenAICompatBackend
        from marlowe.stages import _require_bf16_backend

        monkeypatch.setattr(
            OpenAICompatBackend,
            "generate",
            lambda self, *a, **k: Completion("", "ready.", 2, "stop"),
        )
        ctx = self._ctx(tmp_path, hosted_base_url="https://example.invalid/v1", hosted_model="m")
        backend = _require_bf16_backend(ctx)
        assert backend is not None
        assert backend.model == "m"

    def test_probe_runs_before_any_conversion(self) -> None:
        """Ordering is the whole point: the check must precede the expensive work."""
        import inspect

        from marlowe.stages import stage2_baselines

        src = inspect.getsource(stage2_baselines)
        assert src.index("_require_bf16_backend") < src.index("convert_to_gguf")
        assert src.index("_require_bf16_backend") < src.index("build_reference")

    def test_cli_exposes_the_escape_hatch(self) -> None:
        from marlowe.cli import build_parser

        args = build_parser().parse_args(
            ["run", "stage2-baselines", "--config", "configs/marlowe-22b.yaml",
             "--allow-missing-bf16-baseline"]
        )
        assert args.allow_missing_bf16_baseline is True

    def test_flag_defaults_to_unset(self) -> None:
        from marlowe.cli import build_parser

        args = build_parser().parse_args(
            ["run", "stage2-baselines", "--config", "configs/marlowe-22b.yaml"]
        )
        assert args.allow_missing_bf16_baseline is None


class TestGpuBackendGate:
    """Generation-bound stages must refuse a CPU-only llama.cpp, with the cost stated.

    The repetition harness is ~410K tokens per quant variant. CPU decode on a ~10 GB model is
    memory-bandwidth bound, so this is a 10x wall-clock difference, not a nuisance.
    """

    def test_projection_matches_the_measured_rates(self) -> None:
        from marlowe.eval.kl import CPU_DECODE_TOK_S, GPU_DECODE_TOK_S, generation_hours

        cpu_h = generation_hours(200, 2048, CPU_DECODE_TOK_S)
        gpu_h = generation_hours(200, 2048, GPU_DECODE_TOK_S)
        assert 27 < cpu_h < 30, f"one variant on CPU should be ~28 h, got {cpu_h:.1f}"
        assert 2.5 < gpu_h < 3.2, f"one variant on GPU should be under 3 h, got {gpu_h:.1f}"

    def test_message_names_both_wall_clocks_and_the_fix(self) -> None:
        from marlowe.eval.kl import gpu_requirement_message

        msg = gpu_requirement_message(
            "stage0-bitwidth", n_variants=4, n_completions=200, max_tokens=2048
        )
        assert "CPU decode" in msg and "GPU decode" in msg
        assert "days" in msg
        assert "--allow-cpu-llamacpp" in msg
        assert "cuda" in msg.lower()

    def test_raises_when_cpu_only(self, monkeypatch) -> None:
        from marlowe.eval import kl as kleval

        monkeypatch.setattr(
            kleval, "detect_backend", lambda: kleval.BackendInfo(devices=(), binary="x")
        )
        with pytest.raises(kleval.LlamaCppCpuOnly, match="CPU-only"):
            kleval.require_gpu_backend("stage0-bitwidth")

    def test_escape_hatch_allows_cpu(self, monkeypatch) -> None:
        from marlowe.eval import kl as kleval

        monkeypatch.setattr(
            kleval, "detect_backend", lambda: kleval.BackendInfo(devices=(), binary="x")
        )
        assert not kleval.require_gpu_backend("stage0-bitwidth", allow_cpu=True).has_gpu

    def test_passes_with_a_gpu(self, monkeypatch) -> None:
        from marlowe.eval import kl as kleval

        monkeypatch.setattr(
            kleval,
            "detect_backend",
            lambda: kleval.BackendInfo(devices=("CUDA0: RTX 4080 SUPER",), binary="x"),
        )
        info = kleval.require_gpu_backend("stage0-bitwidth")
        assert info.has_gpu
        assert info.decode_tok_s == kleval.GPU_DECODE_TOK_S

    def test_no_devices_means_no_gpu(self) -> None:
        from marlowe.eval.kl import BackendInfo

        assert not BackendInfo(devices=(), binary="x").has_gpu

    @pytest.mark.parametrize("stage", ["stage0-bitwidth", "stage2-baselines", "stage7-ship"])
    def test_the_three_decode_bound_stages_gate_on_it(self, stage: str) -> None:
        import inspect

        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        src = inspect.getsource(REGISTRY[stage].fn)
        assert "require_gpu_backend" in src, f"{stage} does not check the llama.cpp backend"
        assert "allow_cpu_llamacpp" in src

    def test_gpu_builds_are_preferred_on_the_search_path(self) -> None:
        from marlowe.eval.kl import _LOCAL_BIN_DIRS

        assert "llama-cuda" in str(_LOCAL_BIN_DIRS[0]), "the CUDA build must be tried first"


class TestQ8ReferenceDefault:
    """The KL reference is built from Q8_0, not bf16.

    bf16 is 56 GB against 32 GB of RAM, so llama.cpp mmaps and pages from disk for the whole
    pass. Q8_0 is ~28.6 GB and near-lossless, and since every number in the project is a
    relative comparison against the same reference file, the substitution costs nothing.
    """

    def test_default_is_q8_0(self) -> None:
        from marlowe.eval.kl import REFERENCE_OUTTYPE

        assert REFERENCE_OUTTYPE == "q8_0"

    def test_stage2_uses_the_constant_and_allows_override(self) -> None:
        import inspect

        from marlowe.stages import stage2_baselines

        src = inspect.getsource(stage2_baselines)
        assert "REFERENCE_OUTTYPE" in src
        assert "reference_outtype" in src, "must be overridable for a machine with the RAM"

    def test_cli_exposes_the_override(self) -> None:
        from marlowe.cli import build_parser

        args = build_parser().parse_args(
            ["run", "stage2-baselines", "--config", "configs/marlowe-22b.yaml",
             "--reference-outtype", "bf16"]
        )
        assert args.reference_outtype == "bf16"

    def test_override_defaults_to_unset(self) -> None:
        from marlowe.cli import build_parser

        args = build_parser().parse_args(
            ["run", "stage2-baselines", "--config", "configs/marlowe-22b.yaml"]
        )
        assert args.reference_outtype is None

    def test_q8_0_is_a_converter_outtype(self) -> None:
        """It must be producible directly by convert_hf_to_gguf, not via a second step."""
        from marlowe.quantize import find_converter

        src = find_converter().read_text(encoding="utf-8", errors="replace")
        assert '"q8_0"' in src

    def test_bf16_repetition_baseline_is_unaffected(self) -> None:
        """It comes from the hosted endpoint, not from any local GGUF."""
        import inspect

        from marlowe.stages import _require_bf16_backend

        assert "OpenAICompatBackend" in inspect.getsource(_require_bf16_backend)


class TestDeleteTempsDefault:
    """519 GB retained does not fit in ~350 GB free, so the safe path must not need a flag."""

    def _ctx(self, tmp_path, **extra):
        from marlowe.config import load_run_config
        from marlowe.manifest import Manifest
        from marlowe.pipeline import StageContext

        return StageContext(
            name="t",
            cfg=load_run_config("configs/marlowe-22b.yaml"),
            run_dir=tmp_path,
            manifest=Manifest(stage="t"),
            extra=extra,
        )

    def test_default_deletes(self, tmp_path) -> None:
        ctx = self._ctx(tmp_path)
        assert ctx.keep_intermediates is False
        f = tmp_path / "big.gguf"
        f.write_bytes(b"x" * 1024)
        assert ctx.drop_temp(f, why="test") is True
        assert not f.exists()
        assert any("deleted big.gguf" in n for n in ctx.manifest.notes)

    def test_flag_retains(self, tmp_path) -> None:
        ctx = self._ctx(tmp_path, keep_intermediates=True)
        f = tmp_path / "big.gguf"
        f.write_bytes(b"x" * 1024)
        assert ctx.drop_temp(f, why="test") is False
        assert f.exists()

    def test_handles_directories(self, tmp_path) -> None:
        ctx = self._ctx(tmp_path)
        d = tmp_path / "merged"
        d.mkdir()
        (d / "a.safetensors").write_bytes(b"y" * 512)
        assert ctx.drop_temp(d, why="test") is True
        assert not d.exists()

    def test_missing_path_is_a_noop(self, tmp_path) -> None:
        assert self._ctx(tmp_path).drop_temp(tmp_path / "nope", why="test") is False

    def test_note_explains_how_to_opt_out(self, tmp_path) -> None:
        ctx = self._ctx(tmp_path)
        f = tmp_path / "x.gguf"
        f.write_bytes(b"z")
        ctx.drop_temp(f, why="regenerable")
        assert any("--keep-intermediates" in n for n in ctx.manifest.notes)

    def test_cli_flag_is_opt_in(self) -> None:
        from marlowe.cli import build_parser

        base = build_parser().parse_args(
            ["run", "stage0-bitwidth", "--config", "configs/marlowe-22b.yaml"]
        )
        assert base.keep_intermediates is None  # absent -> default policy (delete)
        opted = build_parser().parse_args(
            ["run", "stage0-bitwidth", "--config", "configs/marlowe-22b.yaml",
             "--keep-intermediates"]
        )
        assert opted.keep_intermediates is True

    def test_persistent_artefacts_are_never_dropped(self) -> None:
        """Nothing a later stage needs may be dropped.

        Parses the first argument of every ctx.drop_temp() call rather than grepping lines --
        the reason strings legitimately mention reference.kld, and a substring check on the
        whole line reads those as the target.
        """
        import ast
        import inspect

        import marlowe.stages as st

        #: variables holding artefacts that must survive their stage
        forbidden = {"ref", "merged", "out_hf", "healed", "cache", "adapters"}
        targets: list[str] = []
        for node in ast.walk(ast.parse(inspect.getsource(st))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "drop_temp"
                and node.args
            ):
                targets.append(ast.unparse(node.args[0]))

        assert targets, "no drop_temp calls found; the delete-temps policy is not wired in"
        for t in targets:
            root = t.split(".")[0].split("[")[0].split("(")[0]
            assert root not in forbidden, (
                f"drop_temp({t}) targets a persistent artefact; a later stage needs it"
            )

    def test_budget_marks_the_next_rungs_parent_as_persistent(self) -> None:
        from marlowe.preflight import default_rungs, disk_budget

        merged = [i for i in disk_budget(default_rungs()) if "next rung" in i.what]
        assert merged, "the merged checkpoint must appear in the budget"
        assert all(i.persists for i in merged), "it is the next rung's parent; never a temp"

    def test_cleaned_peak_is_well_under_retained(self) -> None:
        from marlowe.preflight import budget_totals, default_rungs, disk_budget

        t = budget_totals(disk_budget(default_rungs()))
        assert t["peak_if_cleaned"] < t["total_if_nothing_deleted"] / 2


class TestSurgeryIsIdempotent:
    """`write_checkpoint` collided with its own previous output on a re-run.

    It writes `model-00001.safetensors` and renames to `model-00001-of-000NN.safetensors`
    once N is known. A second run found the renamed file already there and raised
    FileExistsError -- after the entire streaming pass had completed. Stages are re-runnable
    by design and `--force` re-runs them deliberately, so this had to be fixed rather than
    worked around.
    """

    def test_rerun_into_a_populated_directory(self, mini_checkpoint, tmp_path) -> None:
        out = tmp_path / "child"
        first = run_surgery(mini_checkpoint, out, [5], shard_size_gb=0.000001)
        shards_before = sorted(p.name for p in out.glob("*.safetensors"))
        assert shards_before

        second = run_surgery(mini_checkpoint, out, [5], shard_size_gb=0.000001)
        assert second["actual_params"] == first["actual_params"]
        assert sorted(p.name for p in out.glob("*.safetensors")) == shards_before
        verify_checkpoint(out)

    def test_rerun_with_a_different_cut_leaves_no_orphans(self, tmp_path) -> None:
        """A shorter second run must not leave the first run's extra shards behind."""
        # 20 layers: 5 periods, three of them unprotected, so a 3-cut is admissible.
        src = write_checkpoint(tmp_path / "p20", layer_types_for(20))
        out = tmp_path / "child"
        run_surgery(src, out, positional_selection(Layout.from_config(load_config(src)), 1),
                    shard_size_gb=0.000001)
        cuts = positional_selection(Layout.from_config(load_config(src)), 3)
        run_surgery(src, out, cuts, shard_size_gb=0.000001)
        result = verify_checkpoint(out)
        assert result["n_layers"] == 17
        # Every shard on disk must be referenced by the index; orphans would be silently
        # loaded by some readers and ignored by others.
        import json as _json

        with (out / "model.safetensors.index.json").open(encoding="utf-8") as f:
            referenced = set(_json.load(f)["weight_map"].values())
        assert {p.name for p in out.glob("*.safetensors")} == referenced

    def test_only_our_own_files_are_removed(self, tmp_path) -> None:
        from marlowe.surgery import clear_previous_shards

        out = tmp_path / "d"
        out.mkdir()
        for name in ("model-00001-of-00002.safetensors", "model-00002.safetensors",
                     "model.safetensors.index.json"):
            (out / name).write_bytes(b"x")
        for name in ("config.json", "tokenizer.json", "adapter_model.safetensors",
                     "notes.txt"):
            (out / name).write_bytes(b"y")

        assert clear_previous_shards(out) == 3
        assert {p.name for p in out.iterdir()} == {
            "config.json", "tokenizer.json", "adapter_model.safetensors", "notes.txt"
        }

    def test_empty_directory_is_a_noop(self, tmp_path) -> None:
        from marlowe.surgery import clear_previous_shards

        out = tmp_path / "d"
        out.mkdir()
        assert clear_previous_shards(out) == 0


class TestGgufProbeUsesRealNaming:
    """The probe called a correct conversion broken.

    llama.cpp emits the DeltaNet fused projection as ``blk.N.attn_qkv`` and its output gate
    as ``blk.N.attn_gate`` -- ``attn_`` prefixes on the *recurrent* path. The first classifier
    read those as attention markers and labelled every DeltaNet layer "attention+linear",
    failing Stage 1 on a GGUF that was in fact correct.

    The fix is twofold: prefer the explicit ``recurrent_layers`` mask, which is what the
    loader actually uses, and make the tensor-name fallback treat ``ssm_*`` as conclusive.
    """

    #: Verbatim from the Stage 1 GGUF built from real Qwen3.8-27B weights.
    LINEAR_TENSORS: ClassVar[list[str]] = [
        "attn_gate", "attn_norm", "attn_qkv", "ffn_down", "ffn_gate", "ffn_up",
        "post_attention_norm", "ssm_alpha", "ssm_beta", "ssm_conv1d", "ssm_dt",
        "ssm_norm", "ssm_out",
    ]
    ATTENTION_TENSORS: ClassVar[list[str]] = [
        "attn_k", "attn_k_norm", "attn_norm", "attn_output", "attn_q", "attn_q_norm",
        "attn_v", "ffn_down", "ffn_gate", "ffn_up", "post_attention_norm",
    ]

    def _info(self, layer_types):
        from marlowe.quantize import GGUFInfo

        names = []
        for i, t in enumerate(layer_types):
            tensors = (
                self.LINEAR_TENSORS if t == "linear_attention" else self.ATTENTION_TENSORS
            )
            names += [f"blk.{i}.{s}.weight" for s in tensors]
        return GGUFInfo(path="x", version=3, tensor_names=names)

    def test_deltanet_layers_are_not_called_attention(self) -> None:
        from marlowe.quantize import gguf_layer_families

        fams = gguf_layer_families(self._info(["linear_attention"] * 3 + ["full_attention"]))
        assert fams[0] == {"linear"}, f"attn_qkv/attn_gate misread the recurrent path: {fams[0]}"
        assert fams[1] == {"linear"}
        assert fams[2] == {"linear"}
        assert fams[3] == {"attention"}

    def test_no_layer_is_ambiguous(self) -> None:
        from marlowe.quantize import gguf_layer_families

        types = (["linear_attention"] * 3 + ["full_attention"]) * 4
        for i, fam in gguf_layer_families(self._info(types)).items():
            assert len(fam) == 1, f"block {i} classified as {fam}"

    def test_explicit_mask_is_preferred_over_tensor_names(self) -> None:
        """Even if the names were ambiguous, the authoritative key wins."""
        from marlowe.quantize import GGUFInfo, gguf_recurrent_mask

        info = GGUFInfo(
            path="x",
            version=3,
            metadata={"qwen35.attention.recurrent_layers": [True, True, True, False]},
        )
        assert gguf_recurrent_mask(info) == [True, True, True, False]

    def test_absent_mask_returns_none(self) -> None:
        from marlowe.quantize import GGUFInfo, gguf_recurrent_mask

        assert gguf_recurrent_mask(GGUFInfo(path="x", version=3, metadata={})) is None

    def test_mask_detects_a_genuinely_wrong_layout(self) -> None:
        """The probe must still fail when the converter really did regenerate an interval."""
        from marlowe.quantize import GGUFInfo, gguf_recurrent_mask

        # what (i+1) % 4 != 0 produces over 6 layers, against a real 3/1/2 layout
        info = GGUFInfo(
            path="x",
            version=3,
            metadata={"qwen35.attention.recurrent_layers": [True, True, True, False, True, True]},
        )
        mask = gguf_recurrent_mask(info)
        assert mask is not None
        real = [True, True, True, False, True, False]
        assert mask != real, "fixture must differ or the test proves nothing"


def _stub_model_plumbing(monkeypatch) -> None:
    """Stub the base load and LoRA wrap for search tests.

    These exercise selection logic -- ordering, gating, short-circuiting -- not the model
    plumbing. Without this they would try to load safetensors from a fake path.
    """
    from marlowe import heal

    monkeypatch.setattr(heal, "load_student", lambda *a, **k: object())
    monkeypatch.setattr(heal, "attach_lora", lambda base, cfg: _StubWrapped())
    monkeypatch.setattr(heal, "_free_cuda", lambda: None)


class _StubWrapped:
    def unload(self):
        return object()


class TestMemoryPlanSearch:
    """Stage 6's memory config is searched by measurement, not assumed.

    Sizing by arithmetic gets the order of magnitude right and the last gigabyte wrong, and
    the last gigabyte is the whole question on a 16 GB card. The ordering is by *quality*
    cost, not memory saved: an NF4 lm_head frees the most and costs the most, because the
    loss is top-K KL against teacher logits and merge_adapters loads the base in bf16 -- so
    the adapter would learn to cancel noise that does not survive the merge.
    """

    def test_candidates_are_ordered_best_quality_first(self) -> None:
        from marlowe.heal import MEMORY_CANDIDATES

        assert MEMORY_CANDIDATES[0].quantize_lm_head is False, (
            "the fp16 head must be tried first; it is the expensive-to-lose one"
        )
        # Once the head is given up, optimizer precision is bought back before being spent.
        nf4 = [c for c in MEMORY_CANDIDATES if c.quantize_lm_head]
        assert nf4[0].optimizer_8bit is False
        assert nf4[-1].optimizer_8bit is True

    def test_every_candidate_keeps_the_free_savings(self) -> None:
        """CPU embeddings and loss chunking are near-free; no candidate gives them up."""
        from marlowe.config import HealConfig
        from marlowe.heal import MEMORY_CANDIDATES

        base = HealConfig()
        for cand in MEMORY_CANDIDATES:
            cfg = cand.apply(base)
            assert cfg.loss_chunk == base.loss_chunk

    def test_shortened_candidates_descend_in_length(self) -> None:
        """Sequence length is a real cost, so candidates spend it monotonically."""
        from marlowe.config import HealConfig
        from marlowe.heal import MEMORY_CANDIDATES

        base = HealConfig()
        lengths = [
            c.apply(base).seq_len for c in MEMORY_CANDIDATES if not c.quantize_lm_head
        ]
        assert lengths == sorted(lengths, reverse=True), (
            "the fp16 ladder must give up context gradually, never jump back up"
        )
        assert lengths == [2048, 1536, 1024, 1024]  # the last pair differ by LoRA rank

    def test_shortening_the_sequence_is_preferred_over_quantising_the_head(self) -> None:
        """Halving context is a real cost; corrupting the loss target is a worse one."""
        from marlowe.heal import MEMORY_CANDIDATES

        names = [c.name for c in MEMORY_CANDIDATES]
        assert names.index("fp16-head-1024") < names.index("nf4-head-adamw32")

    def test_apply_does_not_mutate_the_original(self) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import MEMORY_CANDIDATES

        base = HealConfig(quantize_lm_head=False)
        MEMORY_CANDIDATES[-1].apply(base)
        assert base.quantize_lm_head is False

    def test_search_takes_the_first_that_fits(self, monkeypatch) -> None:
        from marlowe import heal
        from marlowe.config import HealConfig

        calls: list[bool] = []

        def fake_probe(path, cfg, *, n_steps=8, max_gpu_gb=None, base=None):
            calls.append(cfg.quantize_lm_head)
            headroom = 0.2 if not cfg.quantize_lm_head else 2.0
            return heal.ProbeResult(
                tok_s=180.0, peak_vram_gb=16.0 - headroom, total_vram_gb=16.0,
                step_s=1.0, seq_len=cfg.seq_len, n_steps=n_steps, lora_params=176_000_000,
            )

        monkeypatch.setattr(heal, "probe_training", fake_probe)
        _stub_model_plumbing(monkeypatch)
        res = heal.search_memory_plan("x", HealConfig(), margin_gb=0.8)

        # NF4 candidates need explicit approval, so without it the search stops here.
        assert res.chosen is None
        assert res.exhausted_without_approval
        n_fp16 = sum(1 for c in heal.MEMORY_CANDIDATES if not c.quantize_lm_head)
        assert calls == [False] * n_fp16, "gated candidates must not be probed"

        calls.clear()
        approved = heal.search_memory_plan(
            "x", HealConfig(), margin_gb=0.8, allow_quantized_head=True
        )
        assert approved.chosen is not None
        assert approved.chosen.name == "nf4-head-adamw32"
        assert calls == [False] * n_fp16 + [True], "must not probe further once one fits"

    def test_search_prefers_fp16_head_when_it_fits(self, monkeypatch) -> None:
        from marlowe import heal
        from marlowe.config import HealConfig

        monkeypatch.setattr(
            heal, "probe_training",
            lambda p, c, **k: heal.ProbeResult(
                tok_s=150.0, peak_vram_gb=13.0, total_vram_gb=16.0, step_s=1.0,
                seq_len=c.seq_len, n_steps=4, lora_params=1,
            ),
        )
        _stub_model_plumbing(monkeypatch)
        res = heal.search_memory_plan("x", HealConfig(), margin_gb=0.8)
        assert res.chosen is not None and res.chosen.name == "fp16-head-2048"
        assert res.config is not None and res.config.quantize_lm_head is False

    def test_oom_is_treated_as_does_not_fit(self, monkeypatch) -> None:
        from marlowe import heal
        from marlowe.config import HealConfig

        def fake_probe(path, cfg, **k):
            if not cfg.quantize_lm_head:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.50 GiB")
            return heal.ProbeResult(
                tok_s=200.0, peak_vram_gb=13.0, total_vram_gb=16.0, step_s=1.0,
                seq_len=cfg.seq_len, n_steps=4, lora_params=1,
            )

        monkeypatch.setattr(heal, "probe_training", fake_probe)
        _stub_model_plumbing(monkeypatch)
        res = heal.search_memory_plan(
            "x", HealConfig(), margin_gb=0.5, allow_quantized_head=True
        )
        assert res.chosen is not None and res.chosen.quantize_lm_head is True
        assert any("OOM" in outcome for _, outcome in res.attempts)

    def test_non_oom_errors_propagate(self, monkeypatch) -> None:
        from marlowe import heal
        from marlowe.config import HealConfig

        def boom(path, cfg, **k):
            raise ValueError("a real bug, not a memory problem")

        _stub_model_plumbing(monkeypatch)
        monkeypatch.setattr(heal, "probe_training", boom)
        with pytest.raises(ValueError, match="a real bug"):
            heal.search_memory_plan("x", HealConfig())

    def test_nothing_fits_is_reported_not_silently_accepted(self, monkeypatch) -> None:
        from marlowe import heal
        from marlowe.config import HealConfig

        monkeypatch.setattr(
            heal, "probe_training",
            lambda p, c, **k: heal.ProbeResult(
                tok_s=1.0, peak_vram_gb=15.9, total_vram_gb=16.0, step_s=1.0,
                seq_len=c.seq_len, n_steps=1, lora_params=1,
            ),
        )
        _stub_model_plumbing(monkeypatch)
        # With approval granted, exhausting every candidate is a plain "nothing fits" --
        # distinct from stopping at the approval gate, which has its own message.
        res = heal.search_memory_plan(
            "x", HealConfig(), margin_gb=0.8, allow_quantized_head=True
        )
        assert res.chosen is None
        assert not res.blocked
        assert "NOTHING FIT" in res.render(50_000_000)


class TestLmHeadMitigation:
    """When the head ends up NF4, put trainable capacity where the noise enters."""

    def test_fp16_head_needs_no_mitigation(self) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import effective_lora_targets

        targets, save_full = effective_lora_targets(HealConfig(quantize_lm_head=False))
        assert "lm_head" not in targets
        assert save_full == []

    def test_nf4_head_gets_lora_and_a_trained_final_norm(self) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import FINAL_NORM_MODULE, effective_lora_targets

        targets, save_full = effective_lora_targets(HealConfig(quantize_lm_head=True))
        assert "lm_head" in targets
        assert save_full == [FINAL_NORM_MODULE]

    def test_final_norm_path_is_exact_not_a_suffix(self) -> None:
        """A bare norm target also matches input_layernorm, post_attention_layernorm and
        linear_attn.norm -- three per layer, verified against a real qwen3_5 model."""
        from marlowe.heal import FINAL_NORM_MODULE

        assert FINAL_NORM_MODULE == "model.norm"
        for decoy in ("input_layernorm", "post_attention_layernorm", "linear_attn.norm"):
            assert decoy.endswith("norm")
            assert not decoy.endswith(FINAL_NORM_MODULE)

    def test_bare_norm_is_refused(self) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import effective_lora_targets

        with pytest.raises(ValueError, match="matches by suffix"):
            effective_lora_targets(HealConfig(modules_to_save=["norm"]))

    def test_mitigation_is_idempotent(self) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import FINAL_NORM_MODULE, effective_lora_targets

        cfg = HealConfig(quantize_lm_head=True, modules_to_save=[FINAL_NORM_MODULE])
        cfg.target_modules = [*cfg.target_modules, "lm_head"]
        targets, save_full = effective_lora_targets(cfg)
        assert targets.count("lm_head") == 1
        assert save_full.count(FINAL_NORM_MODULE) == 1

    def test_default_config_does_not_quantize_the_head(self) -> None:
        """bitsandbytes skips lm_head by default and that default is correct here."""
        from marlowe.config import HealConfig

        assert HealConfig().quantize_lm_head is False


class TestMultiWidthShipGate:
    """The deliverable is a base that quantises well, not one that fits a card.

    Under-healed weights carry larger activation outliers and degrade unevenly across
    quantisation schemes, so passing at one bit-width proves nothing about the others. The
    gate requires every width in SHIP_BIT_WIDTHS.
    """

    def _baselines(self):
        from marlowe.report import CheckpointRecord

        return (
            CheckpointRecord("iq3_xxs", kl_mean=0.080, rep32=0.30),
            CheckpointRecord("bf16", kl_mean=0.0, rep32=0.05),
        )

    def _good(self, name):
        from marlowe.report import CheckpointRecord

        return CheckpointRecord(name, quant=name, kl_mean=0.05, rep32=0.04)

    def test_all_three_widths_are_required(self) -> None:
        from marlowe.quantize import SHIP_BIT_WIDTHS

        assert set(SHIP_BIT_WIDTHS) == {"q4_K_M", "iq4_xs", "iq3_m"}

    def test_passes_when_every_width_passes(self) -> None:
        from marlowe.quantize import SHIP_BIT_WIDTHS
        from marlowe.report import ship_gate_multi

        iq3, bf16 = self._baselines()
        cands = {w: self._good(w) for w in SHIP_BIT_WIDTHS}
        res = ship_gate_multi(
            cands, required_widths=SHIP_BIT_WIDTHS,
            iq3_xxs_baseline=iq3, bf16_baseline=bf16,
        )
        assert res.passed

    def test_one_failing_width_fails_the_whole_gate(self) -> None:
        """The specific thing this catches: fine at Q4, falls apart at IQ3."""
        from marlowe.quantize import SHIP_BIT_WIDTHS
        from marlowe.report import CheckpointRecord, ship_gate_multi

        iq3, bf16 = self._baselines()
        cands = {w: self._good(w) for w in SHIP_BIT_WIDTHS}
        cands["iq3_m"] = CheckpointRecord("iq3_m", quant="iq3_m", kl_mean=0.05, rep32=0.40)
        res = ship_gate_multi(
            cands, required_widths=SHIP_BIT_WIDTHS,
            iq3_xxs_baseline=iq3, bf16_baseline=bf16,
        )
        assert not res.passed
        assert res.per_width["q4_K_M"].passed
        assert not res.per_width["iq3_m"].passed
        assert "DO NOT SHIP" in res.render()

    def test_a_width_that_was_never_built_is_a_failure(self) -> None:
        """Absence is not a pass. A missing width means an unproven claim."""
        from marlowe.quantize import SHIP_BIT_WIDTHS
        from marlowe.report import ship_gate_multi

        iq3, bf16 = self._baselines()
        cands = {w: self._good(w) for w in SHIP_BIT_WIDTHS if w != "q4_K_M"}
        res = ship_gate_multi(
            cands, required_widths=SHIP_BIT_WIDTHS,
            iq3_xxs_baseline=iq3, bf16_baseline=bf16,
        )
        assert not res.passed
        assert "q4_K_M" in res.missing
        assert "never built" in res.render()

    def test_ship_recipes_cover_every_gated_width(self) -> None:
        from marlowe.quantize import SHIP_BIT_WIDTHS, ship_recipes

        names = {r.name for r in ship_recipes()}
        assert set(SHIP_BIT_WIDTHS) <= names, (
            "the gate requires widths the recipes do not build; Stage 7 would fail on "
            "missing measurements rather than on quality"
        )

    def test_configs_do_not_pin_a_narrower_quant_list(self) -> None:
        """An explicit quants list would override ship_recipes and starve the gate."""
        from marlowe.config import load_run_config
        from marlowe.quantize import SHIP_BIT_WIDTHS, ship_recipes

        for name in ("marlowe-22b", "marlowe-18b"):
            cfg = load_run_config(f"configs/{name}.yaml")
            built = {r.name for r in (cfg.quants or ship_recipes())}
            missing = set(SHIP_BIT_WIDTHS) - built
            assert not missing, f"{name} would never build {sorted(missing)}"


class TestPromptSetProvenance:
    """Stage 0's curve and Stage 7's gate are comparable only on identical prompts."""

    def test_fingerprint_records_identity_and_trigger_count(self) -> None:
        from marlowe.eval.repetition import fingerprint_prompts, load_prompts

        prompts = load_prompts("data/circling_prompts.jsonl")
        ps = fingerprint_prompts("data/circling_prompts.jsonl", prompts)
        assert len(ps.sha256) == 16
        assert ps.n_prompts == len(prompts)
        assert ps.n_known_triggers == sum(1 for p in prompts if p.known_trigger)

    def test_synthetic_set_is_flagged_as_unvalidated(self) -> None:
        from marlowe.eval.repetition import PromptSet

        assert not PromptSet("p", "abc", 9, 0).validated
        assert PromptSet("p", "abc", 9, 3).validated

    def test_reports_carry_the_hash_and_a_caveat(self) -> None:
        from marlowe.eval.repetition import (
            Completion,
            Prompt,
            PromptSet,
            run_repetition,
        )

        class Fake:
            name = "fake"
            extra_body: ClassVar[dict] = {}

            def generate(self, prompt, max_tokens, sampling, seed):
                return Completion("", " ".join(f"w{i}" for i in range(200)), 200, "stop")

        ps = PromptSet("data/x.jsonl", "deadbeefdeadbeef", 2, 0)
        report = run_repetition(
            Fake(), [Prompt("a", "hi"), Prompt("b", "yo")],
            label="t", n_completions=4, prompt_set=ps,
        )
        assert report.prompt_set["sha256"] == "deadbeefdeadbeef"
        assert "SYNTHETIC PROMPT SET" in report.caveat
        assert "[synthetic prompts]" in report.headline()

    def test_validated_set_gets_no_caveat(self) -> None:
        from marlowe.eval.repetition import (
            Completion,
            Prompt,
            PromptSet,
            run_repetition,
        )

        class Fake:
            name = "fake"
            extra_body: ClassVar[dict] = {}

            def generate(self, prompt, max_tokens, sampling, seed):
                return Completion("", " ".join(f"w{i}" for i in range(200)), 200, "stop")

        ps = PromptSet("data/x.jsonl", "cafebabecafebabe", 2, 2)
        report = run_repetition(
            Fake(), [Prompt("a", "hi", known_trigger=True)],
            label="t", n_completions=4, prompt_set=ps,
        )
        assert report.caveat == ""

    def test_mismatched_prompt_sets_are_detectable(self) -> None:
        from marlowe.eval.repetition import RepetitionReport

        a = RepetitionReport("b", "a", 1, 1, "thinking", "tok", prompt_set={"sha256": "aaa"})
        b = RepetitionReport("b", "b", 1, 1, "thinking", "tok", prompt_set={"sha256": "aaa"})
        c = RepetitionReport("b", "c", 1, 1, "thinking", "tok", prompt_set={"sha256": "zzz"})
        assert a.comparable_with(b)
        assert not a.comparable_with(c)


class TestStage0IsAControlCurve:
    """Reframed: the go/no-go question was answered outside this pipeline."""

    def test_description_does_not_read_as_a_gate(self) -> None:
        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        desc = REGISTRY["stage0-bitwidth"].description
        assert "CONTROL CURVE" in desc
        assert "go/no-go" in desc.lower()

    def test_docstring_states_the_confound_it_resolves(self) -> None:
        import inspect

        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        doc = inspect.getdoc(REGISTRY["stage0-bitwidth"].fn) or ""
        assert "confounded" in doc
        assert "NOT a go/no-go" in doc


class TestLongContextAnnealRefusal:
    """A knob documented at 16384 that needs ~23 GB on a 16 GB card must refuse, loudly.

    Refusing is the point. Silently substituting a length that fits would hand back a
    shortened anneal nobody asked for, and the operator would not know the long-range
    exercise they were relying on had been reduced.
    """

    def test_disabled_anneal_is_never_validated(self) -> None:
        from marlowe.config import HealConfig

        HealConfig(long_context_tokens=0, long_context_seq_len=16384).validate_long_context(
            vram_gb=17.17
        )

    def test_16384_is_refused_on_a_16gb_card(self) -> None:
        from marlowe.config import HealConfig

        cfg = HealConfig(long_context_tokens=5_000_000, long_context_seq_len=16384)
        with pytest.raises(ValueError) as exc:
            cfg.validate_long_context(vram_gb=17.17)
        msg = str(exc.value)
        assert "UNAVAILABLE" in msg
        assert "not being silently shortened" in msg

    def test_message_shows_the_arithmetic(self) -> None:
        from marlowe.config import HealConfig

        cfg = HealConfig(long_context_tokens=1, long_context_seq_len=16384)
        with pytest.raises(ValueError) as exc:
            cfg.validate_long_context(vram_gb=17.17)
        msg = str(exc.value)
        assert "weights + optimizer" in msg
        assert "activations" in msg
        assert "does not shrink with sequence length" in msg
        assert "gradient accumulation does not reduce it" in msg

    def test_message_names_a_ceiling_that_actually_fits(self) -> None:
        """The suggested fallback must itself clear the margin, or it is worse than useless."""
        import re

        from marlowe.config import HealConfig, estimate_peak_gb

        cfg = HealConfig(long_context_tokens=1, long_context_seq_len=16384)
        with pytest.raises(ValueError) as exc:
            cfg.validate_long_context(vram_gb=17.17)
        m = re.search(r"at most ~(\d+)", str(exc.value))
        assert m, "no ceiling suggested"
        ceiling = int(m.group(1))
        assert estimate_peak_gb(ceiling, cfg) <= 17.17 - 1.0

    def test_a_fitting_anneal_is_allowed(self) -> None:
        from marlowe.config import HealConfig

        HealConfig(long_context_tokens=1, long_context_seq_len=512).validate_long_context(
            vram_gb=17.17
        )

    def test_estimator_includes_the_cuda_context(self) -> None:
        """The same torch-vs-device gap that made the probe wrong would make this wrong."""
        from marlowe.config import _EST_CUDA_CONTEXT_GB, HealConfig, estimate_peak_gb

        assert _EST_CUDA_CONTEXT_GB > 0
        cfg = HealConfig()
        # Calibrated against the measured 22B peak: 16.64 GB device-level at seq_len 2048.
        assert abs(estimate_peak_gb(2048, cfg) - 16.64) < 0.35

    def test_activation_memory_is_linear_in_sequence_length(self) -> None:
        from marlowe.config import _activation_gb

        a1, a2 = _activation_gb(1024), _activation_gb(2048)
        assert abs((a2 - 0.25) / (a1 - 0.25) - 2.0) < 0.01

    def test_shipped_configs_do_not_enable_an_impossible_anneal(self) -> None:
        from marlowe.config import load_run_config

        for name in ("marlowe-22b", "marlowe-18b"):
            cfg = load_run_config(f"configs/{name}.yaml", vram_gb=17.17)
            assert cfg.heal.long_context_tokens == 0

    def test_load_refuses_an_impossible_anneal(self, tmp_path) -> None:
        from marlowe.config import load_run_config

        p = tmp_path / "bad.yaml"
        p.write_text(
            "name: x\nparent: y\nn_cuts: 1\n"
            "heal:\n  long_context_tokens: 5000000\n  long_context_seq_len: 16384\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="UNAVAILABLE"):
            load_run_config(p, vram_gb=17.17)


class TestSequenceLengthLadder:
    def test_1536_sits_between_2048_and_1024(self) -> None:
        from marlowe.heal import MEMORY_CANDIDATES

        names = [c.name for c in MEMORY_CANDIDATES]
        assert names.index("fp16-head-2048") < names.index("fp16-head-1536")
        assert names.index("fp16-head-1536") < names.index("fp16-head-1024")

    def test_all_fp16_candidates_precede_every_nf4_one(self) -> None:
        """Shortening context is a real cost; corrupting the loss target is a worse one."""
        from marlowe.heal import MEMORY_CANDIDATES

        first_nf4 = next(
            i for i, c in enumerate(MEMORY_CANDIDATES) if c.quantize_lm_head
        )
        assert all(not c.quantize_lm_head for c in MEMORY_CANDIDATES[:first_nf4])

    def test_probe_all_does_not_short_circuit(self, monkeypatch) -> None:
        """Choosing short-circuits; reporting must not, or the table has holes."""
        from marlowe import heal
        from marlowe.config import HealConfig

        seen: list[int] = []

        def fake(path, cfg, **k):
            seen.append(cfg.seq_len)
            return heal.ProbeResult(
                tok_s=100.0, peak_vram_gb=10.0, total_vram_gb=17.17, step_s=1.0,
                seq_len=cfg.seq_len, n_steps=4, lora_params=1,
            )

        monkeypatch.setattr(heal, "probe_training", fake)
        _stub_model_plumbing(monkeypatch)

        res = heal.search_memory_plan(
            "x", HealConfig(), probe_all=True, allow_quantized_head=True
        )
        assert len(seen) == len(heal.MEMORY_CANDIDATES)
        assert res.chosen is not None and res.chosen.name == "fp16-head-2048"
        assert len(res.attempts) == len(heal.MEMORY_CANDIDATES)

        # Without approval, probe_all still walks the fp16 ladder but never measures a
        # gated candidate: reporting must not spend GPU time on a non-option.
        seen.clear()
        heal.search_memory_plan("x", HealConfig(), probe_all=True)
        n_fp16 = sum(1 for c in heal.MEMORY_CANDIDATES if not c.quantize_lm_head)
        assert len(seen) == n_fp16


class TestQuantizedHeadNeedsApproval:
    """The fp16 ladder exhausting is a decision point, not the next rung down."""

    def _tight(self, headroom: float):
        from marlowe import heal

        def probe(path, cfg, **k):
            return heal.ProbeResult(
                tok_s=120.0, peak_vram_gb=17.17 - headroom, total_vram_gb=17.17,
                step_s=1.0, seq_len=cfg.seq_len, n_steps=4, lora_params=1,
            )

        return probe

    def test_rank16_precedes_every_quantized_candidate(self) -> None:
        from marlowe.heal import MEMORY_CANDIDATES

        names = [c.name for c in MEMORY_CANDIDATES]
        first_nf4 = min(i for i, c in enumerate(MEMORY_CANDIDATES) if c.quantize_lm_head)
        assert names.index("fp16-head-1024-rank16") < first_nf4

    def test_rank16_halves_the_adapter_not_the_target(self) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import MEMORY_CANDIDATES

        cand = next(c for c in MEMORY_CANDIDATES if c.name == "fp16-head-1024-rank16")
        cfg = cand.apply(HealConfig(lora_rank=32))
        assert cfg.lora_rank == 16
        assert cfg.quantize_lm_head is False, "the loss target must stay intact"

    def test_only_quantized_candidates_are_gated(self) -> None:
        from marlowe.heal import MEMORY_CANDIDATES

        for c in MEMORY_CANDIDATES:
            assert c.requires_approval == c.quantize_lm_head

    def test_search_stops_instead_of_selecting_a_quantized_head(self, monkeypatch) -> None:
        from marlowe import heal
        from marlowe.config import HealConfig

        monkeypatch.setattr(heal, "probe_training", self._tight(0.5))
        _stub_model_plumbing(monkeypatch)
        res = heal.search_memory_plan("x", HealConfig(), margin_gb=1.0)

        assert res.chosen is None
        assert res.exhausted_without_approval
        assert {n for n, _ in res.blocked} == {"nf4-head-adamw32", "nf4-head-adamw8"}
        text = res.render(50_000_000)
        assert "LADDER IS EXHAUSTED" in text
        assert "--allow-quantized-head" in text

    def test_gated_candidates_are_never_probed_without_approval(self, monkeypatch) -> None:
        """Blocked means not measured: probing them would waste GPU time on a non-option."""
        from marlowe import heal
        from marlowe.config import HealConfig

        seen: list[bool] = []

        def probe(path, cfg, **k):
            seen.append(cfg.quantize_lm_head)
            return heal.ProbeResult(
                tok_s=1.0, peak_vram_gb=16.7, total_vram_gb=17.17, step_s=1.0,
                seq_len=cfg.seq_len, n_steps=1, lora_params=1,
            )

        monkeypatch.setattr(heal, "probe_training", probe)
        _stub_model_plumbing(monkeypatch)
        heal.search_memory_plan("x", HealConfig(), margin_gb=1.0)
        assert not any(seen), "an approval-gated candidate was probed"

    def test_approval_lets_the_search_continue(self, monkeypatch) -> None:
        from marlowe import heal
        from marlowe.config import HealConfig

        def probe(path, cfg, **k):
            headroom = 2.0 if cfg.quantize_lm_head else 0.5
            return heal.ProbeResult(
                tok_s=120.0, peak_vram_gb=17.17 - headroom, total_vram_gb=17.17,
                step_s=1.0, seq_len=cfg.seq_len, n_steps=4, lora_params=1,
            )

        monkeypatch.setattr(heal, "probe_training", probe)
        _stub_model_plumbing(monkeypatch)
        res = heal.search_memory_plan(
            "x", HealConfig(), margin_gb=1.0, allow_quantized_head=True
        )
        assert res.chosen is not None and res.chosen.name == "nf4-head-adamw32"
        assert not res.blocked

    def test_stage5_refuses_rather_than_trading_silently(self) -> None:
        import inspect

        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        src = inspect.getsource(REGISTRY["stage5-teacher"].fn)
        assert "exhausted_without_approval" in src
        assert "allow_quantized_head" in src

    def test_cli_exposes_the_flag_on_both_commands(self) -> None:
        from marlowe.cli import build_parser

        fit = build_parser().parse_args(
            ["fitcheck", "--model", "m", "--allow-quantized-head"]
        )
        assert fit.allow_quantized_head is True
        run = build_parser().parse_args(
            ["run", "stage5-teacher", "--config", "configs/marlowe-22b.yaml",
             "--allow-quantized-head"]
        )
        assert run.allow_quantized_head is True


class TestNoTorchAccountingInFitDecisions:
    """The torch-vs-device confusion appeared twice; pin that it cannot appear a third time."""

    def test_probe_decides_on_device_level_memory(self) -> None:
        import inspect

        from marlowe.heal import probe_training

        src = inspect.getsource(probe_training)
        assert "mem_get_info" in src
        assert "peak_vram_gb=peak_device_used" in src

    def test_torch_accounting_is_recorded_but_not_decisive(self) -> None:
        from marlowe.heal import ProbeResult

        r = ProbeResult(
            tok_s=1.0, peak_vram_gb=16.0, total_vram_gb=17.0, step_s=1.0,
            seq_len=1024, n_steps=1, lora_params=1,
            torch_allocated_gb=13.0, torch_reserved_gb=14.0,
        )
        # headroom must come from the device figure, not either torch figure
        assert abs(r.headroom_gb() - 1.0) < 1e-9

    def test_preflight_probes_the_device_not_torch_accounting(self) -> None:
        import inspect

        from marlowe.preflight import probe

        src = inspect.getsource(probe)
        assert "mem_get_info" in src
        assert "max_memory_allocated" not in src

    def test_estimator_accounts_for_the_cuda_context(self) -> None:
        import inspect

        from marlowe.config import estimate_peak_gb

        assert "_EST_CUDA_CONTEXT_GB" in inspect.getsource(estimate_peak_gb)


class TestMemoryPlanPersistence:
    """The plan must restore every field the search varied.

    lora_rank was a candidate override that save_memory_plan did not persist. Stage 6 would
    have reloaded seq_len and head precision correctly and silently reverted the rank that
    made the configuration fit -- OOMing on a plan whose own record said it had been measured
    as fitting. The field list is now derived from the dataclass so it cannot drift again.
    """

    def test_persisted_fields_cover_every_override(self) -> None:
        import dataclasses

        from marlowe.heal import MemoryCandidate, _candidate_override_fields

        fixed = {"name", "rationale", "requires_approval"}
        overridable = {
            f.name for f in dataclasses.fields(MemoryCandidate) if f.name not in fixed
        }
        assert set(_candidate_override_fields()) == overridable

    def test_roundtrip_preserves_lora_rank(self, tmp_path) -> None:
        from marlowe.config import HealConfig
        from marlowe.heal import (
            MEMORY_CANDIDATES,
            ProbeResult,
            SearchResult,
            load_memory_plan,
            save_memory_plan,
        )

        cand = next(c for c in MEMORY_CANDIDATES if c.name == "fp16-head-1024-rank16")
        base = HealConfig(lora_rank=32, seq_len=2048)
        chosen = cand.apply(base)
        search = SearchResult(
            chosen=cand,
            config=chosen,
            probe=ProbeResult(
                tok_s=1.0, peak_vram_gb=15.0, total_vram_gb=17.17, step_s=1.0,
                seq_len=1024, n_steps=4, lora_params=1,
            ),
        )
        path = tmp_path / "memory_plan.json"
        save_memory_plan(path, search)

        restored, _ = load_memory_plan(path, base)
        assert restored.lora_rank == 16, "the rank that made it fit must survive the plan"
        assert restored.seq_len == 1024
        assert restored.quantize_lm_head is False

    def test_a_plan_missing_a_field_is_refused(self, tmp_path) -> None:
        """An older plan file must not silently restore a configuration never measured."""
        import json

        from marlowe.config import HealConfig
        from marlowe.heal import load_memory_plan

        path = tmp_path / "memory_plan.json"
        path.write_text(
            json.dumps({"candidate": "fp16-head-1024", "seq_len": 1024}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="predates fields"):
            load_memory_plan(path, HealConfig())


class TestLadderRungsProbeIndependently:
    """The 18B must size itself, not inherit the 22B's plan.

    Shape determines the envelope. At 41 layers the weights drop by roughly 2.4 GB, which
    likely reopens rank 32 or a longer sequence -- and inheriting rank 16 would give the
    *harder* healing job less adapter capacity than the easier one.
    """

    def test_child_runs_in_its_own_run_dir(self) -> None:
        import inspect

        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        src = inspect.getsource(REGISTRY["stage8-ladder"].fn)
        assert "run_dir=ctx.run_dir / child.name" in src, (
            "the child must not share a run_dir with rung 1, or it inherits its memory plan"
        )

    def test_plan_paths_do_not_collide(self, tmp_path) -> None:
        from marlowe.config import load_run_config
        from marlowe.heal import MEMORY_PLAN_FILE
        from marlowe.manifest import Manifest
        from marlowe.pipeline import StageContext

        parent_cfg = load_run_config("configs/marlowe-22b.yaml", vram_gb=17.17)
        child_cfg = load_run_config("configs/marlowe-18b.yaml", vram_gb=17.17)

        parent = StageContext("s", parent_cfg, tmp_path, Manifest(stage="s"))
        child = StageContext("s", child_cfg, tmp_path / child_cfg.name, Manifest(stage="s"))

        assert (parent.metrics_dir / MEMORY_PLAN_FILE) != (
            child.metrics_dir / MEMORY_PLAN_FILE
        )

    def test_child_config_starts_at_full_capacity(self) -> None:
        """The 18B searches down from rank 32; it does not begin where rung 1 ended."""
        from marlowe.config import load_run_config

        child = load_run_config("configs/marlowe-18b.yaml", vram_gb=17.17)
        assert child.heal.lora_rank == 32
        assert child.heal.quantize_lm_head is False
        assert child.heal.seq_len == 2048

    def test_stage5_searches_rather_than_loading(self) -> None:
        """Rung 2's Stage 5 must run its own search against the 18B checkpoint."""
        import inspect

        import marlowe.stages  # noqa: F401
        from marlowe.pipeline import REGISTRY

        src = inspect.getsource(REGISTRY["stage5-teacher"].fn)
        assert "search_memory_plan" in src
        assert "load_memory_plan" not in src, (
            "Stage 5 decides the plan; loading one would inherit another rung's envelope"
        )

    def test_the_18b_is_a_smaller_envelope(self) -> None:
        """Sanity: fewer layers really does free memory, so re-probing can find more room."""
        from marlowe.arch import ArchDims, Layout, positional_selection

        d = ArchDims()
        parent = Layout.qwen38_27b()
        r1, _ = parent.apply(positional_selection(parent, 12))
        r2, _ = r1.apply(positional_selection(r1, 11))
        nf4 = 4.127 / 8
        w1 = (d.total_params(r1.layer_types) - d.embedding_params) * nf4 / 1e9
        w2 = (d.total_params(r2.layer_types) - d.embedding_params) * nf4 / 1e9
        assert w1 - w2 > 1.5, f"expected the 18B to free >1.5 GB, got {w1 - w2:.2f}"
