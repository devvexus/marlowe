"""Manifests, config loading, the ship gate, GGUF reading, and selection logic."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from marlowe.arch import Layout
from marlowe.config import load_run_config
from marlowe.manifest import Manifest, fingerprint_path, hash_obj
from marlowe.report import CheckpointRecord, render_table, ship_gate
from marlowe.score import oneshot_select

# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


class TestManifest:
    def test_roundtrip(self, tmp_path) -> None:
        m = Manifest(stage="s", config_hash="abc")
        m.outputs["out"] = str(tmp_path / "f.txt")
        m.succeed(kl=0.1)
        m.save(tmp_path)
        back = Manifest.load(tmp_path, "s")
        assert back is not None
        assert back.status == "ok"
        assert back.metrics["kl"] == 0.1

    def test_missing_returns_none(self, tmp_path) -> None:
        assert Manifest.load(tmp_path, "nope") is None

    def test_torn_write_is_treated_as_absent(self, tmp_path) -> None:
        p = Manifest.path_for(tmp_path, "s")
        p.parent.mkdir(parents=True)
        p.write_text("{not json", encoding="utf-8")
        assert Manifest.load(tmp_path, "s") is None

    def test_skips_when_current(self, tmp_path) -> None:
        f = tmp_path / "out.bin"
        f.write_bytes(b"x")
        m = Manifest(stage="s", config_hash="h")
        m.inputs["a"] = fingerprint_path(f)
        m.outputs["o"] = str(f)
        m.succeed()
        skip, reason = m.is_current(m.inputs, "h")
        assert skip, reason

    def test_reruns_when_config_changes(self, tmp_path) -> None:
        m = Manifest(stage="s", config_hash="h")
        m.succeed()
        skip, reason = m.is_current({}, "different")
        assert not skip
        assert "config changed" in reason

    def test_reruns_when_an_output_vanished(self, tmp_path) -> None:
        """A manifest is a claim; the files are the evidence."""
        f = tmp_path / "out.bin"
        f.write_bytes(b"x")
        m = Manifest(stage="s", config_hash="h")
        m.outputs["o"] = str(f)
        m.succeed()
        f.unlink()
        skip, reason = m.is_current({}, "h")
        assert not skip
        assert "outputs missing" in reason

    def test_reruns_when_an_input_changed(self, tmp_path) -> None:
        f = tmp_path / "in.bin"
        f.write_bytes(b"x")
        m = Manifest(stage="s", config_hash="h")
        m.inputs["a"] = fingerprint_path(f)
        m.succeed()
        f.write_bytes(b"xxxxx")
        skip, reason = m.is_current({"a": fingerprint_path(f)}, "h")
        assert not skip
        assert "changed" in reason

    def test_failed_manifest_never_skips(self, tmp_path) -> None:
        m = Manifest(stage="s", config_hash="h")
        m.fail(RuntimeError("boom"))
        skip, reason = m.is_current({}, "h")
        assert not skip
        assert "failed" in reason

    def test_fingerprint_dir_is_stable(self, tmp_path) -> None:
        d = tmp_path / "d"
        d.mkdir()
        (d / "a").write_bytes(b"1")
        assert fingerprint_path(d) == fingerprint_path(d)

    def test_hash_obj_is_key_order_independent(self) -> None:
        assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_loads_both_shipped_configs(self) -> None:
        for name in ("marlowe-22b", "marlowe-18b"):
            cfg = load_run_config(Path("configs") / f"{name}.yaml")
            assert cfg.name == name
            assert cfg.repetition.preset == "thinking"

    def test_22b_targets_match_the_accounting(self) -> None:
        from marlowe.arch import ArchDims

        cfg = load_run_config("configs/marlowe-22b.yaml")
        d = ArchDims()
        base = d.total_params(Layout.qwen38_27b().layer_types)
        got = (base - cfg.n_cuts * d.linear_block_params) / 1e9
        assert got == pytest.approx(cfg.expected_params, abs=0.001)
        assert 64 - cfg.n_cuts == cfg.expected_layers

    def test_18b_targets_match_the_accounting(self) -> None:
        from marlowe.arch import ArchDims

        c22 = load_run_config("configs/marlowe-22b.yaml")
        c18 = load_run_config("configs/marlowe-18b.yaml")
        d = ArchDims()
        base = d.total_params(Layout.qwen38_27b().layer_types)
        got = (base - (c22.n_cuts + c18.n_cuts) * d.linear_block_params) / 1e9
        assert got == pytest.approx(c18.expected_params, abs=0.001)
        assert 64 - c22.n_cuts - c18.n_cuts == c18.expected_layers

    def test_18b_requires_a_healed_parent(self) -> None:
        """The guard against a direct 27B -> 18B cut."""
        assert load_run_config("configs/marlowe-18b.yaml").requires_healed_parent is True

    def test_rejects_unknown_keys(self, tmp_path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("name: x\nparent: y\nn_cuts: 1\ntypo_key: 3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown top-level keys"):
            load_run_config(p)

    def test_rejects_unknown_nested_keys(self, tmp_path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("name: x\nparent: y\nn_cuts: 1\nscore:\n  nonsense: 3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown keys for ScoreConfig"):
            load_run_config(p)

    def test_stage_hash_isolates_sections(self) -> None:
        """A healing tweak must not invalidate a finished eight-hour scoring run."""
        a = load_run_config("configs/marlowe-22b.yaml")
        b = load_run_config("configs/marlowe-22b.yaml")
        b.heal.learning_rate = 5e-5
        assert a.stage_hash("score") == b.stage_hash("score")
        assert a.stage_hash("heal") != b.stage_hash("heal")

    def test_config_hash_is_sensitive_to_everything(self) -> None:
        a = load_run_config("configs/marlowe-22b.yaml")
        b = load_run_config("configs/marlowe-22b.yaml")
        b.n_cuts = 13
        assert a.config_hash() != b.config_hash()


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


class TestOneshotSelect:
    def test_picks_the_cheapest_subject_to_constraints(self) -> None:
        layout = Layout.qwen38_27b()
        scores = {i: float(i) for i in layout.candidates()}  # cheapest = lowest index
        picked = oneshot_select(scores, layout, 12)
        assert len(picked) == 12
        layout.validate_removal(picked)

    def test_honours_max_per_period(self) -> None:
        layout = Layout.qwen38_27b()
        # Make one period look uniformly cheap; the cap must still bind at 2.
        scores = {i: 100.0 for i in layout.candidates()}
        for i in (4, 5, 6):
            scores[i] = 0.0
        picked = oneshot_select(scores, layout, 6)
        assert len({4, 5, 6} & set(picked)) == 2

    def test_raises_when_constraints_cannot_be_met(self) -> None:
        layout = Layout.qwen38_27b()
        scores = {i: float(i) for i in layout.candidates()}
        with pytest.raises(ValueError, match="constraints admit only"):
            oneshot_select(scores, layout, 40)

    def test_never_selects_an_attention_layer(self) -> None:
        layout = Layout.repeating(["linear_attention"] * 3 + ["sparse_attention"], 8)
        scores = {i: float(i) for i in layout.candidates()}
        for i in oneshot_select(scores, layout, 6):
            assert layout.layer_types[i] == "linear_attention"


# ---------------------------------------------------------------------------
# ship gate
# ---------------------------------------------------------------------------


class TestShipGate:
    def _base(self) -> tuple[CheckpointRecord, CheckpointRecord]:
        return (
            CheckpointRecord("iq3_xxs", kl_mean=0.080, rep32=0.30),
            CheckpointRecord("bf16", kl_mean=0.0, rep32=0.05),
        )

    def test_passes_when_both_criteria_met(self) -> None:
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.050, rep32=0.04)
        r = ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16)
        assert r.passed
        assert all("FAIL" not in x for x in r.reasons)

    def test_fails_on_kl_alone(self) -> None:
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.090, rep32=0.01)
        r = ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16)
        assert not r.passed
        assert any("FAIL kl_vs_iq3_xxs" in x for x in r.reasons)

    def test_fails_on_repetition_alone(self) -> None:
        """The specific failure the project exists to prevent: good KL, still loops."""
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.010, rep32=0.40)
        r = ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16)
        assert not r.passed
        assert any("FAIL repetition_vs_bf16" in x for x in r.reasons)

    def test_equal_repetition_passes(self) -> None:
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.01, rep32=0.05)
        assert ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16).passed

    def test_equal_kl_fails(self) -> None:
        """KL must be strictly lower: matching the current build is not an improvement."""
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.080, rep32=0.01)
        assert not ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16).passed

    def test_missing_measurement_fails_closed(self) -> None:
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.01)  # no repetition measured
        r = ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16)
        assert not r.passed
        assert any("never measured" in x for x in r.reasons)

    def test_needle_regression_warns_without_blocking(self) -> None:
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.01, rep32=0.01, needle_128k=0.4)
        r = ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16)
        assert r.passed
        assert any("WARN needle" in x for x in r.reasons)

    def test_oversize_warns(self) -> None:
        iq3, bf16 = self._base()
        cand = CheckpointRecord("cand", kl_mean=0.01, rep32=0.01, size_gb=11.5)
        r = ship_gate(cand, iq3_xxs_baseline=iq3, bf16_baseline=bf16)
        assert any("10.0 GB weight budget" in x for x in r.reasons)


class TestTable:
    def test_renders_missing_values_as_dashes(self) -> None:
        out = render_table([CheckpointRecord("only-a-label")])
        assert "only-a-label" in out
        assert "--" in out

    def test_renders_numbers(self) -> None:
        out = render_table([CheckpointRecord("x", kl_mean=0.1234, rep32=0.5)])
        assert "0.1234" in out


# ---------------------------------------------------------------------------
# GGUF reader
# ---------------------------------------------------------------------------


def _write_gguf(path: Path, kv: dict[str, str], tensors: list[str]) -> Path:
    """Minimal GGUF writer, enough to exercise the reader."""

    def gstr(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    body = struct.pack("<I", 0x46554747) + struct.pack("<I", 3)
    body += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, v in kv.items():
        body += gstr(k) + struct.pack("<I", 8) + gstr(v)
    for name in tensors:
        body += gstr(name) + struct.pack("<I", 1) + struct.pack("<Q", 4)
        body += struct.pack("<I", 0) + struct.pack("<Q", 0)
    path.write_bytes(body)
    return path


class TestGGUF:
    def test_reads_metadata_and_tensors(self, tmp_path) -> None:
        from marlowe.quantize import read_gguf

        p = _write_gguf(
            tmp_path / "m.gguf",
            {"general.architecture": "qwen3_5"},
            ["blk.0.attn_q.weight", "blk.1.ssm_in.weight"],
        )
        info = read_gguf(p)
        assert info.metadata["general.architecture"] == "qwen3_5"
        assert info.n_tensors == 2

    def test_rejects_a_non_gguf(self, tmp_path) -> None:
        from marlowe.quantize import read_gguf

        p = tmp_path / "x.gguf"
        p.write_bytes(b"NOPE" + b"\x00" * 40)
        with pytest.raises(ValueError, match="not a GGUF"):
            read_gguf(p)

    def test_layer_families(self, tmp_path) -> None:
        from marlowe.quantize import gguf_layer_families, read_gguf

        p = _write_gguf(
            tmp_path / "m.gguf",
            {},
            [
                "blk.0.ssm_in.weight", "blk.0.ffn_gate.weight",
                "blk.1.ssm_in.weight",
                "blk.2.ssm_in.weight",
                "blk.3.attn_q.weight", "blk.3.attn_k.weight",
            ],
        )
        fams = gguf_layer_families(read_gguf(p))
        assert fams[0] == {"linear"}
        assert fams[3] == {"attention"}

    def test_probe_detects_a_regenerated_layout(self, tmp_path, mini_checkpoint) -> None:
        """The Stage 1 blocker: converter rebuilt a uniform 3:1 over the wrong count."""
        from marlowe.quantize import probe_converter

        # Source declares 12 layers as [lin,lin,lin,full] x3. Simulate a converter that
        # produced a stack shifted by one, which still looks like a plausible 3:1.
        names = []
        for i in range(12):
            fam = "attn_q" if i % 4 == 0 else "ssm_in"
            names.append(f"blk.{i}.{fam}.weight")
        p = _write_gguf(tmp_path / "bad.gguf", {"general.architecture": "qwen3_5"}, names)
        result = probe_converter(mini_checkpoint, p)
        assert not result.ok
        assert result.mismatches
        assert "full_attention_interval" in result.detail

    def test_probe_accepts_a_faithful_conversion(self, tmp_path, mini_checkpoint,
                                                 mini_layer_types) -> None:
        from marlowe.quantize import probe_converter

        names = []
        for i, t in enumerate(mini_layer_types):
            fam = "ssm_in" if t == "linear_attention" else "attn_q"
            names.append(f"blk.{i}.{fam}.weight")
        p = _write_gguf(tmp_path / "good.gguf", {"general.architecture": "qwen3_5"}, names)
        result = probe_converter(mini_checkpoint, p)
        assert result.ok, result.detail

    def test_probe_detects_a_wrong_block_count(self, tmp_path, mini_checkpoint) -> None:
        from marlowe.quantize import probe_converter

        names = [f"blk.{i}.ssm_in.weight" for i in range(8)]
        p = _write_gguf(tmp_path / "short.gguf", {}, names)
        result = probe_converter(mini_checkpoint, p)
        assert not result.ok
        assert "did not read layer_types" in result.detail


class TestModelfile:
    def test_pins_the_thinking_preset(self, tmp_path) -> None:
        from marlowe.quantize import write_modelfile

        p = write_modelfile(tmp_path / "m.gguf", tmp_path / "Modelfile", name="marlowe-22b")
        text = p.read_text(encoding="utf-8")
        assert "PARAMETER presence_penalty 0.0" in text
        assert "PARAMETER temperature 1.0" in text
        assert "PARAMETER top_p 0.95" in text
        assert "PARAMETER top_k 20" in text
