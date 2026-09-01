"""Importance matrices are per-model, and IQ recipes refuse to run without one.

Stage 0 converted the 27B parent -- 54 GB, seven minutes -- and then failed on its first
recipe with "this quantization requires an importance matrix!". ``quantize()`` had carried an
``imatrix`` parameter since it was written, and no call site ever passed one: the same shape
as ``lora_rank`` being a candidate override that persistence did not carry.

The per-model rule is the part worth protecting. An imatrix records which weights carried
activation magnitude on the calibration text; prune 12 layers and the survivors carry
different activations. Reusing a parent's matrix for a child would produce a quantisation
tuned for a model that no longer exists -- and it would load, run, and be quietly worse.
"""

from __future__ import annotations

import pytest

from marlowe.imatrix import (
    IMATRIX_CTX,
    gguf_fingerprint,
    imatrix_for,
    recipe_needs_imatrix,
)
from marlowe.quantize import QuantConfig, quantize


class TestWhichRecipesNeedOne:
    @pytest.mark.parametrize("t", ["iq3_xxs", "iq3_s", "iq3_m", "IQ2_XS", "iq1_s"])
    def test_iq_class_requires_an_imatrix(self, t: str) -> None:
        assert recipe_needs_imatrix(t)

    @pytest.mark.parametrize("t", ["q4_K_M", "q4_K_S", "iq4_xs", "q5_K", "q8_0", "bf16"])
    def test_others_do_not(self, t: str) -> None:
        assert not recipe_needs_imatrix(t)

    def test_iq4_is_not_gated(self) -> None:
        """IQ4_XS is in SHIP_BIT_WIDTHS and llama.cpp builds it without a matrix.

        Gating it would block the ship gate on an input it does not need.
        """
        assert not recipe_needs_imatrix("iq4_xs")


class TestQuantizeRefusesWithoutOne:
    def test_iq_base_type_refuses(self, tmp_path) -> None:
        src = tmp_path / "src.gguf"
        src.write_bytes(b"x")
        with pytest.raises(ValueError, match="importance matrix"):
            quantize(src, tmp_path / "o.gguf", QuantConfig(name="iq3_xxs", base_type="iq3_xxs"))

    def test_an_iq_tensor_override_also_refuses(self, tmp_path) -> None:
        """The custom mixes set ffn_.* to iq3_xxs over a non-IQ base.

        Gating only on base_type would let those through to fail in llama.cpp.
        """
        src = tmp_path / "src.gguf"
        src.write_bytes(b"x")
        recipe = QuantConfig(
            name="mix", base_type="q4_K_M", tensor_types={"ffn_.*": "iq3_xxs"}
        )
        with pytest.raises(ValueError, match="importance matrix"):
            quantize(src, tmp_path / "o.gguf", recipe)

    def test_a_missing_imatrix_file_is_named(self, tmp_path) -> None:
        src = tmp_path / "src.gguf"
        src.write_bytes(b"x")
        with pytest.raises(FileNotFoundError):
            quantize(
                src, tmp_path / "o.gguf",
                QuantConfig(name="iq3_m", base_type="iq3_m"),
                imatrix=tmp_path / "absent.dat",
            )


class TestPerModelCaching:
    def _gguf(self, path, payload: bytes):
        path.write_bytes(payload)
        return path

    def test_different_models_get_different_matrices(self, tmp_path) -> None:
        a = self._gguf(tmp_path / "a.gguf", b"A" * 4096)
        b = self._gguf(tmp_path / "b.gguf", b"B" * 4096)
        assert gguf_fingerprint(a) != gguf_fingerprint(b), (
            "a child must not inherit its parent's imatrix"
        )

    def test_same_bytes_hash_the_same(self, tmp_path) -> None:
        a = self._gguf(tmp_path / "a.gguf", b"A" * 4096)
        b = self._gguf(tmp_path / "b.gguf", b"A" * 4096)
        assert gguf_fingerprint(a) == gguf_fingerprint(b)

    def test_a_changed_tail_changes_the_fingerprint(self, tmp_path) -> None:
        """Two checkpoints of the same shape differ in their tensor data, not their header."""
        big = b"H" * (1 << 20) + b"M" * (1 << 20) + b"T" * (1 << 20)
        a = self._gguf(tmp_path / "a.gguf", big)
        b = self._gguf(tmp_path / "b.gguf", big[:-1] + b"X")
        assert gguf_fingerprint(a) != gguf_fingerprint(b)

    def test_cache_hit_skips_the_build(self, tmp_path, monkeypatch) -> None:
        src = self._gguf(tmp_path / "m.gguf", b"M" * 4096)
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / f"imatrix-{gguf_fingerprint(src)}.dat").write_bytes(b"cached")

        def boom(*a, **k):
            raise AssertionError("should not rebuild when the cache holds this model")

        monkeypatch.setattr("marlowe.imatrix.build_imatrix", boom)
        got = imatrix_for(src, tmp_path / "corpus.txt", cache)
        assert got.read_bytes() == b"cached"

    def test_a_different_model_misses_the_cache(self, tmp_path, monkeypatch) -> None:
        src = self._gguf(tmp_path / "m.gguf", b"M" * 4096)
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "imatrix-deadbeefdeadbeef.dat").write_bytes(b"someone elses")
        called: dict[str, object] = {}

        def fake(src_gguf, corpus, out, **kw):
            called["ctx"] = kw.get("ctx", IMATRIX_CTX)
            out.write_bytes(b"fresh")
            return out

        monkeypatch.setattr("marlowe.imatrix.build_imatrix", fake)
        got = imatrix_for(src, tmp_path / "corpus.txt", cache)
        assert got.read_bytes() == b"fresh"
        assert called["ctx"] == 4096, "512 under-weights the 16 context-dependent attn layers"


class TestSegmentedBuild:
    """An imatrix pass must survive being interrupted.

    A kernel panic killed one at 120 of ~420 chunks. Segmenting banks each finished part on
    disk and chains it into the next with --in-file, so a crash costs one segment rather than
    the whole pass. --chunks cannot do this: it caps chunks read from the START of the file,
    so a rerun re-reads the same opening text instead of advancing.
    """

    def test_split_preserves_every_byte(self, tmp_path) -> None:
        from marlowe.imatrix import split_corpus

        src = tmp_path / "c.txt"
        src.write_text("\n".join(f"line {i} " + "x" * 60 for i in range(500)), encoding="utf-8")
        parts = split_corpus(src, tmp_path / "seg", 4)
        assert len(parts) == 4
        joined = "".join(p.read_text(encoding="utf-8") for p in parts)
        assert joined == src.read_text(encoding="utf-8"), "splitting must not drop text"

    def test_split_breaks_on_line_boundaries(self, tmp_path) -> None:
        """A chunk cut mid-token would feed llama-imatrix a corrupt fragment."""
        from marlowe.imatrix import split_corpus

        src = tmp_path / "c.txt"
        src.write_text("\n".join(f"sentence {i}" for i in range(200)) + "\n", encoding="utf-8")
        for p in split_corpus(src, tmp_path / "seg", 3):
            body = p.read_text(encoding="utf-8")
            assert body.startswith("sentence") or body == ""
            assert body.endswith("\n")

    def test_segments_chain_and_resume(self, tmp_path, monkeypatch) -> None:
        """A completed segment is skipped on rerun, and each merges the one before it."""
        import marlowe.imatrix as im

        src = tmp_path / "m.gguf"
        src.write_bytes(b"M" * 4096)
        corpus = tmp_path / "c.txt"
        corpus.write_text("\n".join(f"line {i}" for i in range(400)) + "\n", encoding="utf-8")

        merges: list[str | None] = []

        def fake_build(src_gguf, part, out, *, merge_from=None, **kw):
            merges.append(None if merge_from is None else str(merge_from))
            out.write_bytes(b"seg")
            return out

        monkeypatch.setattr(im, "build_imatrix", fake_build)
        monkeypatch.setattr(im, "describe_imatrix", lambda p: {"imatrix.chunk_count": 50})

        out = tmp_path / "final.dat"
        im.build_imatrix_segmented(src, corpus, out, segments=3)
        assert out.exists()
        assert merges[0] is None, "the first segment starts fresh"
        assert all(m is not None for m in merges[1:]), "later segments must merge the previous"

        # Rerun: everything is cached, nothing rebuilds.
        merges.clear()
        im.build_imatrix_segmented(src, corpus, out, segments=3)
        assert merges == [], "a completed run must not redo finished segments"
