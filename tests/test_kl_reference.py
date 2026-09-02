"""The KL reference is permanent, so the ways it can be silently wrong matter most.

Three failure modes, all of which produce a plausible-looking number rather than an error:

* The reference shares documents with the healing corpus, so the ship gate scores
  memorisation instead of generalisation.
* The reference is built at a context nobody intended. llama.cpp adopts a base file's
  ``n_ctx`` without warning, and a 4096 file is byte-indistinguishable from an 8192 one.
* Its size is projected from the wrong token count. A chunk *consumes* ``n_ctx`` corpus
  tokens and *scores* ``n_ctx/2`` of them; swapping the two doubles or halves the disk
  estimate, and at ``-c 8192`` the scored count is 4096 -- numerically identical to the old
  default context, so the slip looks like a correct answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_corpora import assert_disjoint, heal_document_hashes  # noqa: E402
from marlowe.eval import kl  # noqa: E402

# ---------------------------------------------------------------------------
# disjointness
# ---------------------------------------------------------------------------


def _heal_file(tmp_path: Path, texts: list[str]) -> Path:
    p = tmp_path / "heal_corpus.jsonl"
    p.write_text(
        "".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8"
    )
    return p


def test_the_assertion_fires_on_a_deliberately_overlapping_pair(tmp_path: Path) -> None:
    """The test that matters: construct the collision on purpose and confirm it is caught.

    A disjointness check that has only ever been run against disjoint inputs is not evidence
    of anything.
    """
    shared = "A proof that every finite integral domain is a field."
    heal = _heal_file(tmp_path, ["unrelated healing document", shared])
    rows = [{"text": "held-out document"}, {"text": shared}]

    with pytest.raises(RuntimeError, match="shares 1 document"):
        assert_disjoint(rows, heal_document_hashes(heal))


def test_it_fires_even_when_the_overlap_is_a_single_document_among_many(
    tmp_path: Path,
) -> None:
    """99% held out is not held out."""
    heal = _heal_file(tmp_path, [f"heal doc {i}" for i in range(200)])
    rows = [{"text": f"reference doc {i}"} for i in range(199)] + [{"text": "heal doc 7"}]

    with pytest.raises(RuntimeError):
        assert_disjoint(rows, heal_document_hashes(heal))


def test_whitespace_does_not_launder_an_overlapping_document(tmp_path: Path) -> None:
    """Hashing is on stripped text, so trailing newlines cannot smuggle a duplicate past."""
    heal = _heal_file(tmp_path, ["shared body text"])
    rows = [{"text": "  shared body text\n\n"}]

    with pytest.raises(RuntimeError):
        assert_disjoint(rows, heal_document_hashes(heal))


def test_a_genuinely_disjoint_pair_passes(tmp_path: Path) -> None:
    heal = _heal_file(tmp_path, ["heal a", "heal b"])
    assert_disjoint([{"text": "ref a"}, {"text": "ref b"}], heal_document_hashes(heal))


def test_a_missing_healing_corpus_does_not_silently_pass(tmp_path: Path) -> None:
    """An absent file yields an empty hash set, which would wave everything through.

    Recorded rather than asserted-against: the builder must check the corpus exists before
    trusting this, because "no collisions" and "nothing to collide with" look identical here.
    """
    assert heal_document_hashes(tmp_path / "does-not-exist.jsonl") == set()


# ---------------------------------------------------------------------------
# the two token counts
# ---------------------------------------------------------------------------


def test_a_chunk_consumes_the_full_context_but_scores_half() -> None:
    assert kl.kld_tokens_per_chunk(8192) == 4095
    assert kl.kld_tokens_per_chunk(4096) == 2047
    assert kl.kld_chunks_for_corpus(100_000, 8192) == 12
    assert kl.kld_chunks_for_corpus(100_000, 4096) == 24


def test_the_scored_count_is_not_the_chunk_divisor() -> None:
    """The specific slip this guards: dividing a corpus by the *scored* count.

    At -c 8192 the scored count is 4096, so the wrong divisor yields 24 chunks for a
    100K-token corpus -- exactly double the truth, and a number that reads as reasonable.
    """
    ctx = 8192
    wrong = 100_000 // kl.kld_tokens_per_chunk(ctx)
    assert wrong == 24
    assert kl.kld_chunks_for_corpus(100_000, ctx) == 12
    assert wrong == 2 * kl.kld_chunks_for_corpus(100_000, ctx)


def test_projected_size_scales_with_corpus_tokens_not_with_context() -> None:
    """Halving the context doubles the chunks and halves each one; the total barely moves."""
    at_8192 = kl.kld_expected_bytes(248_320, 8192, kl.kld_chunks_for_corpus(300_000, 8192))
    at_4096 = kl.kld_expected_bytes(248_320, 4096, kl.kld_chunks_for_corpus(300_000, 4096))
    assert abs(at_8192 - at_4096) / at_8192 < 0.02


def test_the_reference_is_budgeted_at_its_real_size() -> None:
    """It was itemised at a flat 2 GB, which is wrong by ~36x."""
    from marlowe import preflight

    assert preflight.kl_reference_bytes() > 70 * 1000**3


# ---------------------------------------------------------------------------
# the sidecar, and what it refuses
# ---------------------------------------------------------------------------


def _reference(tmp_path: Path, **overrides: object) -> tuple[Path, Path]:
    corpus = tmp_path / "kl_reference.txt"
    corpus.write_text("held-out text", encoding="utf-8")
    ref = tmp_path / "reference.kld"
    ref.write_bytes(b"_logits_" + b"\0" * 12)
    meta = {
        "ctx": 8192,
        "chunks": 36,
        "scored_tokens": 147_420,
        "corpus": str(corpus),
        "corpus_sha256": kl.sha256_file(corpus),
    }
    meta.update(overrides)
    kl.sidecar_path(ref).write_text(json.dumps(meta), encoding="utf-8")
    return ref, corpus


def test_measuring_at_a_context_the_reference_was_not_built_at_is_refused(
    tmp_path: Path,
) -> None:
    """llama.cpp would quietly use the base file's context and report a valid-looking KL."""
    ref, corpus = _reference(tmp_path)
    with pytest.raises(ValueError, match="built at n_ctx=8192"):
        kl.measure("model.gguf", corpus, ref, ctx=4096)


def test_a_reference_without_a_sidecar_is_refused(tmp_path: Path) -> None:
    ref, corpus = _reference(tmp_path)
    kl.sidecar_path(ref).unlink()
    with pytest.raises(RuntimeError, match="no sidecar"):
        kl.measure("model.gguf", corpus, ref)


def test_measuring_against_a_different_corpus_is_refused(tmp_path: Path) -> None:
    """KL is only defined over the same tokens."""
    ref, _ = _reference(tmp_path)
    other = tmp_path / "other.txt"
    other.write_text("some other text entirely", encoding="utf-8")
    with pytest.raises(ValueError, match="not the corpus"):
        kl.measure("model.gguf", other, ref)


def test_a_missing_reference_is_refused_before_the_sidecar_is_consulted(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        kl.measure("model.gguf", tmp_path / "c.txt", tmp_path / "absent.kld")
