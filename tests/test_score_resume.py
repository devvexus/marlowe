"""The candidate sweep is resumable.

Stage 3 is ~8 hours and wrote nothing until it returned, so a crash at hour seven cost seven
hours. This machine kernel-panicked (bugcheck 0x139) under sustained GPU load partway through
an imatrix pass, which is what made the risk concrete rather than theoretical.

Each candidate's KL is independent of the others, so scoring one is a natural commit point --
about fifteen minutes of work. What must be true: a resumed sweep skips what it already has,
produces the same answer as an uninterrupted one, and never resumes from a checkpoint that
answers a different question.
"""

from __future__ import annotations

import json

from marlowe.score import score_candidates


class _Harness:
    """Minimal stand-in: layers can be swapped, and scoring is deterministic per layer."""

    returns_tuple = False

    def __init__(self, n: int = 6) -> None:
        self.layers = [f"layer{i}" for i in range(n)]


def _patch(monkeypatch, calls: list[int]) -> None:
    """Make scoring deterministic and record which candidates were actually computed."""
    import marlowe.score as sc

    monkeypatch.setattr(sc, "make_identity", lambda _t: "identity")

    def fake_forward(h, batch, pos):
        # Which layer is currently ablated determines the "logprobs".
        return next(i for i, v in enumerate(h.layers) if v == "identity")

    def fake_kl(ref, got):
        calls.append(got)
        return float(got) + 1.0, 1

    monkeypatch.setattr(sc, "forward_logprobs_at", fake_forward)
    monkeypatch.setattr(sc, "kl_against", fake_kl)


def _run(tmp_path, monkeypatch, cands, ckpt=None):
    calls: list[int] = []
    _patch(monkeypatch, calls)
    h = _Harness()
    scores = score_candidates(
        h, cands, batches=[object()], positions=[[0]], ref_logprobs=[0.0],
        checkpoint=ckpt,
    )
    return scores, calls


def test_a_completed_sweep_writes_every_candidate(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "partial.json"
    scores, _ = _run(tmp_path, monkeypatch, [1, 2, 3], ckpt)
    assert set(scores) == {1, 2, 3}
    assert set(json.loads(ckpt.read_text())) == {"1", "2", "3"}


def test_resume_skips_work_already_done(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "partial.json"
    ckpt.write_text(json.dumps({"1": 2.0, "2": 3.0}), encoding="utf-8")
    scores, calls = _run(tmp_path, monkeypatch, [1, 2, 3], ckpt)
    assert set(scores) == {1, 2, 3}
    assert calls == [3], "only the unscored candidate should be recomputed"


def test_resume_reproduces_the_uninterrupted_answer(tmp_path, monkeypatch) -> None:
    """A resumed sweep must not produce a different profile from one that never crashed."""
    full, _ = _run(tmp_path, monkeypatch, [1, 2, 3], tmp_path / "a.json")

    ckpt = tmp_path / "b.json"
    partial, _ = _run(tmp_path, monkeypatch, [1, 2], ckpt)  # "crash" after two
    resumed, _ = _run(tmp_path, monkeypatch, [1, 2, 3], ckpt)
    assert resumed == full


def test_a_checkpoint_for_other_candidates_is_ignored(tmp_path, monkeypatch) -> None:
    """Greedy mode scores on top of prior cuts; that is a different question.

    Adopting scores computed under a different ablation prefix would silently mix two
    experiments and produce a damage profile describing neither.
    """
    ckpt = tmp_path / "partial.json"
    ckpt.write_text(json.dumps({"41": 99.0, "1": 2.0}), encoding="utf-8")
    scores, calls = _run(tmp_path, monkeypatch, [1, 2], ckpt)
    assert set(scores) == {1, 2}, "layer 41 is not in this sweep and must not appear"
    assert calls == [2]


def test_no_checkpoint_still_works(tmp_path, monkeypatch) -> None:
    scores, calls = _run(tmp_path, monkeypatch, [1, 2], None)
    assert set(scores) == {1, 2}
    assert calls == [1, 2]
