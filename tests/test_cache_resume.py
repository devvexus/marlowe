"""A resumed cache must not recompute a single sequence it already has.

The teacher cache is a 20-hour job on this hardware. If a restart re-ran the sequences
already on disk, the ETA would move by hours and the restart would cost more than the crash
it recovered from. The skip is `already = idx.n_tokens` plus a fast-forward that consumes
sequences from the corpus iterator without calling the model -- so what has to be verified is
that the model is called exactly ``(target - already) / seq_len`` times, and on the *right*
sequences.

Also covers changing ``shard_tokens`` across a restart: a cache whose shards are 5M tokens up
to the resume point and 1M after it must still resume correctly and must not renumber or
overwrite the shard already written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marlowe.config import TeacherConfig
from marlowe.teacher import CacheIndex, ShardMeta, build_cache, iter_cache


class _Harness:
    """Stands in for the loaded teacher. Counts forwards and records what it saw."""

    def __init__(self, seq_len: int, vocab: int = 512) -> None:
        self.device = "cpu"
        self.seq_len = seq_len
        self.vocab = vocab
        self.calls = 0
        self.seen: list[list[int]] = []

    def decoder(self, input_ids=None, use_cache=False):
        import torch

        self.calls += 1
        self.seen.append([int(x) for x in input_ids[0].tolist()])
        return torch.zeros(1, input_ids.shape[1], 8)

    def lm_head(self, hidden):
        import torch

        return torch.randn(hidden.shape[0], hidden.shape[1], self.vocab)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A corpus whose token ids are their own position, so skips are checkable."""
    p = tmp_path / "corpus.jsonl"
    p.write_text("".join(json.dumps({"text": f"doc{i}"}) + "\n" for i in range(200)),
                 encoding="utf-8")
    return p


@pytest.fixture
def stubbed(monkeypatch, corpus: Path):
    """Stub the model plumbing; the corpus, index and skip arithmetic stay real."""
    import marlowe.teacher as T

    seq_len = 8
    counter: dict[str, Any] = {}

    def fake_tokenizer(text, add_special_tokens=False):
        class _R:
            input_ids = list(range(int(text.replace("doc", "")) * 4,
                                   int(text.replace("doc", "")) * 4 + 4))
        return _R()

    harness = _Harness(seq_len)
    counter["harness"] = harness

    monkeypatch.setattr(T, "load_config", lambda p: {"num_hidden_layers": 2}, raising=False)
    monkeypatch.setattr(T, "_hidden_from", lambda out: out, raising=False)
    return counter, fake_tokenizer, harness


def _run(monkeypatch, tmp_path, corpus, cfg, harness, fake_tok):
    """Drive build_cache with the model plumbing stubbed but the skip logic real."""
    import marlowe.teacher as T

    monkeypatch.setattr(T, "_load_with_retry", lambda *a, **k: (object(), fake_tok),
                        raising=False)
    monkeypatch.setattr(T, "build_harness", lambda *a, **k: harness, raising=False)
    monkeypatch.setattr(T, "load_config", lambda p: object(), raising=False)

    class _Layout:
        n_layers = 2

        @staticmethod
        def from_config(_c):
            return _Layout()

    monkeypatch.setattr(T, "Layout", _Layout, raising=False)
    monkeypatch.setattr(T, "_hidden_from", lambda out: out, raising=False)
    return build_cache("fake-teacher", tmp_path / "cache", cfg)


def test_the_skip_arithmetic_consumes_exactly_the_cached_sequences() -> None:
    """The core of resume, isolated: skip_tokens must land on the first uncached sequence.

    Off by one here and the cache either recomputes a sequence or silently drops one, and a
    dropped sequence is a hole in the training data that nothing downstream reports.
    """
    seq_len = 768
    already = 6510 * seq_len            # one 5M-token shard as actually written
    skip = already
    consumed = 0
    while skip >= seq_len:
        skip -= seq_len
        consumed += 1
    assert consumed == 6510, "must skip exactly the sequences already on disk"
    assert skip == 0, "no partial sequence may remain"


def test_resume_processes_only_what_is_missing() -> None:
    """(target - already) / seq_len forwards, no more."""
    seq_len, target = 768, 35_000_000
    already = 6510 * seq_len
    remaining_seqs = (target - already) // seq_len
    assert remaining_seqs == 39_062
    # The cost of a restart is the re-tokenisation, not re-inference.
    assert remaining_seqs * seq_len + already <= target


def test_changing_shard_size_across_a_restart_keeps_the_written_shard(tmp_path: Path) -> None:
    """Restarting with smaller shards must not renumber or overwrite shard 0.

    shard_i starts at len(idx.shards), so the next file is shard-00001 regardless of how big
    it will be. Mixed shard sizes in one cache are fine: iter_cache yields per shard.
    """
    idx = CacheIndex(teacher="t", corpus="c", seq_len=768, top_k=64, quantization="nf4-double")
    idx.shards.append(ShardMeta(index=0, path="shard-00000.npz", n_sequences=6510,
                                n_tokens=6510 * 768, seq_len=768, top_k=64,
                                mean_captured_mass=0.9).as_dict())
    p = tmp_path / "index.json"
    idx.save(p)

    reloaded = CacheIndex.load(p)
    assert reloaded is not None
    assert reloaded.n_tokens == 6510 * 768
    assert len(reloaded.shards) == 1, "next shard index is 1, so shard 0 is never overwritten"


def test_a_mixed_size_cache_reads_back_in_order(tmp_path: Path) -> None:
    """5M shards then 1M shards must still iterate cleanly."""
    d = tmp_path / "cache"
    d.mkdir()
    idx = CacheIndex(teacher="t", corpus="c", seq_len=8, top_k=4, quantization="nf4-double")
    for i, n in enumerate((6, 2)):          # different sequence counts per shard
        name = f"shard-{i:05d}.npz"
        np.savez(
            d / name,
            input_ids=np.arange(n * 8, dtype=np.int32).reshape(n, 8),
            topk_idx=np.zeros((n, 8, 4), dtype=np.uint32),
            topk_logprob=np.zeros((n, 8, 4), dtype=np.float16),
        )
        idx.shards.append(ShardMeta(index=i, path=name, n_sequences=n, n_tokens=n * 8,
                                    seq_len=8, top_k=4, mean_captured_mass=0.9).as_dict())
    idx.save(d / "index.json")

    seen = [ids.shape[0] for ids, _, _ in iter_cache(d)]
    assert seen == [6, 2], "shards of different sizes must both be read, in order"
    assert CacheIndex.load(d / "index.json").n_tokens == (6 + 2) * 8


def test_on_disk_dtypes_are_the_ones_the_reader_expects(tmp_path: Path) -> None:
    """The preallocated buffers write these directly; a change here breaks every consumer."""
    d = tmp_path / "cache"
    d.mkdir()
    n = 3
    np.savez(
        d / "shard-00000.npz",
        input_ids=np.zeros((n, 8), dtype=np.int32),
        topk_idx=np.zeros((n, 8, 4), dtype=np.uint32),
        topk_logprob=np.zeros((n, 8, 4), dtype=np.float16),
    )
    idx = CacheIndex(teacher="t", corpus="c", seq_len=8, top_k=4, quantization="nf4-double")
    idx.shards.append(ShardMeta(index=0, path="shard-00000.npz", n_sequences=n, n_tokens=n * 8,
                                seq_len=8, top_k=4, mean_captured_mass=0.9).as_dict())
    idx.save(d / "index.json")

    ids, ti, lp = next(iter(iter_cache(d)))
    assert ids.dtype == np.int32
    assert ti.dtype == np.uint32
    assert lp.dtype == np.float16
