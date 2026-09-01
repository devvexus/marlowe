"""Every GGUF this pipeline builds drops the MTP head.

Stage 0 converted the 27B parent to bf16 -- 54 GB, five minutes -- and then failed on the
first quantisation with:

    key qwen35.attention.recurrent_layers has wrong array length; expected 65, got 64

llama.cpp validates that array against ``block_count``, and ``block_count`` counts the MTP
block (the parent writes ``block_count=65`` with ``blk.64.nextn.*``), while the converter
writes one flag per ``num_hidden_layers``. The two disagree by exactly the MTP layer.

It is also simply what the pipeline wants: ``drop_mtp`` is true for every rung because the
draft head is invalid once layers are removed, and a parent reference that keeps it is not
structurally comparable to the children it is the reference for.

This went unnoticed because every prior conversion was of a *pruned child*, which has no MTP
tensors -- so the old auto-detection passed ``--no-mtp`` for exactly the checkpoints that did
not need it, and withheld it from the one that did.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marlowe import quantize as q


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Capture the converter argv without running it."""
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"gguf")

        class P:
            returncode = 0
            stdout = ""
            stderr = ""

        return P()

    monkeypatch.setattr(q.subprocess, "run", fake_run)
    monkeypatch.setattr(q, "find_converter", lambda: tmp_path / "convert_hf_to_gguf.py")
    return seen


def test_mtp_is_dropped_even_when_the_checkpoint_has_one(captured, tmp_path, monkeypatch):
    """The regression: the parent HAS an MTP head, and must still be converted without it."""
    monkeypatch.setattr(q, "checkpoint_has_mtp", lambda p: True)
    q.convert_to_gguf(tmp_path, tmp_path / "out.gguf")
    assert "--no-mtp" in captured["cmd"], (
        "a checkpoint carrying an MTP head must still convert without it, or llama-quantize "
        "rejects the recurrent_layers array after the full conversion is written"
    )


def test_mtp_is_dropped_when_the_checkpoint_has_none(captured, tmp_path, monkeypatch):
    """The pruned-child case, which is why the converter needs the flag at all."""
    monkeypatch.setattr(q, "checkpoint_has_mtp", lambda p: False)
    q.convert_to_gguf(tmp_path, tmp_path / "out.gguf")
    assert "--no-mtp" in captured["cmd"]


def test_keeping_mtp_requires_asking_for_it(captured, tmp_path, monkeypatch):
    monkeypatch.setattr(q, "checkpoint_has_mtp", lambda p: True)
    q.convert_to_gguf(tmp_path, tmp_path / "out.gguf", no_mtp=False)
    assert "--no-mtp" not in captured["cmd"]
