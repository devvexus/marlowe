"""A failed attempt is not a completed trace.

The generator appended every result to one file, errors included, and resume treated any row
carrying an id as done. When llama-server began rejecting requests -- ``max_tokens`` plus the
prompt exceeded its 4096-token slot -- 2988 prompts were marked attempted in about ninety
seconds, and a resume would then have skipped all of them. The output would have been a
corpus one percent of the intended size, with a manifest reporting success.

That is the failure mode this project keeps meeting: not a crash, a plausible artefact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_traces as gt  # noqa: E402


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _resume_state(out: Path) -> tuple[set[str], int]:
    """Reproduce the generator's resume scan: which ids count as done, and how many dropped."""
    done: set[str] = set()
    failed = 0
    kept: list[str] = []
    with out.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" in row:
                failed += 1
                continue
            done.add(row["id"])
            kept.append(line.rstrip("\n"))
    if failed:
        body = "\n".join(kept)
        out.write_text(body + ("\n" if kept else ""), encoding="utf-8")
    return done, failed


def test_error_rows_are_retried_not_counted_as_done(tmp_path: Path) -> None:
    """The exact shape of the bug: 2 real traces, 5 rejections, 7 rows."""
    out = tmp_path / "traces_raw.jsonl"
    _write(out, [
        {"id": "arxiv-0001", "completion_tokens": 3000, "reasoning": "..."},
        {"id": "arxiv-0002", "error": "HTTPError: HTTP Error 503: Service Unavailable"},
        {"id": "arxiv-0003", "error": "HTTPError: HTTP Error 503: Service Unavailable"},
        {"id": "pi-0001", "completion_tokens": 2500, "reasoning": "..."},
        {"id": "pi-0002", "error": "URLError: connection refused"},
        {"id": "tool-0001", "error": "HTTPError: HTTP Error 500: Internal Server Error"},
        {"id": "math-0001", "error": "HTTPError: HTTP Error 503: Service Unavailable"},
    ])
    prompts = [{"id": i, "kind": "x", "prompt": "p"} for i in
               ("arxiv-0001", "arxiv-0002", "arxiv-0003", "pi-0001", "pi-0002",
                "tool-0001", "math-0001")]

    done, failed = _resume_state(out)

    assert done == {"arxiv-0001", "pi-0001"}, "only successful traces are done"
    assert failed == 5
    assert len(done) != 7, "the bug: 7 rows read as 7 completions"

    todo = [p for p in prompts if p["id"] not in done]
    assert len(todo) == 5, "every failed attempt must be retried"
    assert {p["id"] for p in todo} == {
        "arxiv-0002", "arxiv-0003", "pi-0002", "tool-0001", "math-0001"
    }


def test_the_file_is_rewritten_without_the_failures(tmp_path: Path) -> None:
    """Left in place, they would be re-counted on the next resume and re-dropped forever."""
    out = tmp_path / "traces_raw.jsonl"
    _write(out, [
        {"id": "a", "completion_tokens": 10},
        {"id": "b", "error": "HTTPError: HTTP Error 503: Service Unavailable"},
        {"id": "c", "completion_tokens": 20},
    ])
    _resume_state(out)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["id"] for r in rows] == ["a", "c"]
    assert all("error" not in r for r in rows)

    # Idempotent: a second resume finds nothing to drop and the same two completions.
    done2, failed2 = _resume_state(out)
    assert done2 == {"a", "c"} and failed2 == 0


def test_a_clean_file_is_left_untouched(tmp_path: Path) -> None:
    out = tmp_path / "traces_raw.jsonl"
    original = [{"id": "a", "completion_tokens": 1}, {"id": "b", "completion_tokens": 2}]
    _write(out, original)
    before = out.read_text(encoding="utf-8")
    done, failed = _resume_state(out)
    assert done == {"a", "b"} and failed == 0
    assert out.read_text(encoding="utf-8") == before, "no rewrite when there is nothing to drop"


def test_an_all_failures_file_resumes_from_zero(tmp_path: Path) -> None:
    """The real incident: every row an error. Resume must retry everything, not finish."""
    out = tmp_path / "traces_raw.jsonl"
    _write(out, [{"id": f"p-{i:04d}", "error": "HTTPError: HTTP Error 503: Service Unavailable"}
                 for i in range(2988)])
    done, failed = _resume_state(out)
    assert done == set()
    assert failed == 2988
    assert out.read_text(encoding="utf-8") == "", "nothing completed, so nothing is kept"


@pytest.mark.parametrize("err", [
    "HTTPError: HTTP Error 503: Service Unavailable",
    "HTTPError: HTTP Error 500: Internal Server Error",
    "URLError: <urlopen error [WinError 10061] No connection could be made>",
])
def test_transient_server_errors_are_retried_rather_than_consuming_the_prompt(err: str) -> None:
    """503 means "no slot free", not "this prompt is impossible"."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            return {"error": err}
        return {"completion_tokens": 100, "reasoning": "ok"}

    out = gt._retrying(flaky, attempts=5, base_delay=0.0)
    assert "error" not in out
    assert calls["n"] == 3


def test_a_permanent_error_is_not_retried_forever() -> None:
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        return {"error": "ValueError: prompt is malformed"}

    out = gt._retrying(broken, attempts=5, base_delay=0.0)
    assert "error" in out
    assert calls["n"] == 1, "a non-transient error must not burn the retry budget"
