"""Structured logging.

A three-day training run has to be diagnosable from logs alone, so every stage writes
newline-delimited JSON to ``<run_dir>/logs/<stage>.jsonl`` alongside human-readable console
output. The JSONL is what ``report.py`` and any post-hoc analysis read; the console stream
is for watching.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_JSON_SINKS: list[Path] = []
_RUN_CONTEXT: dict[str, Any] = {}


class _ConsoleFormatter(logging.Formatter):
    """Compact console lines: ``12:04:31 INFO  score  message  key=value``."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        extra = getattr(record, "fields", None)
        tail = ""
        if extra:
            tail = "  " + " ".join(f"{k}={_fmt(v)}" for k, v in extra.items())
        base = f"{ts} {record.levelname:<5} {record.name:<12} {record.getMessage()}{tail}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def setup(stage: str, run_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """Install console + JSONL handlers. Idempotent per process."""
    root = logging.getLogger("marlowe")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ConsoleFormatter())
    root.addHandler(console)

    _JSON_SINKS.clear()
    if run_dir is not None:
        log_dir = Path(run_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _JSON_SINKS.append(log_dir / f"{stage}.jsonl")

    _RUN_CONTEXT.clear()
    _RUN_CONTEXT.update({"stage": stage, "pid": os.getpid()})


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"marlowe.{name}")


def event(logger: logging.Logger, msg: str, /, **fields: Any) -> None:
    """Log a structured event at INFO to console and to the stage JSONL.

    Use this rather than ``logger.info(f"...")`` for anything a later analysis might want to
    parse -- metrics, timings, decisions, resource readings. ``level`` is deliberately not a
    parameter here: fields are splatted in from metric dicts, and a stray "level" key would
    silently become the log level instead of a field. Use :func:`event_at` for other levels.
    """
    event_at(logger, logging.INFO, msg, **fields)


def event_at(logger: logging.Logger, level: int, msg: str, /, **fields: Any) -> None:
    """:func:`event` at an explicit level."""
    logger.log(level, msg, extra={"fields": fields})
    if not _JSON_SINKS:
        return
    rec = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "level": logging.getLevelName(level),
        "logger": logger.name,
        "msg": msg,
        **_RUN_CONTEXT,
        **fields,
    }
    line = json.dumps(rec, default=str)
    for sink in _JSON_SINKS:
        with sink.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@contextmanager
def timed(logger: logging.Logger, what: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and emit start/finish events. Yields a dict for extra result fields."""
    t0 = time.time()
    out: dict[str, Any] = {}
    event(logger, f"{what}: start", **fields)
    try:
        yield out
    except BaseException as exc:
        event_at(
            logger,
            logging.ERROR,
            f"{what}: FAILED",
            elapsed_s=round(time.time() - t0, 2),
            error=f"{type(exc).__name__}: {exc}",
            **fields,
        )
        raise
    event(logger, f"{what}: done", elapsed_s=round(time.time() - t0, 2), **fields, **out)
