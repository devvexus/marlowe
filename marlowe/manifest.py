"""Stage manifests: the pipeline's state machine.

Every stage writes ``<run_dir>/manifests/<stage>.json`` recording its inputs, the hash of
the config that produced it, the git SHA of the code, its outputs, and its metrics. A stage
is skipped when its manifest is *current*: it exists, it succeeded, its input fingerprints
still match, and its declared outputs are still on disk.

That last clause matters. A manifest alone is not evidence -- a deleted shard or a truncated
GGUF must re-run the stage, not be trusted because a JSON file says it finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def git_sha(repo: str | Path | None = None) -> str:
    """Short SHA plus a dirty marker, or ``"unknown"`` outside a repo."""
    cwd = str(repo) if repo else os.getcwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=15
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def hash_obj(obj: Any) -> str:
    """Stable SHA256 of any JSON-serialisable object (sorted keys, no whitespace)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fingerprint_path(path: str | Path, *, deep: bool = False) -> dict[str, Any]:
    """Cheap identity for a file or directory.

    Default is (size, mtime, name) rolled up -- hashing 55 GB of safetensors on every stage
    entry would cost more than the stages do. ``deep=True`` content-hashes, for small files
    where it is affordable and worth it (configs, rankings).
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    if p.is_file():
        st = p.stat()
        out: dict[str, Any] = {
            "path": str(p),
            "exists": True,
            "kind": "file",
            "size": st.st_size,
            "mtime": int(st.st_mtime),
        }
        if deep:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out["sha256"] = h.hexdigest()[:16]
        return out

    entries = []
    total = 0
    for child in sorted(p.rglob("*")):
        if child.is_file():
            st = child.stat()
            entries.append((str(child.relative_to(p)), st.st_size, int(st.st_mtime)))
            total += st.st_size
    return {
        "path": str(p),
        "exists": True,
        "kind": "dir",
        "n_files": len(entries),
        "total_size": total,
        "digest": hash_obj(entries),
    }


@dataclass
class Manifest:
    """One stage's record. Serialised to JSON; read back to decide skip-vs-run."""

    stage: str
    status: str = "running"  # running | ok | failed
    version: int = MANIFEST_VERSION
    started: float = field(default_factory=time.time)
    finished: float | None = None
    git_sha: str = field(default_factory=git_sha)
    config_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    #: name -> fingerprint of each declared input
    inputs: dict[str, Any] = field(default_factory=dict)
    #: name -> path of each declared output; existence is re-checked on resume
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    # -- io -----------------------------------------------------------------

    @staticmethod
    def path_for(run_dir: str | Path, stage: str) -> Path:
        return Path(run_dir) / "manifests" / f"{stage}.json"

    def save(self, run_dir: str | Path) -> Path:
        p = self.path_for(run_dir, self.stage)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        tmp.replace(p)  # atomic: a crash mid-write must not corrupt the state machine
        return p

    @classmethod
    def load(cls, run_dir: str | Path, stage: str) -> Manifest | None:
        p = cls.path_for(run_dir, stage)
        if not p.exists():
            return None
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return None  # torn write from a hard kill; treat as absent and re-run
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    # -- state machine ------------------------------------------------------

    def is_current(self, inputs: dict[str, Any], config_hash: str) -> tuple[bool, str]:
        """Can this stage be skipped? Returns (skip, reason)."""
        if self.status != "ok":
            return False, f"previous run status={self.status}"
        if self.version != MANIFEST_VERSION:
            return False, f"manifest schema {self.version} != {MANIFEST_VERSION}"
        if self.config_hash != config_hash:
            return False, f"config changed ({self.config_hash} -> {config_hash})"
        for name, fp in inputs.items():
            if name not in self.inputs:
                return False, f"new input {name!r}"
            if self.inputs[name] != fp:
                return False, f"input {name!r} changed"
        missing = [n for n, p in self.outputs.items() if not Path(p).exists()]
        if missing:
            return False, f"outputs missing on disk: {missing}"
        return True, "up to date"

    def succeed(self, **metrics: Any) -> None:
        self.status = "ok"
        self.finished = time.time()
        self.metrics.update(metrics)

    def fail(self, exc: BaseException) -> None:
        self.status = "failed"
        self.finished = time.time()
        self.error = f"{type(exc).__name__}: {exc}"

    @property
    def elapsed_s(self) -> float | None:
        return None if self.finished is None else self.finished - self.started


def load_all(run_dir: str | Path) -> dict[str, Manifest]:
    """Every manifest in a run directory, keyed by stage name."""
    d = Path(run_dir) / "manifests"
    if not d.is_dir():
        return {}
    out: dict[str, Manifest] = {}
    for p in sorted(d.glob("*.json")):
        m = Manifest.load(run_dir, p.stem)
        if m is not None:
            out[p.stem] = m
    return out
