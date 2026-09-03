"""Verify a checkpoint's bytes on disk, by reading them.

An earlier attempt at this used ``safe_open`` plus ``get_tensor`` and reported 18 shards and
27.78 billion elements clean in **1.8 seconds** -- roughly 30 GB/s, which is memory bandwidth,
not disk. ``get_tensor`` hands back mmap-backed views, so nothing was faulted in and nothing
was verified. It would have been a pass that tested nothing, which is worse than no check.

This reads each shard sequentially in fixed chunks and hashes the bytes, so every page is
actually touched, and compares against the repository's own LFS sha256 where available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

CHUNK = 32 * 1024 * 1024


def sha256_file(path: Path) -> tuple[str, int]:
    """Sequential chunked read. No mmap, no views -- the bytes go through the CPU."""
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def repo_checksums(repo: str) -> dict[str, str]:
    """LFS sha256 per file from the Hub, or {} when unavailable.

    Absence is reported, never treated as a pass: "no reference to compare against" and
    "compared and matched" are different results.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 - offline is a normal outcome here
        print(f"  (no repo checksums: {type(exc).__name__}: {exc})", flush=True)
        return {}
    out: dict[str, str] = {}
    for s in info.siblings or []:
        lfs = getattr(s, "lfs", None)
        digest = None
        if isinstance(lfs, dict):
            digest = lfs.get("sha256") or lfs.get("oid")
        elif lfs is not None:
            digest = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
        if digest:
            out[s.rfilename] = str(digest)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/marlowe-22b/models/parent")
    ap.add_argument("--repo", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", default="runs/marlowe-22b/metrics/checkpoint_integrity.json")
    args = ap.parse_args()

    root = Path(args.dir)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        print(f"no safetensors in {root}", flush=True)
        return 1

    print(f"reading {len(shards)} shards sequentially from {root}", flush=True)
    reference = repo_checksums(args.repo)
    print(f"  reference checksums available for {len(reference)} files", flush=True)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    total_bytes = 0
    for i, p in enumerate(shards, 1):
        s0 = time.time()
        digest, n = sha256_file(p)
        total_bytes += n
        ref = reference.get(p.name)
        status = "no-reference" if ref is None else ("match" if ref == digest else "MISMATCH")
        el = time.time() - s0
        results.append({
            "file": p.name, "bytes": n, "sha256": digest,
            "reference": ref, "status": status,
            "seconds": round(el, 1), "MB_per_s": round(n / 1e6 / max(el, 1e-9), 1),
        })
        print(f"  [{i}/{len(shards)}] {status:12} {p.name}  "
              f"{n/1e9:.2f} GB in {el:.1f}s ({n/1e6/max(el,1e-9):.0f} MB/s)", flush=True)

    elapsed = time.time() - t0
    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    unverified = [r for r in results if r["status"] == "no-reference"]
    summary = {
        "dir": str(root), "repo": args.repo,
        "shards": len(shards), "total_bytes": total_bytes,
        "elapsed_s": round(elapsed, 1),
        "aggregate_MB_per_s": round(total_bytes / 1e6 / max(elapsed, 1e-9), 1),
        "mismatches": [r["file"] for r in mismatches],
        "unverified_no_reference": [r["file"] for r in unverified],
        "verified_against_repo": len(results) - len(unverified),
        "files": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2), flush=True)
    # A read speed near memory bandwidth would mean the pages were already cached and the
    # read proved less than it appears to. Stated, not hidden.
    if summary["aggregate_MB_per_s"] > 5000:
        print("  NOTE: aggregate read rate exceeds plausible disk throughput; the file was "
              "likely served from page cache, so this is a weaker check than a cold read.",
              flush=True)
    print("INTEGRITY_MISMATCH" if mismatches else "INTEGRITY_OK", flush=True)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
