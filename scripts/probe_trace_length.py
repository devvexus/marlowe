"""How many tokens does a complete trace actually need?

This should have run before any generation. It did not, and 13 hours produced a corpus in
which the median trace is truncated at exactly the cap -- p50 = p90 = max = 4096 -- with 76%
containing a thinking block and no answer at all. A cap chosen below the distribution does
not shorten traces, it destroys them.

Generates with a deliberately large cap and reports the distribution of *completed* traces:
the ones that stopped because the model finished, not because the budget ran out. The answer
is the cap for the real run.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_traces import THINKING, generate_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="super-mtp")
    ap.add_argument("--prompts", default="data/trace_prompts.jsonl")
    ap.add_argument("--out", default="runs/marlowe-22b/metrics/trace_length_probe.jsonl")
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--per-kind", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.prompts).read_text(encoding="utf-8").splitlines()
            if x.strip()]
    by_kind: dict[str, list] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    sample = [p for k in sorted(by_kind) for p in by_kind[k][: args.per_kind]]
    print(f"probing {len(sample)} prompts at cap {args.max_tokens}, "
          f"{args.workers} workers, kinds {sorted(by_kind)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")
    lock = threading.Lock()
    q: queue.Queue = queue.Queue()
    for p in sample:
        q.put(p)

    def worker() -> None:
        while True:
            try:
                p = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            r = generate_one(args.base_url, args.model, p["prompt"],
                             args.max_tokens, 0, args.timeout)
            if "error" in r:
                rec = {"id": p["id"], "kind": p["kind"], "error": r["error"]}
            else:
                think = r.get("reasoning") or ""
                content = r.get("content") or ""
                rec = {
                    "id": p["id"], "kind": p["kind"],
                    "completion_tokens": r["completion_tokens"],
                    "finish_reason": r["finish_reason"],
                    "completed": r["finish_reason"] != "length",
                    "thinking_chars": len(think),
                    "content_chars": len(content),
                    "has_answer": bool(content.strip()),
                    "elapsed_s": round(time.time() - t0, 1),
                }
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print("  " + json.dumps({k: v for k, v in rec.items()
                                         if k in ("id", "kind", "completion_tokens",
                                                  "finish_reason", "has_answer", "elapsed_s")}),
                      flush=True)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    fh.close()
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
