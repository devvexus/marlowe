"""Phase A trace generation from the local parent, via Ollama.

Every trace is stored with its own repetition metrics, and **nothing is filtered here**.
The filter is `rep8 above the parent's own baseline`, and that baseline is a separate 200x2048
run that has not happened yet. Discarding a trace against a guessed threshold destroys GPU
hours that cannot be recovered; recording the metric and filtering later costs a JSON field.
`filter_traces.py` applies the rule once the number exists.

Teacher provenance is recorded, not assumed: these come from `marlowe-dusk:27b-super`, which
is the base 27B with MTP at **IQ3_XXS**, not a bf16 endpoint and not a fine-tune. The student
is therefore bounded by a quantised teacher's behaviour, which is a real cost of having no
hosted endpoint and must be visible in the manifest rather than inferred later from a
puzzling KL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marlowe.eval import repetition as rep  # noqa: E402

THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repeat_penalty": 1.0}



def _wait_healthy(base_url: str, timeout_s: int) -> bool:
    """Block until the server responds, so a cold start is not read as failure.

    Probes /v1/models before /health: llama-server serves both, Ollama serves only the first.
    Polling /health alone against Ollama waits out the entire timeout on a 404 while the
    server is up and idle -- which is exactly what it did.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            for path in ("/v1/models", "/health"):
                try:
                    with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as r:
                        if r.status == 200:
                            return True
                except (urllib.error.URLError, TimeoutError, OSError):
                    continue
        except Exception:
            pass
        time.sleep(5)
    return False


def _retrying(fn, attempts: int = 5, base_delay: float = 4.0):
    """Retry a 503 with backoff instead of consuming the prompt.

    llama-server answers 503 when a request cannot be placed in a slot. Treating that as a
    permanent failure burned 2988 prompts in seconds.
    """
    last: dict[str, Any] = {}
    for i in range(attempts):
        last = fn()
        err = last.get("error", "")
        if "503" not in err and "500" not in err and "10061" not in err:
            return last
        time.sleep(base_delay * (i + 1))
    return last


def generate_one(base_url: str, model: str, prompt: str, max_tokens: int,
                 seed: int, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "seed": seed,
        **{k: v for k, v in THINKING.items() if k not in ("min_p", "repeat_penalty")},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "latency_s": time.time() - t0}

    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    # The thinking block IS the behaviour being transferred, so it is kept verbatim and
    # separately -- reconstructing it from the visible answer later is impossible.
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    usage = body.get("usage") or {}
    return {
        "reasoning": reasoning,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "latency_s": round(time.time() - t0, 2),
    }


def measure(trace: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    """Metrics the filter will need, computed now so the filter never needs the GPU again."""
    text = (trace.get("reasoning") or "") + (trace.get("content") or "")
    toks = text.split()
    return {
        "rep8": rep.ngram_repetition(toks, 8),
        "rep32": rep.ngram_repetition(toks, 32),
        "longest_repeated_run": rep.longest_repeated_run(toks),
        "n_words": len(toks),
        "cap_hit": trace.get("finish_reason") == "length",
        "has_thinking": bool(trace.get("reasoning")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="marlowe-dusk:27b-super")
    ap.add_argument("--quant", default="IQ3_XXS",
                    help="recorded in the manifest; the teacher's precision bounds the student")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--prompts", default="data/trace_prompts.jsonl")
    ap.add_argument("--out", default="data/traces_raw.jsonl")
    ap.add_argument("--manifest", default="runs/marlowe-22b/manifests/traces.json")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="the parent circles at depth; 4096 bounds a runaway trace")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-ready", type=int, default=900)
    args = ap.parse_args()

    if not _wait_healthy(args.base_url, args.timeout_ready):
        print(f"server at {args.base_url} never became healthy; refusing to start", flush=True)
        return 1

    prompts = [json.loads(x) for x in Path(args.prompts).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    out_path = Path(args.out)
    done: set[str] = set()
    failed = 0
    if out_path.exists():
        kept: list[str] = []
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                # A failed attempt is NOT done. Counting error rows as completed marked 2988
                # prompts as attempted in seconds -- the server was rejecting every request
                # because max_tokens plus the prompt exceeded its 4096-token slot -- and a
                # resume would then have skipped all of them, silently generating a corpus
                # 1% the intended size.
                if "error" in row:
                    failed += 1
                    continue
                done.add(row["id"])
                kept.append(line.rstrip("\n"))
        if failed:
            body = "\n".join(kept)
            out_path.write_text(body + ("\n" if kept else ""), encoding="utf-8")
            print(f"dropped {failed} failed attempts from {out_path}; they will be retried",
                  flush=True)
        print(f"resuming: {len(done)} completed traces on disk", flush=True)
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(todo)} prompts to generate from {args.model} ({args.quant}), "
          f"cap {args.max_tokens}, {args.workers} workers", flush=True)

    q: queue.Queue = queue.Queue()
    for p in todo:
        q.put(p)
    lock = threading.Lock()
    stats = {"ok": 0, "error": 0, "tokens": 0, "t0": time.time()}
    fh = out_path.open("a", encoding="utf-8")

    def worker() -> None:
        while True:
            try:
                p = q.get_nowait()
            except queue.Empty:
                return
            r = _retrying(lambda: generate_one(
                args.base_url, args.model, p["prompt"],
                args.max_tokens, args.seed, args.timeout))
            row = {"id": p["id"], "kind": p["kind"], "prompt": p["prompt"], **r}
            if "error" not in r:
                row["metrics"] = measure(r, args.max_tokens)
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if "error" in r:
                    stats["error"] += 1
                else:
                    stats["ok"] += 1
                    stats["tokens"] += r["completion_tokens"]
                n = stats["ok"] + stats["error"]
                if n % 10 == 0:
                    el = time.time() - stats["t0"]
                    print(f"  {n}/{len(todo)}  ok {stats['ok']} err {stats['error']}  "
                          f"{stats['tokens'] / 1e6:.2f}M tok  "
                          f"{stats['tokens'] / max(el, 1):.0f} tok/s agg  "
                          f"eta {(len(todo) - n) * el / max(n, 1) / 3600:.1f} h", flush=True)
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fh.close()

    elapsed = time.time() - stats["t0"]
    manifest = {
        "teacher_model": args.model,
        "teacher_quant": args.quant,
        "teacher_note": (
            "Local quantised parent, not a bf16 endpoint. The student's ceiling is this "
            "model's behaviour, not the bf16 parent's -- a real cost of generating locally."
        ),
        "base_url": args.base_url,
        "sampling": THINKING,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "prompts": args.prompts,
        "prompts_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "generated": stats["ok"], "errors": stats["error"],
        "raw_tokens": stats["tokens"],
        "elapsed_hours": round(elapsed / 3600, 2),
        "aggregate_tok_s": round(stats["tokens"] / max(elapsed, 1), 1),
        "filtered": False,
        "filter_note": "No filter applied. Metrics stored per trace; see filter_traces.py.",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    mp = Path(args.manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print("TRACES_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
