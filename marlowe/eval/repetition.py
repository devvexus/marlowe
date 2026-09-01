"""Generation-based loop detection -- the metric this project exists to move.

The motivating artefact is *circling*: reasoning that loops and fails to terminate, observed
at IQ3_XXS (2.97 bpw) and absent at bf16. That gap is what says the cause is quantisation
rather than the model.

KL cannot see it. KL is measured teacher-forced on fixed text; circling is an autoregressive
failure that only appears when the model samples its own continuations. A checkpoint can
have excellent KL and still loop, and under-healed pruned models loop too -- different cause,
identical symptom. So every checkpoint gets both metrics, and neither substitutes for the
other.

Harness spec (section 6): 200 completions x 2048 tokens, thinking preset pinned, on a fixed
held-out prompt set that includes known circling triggers. Reports n-gram repetition rate at
n=8 and n=32, plus the fraction of generations that hit the token cap without emitting a
stop token.
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from marlowe import logutil
from marlowe.config import SamplingPreset, preset

log = logutil.get("eval.repetition")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def ngram_repetition(tokens: Sequence[int | str], n: int) -> float:
    """Fraction of n-grams that are repeats: ``1 - unique/total``.

    0.0 means every n-gram is distinct. Values near 1.0 mean the generation is almost
    entirely recycled text. n=8 catches phrase-level churn; n=32 catches whole-paragraph
    loops, which is what circling looks like.
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def tail_cycle_period(tokens: Sequence[int | str], window: int = 512, max_period: int = 256) -> int:
    """Smallest period p for which the last ``window`` tokens are exactly periodic.

    A direct circling detector, and stricter than n-gram rate: it requires the *tail* of the
    generation to be a clean repeating cycle, which is the specific failure where the model
    never reaches a stop token because it re-enters the same reasoning loop. Returns 0 when
    no cycle is found.
    """
    tail = list(tokens[-window:])
    if len(tail) < 8:
        return 0
    for p in range(1, min(max_period, len(tail) // 2) + 1):
        if all(tail[i] == tail[i - p] for i in range(p, len(tail))):
            return p
    return 0


def longest_repeated_run(tokens: Sequence[int | str], min_len: int = 8) -> int:
    """Length of the longest block that occurs at least twice. O(n log n) by binary search."""
    if len(tokens) < 2 * min_len:
        return 0

    def has_dup(k: int) -> bool:
        seen: set[tuple[int | str, ...]] = set()
        for i in range(len(tokens) - k + 1):
            g = tuple(tokens[i : i + k])
            if g in seen:
                return True
            seen.add(g)
        return False

    lo, hi, best = min_len, len(tokens) // 2, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if has_dup(mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    prompt_id: str
    text: str
    n_tokens: int
    stop_reason: str  # "stop" | "length" | "error"
    token_ids: list[int] | None = None
    latency_s: float = 0.0
    error: str | None = None

    @property
    def hit_cap(self) -> bool:
        return self.stop_reason == "length"


class Backend(Protocol):
    name: str

    def generate(
        self, prompt: str, max_tokens: int, sampling: SamplingPreset, seed: int
    ) -> Completion: ...


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - local llama-server endpoint
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # noqa: S310 - local/http endpoint
        body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    return body


@dataclass
class LlamaServerBackend:
    """Local llama.cpp server, ``/v1/chat/completions``.

    The chat endpoint is used rather than raw ``/completion`` so the model's own chat
    template applies -- thinking mode is a template behaviour, and hand-rolling the prompt
    is a reliable way to accidentally evaluate the wrong mode.
    """

    base_url: str = "http://127.0.0.1:8080"
    name: str = "llama-server"
    timeout: int = 1800
    #: Merged into the request body. Pin ``reasoning_effort`` and any chat_template_kwargs
    #: here; the value used is recorded in the report.
    extra_body: dict[str, Any] = field(default_factory=dict)

    def generate(
        self, prompt: str, max_tokens: int, sampling: SamplingPreset, seed: int
    ) -> Completion:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "presence_penalty": sampling.presence_penalty,
            "repeat_penalty": sampling.repeat_penalty,
            "seed": seed,
            "stream": False,
            "cache_prompt": False,
            **self.extra_body,
        }
        t0 = time.time()
        try:
            body = _post_json(f"{self.base_url}/v1/chat/completions", payload, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return Completion("", "", 0, "error", latency_s=time.time() - t0, error=str(exc))

        choice = body.get("choices", [{}])[0]
        msg = choice.get("message", {}) or {}
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        usage = body.get("usage", {}) or {}
        return Completion(
            prompt_id="",
            text=text,
            n_tokens=int(usage.get("completion_tokens", 0)),
            stop_reason=str(choice.get("finish_reason", "stop")),
            latency_s=time.time() - t0,
        )


@dataclass
class OpenAICompatBackend:
    """Hosted API, for the bf16 baseline that cannot run locally (Stage 2, step 3)."""

    base_url: str
    model: str
    api_key: str = ""
    name: str = "hosted"
    timeout: int = 1800
    extra_body: dict[str, Any] = field(default_factory=dict)

    def generate(
        self, prompt: str, max_tokens: int, sampling: SamplingPreset, seed: int
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "seed": seed,
            **sampling.as_openai_kwargs(),
            **self.extra_body,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(  # noqa: S310 - operator-supplied endpoint
            f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST"
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return Completion("", "", 0, "error", latency_s=time.time() - t0, error=str(exc))

        choice = body.get("choices", [{}])[0]
        msg = choice.get("message", {}) or {}
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        usage = body.get("usage", {}) or {}
        return Completion(
            prompt_id="",
            text=text,
            n_tokens=int(usage.get("completion_tokens", 0)),
            stop_reason=str(choice.get("finish_reason", "stop")),
            latency_s=time.time() - t0,
        )


def spawn_llama_server(
    gguf: str | Path,
    *,
    port: int = 8080,
    ctx: int = 8192,
    n_gpu_layers: int = 999,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    extra_args: Sequence[str] = (),
    wait_s: int = 600,
) -> subprocess.Popen[bytes]:
    """Start a local llama-server and block until it reports healthy.

    KV cache defaults to Q8 because that is the deployment configuration being evaluated --
    measuring at fp16 KV would flatter the model relative to how it will actually run.
    """
    from marlowe.eval.kl import find_binary

    exe = find_binary("llama-server")
    cmd = [
        exe,
        "-m", str(gguf),
        "--port", str(port),
        "-c", str(ctx),
        "-ngl", str(n_gpu_layers),
        "-ctk", cache_type_k,
        "-ctv", cache_type_v,
        *extra_args,
    ]
    logutil.event(log, "spawning llama-server", port=port, ctx=ctx, model=Path(gguf).name)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as r:
                if r.status == 200:
                    logutil.event(log, "llama-server ready", port=port)
                    return proc
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"llama-server did not become healthy within {wait_s}s")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    id: str
    text: str
    #: True for prompts known to trigger circling on the user's IQ3_XXS build. Reported
    #: separately: an aggregate over easy prompts can hide a regression on the hard ones.
    known_trigger: bool = False
    tags: list[str] = field(default_factory=list)


def load_prompts(path: str | Path) -> list[Prompt]:
    rows: list[Prompt] = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                rows.append(Prompt(id=f"p{i:03d}", text=obj))
            else:
                rows.append(
                    Prompt(
                        id=str(obj.get("id", f"p{i:03d}")),
                        text=obj["text"] if "text" in obj else obj["prompt"],
                        known_trigger=bool(obj.get("known_trigger", False)),
                        tags=list(obj.get("tags", [])),
                    )
                )
    if not rows:
        raise ValueError(f"no prompts in {path}")
    return rows


# ---------------------------------------------------------------------------
# tokenisation for the metric
# ---------------------------------------------------------------------------


def make_tokenizer(model_path: str | None) -> tuple[Any, str]:
    """Return (callable text->token list, description).

    Model-token n-grams are the spec. When the tokenizer is unavailable the harness falls
    back to whitespace words and records that it did -- the numbers stay internally
    comparable but must not be compared against a run that used model tokens.
    """
    if model_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            return (lambda s: tok(s, add_special_tokens=False).input_ids), "model-tokens"
        except Exception as exc:  # noqa: BLE001
            log.warning("tokenizer unavailable (%s); falling back to whitespace words", exc)
    return (lambda s: s.split()), "whitespace-words"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


@dataclass
class RepetitionReport:
    backend: str
    label: str
    n_completions: int
    max_tokens: int
    preset: str
    tokenization: str
    #: mean over completions, keyed by "rep8", "rep32", ...
    repetition: dict[str, float] = field(default_factory=dict)
    repetition_median: dict[str, float] = field(default_factory=dict)
    cap_hit_rate: float = 0.0
    loop_rate: float = 0.0
    mean_tokens: float = 0.0
    errors: int = 0
    by_prompt: list[dict[str, Any]] = field(default_factory=list)
    trigger_subset: dict[str, float] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def headline(self) -> str:
        r8 = self.repetition.get("rep8", float("nan"))
        r32 = self.repetition.get("rep32", float("nan"))
        return (
            f"rep8={r8:.4f} rep32={r32:.4f} cap_hit={self.cap_hit_rate:.3f} "
            f"loop={self.loop_rate:.3f}"
        )


def run_repetition(
    backend: Backend,
    prompts: Sequence[Prompt],
    *,
    label: str,
    n_completions: int = 200,
    max_tokens: int = 2048,
    ngram_sizes: Sequence[int] = (8, 32),
    sampling_preset: str = "thinking",
    seed: int = 0,
    tokenizer_path: str | None = None,
    save_completions: str | Path | None = None,
) -> RepetitionReport:
    """Run the harness. Refuses any preset but thinking.

    Running thinking mode with the non-thinking preset (presence_penalty=1.5) produces
    repetition that looks exactly like quantisation damage -- it would manufacture the
    artefact being measured. If a non-thinking measurement is ever wanted, call the metric
    functions directly and label the result unmistakably.
    """
    sampling = preset(sampling_preset)
    if sampling.name != "thinking":
        raise ValueError(
            f"refusing to run the repetition harness with the {sampling.name!r} preset. "
            f"presence_penalty={sampling.presence_penalty} suppresses exactly the repetition "
            f"this harness measures; the result would be uninterpretable."
        )

    tokenize, tok_desc = make_tokenizer(tokenizer_path)
    rng = random.Random(seed)
    plan = [(prompts[i % len(prompts)], seed + i) for i in range(n_completions)]
    rng.shuffle(plan)

    per_prompt: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    errors = 0

    for n, (p, s) in enumerate(plan):
        c = backend.generate(p.text, max_tokens, sampling, s)
        c.prompt_id = p.id
        if c.stop_reason == "error":
            errors += 1
            log.warning("generation error on %s: %s", p.id, c.error)
            continue

        toks = c.token_ids if c.token_ids else tokenize(c.text)
        row: dict[str, Any] = {
            "prompt_id": p.id,
            "known_trigger": p.known_trigger,
            "n_tokens": c.n_tokens or len(toks),
            "hit_cap": c.hit_cap,
            "cycle": tail_cycle_period(toks),
            "longest_repeat": longest_repeated_run(toks),
        }
        for n_g in ngram_sizes:
            row[f"rep{n_g}"] = ngram_repetition(toks, n_g)
        all_rows.append(row)
        per_prompt.setdefault(p.id, []).append(row)
        if save_completions:
            raw.append({**row, "seed": s, "text": c.text})

        if (n + 1) % 20 == 0 or n + 1 == len(plan):
            logutil.event(
                log,
                "progress",
                done=n + 1,
                of=len(plan),
                label=label,
                rep8_so_far=round(statistics.fmean(r["rep8"] for r in all_rows), 4)
                if all_rows and "rep8" in all_rows[0]
                else None,
            )

    if not all_rows:
        raise RuntimeError(f"every generation failed ({errors} errors); backend unreachable?")

    def agg(key: str, fn: Any) -> float:
        return float(fn([r[key] for r in all_rows]))

    # A generation counts as looping if it never stopped AND its tail is a clean cycle.
    # Either alone is too loose: long answers hit the cap legitimately, and a cycle in the
    # middle of an otherwise-terminating answer is not the failure being chased.
    loops = [r for r in all_rows if r["hit_cap"] and r["cycle"] > 0]

    report = RepetitionReport(
        backend=backend.name,
        label=label,
        n_completions=len(all_rows),
        max_tokens=max_tokens,
        preset=sampling.name,
        tokenization=tok_desc,
        repetition={f"rep{n_g}": agg(f"rep{n_g}", statistics.fmean) for n_g in ngram_sizes},
        repetition_median={
            f"rep{n_g}": agg(f"rep{n_g}", statistics.median) for n_g in ngram_sizes
        },
        cap_hit_rate=agg("hit_cap", statistics.fmean),
        loop_rate=len(loops) / len(all_rows),
        mean_tokens=agg("n_tokens", statistics.fmean),
        errors=errors,
        extra_body=dict(getattr(backend, "extra_body", {})),
    )
    report.by_prompt = [
        {
            "prompt_id": pid,
            "n": len(rows),
            "known_trigger": rows[0]["known_trigger"],
            **{
                f"rep{n_g}": statistics.fmean(r[f"rep{n_g}"] for r in rows)
                for n_g in ngram_sizes
            },
            "cap_hit_rate": statistics.fmean(r["hit_cap"] for r in rows),
        }
        for pid, rows in sorted(per_prompt.items())
    ]

    trig = [r for r in all_rows if r["known_trigger"]]
    if trig:
        report.trigger_subset = {
            "n": float(len(trig)),
            **{
                f"rep{n_g}": statistics.fmean(r[f"rep{n_g}"] for r in trig)
                for n_g in ngram_sizes
            },
            "cap_hit_rate": statistics.fmean(r["hit_cap"] for r in trig),
            "loop_rate": len([r for r in trig if r["hit_cap"] and r["cycle"] > 0]) / len(trig),
        }

    if save_completions:
        out = Path(save_completions)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")

    logutil.event(
        log,
        "repetition report",
        label=label,
        backend=backend.name,
        **{k: round(v, 5) for k, v in report.repetition.items()},
        cap_hit_rate=round(report.cap_hit_rate, 4),
        loop_rate=round(report.loop_rate, 4),
        errors=errors,
    )
    return report
