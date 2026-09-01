#!/usr/bin/env python3
"""Evaluate Unsloth against the validated plain-peft stack. Isolated venv only.

Three measurements on identical data and seed, run in each stack and compared:

1. **tok/s** — is the claimed ~2x real on *this* architecture?
2. **peak VRAM, device-level** — savings might reopen seq 1536 or rank 32; an
   unrecognised-architecture fallback might cost more. Either changes sizing, and the whole
   memory ladder was measured on plain peft.
3. **loss curve** — the deciding one. If Unsloth diverges from plain peft on identical data,
   its kernels are computing something different on ``qwen3_5`` and it is unusable at any
   speed.

Why the third one decides. Unsloth patches model internals and must recognise the
architecture to do so correctly. Gated DeltaNet, the fused ``q_proj`` output gate, and the
float32 recurrent state (``mamba_ssm_dtype``) are all new and unusual. Silent miscompute is
this project's signature failure shape, and a throughput win on wrong gradients is worse than
no win at all.

Also worth watching: whether Unsloth *says* it does not recognise the architecture, or
quietly falls back to a generic path. A fallback typically shows as ~1.1x rather than ~2x,
which is its own answer.

    # never in the working environment
    python -m venv .venv-unsloth
    .venv-unsloth/Scripts/pip install unsloth
    .venv-unsloth/Scripts/python scripts/eval_unsloth.py --model <student> --out a.json
    python scripts/eval_unsloth.py --model <student> --out b.json --stack peft
    python scripts/eval_unsloth.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Loss curves are compared with this tolerance. Kernel-level numerical differences are
#: expected at 1e-3; a systematic gap above this means the two stacks are not computing the
#: same function.
LOSS_TOLERANCE = 0.02

SEED = 1234


def _fixed_batch(vocab: int, seq_len: int, top_k: int, device: Any) -> tuple[Any, Any, Any]:
    """Identical inputs in both stacks. Same seed, same shapes, same values."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(SEED)
    ids = torch.randint(0, vocab, (1, seq_len), generator=g)
    t_idx = torch.randint(0, vocab, (1, seq_len, top_k), generator=g)
    t_lp = torch.log_softmax(torch.randn(1, seq_len, top_k, generator=g), dim=-1)
    return ids.to(device), t_idx.to(device), t_lp.to(device)


def run_stack(
    model_path: str, stack: str, *, steps: int, seq_len: int, lora_rank: int
) -> dict[str, Any]:
    """Run ``steps`` training steps in one stack and record all three measurements."""
    import torch

    from marlowe.config import HealConfig
    from marlowe.heal import build_optimizer, chunked_kl_loss
    from marlowe.score import find_decoder

    cfg = HealConfig(seq_len=seq_len, lora_rank=lora_rank, quantize_lm_head=False)
    torch.manual_seed(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free0, total0 = torch.cuda.mem_get_info(0)
    baseline = total0 - free0

    warnings: list[str] = []
    if stack == "unsloth":
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            from unsloth import FastLanguageModel

            model, _ = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=seq_len,
                load_in_4bit=True,
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=lora_rank, lora_alpha=cfg.lora_alpha,
                target_modules=cfg.target_modules, use_gradient_checkpointing=True,
            )
        warnings = [str(w.message)[:300] for w in caught]
    else:
        from marlowe.heal import load_student

        model = load_student(model_path, cfg)

    decoder = find_decoder(model)
    lm_head = model.get_output_embeddings()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = build_optimizer(params, cfg)
    device = next(p.device for p in params)
    vocab = int(getattr(model.config, "vocab_size", 248320))

    ids, t_idx, t_lp = _fixed_batch(vocab, seq_len, 16, device)
    model.train()

    losses: list[float] = []
    timings: list[float] = []
    for i in range(steps):
        t0 = time.time()
        out = decoder(input_ids=ids, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        loss, _ = chunked_kl_loss(lm_head, hidden, t_idx, t_lp, chunk=cfg.loss_chunk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(float(loss.item()))
        del out, hidden, loss
        if i >= 2:  # discard allocator warm-up and cuBLAS autotuning
            timings.append(time.time() - t0)

    step_s = sum(timings) / max(len(timings), 1)
    return {
        "stack": stack,
        "seq_len": seq_len,
        "lora_rank": lora_rank,
        "steps": steps,
        "tok_s": seq_len / step_s,
        "step_s": step_s,
        "peak_vram_gb": (baseline + torch.cuda.max_memory_reserved()) / 1e9,
        "total_vram_gb": total0 / 1e9,
        "lora_params": sum(p.numel() for p in params),
        "losses": losses,
        "warnings": warnings,
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Decide on all three criteria. Loss divergence vetoes regardless of speed."""
    uns = a if a["stack"] == "unsloth" else b
    ref = b if a["stack"] == "unsloth" else a

    la, lb = uns["losses"], ref["losses"]
    n = min(len(la), len(lb))
    diffs = [abs(la[i] - lb[i]) for i in range(n)]
    max_diff = max(diffs) if diffs else float("inf")
    # A systematic offset matters more than any single spike.
    mean_diff = sum(diffs) / len(diffs) if diffs else float("inf")
    loss_ok = max_diff <= LOSS_TOLERANCE

    speedup = uns["tok_s"] / ref["tok_s"] if ref["tok_s"] else 0.0
    vram_delta = uns["peak_vram_gb"] - ref["peak_vram_gb"]

    verdict: list[str] = []
    if not loss_ok:
        verdict.append(
            f"REJECT: loss diverges (max {max_diff:.4f}, mean {mean_diff:.4f} vs tolerance "
            f"{LOSS_TOLERANCE}). Unsloth's kernels compute something different on this "
            f"architecture. Do not use it at any speed."
        )
    elif speedup < 1.25:
        verdict.append(
            f"REJECT: only {speedup:.2f}x. That is the signature of an unrecognised "
            f"architecture falling back to a generic path -- no win, and patched internals "
            f"for nothing."
        )
    else:
        verdict.append(f"PASS: {speedup:.2f}x faster, loss matches within {max_diff:.4f}.")
        verdict.append(
            f"VRAM {vram_delta:+.2f} GB. If negative, re-run `marlowe fitcheck` -- the whole "
            f"memory ladder was measured on plain peft and more room may reopen seq 1536 or "
            f"rank 32."
        )
    if uns["lora_params"] != ref["lora_params"]:
        verdict.append(
            f"WARNING: adapter sizes differ ({uns['lora_params'] / 1e6:.1f}M vs "
            f"{ref['lora_params'] / 1e6:.1f}M) -- the stacks did not target the same modules, "
            f"so the loss comparison is not like-for-like."
        )
    for w in uns["warnings"]:
        if "not recognized" in w.lower() or "unsupported" in w.lower() or "fallback" in w.lower():
            verdict.append(f"NOTE: Unsloth warned -- {w}")

    return {
        "speedup": speedup,
        "vram_delta_gb": vram_delta,
        "loss_max_diff": max_diff,
        "loss_mean_diff": mean_diff,
        "loss_ok": loss_ok,
        "accept": bool(loss_ok and speedup >= 1.25),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--stack", choices=("unsloth", "peft"), default="unsloth")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq-len", dest="seq_len", type=int, default=1024)
    ap.add_argument("--lora-rank", dest="lora_rank", type=int, default=32)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        result = compare(a, b)
        for k in ("stack", "tok_s", "peak_vram_gb", "lora_params"):
            print(f"  {k:<16} {a.get(k)!s:>22}  {b.get(k)!s:>22}")
        print()
        for line in result["verdict"]:
            print(line)
        return 0 if result["accept"] else 1

    if not args.model:
        ap.error("--model is required unless --compare is given")
    res = run_stack(
        args.model, args.stack,
        steps=args.steps, seq_len=args.seq_len, lora_rank=args.lora_rank,
    )
    print(
        f"{res['stack']}: {res['tok_s']:.1f} tok/s, peak {res['peak_vram_gb']:.2f} GB, "
        f"final loss {res['losses'][-1]:.5f}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
