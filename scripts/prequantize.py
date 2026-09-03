"""Quantise once to NF4 on disk, so nothing loads bf16 through the faulting path again.

Three fresh-process loads of the 27B parent gave one SIGSEGV at tensor 754 of 851 and two
clean completions. The fault is in ``transformers.core_model_loading._materialize_copy``, at
``tensor[...]`` -- the mmap read of a safetensors slice -- and it moves between runs, so it is
environmental rather than a corrupt shard. A large contributing cause was found and removed
(a background process leaking 4.9 million handles, 93% of the machine's total), but the load
was never proven reliable afterwards.

**That path only exists because the checkpoint is bf16 and quantisation happens during the
load.** transformers takes it when ``hf_quantizer is not None and not
hf_quantizer.pre_quantized`` -- which is also why it disables its own thread pool there. A
checkpoint that is *already* NF4 on disk is `pre_quantized`, so the load reads small
quantised tensors by the normal route and never enters ``_materialize_copy`` on a 45 GB mmap
that does not fit in 31 GB of RAM.

So this runs once, with retry, and everything afterwards reads the result.

``lm_head`` stays bf16. The healing loss is top-K KL against teacher logits, so quantising the
head adds noise to exactly the quantity being matched.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marlowe import logutil  # noqa: E402

log = logutil.get("prequantize")


def load_with_retry(
    path: str, *, quantize_lm_head: bool, max_gpu_gb: float | None = None, attempts: int = 3
) -> tuple[Any, Any]:
    """Load bf16 -> NF4, retrying. A SIGSEGV kills the process and never reaches here.

    An exception is retried in-process; a hard crash is not catchable, which is why the
    caller is expected to be run under process-level retry as well.
    """
    import torch

    from marlowe.score import load_4bit

    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            out = load_4bit(path, quantize_lm_head=quantize_lm_head)
            if attempt > 1:
                logutil.event(log, "loaded after retry", attempt=attempt)
            return out
        except BaseException as exc:  # noqa: BLE001 - the case being handled is a crash
            last = exc
            log.warning("load attempt %d/%d failed: %s: %s", attempt, attempts,
                        type(exc).__name__, exc)
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(20)
    raise RuntimeError(f"load failed {attempts} times; last: {type(last).__name__}: {last}")


def describe(model: Any) -> dict[str, Any]:
    """What the saved checkpoint actually is, read off the model rather than the config."""
    import torch

    qcfg = getattr(getattr(model, "config", None), "quantization_config", None)
    if qcfg is not None and not isinstance(qcfg, dict):
        qcfg = qcfg.to_dict() if hasattr(qcfg, "to_dict") else vars(qcfg)

    head = None
    for name in ("lm_head", "output"):
        mod = dict(model.named_modules()).get(name)
        if mod is not None and hasattr(mod, "weight"):
            head = {"module": name, "dtype": str(mod.weight.dtype),
                    "class": type(mod).__name__}
            break

    kinds: dict[str, int] = {}
    for m in model.modules():
        n = type(m).__name__
        if "4bit" in n or "Params4bit" in n or "Linear4" in n:
            kinds[n] = kinds.get(n, 0) + 1
    return {"quantization_config": qcfg, "lm_head": head, "quantized_module_counts": kinds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--quantize-lm-head", action="store_true",
                    help="off by default: the loss is top-K KL against the head's own logits")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    print(f"loading {src} bf16 -> NF4 (lm_head "
          f"{'quantized' if args.quantize_lm_head else 'bf16'})", flush=True)
    t0 = time.time()
    model, tok = load_with_retry(str(src), quantize_lm_head=args.quantize_lm_head)
    load_s = time.time() - t0
    before = describe(model)
    print(f"  loaded in {load_s:.0f}s", flush=True)
    print("  " + json.dumps(before, default=str)[:400], flush=True)

    dst.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    model.save_pretrained(str(dst), safe_serialization=True)
    tok.save_pretrained(str(dst))
    save_s = time.time() - t1
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"  saved in {save_s:.0f}s, {size/1e9:.2f} GB", flush=True)

    del model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    time.sleep(20)  # let driver host backing drain before the verification load

    # --- verify by reloading, not by trusting what was just written --------------------
    print("verifying: reloading the saved checkpoint", flush=True)
    from transformers import AutoModelForCausalLM

    t2 = time.time()
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(dst), device_map={"": 0}, trust_remote_code=True
    )
    reload_s = time.time() - t2
    after = describe(reloaded)
    print(f"  reloaded in {reload_s:.0f}s", flush=True)

    qc = after.get("quantization_config") or {}
    problems = []
    # quant_method is an enum whose str() is "<QuantizationMethod.BITS_AND_BYTES:
    # 'bitsandbytes'>", so a naive string compare reports a false failure on a checkpoint
    # that saved correctly -- which it did, and then drove a pointless retry of a 3-minute job.
    qm = qc.get("quant_method")
    qm_s = getattr(qm, "value", qm)
    if str(qm_s).lower() != "bitsandbytes":
        problems.append(f"quant_method is {qm!r}, expected bitsandbytes")
    if not qc.get("load_in_4bit"):
        problems.append("load_in_4bit is not set on the saved config")
    if str(qc.get("bnb_4bit_quant_type", "")).lower() != "nf4":
        problems.append(f"bnb_4bit_quant_type is {qc.get('bnb_4bit_quant_type')!r}, expected nf4")
    head = after.get("lm_head") or {}
    if not args.quantize_lm_head and "bfloat16" not in str(head.get("dtype", "")):
        problems.append(f"lm_head dtype is {head.get('dtype')!r}, expected bfloat16")
    if not after.get("quantized_module_counts"):
        problems.append("no 4-bit modules found on reload; the save did not preserve quantisation")

    manifest = {
        "src": str(src), "dst": str(dst),
        "load_seconds": round(load_s, 1), "save_seconds": round(save_s, 1),
        "reload_seconds": round(reload_s, 1),
        "bytes": size,
        "quantize_lm_head": args.quantize_lm_head,
        "verified": describe(reloaded),
        "problems": problems,
        "why": (
            "The on-the-fly quantization path is what faulted: transformers takes it when the "
            "checkpoint is not pre_quantized, and it reads a 45 GB bf16 mmap through "
            "_materialize_copy on a 31 GB machine. A pre-quantized checkpoint is loaded by "
            "the normal route and never enters it. Measured: 1 SIGSEGV in 3 fresh loads of "
            "the bf16 parent, at a varying tensor index."
        ),
    }
    mp = Path(args.manifest) if args.manifest else dst / "prequantize_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "verified"}, indent=2,
                     default=str), flush=True)
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}", flush=True)
        print("PREQUANTIZE_FAILED", flush=True)
        return 1
    print("PREQUANTIZE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
