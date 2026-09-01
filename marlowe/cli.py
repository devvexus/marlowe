"""Command-line entry point.

    marlowe doctor                          # what is installed, what is missing
    marlowe plan   --config configs/marlowe-22b.yaml
    marlowe run    stage1-smoke --config configs/marlowe-22b.yaml
    marlowe surgery --src ... --out ... --remove 4,9,13
    marlowe verify --model ./Marlowe-22B    # the rule 3.4 assertion, standalone
    marlowe report --run-dir runs/marlowe-22b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from marlowe import logutil, preflight
from marlowe.arch import ArchDims, Layout, load_config, positional_selection
from marlowe.eval.kl import LlamaCppCpuOnly
from marlowe.pipeline import MissingBaseline, StageBlocked

GB = 1_000_000_000


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="run config YAML")
    p.add_argument("--run-dir", help="override run_dir from the config")
    p.add_argument("-v", "--verbose", action="store_true")


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report the environment honestly, including what will block which stage."""
    import importlib

    from marlowe.eval.kl import have_llamacpp

    res = preflight.probe(args.path or ".")
    print("resources")
    print(f"  {res.describe()}")

    print("\npython packages")
    need = {
        "torch": "all GPU stages",
        "transformers": "all stages",
        "safetensors": "surgery",
        "accelerate": "4-bit loading",
        "bitsandbytes": "stages 3, 5, 6 (NF4) -- no bf16 fallback exists on 16 GB",
        "peft": "stage 6 (QLoRA)",
        "numpy": "teacher cache",
        "yaml": "configs",
        "matplotlib": "plots (optional)",
        "psutil": "preflight",
    }
    missing: list[str] = []
    for mod, why in need.items():
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<16} {getattr(m, '__version__', 'present'):<12} {why}")
        except ImportError:
            print(f"  {mod:<16} {'MISSING':<12} {why}")
            missing.append(mod)

    print("\nllama.cpp")
    gpu_stages_blocked = False
    if have_llamacpp():
        from marlowe.eval.kl import detect_backend, find_binary, gpu_requirement_message

        print(f"  binaries: {Path(find_binary('llama-quantize')).parent}")
        backend = detect_backend()
        print(f"  backend:  {backend.describe()}")
        if not backend.has_gpu:
            gpu_stages_blocked = True
            print()
            print(
                gpu_requirement_message(
                    "Stages 0, 2 and 7", n_variants=8, n_completions=200, max_tokens=2048
                )
            )
    else:
        print("  MISSING -- stages 0, 1, 2, 7 need llama-quantize/perplexity/bench/server")
        print("  set LLAMA_CPP_BIN to the binary directory, or build llama.cpp")
    try:
        from marlowe.quantize import converter_writes_explicit_layout, find_converter

        conv = find_converter()
        print(f"  converter: {conv}")
        ok, detail = converter_writes_explicit_layout(conv)
        print(f"  layout support: {'OK' if ok else 'BROKEN'} -- {detail.splitlines()[0]}")
        if not ok:
            print("    A depth-pruned stack WILL be mis-typed. See vendor/ for the patch.")
    except FileNotFoundError as exc:
        print(f"  converter: MISSING ({exc.args[0].splitlines()[0]})")

    print("\ndisk budget (whole ladder: 27B -> 22B -> 18B)")
    items = preflight.disk_budget(preflight.default_rungs())
    rung = None
    for it in items:
        if it.rung != rung:
            rung = it.rung
            print(f"  {rung}")
        print(f"    {'keep' if it.persists else 'temp'}  {it.what:<44} ~{it.gb:>5.0f} GB")
    totals = preflight.budget_totals(items)
    free_gb = res.free_disk / GB
    print(f"\n  {'persistent (never deletable)':<52} ~{totals['persistent']:>5.0f} GB")
    print(
        f"  {'peak, deleting temps as each stage finishes':<52} "
        f"~{totals['peak_if_cleaned']:>5.0f} GB"
    )
    print(
        f"  {'peak, keeping everything':<52} ~{totals['total_if_nothing_deleted']:>5.0f} GB"
    )
    print(f"  {'free now':<52} ~{free_gb:>5.0f} GB")

    print(
        "\n  Default policy deletes temps as each stage finishes; --keep-intermediates opts out."
    )
    if free_gb >= totals["peak_if_cleaned"]:
        line = "  Default policy fits."
        if free_gb < totals["total_if_nothing_deleted"]:
            line += (
                f" --keep-intermediates would need ~"
                f"{totals['total_if_nothing_deleted']:.0f} GB and does NOT fit."
            )
        print(line)
    else:
        print(
            f"  WARNING: short by ~{totals['peak_if_cleaned'] - free_gb:.0f} GB even with temps "
            f"deleted. The merged rung-1 checkpoint cannot be deleted -- it is rung 2's parent."
        )
    return 1 if (missing or gpu_stages_blocked) else 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Show the layout, the cut budget, and the projected sizes without touching weights."""
    from marlowe.config import load_run_config

    cfg = load_run_config(args.config)
    src = Path(cfg.parent)
    if (src / "config.json").exists():
        raw = load_config(src)
        layout = Layout.from_config(
            raw,
            max_per_period=cfg.max_per_period,
            protect_first_periods=cfg.protect_first_periods,
            protect_last_periods=cfg.protect_last_periods,
        )
        dims = ArchDims.from_config(raw)
    else:
        print(f"parent {cfg.parent} not present locally; using the published Qwen3.8-27B layout")
        layout = Layout.qwen38_27b(
            max_per_period=cfg.max_per_period,
            protect_first_periods=cfg.protect_first_periods,
            protect_last_periods=cfg.protect_last_periods,
        )
        dims = ArchDims()

    base = dims.total_params(layout.layer_types)
    child, _ = layout.apply(positional_selection(layout, cfg.n_cuts))
    child_params = dims.total_params(child.layer_types)

    print(f"config      {args.config}  ({cfg.name})")
    print(f"parent      {cfg.parent}")
    print(f"layout      {layout.describe()}")
    print(f"cuts        {cfg.n_cuts}  (budget {layout.budget()})")
    print(f"parent size {base / 1e9:.4f} B")
    print(f"child  size {child_params / 1e9:.4f} B   ({child.describe()})")
    if cfg.expected_params:
        drift = abs(child_params / 1e9 - cfg.expected_params)
        print(f"expected    {cfg.expected_params:.4f} B   (drift {drift:+.4f} B)")
    print(f"\nblock sizes linear {dims.linear_block_params / 1e6:.1f} M   "
          f"attention {dims.full_block_params / 1e6:.1f} M")
    print(f"after cuts  budget for a further rung: {child.budget()}")

    import marlowe.stages  # noqa: F401  -- registers the stages
    from marlowe.pipeline import BUILD_ORDER, REGISTRY

    print("\nbuild order")
    total = 0.0
    for name in BUILD_ORDER:
        s = REGISTRY[name]
        total += s.estimated_hours
        print(f"  {name:<18} ~{s.estimated_hours:>5.1f} h  {s.description}")
    print(f"  {'TOTAL':<18} ~{total:>5.1f} h ({total / 24:.1f} d)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import marlowe.stages  # noqa: F401
    from marlowe.config import load_run_config
    from marlowe.pipeline import run_all, run_stage

    cfg = load_run_config(args.config)
    extra: dict[str, Any] = {}
    for key in ("iq3_xxs_gguf", "hosted_base_url", "hosted_model", "hosted_api_key",
                "child_config", "max_gpu_gb", "score_parent",
                "allow_missing_bf16_baseline", "allow_cpu_llamacpp",
                "reference_outtype", "keep_intermediates"):
        val = getattr(args, key, None)
        if val is not None:
            extra[key] = val

    if args.stage in ("all", None):
        mans = run_all(cfg, run_dir=args.run_dir, force=args.force, extra=extra)
    else:
        mans = {args.stage: run_stage(
            args.stage, cfg, run_dir=args.run_dir, force=args.force, extra=extra
        )}
    for name, man in mans.items():
        print(f"{name:<18} {man.status:<8} {json.dumps(man.metrics, default=str)[:120]}")
    return 0 if all(m.status == "ok" for m in mans.values()) else 1


def cmd_surgery(args: argparse.Namespace) -> int:
    from marlowe.surgery import run_surgery

    if args.remove:
        removed = sorted(int(x) for x in args.remove.split(","))
    elif args.auto:
        layout = Layout.from_config(
            load_config(args.src),
            max_per_period=args.max_per_period,
            protect_first_periods=args.protect_first,
            protect_last_periods=args.protect_last,
        )
        removed = positional_selection(layout, args.auto)
        print(f"WARNING: --auto ignores measured damage. Score first (stage3). Picked {removed}")
    else:
        print("pass --remove or --auto", file=sys.stderr)
        return 2

    report = run_surgery(
        args.src,
        args.out,
        removed,
        drop_vision=args.drop_vision,
        drop_mtp=args.drop_mtp,
        vision_mode=args.vision_mode,
        max_per_period=args.max_per_period,
        protect_first=args.protect_first,
        protect_last=args.protect_last,
        shard_size_gb=args.shard_size,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the rule 3.4 assertion against a checkpoint on disk."""
    from marlowe.surgery import verify_checkpoint

    try:
        result = verify_checkpoint(args.model)
    except AssertionError as exc:
        print(f"FAILED\n{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Stage 1's converter check, runnable standalone against any HF dir + GGUF pair."""
    from marlowe.quantize import probe_converter

    result = probe_converter(args.model, args.gguf)
    print(json.dumps(result.as_dict(), indent=2))
    return 0 if result.ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    from marlowe.report import collect, render_table, write_report

    records = collect(args.run_dir)
    if not records:
        print(f"no scored checkpoints under {args.run_dir}")
        return 1
    print(render_table(records))
    path = write_report(args.run_dir, records)
    print(f"\nwrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="marlowe", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment and disk report")
    d.add_argument("--path", default=".")
    d.set_defaults(fn=cmd_doctor)

    pl = sub.add_parser("plan", help="layout, budget, projected sizes, build order")
    pl.add_argument("--config", required=True)
    pl.set_defaults(fn=cmd_plan)

    r = sub.add_parser("run", help="run a stage (or all)")
    r.add_argument("stage", nargs="?", default="all")
    r.add_argument("--config", required=True)
    r.add_argument("--run-dir")
    r.add_argument("--force", action="store_true", help="ignore an up-to-date manifest")
    r.add_argument("--iq3-xxs-gguf", dest="iq3_xxs_gguf", help="the user's current build")
    r.add_argument("--hosted-base-url", dest="hosted_base_url", help="for the bf16 baseline")
    r.add_argument("--hosted-model", dest="hosted_model")
    r.add_argument("--hosted-api-key", dest="hosted_api_key")
    r.add_argument(
        "--allow-cpu-llamacpp",
        dest="allow_cpu_llamacpp",
        action="store_true",
        default=None,
        help="run generation-bound stages on a CPU-only llama.cpp. ~10x slower; the "
             "projected wall-clock is logged.",
    )
    r.add_argument(
        "--reference-outtype",
        dest="reference_outtype",
        choices=("q8_0", "bf16", "f16"),
        default=None,
        help="precision of the KL reference model (default q8_0: bf16 is 56 GB against "
             "32 GB of RAM and would page from disk for the whole pass).",
    )
    r.add_argument(
        "--keep-intermediates",
        dest="keep_intermediates",
        action="store_true",
        default=None,
        help="keep large regenerable artefacts (bf16/q8_0 GGUFs, unhealed checkpoints, "
             "quant candidates). Default is to delete them as each stage finishes; the "
             "whole ladder is ~491 GB retained vs ~217 GB cleaned.",
    )
    r.add_argument(
        "--allow-missing-bf16-baseline",
        dest="allow_missing_bf16_baseline",
        action="store_true",
        default=None,
        help="run stage2 without the bf16 repetition baseline. Stages 3-6 still produce "
             "their artefacts, but Stage 7's ship gate cannot pass.",
    )
    r.add_argument("--child-config", dest="child_config", help="stage8: the 18B config")
    r.add_argument("--max-gpu-gb", dest="max_gpu_gb", type=float)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("surgery", help="streaming depth surgery")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--remove", help="comma-separated indices from the scoring pass")
    s.add_argument("--auto", type=int, help="PLACEHOLDER: evenly spaced, ignores measured damage")
    s.add_argument("--drop-vision", action="store_true", default=True)
    s.add_argument("--keep-vision", dest="drop_vision", action="store_false")
    s.add_argument("--drop-mtp", action="store_true", default=True)
    s.add_argument("--keep-mtp", dest="drop_mtp", action="store_false")
    s.add_argument(
        "--vision-mode", choices=("extract-text", "keep-wrapper"), default="extract-text"
    )
    s.add_argument("--max-per-period", type=int, default=2)
    s.add_argument("--protect-first", type=int, default=1)
    s.add_argument("--protect-last", type=int, default=1)
    s.add_argument("--shard-size", type=float, default=4.0, help="GB per output shard")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_surgery)

    v = sub.add_parser("verify", help="assert layer_types matches the tensors present")
    v.add_argument("--model", required=True)
    v.set_defaults(fn=cmd_verify)

    pr = sub.add_parser("probe", help="did the GGUF converter preserve the layout?")
    pr.add_argument("--model", required=True, help="source HF directory")
    pr.add_argument("--gguf", required=True)
    pr.set_defaults(fn=cmd_probe)

    rp = sub.add_parser("report", help="unified metrics table")
    rp.add_argument("--run-dir", required=True)
    rp.set_defaults(fn=cmd_report)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logutil.setup(args.cmd, getattr(args, "run_dir", None))
    try:
        return int(args.fn(args))
    except (
        preflight.ResourceError,
        LlamaCppCpuOnly,
        MissingBaseline,
        StageBlocked,
        FileNotFoundError,
        ValueError,
        AssertionError,
    ) as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
