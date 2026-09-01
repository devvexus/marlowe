"""Stage implementations.

Each stage is small: it resolves paths, calls into the module that does the work, declares
its inputs and outputs, and returns metrics for the manifest. The interesting logic lives in
:mod:`marlowe.score`, :mod:`marlowe.surgery`, :mod:`marlowe.heal` and friends -- this file is
the wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from marlowe import logutil, preflight
from marlowe.arch import ArchDims, Layout, load_config, positional_selection
from marlowe.config import THINKING, QuantConfig
from marlowe.eval import bench
from marlowe.eval import kl as kleval
from marlowe.eval import repetition as rep
from marlowe.pipeline import MissingBaseline, StageContext, register
from marlowe.report import CheckpointRecord, bpw_curve, collect, ship_gate, write_report

log = logutil.get("stages")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def _parent_dir(ctx: StageContext) -> Path:
    """Resolve the parent checkpoint: a local dir, or download it into the run."""
    p = Path(ctx.cfg.parent)
    if p.exists():
        return p
    dest = ctx.models_dir / "parent"
    if (dest / "config.json").exists():
        return dest
    raise FileNotFoundError(
        f"parent checkpoint {ctx.cfg.parent!r} is not a local directory and has not been "
        f"downloaded to {dest}. Fetch it first:\n"
        f"  huggingface-cli download {ctx.cfg.parent} --local-dir {dest}\n"
        f"That is ~56 GB of bf16 safetensors; check free disk before starting."
    )


def _measure_pair(
    ctx: StageContext,
    label: str,
    gguf: Path,
    *,
    tokenizer_path: str | None,
    reference: Path | None,
    corpus: Path,
    n_completions: int | None = None,
) -> dict[str, Any]:
    """Both metrics on one GGUF. Never report one without the other."""
    out: dict[str, Any] = {"label": label, "size_gb": round(gguf.stat().st_size / 1e9, 3)}

    if reference is not None and reference.exists():
        out["kl"] = kleval.measure(gguf, corpus, reference).metrics

    cfgr = ctx.cfg.repetition
    prompts = rep.load_prompts(cfgr.prompts_path)
    proc = rep.spawn_llama_server(gguf, ctx=8192, parallel=cfgr.parallel)
    try:
        report = rep.run_repetition(
            rep.LlamaServerBackend(),
            prompts,
            label=label,
            n_completions=n_completions or cfgr.n_completions,
            max_tokens=cfgr.max_tokens,
            ngram_sizes=cfgr.ngram_sizes,
            sampling_preset=cfgr.preset,
            seed=cfgr.seed,
            tokenizer_path=tokenizer_path,
            save_completions=ctx.metrics_dir / f"completions-{label}.jsonl",
            parallel=cfgr.parallel,
        )
        out["repetition"] = report.as_dict()
    finally:
        proc.terminate()

    _write_json(ctx.metrics_dir / f"{label}.json", out)
    return out


# ---------------------------------------------------------------------------
# Stage 1 -- toolchain smoke test (runs FIRST)
# ---------------------------------------------------------------------------


@register(
    "stage1-smoke",
    "2-cut surgery -> GGUF -> quantize -> generate. Tests the converter blocker.",
    estimated_hours=4.0,
)
def stage1_smoke(ctx: StageContext) -> dict[str, Any]:
    """The unmitigated blocker, tested on day one.

    ``convert_hf_to_gguf.py`` may regenerate the layout from ``full_attention_interval``
    rather than reading a non-uniform ``layer_types``. If it does, everything downstream is
    dead until the converter is patched. A 2-cut model should be near-indistinguishable from
    its parent; if it is incoherent, that is a bug in renumbering or config, not a pruning
    result.
    """
    from marlowe import quantize as q
    from marlowe.surgery import run_surgery

    # Static check first. It costs milliseconds and answers the blocking question outright:
    # a converter with no way to emit an explicit per-layer recurrent mask can only write
    # full_attention_interval, which cannot describe a pruned hybrid stack.
    layout_ok, layout_detail = q.converter_writes_explicit_layout()
    if not layout_ok:
        raise RuntimeError(
            f"BLOCKER (Stage 1): the GGUF converter cannot express a non-uniform layout.\n"
            f"  {layout_detail}\n"
            f"Nothing downstream can proceed until this is fixed."
        )
    ctx.note(f"converter layout support: {layout_detail}")

    src = ctx.declare_input("parent", _parent_dir(ctx))
    layout = Layout.from_config(load_config(src))

    # Two cuts, deliberately from different periods so the layout genuinely stops being
    # uniform -- a single cut, or two from one period, can leave an interval that still
    # happens to describe the stack.
    removed = positional_selection(layout, 2)
    ctx.note(f"smoke test removing layers {removed} from {layout.describe()}")

    out_hf = ctx.declare_output("smoke_hf", ctx.models_dir / "smoke-2cut")
    report = run_surgery(
        src,
        out_hf,
        removed,
        drop_vision=ctx.cfg.drop_vision,
        drop_mtp=ctx.cfg.drop_mtp,
        max_per_period=ctx.cfg.max_per_period,
        protect_first=ctx.cfg.protect_first_periods,
        protect_last=ctx.cfg.protect_last_periods,
    )

    gguf = ctx.declare_output("smoke_gguf", ctx.gguf_dir / "smoke-2cut-bf16.gguf")
    q.convert_to_gguf(out_hf, gguf, outtype="bf16")

    probe = q.probe_converter(out_hf, gguf)
    q.dump_layout_json(gguf, ctx.metrics_dir / "smoke-gguf-layout.json")
    if not probe.ok:
        raise RuntimeError(
            "BLOCKER (Stage 1): the GGUF converter did not preserve the non-uniform "
            f"layer_types.\n  {probe.detail}\n"
            "Patch convert_hf_to_gguf.py to read the explicit layer_types list, place the "
            "patched copy at vendor/convert_hf_to_gguf.py, and upstream it. Nothing "
            "downstream can proceed until this is fixed."
        )
    ctx.note(f"converter probe passed: {probe.detail}")

    qgguf = ctx.declare_output("smoke_quant", ctx.gguf_dir / "smoke-2cut-iq3_m.gguf")
    q.quantize(gguf, qgguf, QuantConfig(name="smoke", base_type="iq3_m"))

    # A short generation is enough: incoherence from a renumbering bug is immediate and
    # total, not subtle.
    proc = rep.spawn_llama_server(qgguf, ctx=4096)
    try:
        backend = rep.LlamaServerBackend()
        from marlowe.config import THINKING

        sample = backend.generate(
            "Explain in three sentences why a fixed-size recurrent state cannot do exact "
            "long-range token retrieval.",
            256,
            THINKING,
            0,
        )
    finally:
        proc.terminate()

    _write_json(ctx.metrics_dir / "smoke-generation.json", {"text": sample.text})
    ctx.note(f"smoke generation ({sample.n_tokens} tokens): {sample.text[:200]!r}")

    return {
        "label": "smoke-2cut",
        "removed": removed,
        "params_b": round(report["projected_params"] / 1e9, 4),
        "layers": len(report["new_layer_types"]),
        "converter_probe": probe.as_dict(),
        "gguf": q.summarize_gguf(qgguf),
    }


# ---------------------------------------------------------------------------
# Stage 0 -- bit-width threshold experiment
# ---------------------------------------------------------------------------


@register(
    "stage0-bitwidth",
    "Where does circling stop? bpw-vs-repetition on the UNPRUNED 27B.",
    requires=("stage1-smoke",),
    config_sections=("repetition", "quants"),
    estimated_hours=4.0,
)
def stage0_bitwidth(ctx: StageContext) -> dict[str, Any]:
    """Decision-critical, and it gates the pruning targets.

    Circling occurs at 2.97 bpw and not at bf16. Nobody knows where in between it stops, and
    that threshold decides whether 22B at IQ3_M (3.66 bpw) is sufficient or whether only 18B
    at IQ4_XS (4.25 bpw) solves the actual problem.

    If a ~10 GB custom mix of the *unpruned* 27B already eliminates circling, that is the
    headline result and it is reported plainly: the user's problem would be solved without
    pruning, and this project becomes a speed and headroom play rather than a rescue.
    """
    from marlowe import quantize as q

    recipes = ctx.cfg.quants or q.stage0_recipes()
    # Decode-bound and the most expensive stage to get wrong: state the cost before the
    # first quantisation rather than discovering it eight variants in.
    kleval.require_gpu_backend(
        "stage0-bitwidth",
        n_variants=len(recipes),
        n_completions=ctx.cfg.repetition.n_completions,
        max_tokens=ctx.cfg.repetition.max_tokens,
        allow_cpu=bool(ctx.extra.get("allow_cpu_llamacpp")),
    )

    src = ctx.declare_input("parent", _parent_dir(ctx))
    dims = ArchDims.from_config(load_config(src))
    layout = Layout.from_config(load_config(src))
    n_params = dims.total_params(layout.layer_types)

    base = ctx.gguf_dir / "parent-bf16.gguf"
    if not base.exists():
        q.convert_to_gguf(src, base, outtype="bf16")
    ctx.declare_output("parent_bf16_gguf", base)

    results: list[dict[str, Any]] = []

    for recipe, path in q.iter_recipe_outputs(ctx.gguf_dir / "stage0", recipes):
        if not path.exists():
            q.quantize(base, path, recipe)
        bpw = q.bits_per_weight(path, n_params)
        row = _measure_pair(
            ctx,
            f"27b-{recipe.name}",
            path,
            tokenizer_path=str(src),
            reference=None,  # KL reference belongs to Stage 2; this stage is repetition-only
            corpus=Path(ctx.cfg.score.calib_path),
        )
        row.update(
            {
                "recipe": recipe.name,
                "base_type": recipe.base_type,
                "tensor_types": recipe.tensor_types,
                "bpw": round(bpw, 3),
                "budget": q.check_deployment_budget(path),
            }
        )
        results.append(row)
        ctx.declare_output(f"quant_{recipe.name}", path)

    # The candidates are ~10 GB each and fully described by their measured metrics; the
    # winning recipe is rebuilt in Stage 7 from the recipe definition.
    for _, path in q.iter_recipe_outputs(ctx.gguf_dir / "stage0", recipes):
        ctx.drop_temp(path, why="stage-0 candidate; metrics recorded, recipe reproducible")
    ctx.drop_temp(base, why="quantisation source; regenerable from the parent safetensors")

    records = [
        CheckpointRecord(
            label=r["label"],
            quant=r["recipe"],
            bpw=r["bpw"],
            size_gb=r["size_gb"],
            rep8=r["repetition"]["repetition"].get("rep8"),
            rep32=r["repetition"]["repetition"].get("rep32"),
            cap_hit_rate=r["repetition"]["cap_hit_rate"],
            loop_rate=r["repetition"]["loop_rate"],
        )
        for r in results
    ]
    bpw_curve(records, ctx.metrics_dir / "stage0-bpw-curve.png")
    _write_json(ctx.metrics_dir / "stage0-results.json", results)

    # Report the threshold plainly, including the case where pruning turns out unnecessary.
    clean = [r for r in results if r["repetition"]["loop_rate"] <= 0.01]
    threshold = min((r["bpw"] for r in clean), default=None)
    ctx.note(
        f"circling threshold: {threshold} bpw"
        if threshold
        else "no tested build eliminated circling; the threshold is above every recipe tried"
    )
    if threshold is not None and threshold <= 3.66:
        ctx.note("IQ3_M (3.66 bpw) is at or above the threshold, so Marlowe-22B is viable.")
    elif threshold is not None:
        ctx.note(
            f"threshold {threshold} bpw is above IQ3_M. Documented fallback: skip 22B and go "
            f"direct to 18B at IQ4_XS (4.25 bpw)."
        )
    custom_clean = [r for r in clean if r["tensor_types"]]
    if custom_clean:
        ctx.note(
            f"A ~10 GB custom mix of the UNPRUNED 27B ({custom_clean[0]['recipe']}) eliminates "
            f"circling. The user's actual problem is solved without pruning; the compression "
            f"project is now a speed and headroom play, not a rescue."
        )

    return {"checkpoints": results, "threshold_bpw": threshold}


# ---------------------------------------------------------------------------
# Stage 2 -- baselines
# ---------------------------------------------------------------------------


def _require_bf16_backend(ctx: StageContext) -> rep.OpenAICompatBackend | None:
    """Resolve and live-check the hosted bf16 endpoint. Fails closed. (Runs first.)

    The bf16 repetition baseline is not one input among several -- it is the definition of
    success. This project exists because circling appears at 2.97 bpw and vanishes at bf16.
    Without that number Stage 7 can report that repetition improved, but not that it reached
    target, and "better than a badly quantised model" is a bar the *unhealed* checkpoint might
    clear on its own.

    So this runs before the multi-hour GGUF conversion and KL reference build, and it sends a
    real request rather than just checking that a flag was passed: an endpoint that is
    configured but wrong is the same 80-hour failure as one that is missing.
    """
    hosted = ctx.extra.get("hosted_base_url")
    allow_missing = bool(ctx.extra.get("allow_missing_bf16_baseline"))

    if not hosted:
        if allow_missing:
            ctx.note(
                "PROCEEDING WITHOUT THE BF16 BASELINE (--allow-missing-bf16-baseline). Stage 7's "
                "ship gate will fail closed on the repetition criterion. Stages 3-6 artefacts "
                "are still produced and still useful."
            )
            return None
        raise MissingBaseline(
            "no hosted bf16 endpoint configured, so the bf16 repetition baseline cannot be "
            "measured.\n\n"
            "  That baseline is the definition of success for this project, not an optional "
            "extra: circling appears at 2.97 bpw and is absent at bf16, and the ship gate "
            "requires the candidate's repetition rate to reach the bf16 level. Without it "
            "Stage 7 can only say repetition improved -- a bar the unhealed checkpoint might "
            "clear on its own.\n\n"
            "  Configuring an endpoint now costs a few dollars. Discovering this at Stage 7 "
            "costs the ~80 hours of scoring, caching and healing in between.\n\n"
            "  Fix:   marlowe run stage2-baselines --config <cfg> \\\n"
            "           --hosted-base-url https://openrouter.ai/api/v1 \\\n"
            "           --hosted-model qwen/qwen3.8-27b \\\n"
            "           --hosted-api-key $OPENROUTER_API_KEY\n"
            "  Or:    --allow-missing-bf16-baseline   (Stages 3-6 only; Stage 7 cannot pass)"
        )

    backend = rep.OpenAICompatBackend(
        base_url=str(hosted),
        model=str(ctx.extra.get("hosted_model", ctx.cfg.parent)),
        api_key=str(ctx.extra.get("hosted_api_key", "")),
        extra_body=dict(ctx.extra.get("hosted_extra_body", {})),
    )
    probe = backend.generate("Reply with the single word: ready.", 32, THINKING, 0)
    if probe.stop_reason == "error":
        if allow_missing:
            ctx.note(f"bf16 endpoint unreachable ({probe.error}); continuing without it")
            return None
        raise MissingBaseline(
            f"the hosted bf16 endpoint at {hosted} did not answer a one-token probe.\n"
            f"  error: {probe.error}\n\n"
            f"  Checked before the GGUF conversion and KL reference build deliberately -- an "
            f"endpoint that is configured but wrong fails exactly as expensively as one that "
            f"was never configured. Verify the URL, model id and key, then re-run.\n"
            f"  Or pass --allow-missing-bf16-baseline to proceed without the ship gate."
        )
    ctx.note(f"bf16 endpoint live: {backend.model} at {hosted}")
    return backend

@register(
    "stage2-baselines",
    "bf16 KL reference, IQ3_XXS KL (the number to beat), repetition on both.",
    requires=("stage1-smoke",),
    config_sections=("repetition",),
    estimated_hours=4.0,
)

def stage2_baselines(ctx: StageContext) -> dict[str, Any]:
    """Three numbers, all required before any pruning.

    The gap between IQ3_XXS and bf16 is the entire opportunity this project targets. If it is
    small, the honest report is that the project may not beat what the user already runs.
    """
    # Preconditions first: hours of conversion follow, and both of these are cheap to check.
    kleval.require_gpu_backend(
        "stage2-baselines",
        n_variants=2,
        n_completions=ctx.cfg.repetition.n_completions,
        max_tokens=ctx.cfg.repetition.max_tokens,
        allow_cpu=bool(ctx.extra.get("allow_cpu_llamacpp")),
    )
    backend = _require_bf16_backend(ctx)

    src = ctx.declare_input("parent", _parent_dir(ctx))
    corpus = ctx.declare_input("kl_corpus", ctx.cfg.score.calib_path, deep=True)

    # The KL reference is built from Q8_0, not bf16. bf16 is 56 GB against 32 GB of RAM, so
    # llama.cpp would mmap and page from disk for the whole pass; Q8_0 is ~28.6 GB and its
    # own KL to bf16 is far below the deltas being measured. Every number in the project is
    # a relative comparison against this same reference file, so the substitution is free.
    # --reference-outtype bf16 overrides it for a machine with the RAM.
    outtype = str(ctx.extra.get("reference_outtype", kleval.REFERENCE_OUTTYPE))
    ref_gguf = ctx.gguf_dir / f"parent-{outtype}.gguf"
    if not ref_gguf.exists():
        from marlowe import quantize as q

        q.convert_to_gguf(src, ref_gguf, outtype=outtype)
    ctx.note(
        f"KL reference model: {ref_gguf.name} "
        f"({ref_gguf.stat().st_size / 1e9:.1f} GB); every checkpoint is compared to this file"
    )

    ref = ctx.declare_output("kl_reference", ctx.metrics_dir / "reference.kld")
    if not ref.exists():
        kleval.build_reference(ref_gguf, corpus, ref)

    results: dict[str, Any] = {}

    iq3 = ctx.extra.get("iq3_xxs_gguf") or ctx.gguf_dir / "stage0" / "iq3_xxs.gguf"
    iq3 = Path(iq3)
    if iq3.exists():
        results["iq3_xxs"] = _measure_pair(
            ctx, "27b-iq3_xxs-baseline", iq3, tokenizer_path=str(src), reference=ref, corpus=corpus
        )
    else:
        # Not fatal, unlike the bf16 baseline: this artefact is produced locally by Stage 0,
        # which runs earlier in BUILD_ORDER, so its absence means a stage was skipped rather
        # than an external dependency going unconfigured. Flip to MissingBaseline here if you
        # want both gate inputs enforced identically.
        ctx.note(
            f"IQ3_XXS build not found at {iq3}. Pass --iq3-xxs-gguf <path> to point at your "
            f"current build; without it the number to beat is unknown and the ship gate's KL "
            f"criterion cannot be evaluated."
        )

    # bf16 repetition goes through a hosted API: the bf16 weights are 55.6 GB and cannot run
    # on this machine at any useful speed.
    if backend is not None:
        prompts = rep.load_prompts(ctx.cfg.repetition.prompts_path)
        report = rep.run_repetition(
            backend,
            prompts,
            label="27b-bf16-hosted",
            n_completions=ctx.cfg.repetition.n_completions,
            max_tokens=ctx.cfg.repetition.max_tokens,
            ngram_sizes=ctx.cfg.repetition.ngram_sizes,
            sampling_preset=ctx.cfg.repetition.preset,
            seed=ctx.cfg.repetition.seed,
            tokenizer_path=str(src),
            save_completions=ctx.metrics_dir / "completions-bf16.jsonl",
        )
        results["bf16"] = {"label": "27b-bf16-hosted", "repetition": report.as_dict()}

    # reference.kld is the persistent artefact -- every checkpoint in the project is compared
    # against those exact bytes. The model that produced it is not needed again.
    ctx.drop_temp(ref_gguf, why="reference.kld is written; the source model is regenerable")

    _write_json(ctx.metrics_dir / "stage2-baselines.json", results)

    if "iq3_xxs" in results and "bf16" in results:
        gap = (
            results["iq3_xxs"]["repetition"]["repetition"]["rep32"]
            - results["bf16"]["repetition"]["repetition"]["rep32"]
        )
        ctx.note(f"rep32 gap IQ3_XXS - bf16 = {gap:+.4f}; this is the opportunity being targeted")
        if gap < 0.01:
            ctx.note(
                "The gap is small. Report plainly: this project may not beat what the user "
                "already runs, and the compression case rests on speed and headroom instead."
            )
    return {"checkpoints": [v for v in results.values() if "label" in v], **results}


# ---------------------------------------------------------------------------
# Stage 3 -- layer scoring
# ---------------------------------------------------------------------------


@register(
    "stage3-score",
    "Ablation-KL damage profile over eligible linear_attention layers (4-bit).",
    requires=("stage1-smoke",),
    config_sections=("score",),
    estimated_hours=8.0,
)
def stage3_score(ctx: StageContext) -> dict[str, Any]:
    """Also writes the standalone layer-redundancy artefact.

    No one has published layer-redundancy measurements on a hybrid linear/full-attention
    stack. Whether DeltaNet layers are more or less redundant than attention layers, and
    whether the 3:1 ratio is load-bearing or arbitrary, are open questions this profile
    answers -- independent of whether the pruning run succeeds.
    """
    from marlowe.score import run_scoring

    parent = ctx.extra.get("score_parent") or _parent_dir(ctx)
    src = ctx.declare_input("parent", parent)
    ctx.declare_input("calib", ctx.cfg.score.calib_path, deep=True)
    out = ctx.declare_output("profile", ctx.metrics_dir / "layer_profile.json")

    profile = run_scoring(
        str(src),
        out.parent,
        ctx.cfg.score,
        ctx.cfg.n_cuts,
        max_per_period=ctx.cfg.max_per_period,
        protect_first=ctx.cfg.protect_first_periods,
        protect_last=ctx.cfg.protect_last_periods,
        max_gpu_gb=ctx.extra.get("max_gpu_gb"),
    )
    ctx.declare_output("profile_plot", ctx.metrics_dir / "layer_profile.png")
    return {
        "selected": profile["selected"],
        "positional_control": profile["positional_control"],
        "forward_passes": profile["forward_passes"],
        "params_b": round(profile["projected_params"] / 1e9, 4),
        "layers": len(profile["child_layout"]),
    }


# ---------------------------------------------------------------------------
# Stage 4 -- surgery
# ---------------------------------------------------------------------------


@register(
    "stage4-surgery",
    "Cut the selected layers; measure the unhealed floor.",
    requires=("stage3-score",),
    estimated_hours=1.0,
)
def stage4_surgery(ctx: StageContext) -> dict[str, Any]:
    """The unhealed checkpoint is the floor healing must climb from, so it gets measured."""
    from marlowe.surgery import run_surgery

    src = ctx.declare_input("parent", _parent_dir(ctx))
    profile_path = ctx.declare_input("profile", ctx.metrics_dir / "layer_profile.json", deep=True)
    with profile_path.open(encoding="utf-8") as f:
        profile = json.load(f)

    removed = list(profile["selected"])
    out_hf = ctx.declare_output("unhealed", ctx.models_dir / f"{ctx.cfg.name}-unhealed")
    report = run_surgery(
        src,
        out_hf,
        removed,
        drop_vision=ctx.cfg.drop_vision,
        drop_mtp=ctx.cfg.drop_mtp,
        max_per_period=ctx.cfg.max_per_period,
        protect_first=ctx.cfg.protect_first_periods,
        protect_last=ctx.cfg.protect_last_periods,
    )

    params_b = report["actual_params"] / 1e9
    if ctx.cfg.expected_params is not None:
        drift = abs(params_b - ctx.cfg.expected_params)
        if drift > 0.05:
            raise AssertionError(
                f"parameter count {params_b:.4f}B differs from the configured target "
                f"{ctx.cfg.expected_params:.4f}B by {drift:.4f}B. Either the selection or the "
                f"accounting is wrong; do not proceed on a checkpoint of unexplained size."
            )
    ctx.note(f"unhealed checkpoint: {params_b:.4f}B, {len(report['new_layer_types'])} layers")
    ctx.note("This is the floor, not a result. It is expected to be bad; heal before judging.")

    return {
        "label": f"{ctx.cfg.name}-unhealed",
        "removed": removed,
        "params_b": round(params_b, 4),
        "layers": len(report["new_layer_types"]),
        "verification": report["verification"],
    }


# ---------------------------------------------------------------------------
# Stage 5 -- teacher cache
# ---------------------------------------------------------------------------


@register(
    "stage5-teacher",
    "Offline top-K logprob cache from the 4-bit teacher.",
    requires=("stage4-surgery",),
    config_sections=("teacher",),
    estimated_hours=6.0,
)
def stage5_teacher(ctx: StageContext) -> dict[str, Any]:
    from marlowe.teacher import build_cache, estimate_size

    src = ctx.declare_input("parent", _parent_dir(ctx))
    ctx.declare_input("corpus", ctx.cfg.teacher.corpus_path, deep=True)
    out = ctx.declare_output("cache", ctx.run_dir / "teacher-cache")

    need = estimate_size(ctx.cfg.teacher.tokens, ctx.cfg.teacher.top_k)
    ctx.note(f"teacher cache will be about {need / 1e9:.1f} GB on disk")

    idx = build_cache(
        str(src), out, ctx.cfg.teacher, max_gpu_gb=ctx.extra.get("max_gpu_gb"), resume=True
    )
    masses = [float(s["mean_captured_mass"]) for s in idx.shards]
    mean_mass = sum(masses) / max(len(masses), 1)
    if mean_mass < 0.8:
        ctx.note(
            f"teacher top-{idx.top_k} captures only {mean_mass:.3f} of the distribution on "
            f"average. A top-K objective is a weak proxy at that coverage; consider raising "
            f"top_k before spending three days healing against it."
        )
    return {
        "tokens": idx.n_tokens,
        "shards": len(idx.shards),
        "top_k": idx.top_k,
        "mean_captured_mass": round(mean_mass, 4),
    }


# ---------------------------------------------------------------------------
# Stage 6 -- healing
# ---------------------------------------------------------------------------


@register(
    "stage6-heal",
    "QLoRA distillation against the cached teacher. Both metrics at every checkpoint.",
    requires=("stage5-teacher",),
    config_sections=("heal", "repetition"),
    estimated_hours=72.0,
)
def stage6_heal(ctx: StageContext) -> dict[str, Any]:
    from marlowe.heal import probe_training, run_healing

    student = ctx.declare_input("unhealed", ctx.models_dir / f"{ctx.cfg.name}-unhealed")
    cache = ctx.declare_input("cache", ctx.run_dir / "teacher-cache")
    out = ctx.declare_output("adapters", ctx.models_dir / f"{ctx.cfg.name}-adapters")

    # Measure before committing three days. The brief's ~200 tok/s assumed Unsloth's memory
    # savings; this stack is plain peft + bitsandbytes, so the number has to be established
    # rather than inherited. Two minutes of real steps answers both "does it fit" and
    # "how long".
    probe = probe_training(str(student), ctx.cfg.heal, max_gpu_gb=ctx.extra.get("max_gpu_gb"))
    for line in probe.render(ctx.cfg.heal.tokens).splitlines():
        ctx.note(line)
    if probe.headroom_gb() < 0.3:
        raise preflight.ResourceError(
            f"only {probe.headroom_gb():.2f} GB VRAM headroom at seq_len "
            f"{ctx.cfg.heal.seq_len}; this will OOM partway through a three-day run. "
            f"Lower heal.seq_len, heal.lora_rank, or heal.loss_chunk."
        )
    scored: list[dict[str, Any]] = []

    def on_checkpoint(path: Path, state: Any) -> dict[str, Any]:
        """Score every checkpoint on both metrics.

        A checkpoint scored on KL alone has not been evaluated. If repetition stays elevated
        while KL improves, that is the signal to keep healing -- it is the metric the project
        exists to move.
        """
        row: dict[str, Any] = {"checkpoint": path.name, "tokens": state.position["tokens_seen"]}
        try:
            row.update(_checkpoint_metrics(ctx, path))
        except Exception as exc:  # noqa: BLE001 - a failed eval must not kill a 3-day run
            log.warning("checkpoint eval failed (training continues): %s", exc)
            row["eval_error"] = str(exc)
        scored.append(row)
        _write_json(ctx.metrics_dir / "heal-checkpoints.json", scored)
        return row

    state = run_healing(
        str(student),
        cache,
        out,
        ctx.cfg.heal,
        max_gpu_gb=ctx.extra.get("max_gpu_gb"),
        on_checkpoint=on_checkpoint,
        resume=True,
    )
    return {
        "probe": {
            "tok_s": round(probe.tok_s, 1),
            "peak_vram_gb": round(probe.peak_vram_gb, 2),
            "projected_hours": round(probe.projected_hours(ctx.cfg.heal.tokens), 1),
        },
        "stopped_reason": state.stopped_reason,
        "tokens_seen": state.position["tokens_seen"],
        "wall_seconds": state.wall_seconds,
        "checkpoints": scored,
        "kl_history": state.kl_history,
    }


def _checkpoint_metrics(ctx: StageContext, adapter: Path) -> dict[str, Any]:
    """Merge, quantize, and score a mid-training checkpoint.

    Deliberately cheap: a reduced completion count, because this runs every 10M tokens during
    a three-day job. The full harness runs at ship time.
    """
    from marlowe import quantize as q
    from marlowe.heal import merge_adapters

    tag = adapter.name
    merged = ctx.models_dir / f"tmp-merged-{tag}"
    gguf = ctx.gguf_dir / f"{tag}-bf16.gguf"
    quant = ctx.gguf_dir / f"{tag}-iq3_m.gguf"

    merge_adapters(str(ctx.models_dir / f"{ctx.cfg.name}-unhealed"), adapter, merged)
    q.convert_to_gguf(merged, gguf, outtype="bf16")
    q.quantize(gguf, quant, QuantConfig(name=tag, base_type="iq3_m"))

    ref = ctx.metrics_dir / "reference.kld"
    row = _measure_pair(
        ctx,
        tag,
        quant,
        tokenizer_path=str(merged),
        reference=ref if ref.exists() else None,
        corpus=Path(ctx.cfg.score.calib_path),
        n_completions=max(40, ctx.cfg.repetition.n_completions // 5),
    )
    out = {
        "eval_kl": (row.get("kl") or {}).get("kl_mean"),
        "rep8": row["repetition"]["repetition"].get("rep8"),
        "rep32": row["repetition"]["repetition"].get("rep32"),
        "cap_hit_rate": row["repetition"]["cap_hit_rate"],
        "loop_rate": row["repetition"]["loop_rate"],
    }
    # Intermediate artefacts are large; keep the GGUF, drop the merged safetensors.
    import shutil

    shutil.rmtree(merged, ignore_errors=True)
    gguf.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------------
# Stage 7 -- ship
# ---------------------------------------------------------------------------


@register(
    "stage7-ship",
    "Merge, needle @128K, quantize candidates, evaluate the ship gate.",
    requires=("stage6-heal",),
    config_sections=("repetition", "quants", "ship"),
    estimated_hours=4.0,
)
def stage7_ship(ctx: StageContext) -> dict[str, Any]:
    """Both ship criteria are required. If either fails, report which one and why."""
    from marlowe import quantize as q
    from marlowe.heal import latest_checkpoint, merge_adapters

    kleval.require_gpu_backend(
        "stage7-ship",
        n_variants=len(ctx.cfg.quants or q.ship_recipes()),
        n_completions=ctx.cfg.repetition.n_completions,
        max_tokens=ctx.cfg.repetition.max_tokens,
        allow_cpu=bool(ctx.extra.get("allow_cpu_llamacpp")),
    )

    unhealed = ctx.declare_input("unhealed", ctx.models_dir / f"{ctx.cfg.name}-unhealed")
    adapters = ctx.declare_input("adapters", ctx.models_dir / f"{ctx.cfg.name}-adapters")
    ckpt = latest_checkpoint(adapters)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoints under {adapters}")

    merged = ctx.declare_output("merged", ctx.models_dir / ctx.cfg.name)
    if not (merged / "config.json").exists():
        merge_adapters(str(unhealed), ckpt, merged)

    dims = ArchDims.from_config(load_config(merged))
    layout = Layout.from_config(load_config(merged))
    n_params = dims.total_params(layout.layer_types)

    base_gguf = ctx.gguf_dir / f"{ctx.cfg.name}-bf16.gguf"
    if not base_gguf.exists():
        q.convert_to_gguf(merged, base_gguf, outtype="bf16")

    probe = q.probe_converter(merged, base_gguf)
    if not probe.ok:
        raise RuntimeError(f"converter probe failed on the ship candidate: {probe.detail}")

    ref = ctx.metrics_dir / "reference.kld"
    corpus = Path(ctx.cfg.score.calib_path)
    candidates: list[dict[str, Any]] = []
    recipes = ctx.cfg.quants or q.ship_recipes()

    for recipe, path in q.iter_recipe_outputs(ctx.gguf_dir / ctx.cfg.name, recipes):
        if not path.exists():
            q.quantize(base_gguf, path, recipe)
        row = _measure_pair(
            ctx,
            f"{ctx.cfg.name}-{recipe.name}",
            path,
            tokenizer_path=str(merged),
            reference=ref if ref.exists() else None,
            corpus=corpus,
        )
        row["recipe"] = recipe.name
        row["bpw"] = round(q.bits_per_weight(path, n_params), 3)
        row["budget"] = q.check_deployment_budget(path)
        row["throughput"] = kleval.throughput(path)

        # Needle before shipping: healing ran at 2048 tokens and the training loss cannot see
        # whether long-context state maintenance survived.
        proc = rep.spawn_llama_server(path, ctx=131072, extra_args=("--no-context-shift",))
        try:
            needle = bench.run_needle(
                rep.LlamaServerBackend(),
                label=row["label"],
                filler_path=ctx.cfg.score.calib_path,
                tokenizer_path=str(merged),
                context_tokens=131072,
            )
            row["needle"] = needle.as_dict()
        finally:
            proc.terminate()

        candidates.append(row)
        ctx.declare_output(f"gguf_{recipe.name}", path)

    _write_json(ctx.metrics_dir / f"stage7-{ctx.cfg.name}.json", candidates)

    # Pick the winner empirically -- lowest KL among builds inside the weight budget, with
    # repetition as the tie-break. Not by bpw, and not by which recipe sounds better.
    def key(r: dict[str, Any]) -> tuple[float, float]:
        return ((r.get("kl") or {}).get("kl_mean", 1e9), r["repetition"]["repetition"]["rep32"])

    inside = [r for r in candidates if r["budget"]["fits"]] or candidates
    winner = min(inside, key=key)
    ctx.note(f"winner on measured KL/repetition: {winner['label']} ({winner['recipe']})")

    baselines = {r.label: r for r in collect(ctx.run_dir)}
    cand_rec = CheckpointRecord(
        label=winner["label"],
        quant=winner["recipe"],
        params_b=round(n_params / 1e9, 4),
        layers=layout.n_layers,
        size_gb=winner["size_gb"],
        bpw=winner["bpw"],
        kl_mean=(winner.get("kl") or {}).get("kl_mean"),
        top1_agreement=(winner.get("kl") or {}).get("top1_agreement"),
        rep8=winner["repetition"]["repetition"].get("rep8"),
        rep32=winner["repetition"]["repetition"].get("rep32"),
        cap_hit_rate=winner["repetition"]["cap_hit_rate"],
        loop_rate=winner["repetition"]["loop_rate"],
        needle_128k=winner.get("needle", {}).get("recall"),
        tok_s=winner.get("throughput", {}).get("tg_tok_s"),
    )
    gate = ship_gate(
        cand_rec,
        iq3_xxs_baseline=baselines.get("27b-iq3_xxs-baseline", CheckpointRecord("iq3_xxs")),
        bf16_baseline=baselines.get("27b-bf16-hosted", CheckpointRecord("bf16")),
    )
    for line in gate.reasons:
        ctx.note(line)

    if gate.passed:
        modelfile = ctx.declare_output("modelfile", ctx.run_dir / f"Modelfile.{ctx.cfg.name}")
        q.write_modelfile(
            ctx.gguf_dir / ctx.cfg.name / f"{winner['recipe']}.gguf", modelfile, name=ctx.cfg.name
        )
    else:
        ctx.note("Ship gate failed. Not writing a Modelfile.")

    records = [*collect(ctx.run_dir), cand_rec]
    write_report(ctx.run_dir, records, gate=gate, title=f"Marlowe: {ctx.cfg.name}")

    return {
        "checkpoints": candidates,
        "winner": winner["label"],
        "gate": gate.as_dict(),
        "shipped": gate.passed,
        "params_b": round(n_params / 1e9, 4),
        "layers": layout.n_layers,
    }


# ---------------------------------------------------------------------------
# Stage 8 -- ladder
# ---------------------------------------------------------------------------


@register(
    "stage8-ladder",
    "Re-score against the HEALED parent and repeat stages 4-7 for the next rung.",
    requires=("stage7-ship",),
    estimated_hours=96.0,
)
def stage8_ladder(ctx: StageContext) -> dict[str, Any]:
    """The ladder step. Re-scoring against the healed parent is a correctness requirement.

    The damage profile shifts after healing; reusing the 27B ranking would select layers whose
    measured redundancy no longer holds. And the ladder itself is non-negotiable: a direct
    27B -> 18B cut removes 36% of depth in one step, outside the band where healing reliably
    recovers, while two sequential ~19% cuts from an already-healed parent stay inside it.
    """
    child_cfg_path = ctx.extra.get("child_config")
    if not child_cfg_path:
        raise ValueError(
            "stage8 needs --child-config pointing at configs/marlowe-18b.yaml. The second rung "
            "is a full run against the healed 22B as its parent."
        )
    from marlowe.config import load_run_config
    from marlowe.pipeline import run_all

    child = load_run_config(child_cfg_path)
    healed = ctx.models_dir / ctx.cfg.name
    if not (healed / "config.json").exists():
        raise FileNotFoundError(f"healed parent {healed} not found; stage7 must complete first")

    if not child.requires_healed_parent:
        raise ValueError(
            f"{child_cfg_path}: requires_healed_parent must be true on the 18B rung. It is the "
            f"guard against a direct 27B -> 18B cut, which is the one thing this project's "
            f"design rules out."
        )
    child.parent = str(healed)
    ctx.note(f"ladder: {ctx.cfg.name} (healed) -> {child.name}, {child.n_cuts} further cuts")
    ctx.note(
        "Marlowe-22B ships independently and first; if this rung disappoints it is "
        "already in hand."
    )

    mans = run_all(
        child,
        run_dir=ctx.run_dir / child.name,
        stages=("stage3-score", "stage4-surgery", "stage5-teacher", "stage6-heal",
                "stage7-ship"),
        extra=ctx.extra,
    )
    return {
        "child": child.name,
        "stages": {k: v.status for k, v in mans.items()},
        "run_dir": str(ctx.run_dir / child.name),
    }
