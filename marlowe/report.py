"""Unified metrics table across checkpoints, and the ship gate.

Every checkpoint gets every metric. The table is the primary output of the project, and the
two columns that decide anything are KL (general fidelity) and repetition (the artefact being
fixed). They are not redundant: a checkpoint can have excellent KL and still loop, so both
appear and neither is allowed to stand in for the other.

**On AAII.** True AAII v4.1.1 is not reproducible here. It is nine evaluations -- GDPval-AA
v2, tau^3-Banking, Terminal-Bench v2.1, SciCode, HLE, GPQA Diamond, CritPt, AA-Omniscience,
AA-LCR -- weighted 34% Agents / 24% Coding / 24% Scientific Reasoning / 18% General, and
AA-Omniscience is a private dataset. This module reports GPQA Diamond and the KL/repetition
suite and does not compute an AAII number. The AAII figures in the project brief are priors,
not measurements, and are labelled as such wherever they appear.

Note also that 58% of AAII weight is agentic plus coding -- long generative traces where
errors compound. Expect measured degradation to exceed what MCQA-style evals suggest, and
read a flat GPQA accordingly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil
from marlowe.manifest import load_all

log = logutil.get("report")

NA = float("nan")


def _fmt(v: Any, spec: str = ".4f", width: int = 0) -> str:
    if v is None or (isinstance(v, float) and v != v):
        s = "--"
    elif isinstance(v, float):
        s = format(v, spec)
    else:
        s = str(v)
    return s.rjust(width) if width else s


@dataclass
class CheckpointRecord:
    """One row of the table. Missing metrics stay missing rather than defaulting to zero."""

    label: str
    stage: str = ""
    params_b: float | None = None
    layers: int | None = None
    quant: str = ""
    size_gb: float | None = None
    bpw: float | None = None

    kl_mean: float | None = None
    kl_median: float | None = None
    top1_agreement: float | None = None

    rep8: float | None = None
    rep32: float | None = None
    cap_hit_rate: float | None = None
    loop_rate: float | None = None

    gpqa: float | None = None
    needle_128k: float | None = None
    tok_s: float | None = None

    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


COLUMNS: list[tuple[str, str, str, int]] = [
    ("label", "checkpoint", "", 22),
    ("params_b", "params B", ".2f", 8),
    ("layers", "layers", "", 6),
    ("quant", "quant", "", 12),
    ("size_gb", "GB", ".2f", 6),
    ("bpw", "bpw", ".2f", 5),
    ("kl_mean", "KL", ".4f", 8),
    ("top1_agreement", "top-1", ".3f", 6),
    ("rep8", "rep8", ".4f", 7),
    ("rep32", "rep32", ".4f", 7),
    ("cap_hit_rate", "cap", ".3f", 6),
    ("loop_rate", "loop", ".3f", 6),
    ("gpqa", "GPQA", ".3f", 6),
    ("needle_128k", "ndl128k", ".3f", 8),
    ("tok_s", "tok/s", ".1f", 7),
]


def render_table(records: Sequence[CheckpointRecord]) -> str:
    header = "  ".join(h.ljust(w) if k == "label" else h.rjust(w) for k, h, _, w in COLUMNS)
    sep = "  ".join("-" * w for _, _, _, w in COLUMNS)
    lines = [header, sep]
    for r in records:
        cells = []
        for key, _, spec, width in COLUMNS:
            val = getattr(r, key)
            cells.append(
                str(val or "").ljust(width) if key == "label" else _fmt(val, spec, width)
            )
        lines.append("  ".join(cells))
    return "\n".join(lines)


def render_markdown(records: Sequence[CheckpointRecord]) -> str:
    head = "| " + " | ".join(h for _, h, _, _ in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = [
        "| "
        + " | ".join(_fmt(getattr(r, k), spec) for k, _, spec, _ in COLUMNS)
        + " |"
        for r in records
    ]
    return "\n".join([head, sep, *rows])


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def record_from_metrics(label: str, stage: str, metrics: dict[str, Any]) -> CheckpointRecord:
    """Build a row from a manifest's ``metrics`` blob, tolerating partial data."""
    rep = metrics.get("repetition", {}) or {}
    kl = metrics.get("kl", {}) or {}
    return CheckpointRecord(
        label=label,
        stage=stage,
        params_b=metrics.get("params_b"),
        layers=metrics.get("layers"),
        quant=metrics.get("quant", ""),
        size_gb=metrics.get("size_gb"),
        bpw=metrics.get("bpw"),
        kl_mean=kl.get("kl_mean", metrics.get("kl_mean")),
        kl_median=kl.get("kl_median"),
        top1_agreement=kl.get("top1_agreement", metrics.get("top1_agreement")),
        rep8=_dig(rep, "repetition", "rep8") or rep.get("rep8"),
        rep32=_dig(rep, "repetition", "rep32") or rep.get("rep32"),
        cap_hit_rate=rep.get("cap_hit_rate", metrics.get("cap_hit_rate")),
        loop_rate=rep.get("loop_rate", metrics.get("loop_rate")),
        gpqa=_dig(metrics, "gpqa", "accuracy"),
        needle_128k=_dig(metrics, "needle", "recall"),
        tok_s=_dig(metrics, "throughput", "tg_tok_s") or metrics.get("tok_s"),
        notes=list(metrics.get("notes", [])),
    )


def collect(run_dir: str | Path) -> list[CheckpointRecord]:
    """Gather every scored checkpoint from a run directory's manifests."""
    records: list[CheckpointRecord] = []
    for stage, man in sorted(load_all(run_dir).items()):
        m = man.metrics or {}
        if "checkpoints" in m and isinstance(m["checkpoints"], list):
            for entry in m["checkpoints"]:
                records.append(
                    record_from_metrics(entry.get("label", stage), stage, entry)
                )
        elif any(k in m for k in ("kl", "repetition", "kl_mean", "params_b")):
            records.append(record_from_metrics(m.get("label", stage), stage, m))
    return records


# ---------------------------------------------------------------------------
# ship gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        head = "SHIP" if self.passed else "DO NOT SHIP"
        return f"{head}\n" + "\n".join(f"  {r}" for r in self.reasons)


def ship_gate(
    candidate: CheckpointRecord,
    *,
    iq3_xxs_baseline: CheckpointRecord,
    bf16_baseline: CheckpointRecord,
) -> GateResult:
    """Stage 7 gate. Both criteria required; neither is negotiable at ship time.

    * KL to the bf16 parent strictly **below** the user's current IQ3_XXS build. Anything
      else means shipping something worse than what is already installed.
    * Repetition rate **at or below** the bf16 baseline. bf16 does not circle, so bf16 is the
      standard; matching it is the whole point of the exercise.
    """
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    passed = True

    def missing(name: str, what: str) -> None:
        nonlocal passed
        passed = False
        reasons.append(f"FAIL {name}: {what} was never measured; the gate cannot be evaluated")
        checks.append({"check": name, "ok": False, "reason": "missing measurement"})

    if candidate.kl_mean is None:
        missing("kl_vs_iq3_xxs", "candidate KL")
    elif iq3_xxs_baseline.kl_mean is None:
        missing("kl_vs_iq3_xxs", "IQ3_XXS baseline KL")
    else:
        ok = candidate.kl_mean < iq3_xxs_baseline.kl_mean
        passed &= ok
        reasons.append(
            f"{'PASS' if ok else 'FAIL'} kl_vs_iq3_xxs: {candidate.kl_mean:.4f} vs baseline "
            f"{iq3_xxs_baseline.kl_mean:.4f} (must be strictly lower)"
        )
        checks.append(
            {
                "check": "kl_vs_iq3_xxs",
                "ok": ok,
                "candidate": candidate.kl_mean,
                "baseline": iq3_xxs_baseline.kl_mean,
            }
        )

    cand_rep = candidate.rep32
    base_rep = bf16_baseline.rep32
    if cand_rep is None:
        missing("repetition_vs_bf16", "candidate repetition")
    elif base_rep is None:
        missing("repetition_vs_bf16", "bf16 baseline repetition")
    else:
        ok = cand_rep <= base_rep
        passed &= ok
        reasons.append(
            f"{'PASS' if ok else 'FAIL'} repetition_vs_bf16: rep32 {cand_rep:.4f} vs bf16 "
            f"{base_rep:.4f} (must be at or below)"
        )
        checks.append(
            {"check": "repetition_vs_bf16", "ok": ok, "candidate": cand_rep, "baseline": base_rep}
        )

    # Advisory, not part of the gate: reported so a marginal ship decision is informed.
    if candidate.needle_128k is not None and candidate.needle_128k < 0.9:
        reasons.append(
            f"WARN needle@128K recall {candidate.needle_128k:.3f}. Healing ran at 2048 tokens; "
            f"this is the signature of unrepaired DeltaNet state maintenance. Documented "
            f"mitigation: a final 5M-token anneal at 16K."
        )
    if candidate.size_gb is not None and candidate.size_gb > 10.0:
        reasons.append(
            f"WARN {candidate.size_gb:.2f} GB exceeds the 10.0 GB weight budget. Fallbacks: "
            f"Q4 K-cache, 24K context, or partial offload."
        )

    return GateResult(passed=bool(passed), reasons=reasons, checks=checks)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

@dataclass
class MultiGateResult:
    """The ship gate across every required bit-width. All must pass."""

    passed: bool
    per_width: dict[str, GateResult] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "missing": self.missing,
            "per_width": {k: v.as_dict() for k, v in self.per_width.items()},
        }

    def render(self) -> str:
        head = "SHIP" if self.passed else "DO NOT SHIP"
        lines = [f"{head} -- gate requires every bit-width below to pass", ""]
        for width, res in self.per_width.items():
            lines.append(f"  [{'PASS' if res.passed else 'FAIL'}] {width}")
            lines.extend(f"    {r}" for r in res.reasons)
        for width in self.missing:
            lines.append(f"  [FAIL] {width}: never built or never measured")
        return "\n".join(lines)


def ship_gate_multi(
    candidates: dict[str, CheckpointRecord],
    *,
    required_widths: Sequence[str],
    iq3_xxs_baseline: CheckpointRecord,
    bf16_baseline: CheckpointRecord,
) -> MultiGateResult:
    """Evaluate the gate at every required bit-width.

    The deliverable is a base that survives quantisation downward, so passing at one width
    proves nothing on its own: under-healed weights carry larger activation outliers and
    degrade unevenly across quantisation schemes. A width that was never built counts as a
    failure, not as an absence.
    """
    per_width: dict[str, GateResult] = {}
    missing: list[str] = []
    for width in required_widths:
        rec = candidates.get(width)
        if rec is None:
            missing.append(width)
            continue
        per_width[width] = ship_gate(
            rec, iq3_xxs_baseline=iq3_xxs_baseline, bf16_baseline=bf16_baseline
        )
    passed = bool(per_width) and not missing and all(g.passed for g in per_width.values())
    return MultiGateResult(passed=passed, per_width=per_width, missing=missing)


AAII_DISCLAIMER = """\
AAII is NOT reported. True AAII v4.1.1 is nine evaluations (GDPval-AA v2, tau^3-Banking,
Terminal-Bench v2.1, SciCode, HLE, GPQA Diamond, CritPt, AA-Omniscience, AA-LCR) weighted
34% Agents / 24% Coding / 24% Scientific Reasoning / 18% General, and AA-Omniscience is a
private dataset. Any AAII figure in the project brief is a prior, not a measurement.

58% of AAII weight is agentic plus coding -- long generative traces where errors compound.
Measured degradation should be expected to exceed what the MCQA-style GPQA column suggests.
"""


def write_report(
    run_dir: str | Path,
    records: Sequence[CheckpointRecord],
    *,
    gate: GateResult | MultiGateResult | None = None,
    title: str = "Marlowe metrics",
) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "report.md"
    parts = [
        f"# {title}",
        "",
        "```",
        render_table(records),
        "```",
        "",
        "## Metrics",
        "",
        render_markdown(records),
        "",
        "## Caveats",
        "",
        AAII_DISCLAIMER,
    ]
    if gate is not None:
        parts += ["## Ship gate", "", "```", gate.render(), "```", ""]
    out.write_text("\n".join(parts), encoding="utf-8")

    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "records": [r.as_dict() for r in records],
                "gate": gate.as_dict() if gate else None,
                "aaii": "not reported; see report.md",
            },
            f,
            indent=2,
        )
    logutil.event(log, "report written", path=str(out), rows=len(records))
    return out


def bpw_curve(records: Sequence[CheckpointRecord], path: str | Path) -> bool:
    """Stage 0's headline artefact: repetition rate against effective bit width."""
    pts = [(r.bpw, r.rep32, r.loop_rate, r.label) for r in records if r.bpw and r.rep32 is not None]
    if len(pts) < 2:
        log.warning("not enough points for a bpw curve (%d)", len(pts))
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping bpw curve")
        return False

    pts.sort(key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#c0392b", label="rep32")
    loops = [(p[0], p[2]) for p in pts if p[2] is not None]
    if loops:
        ax.plot(
            [p[0] for p in loops],
            [p[1] for p in loops],
            "s--",
            color="#2c7fb8",
            label="loop rate",
        )
    for bpw, rep, _, label in pts:
        ax.annotate(label, (bpw, rep), textcoords="offset points", xytext=(4, 5), fontsize=7)
    ax.set_xlabel("effective bits per weight")
    ax.set_ylabel("rate")
    ax.set_title("Stage 0: where does circling stop?\nunpruned 27B, ~10 GB builds, thinking preset")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True
