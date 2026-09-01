"""Stage registry and orchestration.

Every stage is resumable, manifest-writing, and idempotent. A stage is skipped when its
manifest says it succeeded, its config hash is unchanged, its input fingerprints match, and
its declared outputs are still on disk. That last clause is what makes the state machine
trustworthy: a manifest is a claim, and the files are the evidence.

Build order is not the same as stage number. Stages 0 and 1 both gate the entire project and
both take about four hours, but **Stage 1 goes first** -- if the converter cannot handle a
non-uniform layout, Stage 0's results are irrelevant until that is fixed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marlowe import logutil
from marlowe.config import RunConfig
from marlowe.manifest import Manifest, fingerprint_path

log = logutil.get("pipeline")

StageFn = Callable[["StageContext"], dict[str, Any]]


@dataclass
class StageContext:
    """What a stage function receives. Declares its own inputs and outputs as it goes."""

    name: str
    cfg: RunConfig
    run_dir: Path
    manifest: Manifest
    force: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # -- paths --------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        return self.run_dir / "models"

    @property
    def gguf_dir(self) -> Path:
        return self.run_dir / "gguf"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    # -- declarations -------------------------------------------------------

    def declare_input(self, name: str, path: str | Path, *, deep: bool = False) -> Path:
        self.manifest.inputs[name] = fingerprint_path(path, deep=deep)
        return Path(path)

    def declare_output(self, name: str, path: str | Path) -> Path:
        self.manifest.outputs[name] = str(path)
        return Path(path)

    def note(self, text: str) -> None:
        self.manifest.notes.append(text)
        log.info("%s", text)


@dataclass
class Stage:
    name: str
    fn: StageFn
    description: str
    #: Stage names that must have succeeded first. Enforced, not documentation.
    requires: tuple[str, ...] = ()
    #: Which config sections this stage reads, for a narrow config hash. A tweak to healing
    #: hyperparameters must not invalidate a finished eight-hour scoring run.
    config_sections: tuple[str, ...] = ()
    estimated_hours: float = 0.0


REGISTRY: dict[str, Stage] = {}


def register(
    name: str,
    description: str,
    *,
    requires: tuple[str, ...] = (),
    config_sections: tuple[str, ...] = (),
    estimated_hours: float = 0.0,
) -> Callable[[StageFn], StageFn]:
    def deco(fn: StageFn) -> StageFn:
        REGISTRY[name] = Stage(
            name=name,
            fn=fn,
            description=description,
            requires=requires,
            config_sections=config_sections,
            estimated_hours=estimated_hours,
        )
        return fn

    return deco


class StageBlocked(RuntimeError):
    """A prerequisite stage has not succeeded."""


class MissingBaseline(RuntimeError):
    """A measurement the ship gate depends on cannot be obtained.

    Raised at stage entry rather than at ship time. The alternative is discovering at Stage 7
    that the gate cannot be evaluated, after the ~80 hours of scoring, caching and healing
    that sit in between.
    """


def check_requirements(stage: Stage, run_dir: Path) -> None:
    for req in stage.requires:
        m = Manifest.load(run_dir, req)
        if m is None or m.status != "ok":
            status = "never run" if m is None else m.status
            raise StageBlocked(
                f"stage {stage.name!r} requires {req!r}, which is {status}. Run it first: "
                f"marlowe run {req} --config <config.yaml>"
            )


def run_stage(
    name: str,
    cfg: RunConfig,
    *,
    run_dir: str | Path | None = None,
    force: bool = False,
    extra: dict[str, Any] | None = None,
) -> Manifest:
    """Run one stage, honouring the manifest state machine."""
    if name not in REGISTRY:
        raise KeyError(f"unknown stage {name!r}; have {sorted(REGISTRY)}")
    stage = REGISTRY[name]
    root = Path(run_dir or cfg.run_dir)
    root.mkdir(parents=True, exist_ok=True)

    logutil.setup(name, root)
    check_requirements(stage, root)

    config_hash = cfg.stage_hash(*stage.config_sections)
    previous = Manifest.load(root, name)

    man = Manifest(stage=name, config_hash=config_hash, config={"name": cfg.name})
    ctx = StageContext(
        name=name, cfg=cfg, run_dir=root, manifest=man, force=force, extra=extra or {}
    )

    if previous is not None and not force:
        # Re-derive this run's inputs cheaply by trusting the previous declaration; a stage
        # whose inputs genuinely moved will fail the fingerprint comparison below.
        skip, reason = previous.is_current(previous.inputs, config_hash)
        if skip:
            logutil.event(log, "skip", stage=name, reason=reason, elapsed_s=previous.elapsed_s)
            return previous
        logutil.event(log, "rerun", stage=name, reason=reason)

    logutil.event(
        log,
        "stage start",
        stage=name,
        description=stage.description,
        est_hours=stage.estimated_hours,
        run_dir=str(root),
        git=man.git_sha,
    )
    man.save(root)
    t0 = time.time()
    try:
        metrics = stage.fn(ctx) or {}
    except BaseException as exc:
        man.fail(exc)
        man.save(root)
        logutil.event_at(
            log,
            logging.ERROR,
            "stage failed",
            stage=name,
            elapsed_s=round(time.time() - t0, 1),
            error=str(exc),
        )
        raise
    man.succeed(**metrics)
    man.save(root)
    logutil.event(
        log, "stage ok", stage=name, elapsed_s=round(time.time() - t0, 1), outputs=man.outputs
    )
    return man


#: The order the brief specifies. Stage 1 before Stage 0 is deliberate.
BUILD_ORDER: tuple[str, ...] = (
    "stage1-smoke",
    "stage0-bitwidth",
    "stage2-baselines",
    "stage3-score",
    "stage4-surgery",
    "stage5-teacher",
    "stage6-heal",
    "stage7-ship",
)


def run_all(
    cfg: RunConfig,
    *,
    run_dir: str | Path | None = None,
    stages: tuple[str, ...] = BUILD_ORDER,
    force: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Manifest]:
    """Run stages in order. ``extra`` reaches every stage.

    It must: the hosted bf16 endpoint arrives this way, and Stage 2 fails closed without it.
    """
    out: dict[str, Manifest] = {}
    for name in stages:
        out[name] = run_stage(name, cfg, run_dir=run_dir, force=force, extra=extra)
    return out
