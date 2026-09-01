"""Run configuration and the pinned sampling presets.

The sampling preset is not a convenience wrapper. Qwen3.8 ships two presets and running
thinking mode with the non-thinking one (``presence_penalty=1.5``) produces repetition that
looks exactly like quantisation damage -- which is the artefact this whole project exists to
measure. Every eval resolves its preset through :data:`THINKING` or :data:`NON_THINKING` and
records which one it used in its manifest. Nothing relies on a runtime default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from marlowe.manifest import hash_obj


@dataclass(frozen=True)
class SamplingPreset:
    """A fully-specified decode configuration. Every field is explicit on purpose."""

    name: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repeat_penalty: float = 1.0

    def as_llamacpp_args(self) -> list[str]:
        return [
            "--temp", str(self.temperature),
            "--top-p", str(self.top_p),
            "--top-k", str(self.top_k),
            "--min-p", str(self.min_p),
            "--presence-penalty", str(self.presence_penalty),
            "--repeat-penalty", str(self.repeat_penalty),
        ]

    def as_openai_kwargs(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
        }


#: Qwen3.8 thinking preset. All evals in this pipeline run at reasoning_effort=xhigh with
#: this preset pinned.
THINKING = SamplingPreset(
    name="thinking",
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
)

#: Present so the difference is visible in one place. Using it for a thinking-mode eval
#: manufactures the artefact being measured; the repetition harness refuses it.
NON_THINKING = SamplingPreset(
    name="non_thinking",
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
)

PRESETS: dict[str, SamplingPreset] = {p.name: p for p in (THINKING, NON_THINKING)}


def preset(name: str) -> SamplingPreset:
    if name not in PRESETS:
        raise KeyError(f"unknown sampling preset {name!r}; have {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# stage configs
# ---------------------------------------------------------------------------


@dataclass
class ScoreConfig:
    """Stage 3. Defaults encode rule 3.2: score long, score late in the sequence."""

    mode: Literal["oneshot", "greedy"] = "oneshot"
    seq_len: int = 32768
    n_seqs: int = 4
    n_positions: int = 512
    #: Fraction of the sequence to skip before sampling scored positions. 0.25 = last three
    #: quarters. Recurrent-state damage accumulates and is invisible near position zero.
    position_start_frac: float = 0.25
    rescore_every: int = 4  # greedy mode only
    seed: int = 0
    calib_path: str = "data/calib.jsonl"
    load_in_4bit: bool = True
    run_positional_control: bool = True


@dataclass
class TeacherConfig:
    """Stage 5. Offline because teacher and student cannot both be resident on 16 GB."""

    tokens: int = 50_000_000
    top_k: int = 16
    seq_len: int = 2048
    batch_size: int = 4
    corpus_path: str = "data/heal_corpus.jsonl"
    shard_tokens: int = 5_000_000  # one output file per shard, for resumability


@dataclass
class HealConfig:
    """Stage 6. QLoRA against the cached top-K distribution, not next-token CE."""

    tokens: int = 50_000_000
    seq_len: int = 2048
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    #: Mixers first: they are what pruning disturbed. FFN projections included because the
    #: residual stream statistics shift under them too.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    learning_rate: float = 1e-4
    warmup_steps: int = 200
    grad_accum: int = 16
    micro_batch: int = 1
    gradient_checkpointing: bool = True
    #: 8-bit Adam moments. ~1 GB saved on a 176M-parameter LoRA state; on a 16 GB card that
    #: is the difference between fitting at seq_len 2048 and not.
    optimizer_8bit: bool = True
    #: Sequence chunk for the logit/loss computation. 2048 x 248320 would be 1.0 GB in bf16
    #: before any softmax intermediate, so the loss is always chunked; 256 halves the peak
    #: relative to 512 at a small throughput cost.
    loss_chunk: int = 256
    checkpoint_every_tokens: int = 10_000_000
    #: Stop when KL improvement over a checkpoint window falls below this. Banks the compute.
    kl_plateau_delta: float = 0.002
    #: Optional final anneal at long context, mitigating the 2K-healing risk (section 7).
    long_context_tokens: int = 0
    long_context_seq_len: int = 16384


@dataclass
class QuantConfig:
    """Stage 0 and 7. ``tensor_types`` are llama-quantize --tensor-type patterns."""

    name: str
    base_type: str = "iq3_m"
    tensor_types: dict[str, str] = field(default_factory=dict)
    #: Informational: the size this recipe is expected to land at, for the 10 GB budget.
    target_gb: float = 10.2


@dataclass
class RepetitionConfig:
    """The metric this project exists to move. Spec fixed in section 6."""

    n_completions: int = 200
    max_tokens: int = 2048
    ngram_sizes: list[int] = field(default_factory=lambda: [8, 32])
    prompts_path: str = "data/circling_prompts.jsonl"
    preset: str = "thinking"
    seed: int = 0
    #: Concurrency against a local llama-server. 1 is safest on a 16 GB card.
    parallel: int = 1


@dataclass
class ShipCriteria:
    """Stage 7 gate. Both must pass; neither is negotiable at ship time."""

    #: KL to the bf16 parent must be strictly below the user's current IQ3_XXS build.
    max_kl_vs_baseline: Literal["iq3_xxs"] = "iq3_xxs"
    #: Repetition rate must be at or below the bf16 baseline.
    max_repetition_vs: Literal["bf16"] = "bf16"


@dataclass
class RunConfig:
    """One rung of the ladder. ``configs/marlowe-22b.yaml`` and ``-18b.yaml``."""

    name: str
    parent: str  # HF id or local path
    n_cuts: int
    run_dir: str = "runs/default"

    # layout constraints (rule 3.1 / 3.2)
    max_per_period: int = 2
    protect_first_periods: int = 1
    protect_last_periods: int = 1

    drop_vision: bool = True
    drop_mtp: bool = True

    score: ScoreConfig = field(default_factory=ScoreConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    heal: HealConfig = field(default_factory=HealConfig)
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)
    ship: ShipCriteria = field(default_factory=ShipCriteria)
    quants: list[QuantConfig] = field(default_factory=list)

    #: Set on the 18B rung. Refuses a direct 27B -> 18B cut: a 36% depth reduction in one
    #: step is outside the band where healing reliably recovers.
    requires_healed_parent: bool = False

    expected_params: float | None = None
    expected_layers: int | None = None

    def config_hash(self) -> str:
        return hash_obj(asdict(self))

    def stage_hash(self, *stages: str) -> str:
        """Hash only the parts of the config a given stage reads.

        Keeps a tweak to healing hyperparameters from invalidating a finished scoring run.
        """
        sub = {k: v for k, v in asdict(self).items() if k in stages or not _is_stage_field(k)}
        return hash_obj(sub)


_STAGE_FIELDS = {"score", "teacher", "heal", "repetition", "quants", "ship"}


def _is_stage_field(name: str) -> bool:
    return name in _STAGE_FIELDS


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _build(cls: type, data: Any) -> Any:
    """Recursively construct nested dataclasses from plain dicts, rejecting unknown keys."""
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        ftype = known[key].type
        if key == "quants" and isinstance(val, list):
            kwargs[key] = [_build(QuantConfig, v) for v in val]
        elif isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[key] = _build(ftype, val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


_NESTED: dict[str, type] = {
    "score": ScoreConfig,
    "teacher": TeacherConfig,
    "heal": HealConfig,
    "repetition": RepetitionConfig,
    "ship": ShipCriteria,
}


def load_run_config(path: str | Path) -> RunConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(RunConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    for key, val in raw.items():
        if key in _NESTED:
            kwargs[key] = _build(_NESTED[key], val)
        elif key == "quants":
            kwargs[key] = [_build(QuantConfig, v) for v in (val or [])]
        else:
            kwargs[key] = val
    return RunConfig(**kwargs)
