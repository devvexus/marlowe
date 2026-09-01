"""Architecture model for hybrid linear/attention decoder stacks.

Concretely targets Qwen3.8-27B (``Qwen3_5ForConditionalGeneration``), whose
``text_config.layer_types`` is an explicit 64-entry list::

    16 x [ linear_attention, linear_attention, linear_attention, full_attention ]

but nothing here is hardcoded to that shape. Two rules make it portable:

**Removability is a whitelist, not a blacklist.** ``linear_attention`` is the only removable
type. Every other type -- ``full_attention``, ``sparse_attention``, anything this module has
never heard of -- is attention-class and never removable. An unrecognised type fails closed
and says so, rather than being quietly eligible. Qwen3.8-Flash-Next (arch ``qwen4_exp``)
replaces Gated Attention with Qwen Sparse Attention; under a blacklist those layers would
have been silently removable and long-context retrieval would have been destroyed with no
error raised.

**Period structure is derived, not assumed.** A period is a run of layers ending at an
attention-class layer, read off ``layer_types``. Period count, per-period caps, and the
protect-first/protect-last window all follow from that. Qwen3.8-27B gives 16 periods of 4;
Flash-Next's 48 layers give 12 periods of 4; a stack with irregular spacing gives whatever
it actually has. No constant 4 appears anywhere in this file.

Parameter counts are computed from config fields rather than transcribed from a table, so
they stay correct for a pruned child whose layout no longer matches its parent. They
reconstruct the published Qwen3.8-27B figures exactly (see ``tests/test_arch.py``).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

log = logging.getLogger("marlowe.arch")

#: A layer type is any string appearing in ``layer_types``. Deliberately not an enum or a
#: Literal: the whole point is that unknown values must round-trip rather than fail to parse.
LayerType = str

LINEAR: Final[LayerType] = "linear_attention"
FULL: Final[LayerType] = "full_attention"

#: The whitelist. The ONLY removable layer type. Extending this is a correctness decision:
#: a type belongs here only if it maintains a fixed-size recurrent state rather than an
#: exact per-token cache, and only if the stack retains enough of them to keep state.
REMOVABLE_TYPES: Final[frozenset[str]] = frozenset({LINEAR})

#: Attention-class types this module recognises by name. Membership here buys a stronger
#: tensor-consistency check; absence costs nothing but the refinement, because anything not
#: in :data:`REMOVABLE_TYPES` is treated as attention-class regardless.
KNOWN_ATTENTION_TYPES: Final[frozenset[str]] = frozenset(
    {FULL, "sparse_attention", "swa", "sliding_attention", "full_attention_with_sink"}
)


def is_removable(layer_type: LayerType) -> bool:
    """True only for types on the whitelist. Everything else is load-bearing."""
    return layer_type in REMOVABLE_TYPES


def is_attention_class(layer_type: LayerType) -> bool:
    """True for every non-removable type, recognised or not. Periods end at these."""
    return not is_removable(layer_type)


def is_known_type(layer_type: LayerType) -> bool:
    return layer_type in REMOVABLE_TYPES or layer_type in KNOWN_ATTENTION_TYPES


# ---------------------------------------------------------------------------
# tensor naming
# ---------------------------------------------------------------------------

#: ``.layers.<idx>.`` namespaces. The vision tower has one too (depth 27), which is why
#: prefix detection keys on the max index rather than taking the first match.
LAYER_RE: Final[re.Pattern[str]] = re.compile(r"^(.*?\.layers\.)(\d+)(\..*)$")

#: Submodule names (the component directly under ``layers.<idx>.``) that identify a mixer
#: family. Matched as whole components, never as substrings -- ``linear_attn`` contains
#: ``attn``, and substring matching would classify every DeltaNet layer as attention.
LINEAR_MIXER_NAMES: Final[frozenset[str]] = frozenset(
    {"linear_attn", "linear_attention", "gdn", "gated_deltanet", "mamba", "mixer"}
)
ATTENTION_MIXER_NAMES: Final[frozenset[str]] = frozenset(
    {"self_attn", "attn", "attention", "sparse_attn", "sparse_attention", "qsa"}
)
#: Components present in every layer regardless of type; they carry no type evidence.
SHARED_COMPONENT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "mlp",
        "ffn",
        "feed_forward",
        "block_sparse_moe",
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    }
)

VISION_NAME_MARKERS: Final[tuple[str, ...]] = ("visual.", "vision_tower.", "vision_model.")
MTP_NAME_MARKERS: Final[tuple[str, ...]] = ("mtp",)


# ---------------------------------------------------------------------------
# config access
# ---------------------------------------------------------------------------


def text_config(cfg: Any) -> Any:
    """Return the text sub-config, whether nested (multimodal) or flattened (text-only).

    ``surgery --drop-vision`` flattens the nested ``text_config`` into the top level, so both
    shapes occur in this pipeline and every reader must tolerate both.
    """
    if isinstance(cfg, dict):
        sub = cfg.get("text_config")
        return sub if isinstance(sub, dict) else cfg
    sub = getattr(cfg, "text_config", None)
    return sub if sub is not None else cfg


def load_config(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"no config.json in {model_dir}")
    with path.open(encoding="utf-8") as f:
        obj: dict[str, Any] = json.load(f)
    return obj


def _cfg_get(t: Any, key: str, default: Any) -> Any:
    val = t.get(key, default) if isinstance(t, dict) else getattr(t, key, default)
    return default if val is None else val


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchDims:
    """Shape fields that drive parameter counts.

    Defaults are Qwen3.8-27B published values, so the dataclass doubles as the reference the
    unit tests check against. :meth:`from_config` overrides from a real ``config.json`` --
    always prefer that in pipeline code.
    """

    hidden_size: int = 5120
    intermediate_size: int = 17408
    vocab_size: int = 248320
    tie_word_embeddings: bool = False

    # Gated DeltaNet (linear_attention)
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # Gated GQA (full_attention)
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attn_output_gate: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> ArchDims:
        t = text_config(cfg)
        d = cls()
        # head_dim is sometimes omitted and implied by hidden/num_heads. Qwen3.8 states it
        # explicitly (256, larger than 5120/24) so never infer it when it is present.
        return cls(
            hidden_size=int(_cfg_get(t, "hidden_size", d.hidden_size)),
            intermediate_size=int(_cfg_get(t, "intermediate_size", d.intermediate_size)),
            vocab_size=int(_cfg_get(t, "vocab_size", d.vocab_size)),
            tie_word_embeddings=bool(_cfg_get(t, "tie_word_embeddings", d.tie_word_embeddings)),
            linear_num_value_heads=int(
                _cfg_get(t, "linear_num_value_heads", d.linear_num_value_heads)
            ),
            linear_num_key_heads=int(_cfg_get(t, "linear_num_key_heads", d.linear_num_key_heads)),
            linear_key_head_dim=int(_cfg_get(t, "linear_key_head_dim", d.linear_key_head_dim)),
            linear_value_head_dim=int(
                _cfg_get(t, "linear_value_head_dim", d.linear_value_head_dim)
            ),
            linear_conv_kernel_dim=int(
                _cfg_get(t, "linear_conv_kernel_dim", d.linear_conv_kernel_dim)
            ),
            num_attention_heads=int(_cfg_get(t, "num_attention_heads", d.num_attention_heads)),
            num_key_value_heads=int(_cfg_get(t, "num_key_value_heads", d.num_key_value_heads)),
            head_dim=int(_cfg_get(t, "head_dim", d.head_dim)),
            attn_output_gate=bool(_cfg_get(t, "attn_output_gate", d.attn_output_gate)),
        )

    # -- component counts ---------------------------------------------------

    @property
    def ffn_params(self) -> int:
        """SwiGLU: gate + up + down, all [intermediate, hidden]-shaped."""
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def linear_mixer_params(self) -> int:
        """Gated DeltaNet mixer.

        q/k project to ``key_heads * key_head_dim``; v and the output gate both project to
        ``value_heads * value_head_dim``. The separate output-gate projection is what makes
        this mixer 115.9M rather than 84.4M -- omitting it under-counts by 31.5M/layer.
        """
        h = self.hidden_size
        k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        v_dim = self.linear_num_value_heads * self.linear_value_head_dim
        conv_dim = 2 * k_dim + v_dim  # depthwise short conv over q, k, v

        return (
            h * k_dim  # q_proj
            + h * k_dim  # k_proj
            + h * v_dim  # v_proj
            + h * v_dim  # output gate
            + v_dim * h  # out_proj
            + h * self.linear_num_value_heads  # b_proj (beta)
            + h * self.linear_num_value_heads  # a_proj (decay)
            + self.linear_conv_kernel_dim * conv_dim  # conv1d weight (depthwise)
            + conv_dim  # conv1d bias
            + v_dim  # gated RMSNorm
        )

    @property
    def full_mixer_params(self) -> int:
        """Gated GQA mixer.

        ``attn_output_gate`` fuses a multiplicative output gate into q_proj, doubling it to
        [12288, 5120]. That gate is the component Stage 0 protects at higher precision:
        multiplicative error compounds rather than adds.
        """
        h = self.hidden_size
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        q_out = q_dim * 2 if self.attn_output_gate else q_dim
        return (
            h * q_out  # q_proj (+ fused gate)
            + h * kv_dim  # k_proj
            + h * kv_dim  # v_proj
            + q_dim * h  # o_proj
            + self.head_dim  # q_norm
            + self.head_dim  # k_norm
        )

    @property
    def linear_block_params(self) -> int:
        """A whole linear_attention decoder block: mixer + FFN. ~383.3M."""
        return self.linear_mixer_params + self.ffn_params

    @property
    def full_block_params(self) -> int:
        """A whole full_attention decoder block: mixer + FFN. ~372.2M."""
        return self.full_mixer_params + self.ffn_params

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def embedding_total(self) -> int:
        """embed_tokens plus lm_head. Untied on this model, so it is counted twice."""
        return self.embedding_params * (1 if self.tie_word_embeddings else 2)

    def mixer_params(
        self, layer_type: LayerType, overrides: dict[str, int] | None = None
    ) -> tuple[int, bool]:
        """Mixer parameter count for a layer type. Returns (params, exact).

        ``exact`` is False for an attention-class type with no known formula and no override,
        where the gated-GQA count stands in. That makes projected sizes approximate, which is
        a reporting inaccuracy rather than a correctness bug -- unlike guessing at
        removability, which this module refuses to do.
        """
        if overrides and layer_type in overrides:
            return overrides[layer_type], True
        if layer_type == LINEAR:
            return self.linear_mixer_params, True
        if layer_type == FULL:
            return self.full_mixer_params, True
        return self.full_mixer_params, False

    def block_params(
        self, layer_type: LayerType, overrides: dict[str, int] | None = None
    ) -> tuple[int, bool]:
        mixer, exact = self.mixer_params(layer_type, overrides)
        return mixer + self.ffn_params, exact

    def total_params(
        self,
        layer_types: list[LayerType],
        overrides: dict[str, int] | None = None,
        *,
        strict: bool = False,
    ) -> int:
        """Text-only parameter count for a layout (excludes vision tower and MTP head).

        With ``strict=True``, raises rather than approximating an unknown attention type.
        """
        body = 0
        inexact: set[str] = set()
        for t in layer_types:
            params, exact = self.block_params(t, overrides)
            body += params
            if not exact:
                inexact.add(t)
        if inexact:
            msg = (
                f"no parameter formula for attention-class layer type(s) {sorted(inexact)}; "
                f"using the gated-GQA count as a stand-in. Projected sizes are approximate. "
                f"Pass overrides={{'<type>': <mixer_params>}} for an exact figure."
            )
            if strict:
                raise ValueError(msg)
            log.warning("%s", msg)
        return body + self.embedding_total + self.hidden_size  # + final norm


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    """A layer stack plus the constraints governing what may be cut from it.

    Constraint rationale (the five rules):

    * Only :data:`REMOVABLE_TYPES` may be cut. On Qwen3.8-27B just 16 of 64 layers carry a
      KV cache and do exact retrieval, and they hold up 262K-context behaviour essentially
      alone. The whitelist generalises that: anything not proven safe to remove is not.
    * The first and last period are protected; boundary layers do format-critical work that
      ablation KL on mid-sequence positions under-weights.
    * ``max_per_period`` caps removals per period so every period retains at least one
      removable-type layer and the alternation survives.
    """

    layer_types: list[LayerType]
    protect_first_periods: int = 1
    protect_last_periods: int = 1
    max_per_period: int = 2
    #: Layers already gone in an ancestor, in that ancestor's index space. Informational;
    #: carried through manifests so a ladder step can report its full lineage.
    provenance: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            raise ValueError("empty layer_types")
        unknown = sorted({t for t in self.layer_types if not is_known_type(t)})
        if unknown:
            # Not an error: an unrecognised type round-trips fine and is simply never
            # removable. Loud, because it means this stack is newer than this module.
            log.warning(
                "unrecognised layer type(s) %s: treated as attention-class and NEVER "
                "removable. If any of these maintain a fixed-size recurrent state and are "
                "genuinely prunable, add them to arch.REMOVABLE_TYPES deliberately.",
                unknown,
            )

    # -- derived views ------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.layer_types)

    @property
    def n_removable_type(self) -> int:
        """Count of layers whose *type* is removable, ignoring position constraints."""
        return sum(1 for t in self.layer_types if is_removable(t))

    @property
    def n_attention(self) -> int:
        """Count of attention-class layers -- the ones that are never removable."""
        return sum(1 for t in self.layer_types if is_attention_class(t))

    @property
    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for t in self.layer_types:
            counts[t] += 1
        return dict(sorted(counts.items()))

    def periods(self) -> list[list[int]]:
        """Group indices into runs ending at an attention-class layer.

        Derived from the explicit list, never from ``full_attention_interval`` or any assumed
        interval. After pruning the periods have unequal lengths and no interval describes
        them; across model families the period length differs to begin with.
        """
        out: list[list[int]] = []
        cur: list[int] = []
        for i, t in enumerate(self.layer_types):
            cur.append(i)
            if is_attention_class(t):
                out.append(cur)
                cur = []
        if cur:  # trailing removable layers with no closing attention layer
            out.append(cur)
        return out

    def period_of(self, idx: int) -> int:
        for p_i, p in enumerate(self.periods()):
            if idx in p:
                return p_i
        raise KeyError(idx)

    def _open_period_range(self) -> tuple[int, int]:
        n = len(self.periods())
        lo = self.protect_first_periods
        hi = n - self.protect_last_periods
        return lo, max(lo, hi)

    def removable_in(self, period: list[int]) -> list[int]:
        return [i for i in period if is_removable(self.layer_types[i])]

    def candidates(self) -> list[int]:
        """Removable-type layers inside the unprotected window, in index order."""
        lo, hi = self._open_period_range()
        elig: list[int] = []
        for p_i, p in enumerate(self.periods()):
            if lo <= p_i < hi:
                elig.extend(self.removable_in(p))
        return elig

    def period_capacity(self, p_i: int) -> int:
        """How many layers may be cut from period ``p_i`` under all constraints."""
        lo, hi = self._open_period_range()
        if not lo <= p_i < hi:
            return 0
        n_rem = len(self.removable_in(self.periods()[p_i]))
        return max(0, min(self.max_per_period, n_rem - 1))

    def budget(self) -> int:
        """Maximum removals under the current constraints.

        Capped both by ``max_per_period`` and by the requirement that each period keep at
        least one removable-type layer -- the latter binds on an already-pruned parent, where
        some periods are down to a single such layer.
        """
        return sum(self.period_capacity(p) for p in range(len(self.periods())))

    # -- validation ---------------------------------------------------------

    def validate_removal(self, removed: list[int]) -> None:
        """Hard-refuse any removal set that breaks a correctness rule.

        Raises rather than warns: every one of these produces a model that loads, runs, and
        is silently wrong.
        """
        if len(set(removed)) != len(removed):
            dupes = sorted({i for i in removed if removed.count(i) > 1})
            raise ValueError(f"duplicate indices in removal set: {dupes}")

        oob = [i for i in removed if not 0 <= i < self.n_layers]
        if oob:
            raise ValueError(f"indices out of range for a {self.n_layers}-layer stack: {oob}")

        # Rule 3.1, as a whitelist. Never overridable.
        blocked = [
            (i, self.layer_types[i]) for i in removed if not is_removable(self.layer_types[i])
        ]
        if blocked:
            kinds = sorted({t for _, t in blocked})
            unknown = [t for t in kinds if t not in KNOWN_ATTENTION_TYPES]
            detail = (
                f" Type(s) {unknown} are not recognised by this module, so they fail closed: "
                f"removable types are exactly {sorted(REMOVABLE_TYPES)}."
                if unknown
                else ""
            )
            raise ValueError(
                f"refusing to remove attention-class layers {[i for i, _ in blocked]} "
                f"(types {kinds}). Only {self.n_attention} of {self.n_layers} layers carry a "
                f"per-token cache and do exact token-to-token retrieval; removing any of them "
                f"destroys long-context recall in a way short-context evals will not show."
                f"{detail}"
            )

        protected = sorted(set(removed) - set(self.candidates()))
        if protected:
            lo, hi = self._open_period_range()
            raise ValueError(
                f"layers {protected} are outside the removable window (periods "
                f"[{lo}, {hi}) of {len(self.periods())})"
            )

        per_period: dict[int, int] = defaultdict(int)
        for i in removed:
            per_period[self.period_of(i)] += 1
        over = {p: c for p, c in per_period.items() if c > self.max_per_period}
        if over:
            raise ValueError(
                f"max_per_period={self.max_per_period} exceeded: {dict(sorted(over.items()))}"
            )

        # Belt and braces: even within max_per_period, prove no period is emptied of
        # removable-type layers. Binds when pruning an already-pruned parent.
        kept = set(range(self.n_layers)) - set(removed)
        for p_i, p in enumerate(self.periods()):
            had = self.removable_in(p)
            left = [i for i in had if i in kept]
            if had and not left:
                raise ValueError(
                    f"period {p_i} would retain no {LINEAR} layer (had {had}); the "
                    f"alternation would be broken there"
                )

    def apply(self, removed: list[int]) -> tuple[Layout, list[int]]:
        """Return (child layout, kept parent indices). Validates first."""
        self.validate_removal(removed)
        kept = [i for i in range(self.n_layers) if i not in set(removed)]
        child = Layout(
            layer_types=[self.layer_types[i] for i in kept],
            protect_first_periods=self.protect_first_periods,
            protect_last_periods=self.protect_last_periods,
            max_per_period=self.max_per_period,
            provenance=[*self.provenance, *sorted(removed)],
        )
        return child, kept

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Any, **kwargs: Any) -> Layout:
        t = text_config(cfg)
        types = _cfg_get(t, "layer_types", None)
        if not types:
            n = _cfg_get(t, "num_hidden_layers", None)
            raise ValueError(
                f"config has no explicit layer_types (num_hidden_layers={n}). Refusing to "
                f"reconstruct it from full_attention_interval -- that is exactly the silent "
                f"corruption this pipeline guards against."
            )
        return cls(layer_types=[str(x) for x in types], **kwargs)

    @classmethod
    def repeating(
        cls, pattern: list[LayerType] | tuple[LayerType, ...], reps: int, **kwargs: Any
    ) -> Layout:
        """Build a layout from a repeating pattern. For tests and for synthetic fixtures."""
        return cls(layer_types=list(pattern) * reps, **kwargs)

    @classmethod
    def qwen38_27b(cls, **kwargs: Any) -> Layout:
        """The published 64-entry layout: 16 x [lin, lin, lin, full]."""
        return cls.repeating([LINEAR, LINEAR, LINEAR, FULL], 16, **kwargs)

    def describe(self) -> str:
        counts = ", ".join(f"{n} {t}" for t, n in self.type_counts.items())
        periods = self.periods()
        lengths = sorted({len(p) for p in periods})
        shape = f"len {lengths[0]}" if len(lengths) == 1 else f"lens {lengths}"
        return (
            f"{self.n_layers} layers ({counts}), {len(periods)} periods ({shape}), "
            f"{len(self.candidates())} eligible, budget {self.budget()}"
        )


# ---------------------------------------------------------------------------
# positional control (Stage 3)
# ---------------------------------------------------------------------------


def positional_selection(layout: Layout, n_remove: int) -> list[int]:
    """Evenly spaced removals ignoring measured damage -- the Stage 3 control arm.

    If ablation-KL selection does not beat this on unhealed KL, the measurement pass is not
    earning its eight hours, and that is a finding worth reporting.
    """
    pool = layout.candidates()
    if n_remove > len(pool):
        raise ValueError(f"cannot take {n_remove} positional cuts from {len(pool)} candidates")
    if n_remove <= 0:
        return []

    chosen: list[int] = []
    per_period: dict[int, int] = defaultdict(int)
    # Walk evenly spaced offsets; if a pick would violate a constraint, advance to the next
    # free candidate rather than dropping the slot.
    for k in range(n_remove):
        start = round(k * (len(pool) - 1) / (n_remove - 1)) if n_remove > 1 else 0
        for off in range(len(pool)):
            idx = pool[(start + off) % len(pool)]
            if idx in chosen:
                continue
            p = layout.period_of(idx)
            if per_period[p] >= layout.period_capacity(p):
                continue
            chosen.append(idx)
            per_period[p] += 1
            break
        else:
            raise ValueError(f"positional selection stalled at {len(chosen)}/{n_remove}")
    return sorted(chosen)


# ---------------------------------------------------------------------------
# tensor-name analysis
# ---------------------------------------------------------------------------


def detect_text_stack_prefix(
    weight_map: dict[str, str] | list[str], expect_layers: int
) -> tuple[str, dict[str, int]]:
    """Find the ``.layers.`` prefix belonging to the text decoder.

    The vision tower has its own ``.layers.`` namespace (depth 27), so select the prefix
    whose highest index matches the text stack rather than taking the first match. Returns
    ``(prefix, {prefix: layer_count})``; the second element is for logging the namespaces
    that were seen and deliberately left alone.
    """
    names = weight_map.keys() if isinstance(weight_map, dict) else weight_map
    maxima: dict[str, int] = defaultdict(lambda: -1)
    for name in names:
        m = LAYER_RE.match(name)
        if m:
            maxima[m.group(1)] = max(maxima[m.group(1)], int(m.group(2)))
    counts = {p: mx + 1 for p, mx in maxima.items()}
    if not counts:
        raise ValueError("no '.layers.<n>.' namespaces found in the checkpoint")

    hits = [p for p, c in counts.items() if c == expect_layers]
    if not hits:
        found = ", ".join(f"{p}* -> {c} layers" for p, c in sorted(counts.items()))
        raise ValueError(f"no prefix with {expect_layers} layers. Found: {found}")
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous text stack -- {len(hits)} namespaces have {expect_layers} layers: "
            f"{sorted(hits)}. Pass the prefix explicitly."
        )
    return hits[0], counts


def is_vision_tensor(name: str) -> bool:
    return any(m in name for m in VISION_NAME_MARKERS)


def is_mtp_tensor(name: str) -> bool:
    return any(m in name.lower() for m in MTP_NAME_MARKERS)


def layer_signatures(names: list[str], prefix: str) -> dict[int, frozenset[str]]:
    """Map layer index -> the set of submodule names present directly under that index.

    e.g. ``{"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}``. This is the
    raw evidence for what kind of layer sits at each index, and it needs no knowledge of what
    the mixer is called: two layers of the same type have the same signature, and two layers
    of different types do not.
    """
    seen: dict[int, set[str]] = {}
    for name in names:
        m = LAYER_RE.match(name)
        if not m or m.group(1) != prefix:
            continue
        idx = int(m.group(2))
        component = m.group(3).lstrip(".").split(".", 1)[0]
        seen.setdefault(idx, set()).add(component)
    return {i: frozenset(v) for i, v in seen.items()}


def mixer_components(signature: frozenset[str]) -> frozenset[str]:
    """The parts of a signature that carry type evidence, i.e. not FFN or norms."""
    return frozenset(
        c for c in signature if c not in SHARED_COMPONENT_NAMES and "norm" not in c.lower()
    )


def classify_signature(signature: frozenset[str]) -> str | None:
    """Best-effort family for a signature: ``"linear"``, ``"attention"``, or None.

    None means the mixer submodule is named something this module does not recognise, which
    is fine -- the partition check below does not depend on recognising it.
    """
    mixers = mixer_components(signature)
    if mixers & LINEAR_MIXER_NAMES and not (mixers & ATTENTION_MIXER_NAMES):
        return "linear"
    if mixers & ATTENTION_MIXER_NAMES and not (mixers & LINEAR_MIXER_NAMES):
        return "attention"
    return None


def assert_layer_types_match_tensors(
    layer_types: list[LayerType], names: list[str], prefix: str
) -> None:
    """Prove a rewritten ``layer_types`` describes the tensors actually present. (Rule 3.4)

    This single assertion catches the whole class of renumbering and config bugs: an
    off-by-one in the remap, a stale ``full_attention_interval`` regenerating a uniform
    layout, a period boundary that moved. Each of those otherwise yields a checkpoint that
    loads cleanly, runs at full speed, and emits fluent nonsense.

    Three independent checks, in increasing specificity:

    1. **Coverage.** Every declared index has tensors; no tensors sit past the declared end.
    2. **Partition.** Each declared type maps to exactly one tensor signature, and no two
       types share a signature. This is the generic check -- it needs no knowledge of what
       the mixer submodules are called, so it works unchanged on a stack whose attention
       layers are Qwen Sparse Attention rather than Gated Attention. A shifted or misaligned
       ``layer_types`` splits some type across two signatures and is caught here.
    3. **Family.** Where the mixer submodule name *is* recognised, additionally require that
       a ``linear_attention`` layer looks linear and an attention-class layer looks like
       attention. This catches a wholesale swap that the partition check alone would pass.

    Runs after every surgery and in CI.
    """
    sigs = layer_signatures(names, prefix)
    if not sigs:
        raise AssertionError(f"no tensors found under prefix {prefix!r}")

    # 1. coverage
    missing = sorted(set(range(len(layer_types))) - set(sigs))
    if missing:
        raise AssertionError(
            f"layer_types declares {len(layer_types)} layers but indices {missing} have no "
            f"tensors under {prefix!r}"
        )
    extra = sorted(i for i in sigs if i >= len(layer_types))
    if extra:
        raise AssertionError(
            f"tensors exist at indices {extra} but layer_types has only {len(layer_types)} "
            f"entries -- renumbering left a gap or an orphan"
        )

    # 2. partition: declared type <-> tensor signature must be a bijection
    by_type: dict[str, dict[frozenset[str], list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, declared in enumerate(layer_types):
        by_type[declared][mixer_components(sigs[i])].append(i)

    problems: list[str] = []
    for declared, groups in sorted(by_type.items()):
        if len(groups) > 1:
            detail = "; ".join(
                f"{sorted(sig) or '<none>'} at layers {idxs[:6]}"
                + ("..." if len(idxs) > 6 else "")
                for sig, idxs in sorted(groups.items(), key=lambda kv: min(kv[1]))
            )
            problems.append(
                f"  type {declared!r} spans {len(groups)} different tensor signatures: {detail}"
            )
        for sig, idxs in groups.items():
            if not sig:
                problems.append(
                    f"  type {declared!r} at layers {idxs[:6]} has no mixer tensors at all "
                    f"(only shared components)"
                )

    sig_to_types: dict[frozenset[str], set[str]] = defaultdict(set)
    for declared, groups in by_type.items():
        for sig in groups:
            sig_to_types[sig].add(declared)
    for sig, types in sorted(sig_to_types.items(), key=lambda kv: sorted(kv[1])):
        if len(types) > 1:
            problems.append(
                f"  types {sorted(types)} share one tensor signature {sorted(sig)}: "
                f"layer_types claims a distinction the weights do not have"
            )

    # 3. family refinement, only where the mixer name is recognised
    for i, declared in enumerate(layer_types):
        family = classify_signature(sigs[i])
        if family is None:
            continue
        want = "linear" if is_removable(declared) else "attention"
        if family != want:
            problems.append(
                f"  layer {i}: declared {declared!r} (expects a {want} mixer) but tensors are "
                f"{family}: {sorted(mixer_components(sigs[i]))}"
            )

    if problems:
        raise AssertionError(
            "layer_types does not match the tensors present:\n" + "\n".join(problems)
        )
