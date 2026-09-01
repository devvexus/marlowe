"""Streaming depth surgery. No GPU, no full model load.

Reads the source safetensors one tensor at a time, drops the tensors belonging to removed
layers, renumbers the survivors, rewrites the config, and writes new shards. Peak memory is
one output shard buffer plus one tensor -- ``--shard-size`` bounds it directly. (The
original script's docstring claimed ~250 MB; that is the *read* granularity. The write side
buffers a whole shard because ``safetensors.save_file`` takes a complete dict.)

Only linear_attention layers are removable. All 16 full_attention layers are preserved --
they carry the KV cache and every exact retrieval, and losing one wrecks long-context recall
in a way short benchmarks will not show you.

Two things here are load-bearing and easy to get wrong:

* **Renumbering.** Two caches are keyed by ``layer_idx``: the KV cache for full_attention and
  the conv/recurrent state cache for DeltaNet. On disk this is the tensor-name index; in a
  live module tree it is the ``layer_idx`` attribute. Both must be reset.
* **The consistency assertion.** After writing, every entry of the new ``layer_types`` is
  checked against the tensors actually present at that index. This is the one assertion that
  catches the entire class of renumbering and config bugs, all of which otherwise produce a
  checkpoint that loads cleanly and emits fluent nonsense.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from marlowe import logutil, preflight
from marlowe.arch import (
    LAYER_RE,
    ArchDims,
    LayerType,
    Layout,
    assert_layer_types_match_tensors,
    detect_text_stack_prefix,
    is_mtp_tensor,
    is_vision_tensor,
    text_config,
)

log = logutil.get("surgery")

VisionMode = Literal["extract-text", "keep-wrapper"]

#: Files copied verbatim from the parent when they exist.
COPY_PREFIXES = (
    "tokenizer",
    "vocab",
    "merges",
    "chat_template",
    "generation_config",
    "special_tokens",
    "added_tokens",
)
VISION_COPY_PREFIXES = ("preprocessor", "video_preprocessor", "processor")


# ---------------------------------------------------------------------------
# reading the source
# ---------------------------------------------------------------------------


def read_provenance(src: str | Path) -> list[int]:
    """Cut lineage from a parent's ``pruning_report.json``, or empty for an original model.

    Provenance is otherwise in-memory only, so without this the second rung of the ladder
    would report itself as a single cut from an unremarkable 52-layer model rather than as
    27B -> 22B -> 18B. Each entry is in *its own* ancestor's index space, which is why the
    list is kept flat and append-only rather than remapped.
    """
    report = Path(src) / "pruning_report.json"
    if not report.exists():
        return []
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    prov = data.get("provenance")
    return [int(i) for i in prov] if isinstance(prov, list) else []


def load_index(src: str | Path) -> tuple[dict[str, str], bool]:
    """Return (tensor name -> shard filename, is_sharded)."""
    src = Path(src)
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        with idx.open(encoding="utf-8") as f:
            wm: dict[str, str] = json.load(f)["weight_map"]
        return wm, True
    single = src / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"no safetensors found in {src} (looked for model.safetensors.index.json "
            f"and model.safetensors)"
        )
    from safetensors import safe_open

    with safe_open(str(single), framework="pt") as f:
        keys = list(f.keys())
    return dict.fromkeys(keys, "model.safetensors"), False


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


@dataclass
class SurgeryPlan:
    """Everything decided before a byte is written. Fully inspectable under --dry-run."""

    prefix: str
    removed: list[int]
    kept: list[int]
    parent_layout: Layout
    child_layout: Layout
    #: old tensor name -> new tensor name
    mapping: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    namespaces: dict[str, int] = field(default_factory=dict)
    text_root: str = ""
    new_root: str = ""

    @property
    def n_removed(self) -> int:
        return len(self.removed)

    def summary(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "removed": self.removed,
            "n_kept_layers": len(self.kept),
            "child": self.child_layout.describe(),
            "tensors_kept": len(self.mapping),
            "tensors_dropped": len(self.dropped),
            "rename_root": f"{self.text_root} -> {self.new_root}"
            if self.text_root != self.new_root
            else "(none)",
        }


def _rename_root(name: str, text_root: str, new_root: str) -> str:
    """Move a tensor out of the multimodal wrapper namespace into a text-only one."""
    if text_root == new_root or not name.startswith(text_root):
        return name
    out = new_root + name[len(text_root) :]
    # lm_head lives at the top level in a text-only checkpoint, never under model.
    if out.startswith("model.lm_head."):
        out = out[len("model.") :]
    return out


def plan_surgery(
    weight_map: dict[str, str],
    layout: Layout,
    removed: list[int],
    *,
    drop_vision: bool,
    drop_mtp: bool,
    vision_mode: VisionMode = "extract-text",
    prefix: str | None = None,
) -> SurgeryPlan:
    """Build the full old-name -> new-name mapping. Validates the removal set first."""
    child_layout, kept = layout.apply(removed)  # raises on any rule violation

    if prefix is None:
        prefix, namespaces = detect_text_stack_prefix(weight_map, layout.n_layers)
    else:
        namespaces = {}
    remap = {old: new for new, old in enumerate(kept)}

    # "model.language_model.layers." -> text root "model.language_model."
    text_root = prefix[: -len("layers.")]
    new_root = text_root
    if drop_vision and vision_mode == "extract-text":
        # Collapse the wrapper namespace so the result is an ordinary text-only checkpoint.
        # Derived from the detected prefix, never hardcoded; a no-op when already flat.
        head, _, _ = text_root.rstrip(".").rpartition(".")
        new_root = f"{head}." if head else "model."

    mapping: dict[str, str] = {}
    dropped: list[str] = []
    for name in weight_map:
        if drop_vision and is_vision_tensor(name):
            dropped.append(name)
            continue
        if drop_mtp and is_mtp_tensor(name):
            dropped.append(name)
            continue
        m = LAYER_RE.match(name)
        if m and m.group(1) == prefix:
            idx = int(m.group(2))
            if idx not in remap:
                dropped.append(name)
                continue
            renumbered = f"{m.group(1)}{remap[idx]}{m.group(3)}"
            mapping[name] = _rename_root(renumbered, text_root, new_root)
        else:
            mapping[name] = _rename_root(name, text_root, new_root)

    collisions = len(mapping) - len(set(mapping.values()))
    if collisions:
        seen: set[str] = set()
        dupes = sorted({v for v in mapping.values() if v in seen or seen.add(v)})  # type: ignore[func-returns-value]
        raise ValueError(f"rename produced {collisions} colliding names, e.g. {dupes[:5]}")

    return SurgeryPlan(
        prefix=prefix,
        removed=sorted(removed),
        kept=kept,
        parent_layout=layout,
        child_layout=child_layout,
        mapping=mapping,
        dropped=dropped,
        namespaces=namespaces,
        text_root=text_root,
        new_root=new_root,
    )


# ---------------------------------------------------------------------------
# config rewriting
# ---------------------------------------------------------------------------


def rewrite_config(
    cfg: dict[str, Any],
    plan: SurgeryPlan,
    *,
    drop_vision: bool,
    drop_mtp: bool,
    vision_mode: VisionMode = "extract-text",
) -> dict[str, Any]:
    """Produce the child config. Pure function -- takes and returns a dict.

    ``full_attention_interval`` is removed unconditionally (rule 3.4). After pruning the
    layout is not uniform and no interval describes it; some runtimes *regenerate*
    ``layer_types`` from that field instead of reading the explicit list, which rebuilds the
    original 3:1 pattern over the wrong number of layers and loads without complaint.
    """
    cfg = json.loads(json.dumps(cfg))  # deep copy; never mutate the caller's parsed config
    t = text_config(cfg)

    t["layer_types"] = list(plan.child_layout.layer_types)
    t["num_hidden_layers"] = plan.child_layout.n_layers

    # Rule 3.4. Pop from both levels: it can appear on either in composite configs.
    t.pop("full_attention_interval", None)
    cfg.pop("full_attention_interval", None)

    if drop_mtp:
        t["mtp_num_hidden_layers"] = 0
        t.pop("mtp_use_dedicated_embeddings", None)
        cfg.pop("mtp_num_hidden_layers", None)

    if drop_vision:
        cfg.pop("vision_config", None)
        cfg.pop("deepstack_visual_indexes", None)
        t.pop("deepstack_visual_indexes", None)
        if vision_mode == "extract-text":
            # Flatten to an ordinary text-only causal LM config. Paired with the tensor
            # namespace collapse in plan_surgery -- doing one without the other yields a
            # config and a state dict that disagree about where the decoder lives.
            flat = dict(t)
            for key, val in cfg.items():
                if key not in ("text_config", "vision_config") and key not in flat:
                    flat[key] = val
            flat["architectures"] = ["Qwen3_5ForCausalLM"]
            flat["model_type"] = t.get("model_type", "qwen3_5_text")
            flat.pop("text_config", None)
            flat["language_model_only"] = True
            cfg = flat
        else:
            cfg["language_model_only"] = True

    return cfg


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _iter_source(src: Path, plan: SurgeryPlan, by_shard: dict[str, list[str]]) -> Iterator[
    tuple[str, Any]
]:
    """Yield (new_name, tensor), opening each source shard exactly once."""
    from safetensors import safe_open

    for shard in sorted(by_shard):
        with safe_open(str(src / shard), framework="pt") as f:
            for old in by_shard[shard]:
                yield plan.mapping[old], f.get_tensor(old)


#: Shards this module writes. Matches both the in-progress name and the final N-of-M form.
_OUR_SHARD_RE = re.compile(r"^model-\d{5}(-of-\d{5})?\.safetensors$")


def clear_previous_shards(out: str | Path) -> int:
    """Remove shards from an earlier run of this function. Returns the count removed.

    Surgery must be idempotent: stages are re-runnable and ``--force`` re-runs them
    deliberately. Without this, a second run writes ``model-00001.safetensors``, then fails
    renaming it to ``model-00001-of-000NN.safetensors`` because the previous run's output is
    already sitting there -- after having spent the whole streaming pass.

    Deliberately narrow: only files matching the exact naming this module produces, plus its
    index. Anything else in the directory is someone else's and is left alone.
    """
    out = Path(out)
    removed = 0
    for p in out.iterdir():
        if not p.is_file():
            continue
        if _OUR_SHARD_RE.match(p.name) or p.name == "model.safetensors.index.json":
            p.unlink()
            removed += 1
    if removed:
        logutil.event(log, "cleared previous output", dir=str(out), files=removed)
    return removed


def write_checkpoint(
    src: str | Path,
    out: str | Path,
    plan: SurgeryPlan,
    weight_map: dict[str, str],
    *,
    shard_size_gb: float = 4.0,
) -> tuple[dict[str, str], int]:
    """Stream tensors into new shards. Returns (new weight_map, total bytes)."""
    from safetensors.torch import save_file

    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clear_previous_shards(out)

    by_shard: dict[str, list[str]] = defaultdict(list)
    for old in plan.mapping:
        by_shard[weight_map[old]].append(old)

    limit = int(shard_size_gb * 1e9)
    buf: dict[str, Any] = {}
    buf_bytes = 0
    shard_i = 0
    new_map: dict[str, str] = {}
    total = 0
    written: list[str] = []

    def flush() -> None:
        nonlocal buf, buf_bytes, shard_i
        if not buf:
            return
        shard_i += 1
        fn = f"model-{shard_i:05d}.safetensors"
        save_file(buf, str(out / fn), metadata={"format": "pt"})
        for k in buf:
            new_map[k] = fn
        written.append(fn)
        logutil.event(log, "shard written", file=fn, gb=round(buf_bytes / 1e9, 2), tensors=len(buf))
        buf, buf_bytes = {}, 0

    for new_name, tensor in _iter_source(src, plan, by_shard):
        nbytes = tensor.numel() * tensor.element_size()
        buf[new_name] = tensor
        buf_bytes += nbytes
        total += nbytes
        if buf_bytes >= limit:
            flush()
    flush()

    # Rename to the standard N-of-M convention now that M is known.
    final: dict[str, str] = {}
    for i, fn in enumerate(written, start=1):
        dst = f"model-{i:05d}-of-{shard_i:05d}.safetensors"
        (out / fn).rename(out / dst)
        final[fn] = dst
    new_map = {k: final[v] for k, v in new_map.items()}

    with (out / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, f, indent=2)
    return new_map, total


def copy_aux_files(src: str | Path, out: str | Path, *, keep_vision_files: bool) -> list[str]:
    src, out = Path(src), Path(out)
    copied: list[str] = []
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith(COPY_PREFIXES) or (
            keep_vision_files and name.startswith(VISION_COPY_PREFIXES)
        ):
            shutil.copy2(p, out / name)
            copied.append(name)
    return copied


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify_checkpoint(out: str | Path) -> dict[str, Any]:
    """Re-open the written checkpoint and prove config and tensors agree. (Rule 3.4)

    Deliberately reads from disk rather than trusting in-memory state: the thing being
    verified is the artefact, not the plan that produced it.
    """
    out = Path(out)
    with (out / "config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    t = text_config(cfg)

    layer_types: list[LayerType] = list(t["layer_types"])
    n_declared = t["num_hidden_layers"]
    if n_declared != len(layer_types):
        raise AssertionError(
            f"num_hidden_layers={n_declared} but layer_types has {len(layer_types)} entries"
        )
    if "full_attention_interval" in t or "full_attention_interval" in cfg:
        raise AssertionError(
            "full_attention_interval survived the rewrite. A runtime that regenerates "
            "layer_types from it will build the wrong stack and never tell you."
        )

    weight_map, _ = load_index(out)
    prefix, namespaces = detect_text_stack_prefix(weight_map, len(layer_types))
    assert_layer_types_match_tensors(layer_types, list(weight_map), prefix)

    dims = ArchDims.from_config(cfg)
    params = dims.total_params(layer_types)
    child = Layout(layer_types=layer_types)
    result = {
        "prefix": prefix,
        "namespaces": namespaces,
        "n_layers": len(layer_types),
        "type_counts": child.type_counts,
        "n_attention": child.n_attention,
        "n_periods": len(child.periods()),
        "params": params,
        "params_b": round(params / 1e9, 4),
        "n_tensors": len(weight_map),
    }
    logutil.event(log, "verify ok", **result)
    return result


def renumber_live_modules(layers: Any) -> int:
    """Reset ``layer_idx`` on every submodule of a live ``nn.ModuleList``. (Rule 3.3)

    Only needed when surgery happens on an instantiated model rather than on disk. Both KV
    and conv/recurrent caches key off this attribute; miss one and the model generates
    fluent, confident nonsense with no error raised.
    """
    fixed = 0
    for new_idx, block in enumerate(layers):
        # nn.Module.modules() already yields the block itself first, so do not prepend it --
        # the assignment would be idempotent but the returned count would be inflated, and
        # this count is the only evidence in the log that renumbering actually ran.
        for mod in block.modules():
            if hasattr(mod, "layer_idx"):
                mod.layer_idx = new_idx
                fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def run_surgery(
    src: str | Path,
    out: str | Path,
    removed: list[int],
    *,
    drop_vision: bool = True,
    drop_mtp: bool = True,
    vision_mode: VisionMode = "extract-text",
    max_per_period: int = 2,
    protect_first: int = 1,
    protect_last: int = 1,
    shard_size_gb: float = 4.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full surgery: plan, check resources, write, verify. Returns a report dict."""
    src, out = Path(src), Path(out)
    with (src / "config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)

    layout = Layout.from_config(
        cfg,
        max_per_period=max_per_period,
        protect_first_periods=protect_first,
        protect_last_periods=protect_last,
        provenance=read_provenance(src),
    )
    dims = ArchDims.from_config(cfg)
    weight_map, sharded = load_index(src)

    plan = plan_surgery(
        weight_map,
        layout,
        removed,
        drop_vision=drop_vision,
        drop_mtp=drop_mtp,
        vision_mode=vision_mode,
    )

    logutil.event(log, "parent layout", layout=layout.describe(), sharded=sharded)
    for p, n in sorted(plan.namespaces.items()):
        if p != plan.prefix:
            logutil.event(log, "other .layers. namespace, untouched", ns=p, layers=n)
    logutil.event(log, "plan", **plan.summary())

    child_params = dims.total_params(plan.child_layout.layer_types)
    logutil.event(
        log,
        "projected size",
        params_b=round(child_params / 1e9, 4),
        parent_b=round(dims.total_params(layout.layer_types) / 1e9, 4),
        removed_b=round(len(plan.removed) * dims.linear_block_params / 1e9, 4),
    )

    report: dict[str, Any] = {
        "src": str(src),
        "out": str(out),
        "removed_layers": plan.removed,
        "kept_layers": plan.kept,
        "new_layer_types": list(plan.child_layout.layer_types),
        "provenance": plan.child_layout.provenance,
        "dropped_vision": drop_vision,
        "dropped_mtp": drop_mtp,
        "vision_mode": vision_mode if drop_vision else None,
        "projected_params": child_params,
        "tensors_kept": len(plan.mapping),
        "tensors_dropped": len(plan.dropped),
    }

    if dry_run:
        report["dry_run"] = True
        return report

    preflight.require_disk_for_checkpoint(out.parent if out.parent.exists() else ".", child_params)

    with logutil.timed(log, "write shards", out=str(out)):
        _, total = write_checkpoint(src, out, plan, weight_map, shard_size_gb=shard_size_gb)

    new_cfg = rewrite_config(
        cfg, plan, drop_vision=drop_vision, drop_mtp=drop_mtp, vision_mode=vision_mode
    )
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2)

    copied = copy_aux_files(src, out, keep_vision_files=not drop_vision)
    logutil.event(log, "aux files copied", files=copied)

    verification = verify_checkpoint(out)
    report.update(
        {
            "total_bytes": total,
            "actual_params": verification["params"],
            "verification": verification,
            "aux_copied": copied,
        }
    )
    with (out / "pruning_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logutil.event(
        log,
        "surgery complete",
        gb=round(total / 1e9, 2),
        params_b=round(verification["params"] / 1e9, 4),
        layers=verification["n_layers"],
    )
    log.warning("This checkpoint loads but is NOT healed. Do not judge it before Stage 6.")
    return report
