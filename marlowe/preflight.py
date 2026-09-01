"""Resource checks that fail loudly, with the actual number required.

The hardware here is fixed and tight: a 16 GB RTX 4080 Super and 32 GB of system RAM
against a model whose bf16 weights are 55.6 GB. Most ways this pipeline can fail are
resource failures, and they are far cheaper to catch at stage entry than six hours into a
scoring run.

The bf16 residency assertion (:func:`assert_no_bf16_resident`) is a correctness check, not
a performance one: no stage in this pipeline is permitted to hold full-precision weights.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marlowe import logutil

log = logutil.get("preflight")

GB = 1_000_000_000

#: Published sizes, used for the error messages. bf16 27B = 27.74e9 * 2 bytes.
BF16_27B_BYTES = 55_600_000_000
NF4_TEXT_ONLY_BYTES = 14_600_000_000


class ResourceError(RuntimeError):
    """Raised when the machine cannot satisfy a stage's hard requirement."""


@dataclass(frozen=True)
class Resources:
    free_disk: int
    total_ram: int
    available_ram: int
    vram_total: int
    vram_free: int
    gpu_name: str

    def describe(self) -> str:
        return (
            f"disk {self.free_disk / GB:.1f} GB free | "
            f"RAM {self.available_ram / GB:.1f}/{self.total_ram / GB:.1f} GB | "
            f"VRAM {self.vram_free / GB:.1f}/{self.vram_total / GB:.1f} GB ({self.gpu_name})"
        )


def probe(path: str | Path = ".") -> Resources:
    """Read current disk/RAM/VRAM. Never raises; absent GPU reports zeros."""
    free_disk = shutil.disk_usage(str(path)).free

    total_ram = available_ram = 0
    try:
        import psutil

        vm = psutil.virtual_memory()
        total_ram, available_ram = vm.total, vm.available
    except ImportError:  # pragma: no cover - psutil is a hard dep, this is belt and braces
        log.warning("psutil unavailable; RAM checks skipped")

    vram_total = vram_free = 0
    gpu_name = "none"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_free, vram_total = torch.cuda.mem_get_info(0)
    except Exception as exc:  # noqa: BLE001 - a broken CUDA install must not mask the check
        log.warning("could not query VRAM: %s", exc)

    return Resources(free_disk, total_ram, available_ram, vram_total, vram_free, gpu_name)


def _fail(what: str, need: int, have: int, hint: str) -> None:
    raise ResourceError(
        f"insufficient {what}: need {need / GB:.1f} GB, have {have / GB:.1f} GB "
        f"(short by {(need - have) / GB:.1f} GB). {hint}"
    )


def require(
    *,
    disk_gb: float = 0.0,
    ram_gb: float = 0.0,
    vram_gb: float = 0.0,
    path: str | Path = ".",
    what: str = "stage",
) -> Resources:
    """Assert headroom for a stage. Raises :class:`ResourceError` with the shortfall."""
    r = probe(path)
    logutil.event(
        log,
        "resource check",
        stage=what,
        need_disk_gb=disk_gb,
        need_ram_gb=ram_gb,
        need_vram_gb=vram_gb,
        have=r.describe(),
    )

    if disk_gb and r.free_disk < disk_gb * GB:
        _fail(
            f"disk at {Path(path).resolve()}",
            int(disk_gb * GB),
            r.free_disk,
            "Free space or point --run-dir / model paths at another volume.",
        )
    if ram_gb and r.available_ram < ram_gb * GB:
        _fail(
            "system RAM",
            int(ram_gb * GB),
            r.available_ram,
            "Close other processes; this stage streams but still needs a working set.",
        )
    if vram_gb:
        if r.vram_total == 0:
            raise ResourceError(
                f"{what} needs {vram_gb:.1f} GB of VRAM but no CUDA device was found. "
                f"Install a CUDA-enabled torch build or run this stage on a GPU machine."
            )
        if r.vram_free < vram_gb * GB:
            _fail(
                f"VRAM on {r.gpu_name}",
                int(vram_gb * GB),
                r.vram_free,
                "Free the GPU (close other CUDA processes) or lower batch/sequence length.",
            )
    return r


def require_disk_for_checkpoint(
    path: str | Path, params: float, *, dtype_bytes: int = 2, slack: float = 1.15
) -> None:
    """Assert room to write a checkpoint of ``params`` parameters at ``dtype_bytes`` each."""
    need = params * dtype_bytes * slack
    r = probe(path)
    if r.free_disk < need:
        _fail(
            f"disk at {Path(path).resolve()}",
            int(need),
            r.free_disk,
            f"Writing {params / 1e9:.2f}B params at {dtype_bytes}B each needs "
            f"{need / GB:.1f} GB including {int((slack - 1) * 100)}% slack.",
        )


# ---------------------------------------------------------------------------
# the bf16 residency rule
# ---------------------------------------------------------------------------


def assert_no_bf16_resident(model: Any, *, allow_params: int = 200_000_000) -> None:
    """Assert no stage is holding full-precision weights. (Section 1, rule 1)

    A 4-bit load leaves the big matmul weights as ``uint8`` blocks; what stays in bf16/fp32
    is norms, biases, embeddings kept out of the quantisation set, and -- deliberately --
    the DeltaNet recurrent state (``mamba_ssm_dtype: float32``). ``allow_params`` is the
    ceiling for that legitimate residue.

    Raises if a large tensor came back at 16 or 32 bits, which means the quantisation config
    was ignored and the run is about to OOM or silently thrash to disk.
    """
    import torch

    wide = (torch.bfloat16, torch.float16, torch.float32, torch.float64)
    resident = 0
    worst: list[tuple[str, int, str]] = []
    for name, p in model.named_parameters():
        if p.dtype in wide:
            resident += p.numel()
            worst.append((name, p.numel(), str(p.dtype)))

    if resident > allow_params:
        worst.sort(key=lambda t: -t[1])
        top = "\n".join(f"    {n}  {c / 1e6:.1f}M  {d}" for n, c, d in worst[:8])
        raise ResourceError(
            f"bf16/fp32 residency violation: {resident / 1e9:.2f}B parameters are held at "
            f"full precision (ceiling {allow_params / 1e6:.0f}M). No stage in this pipeline "
            f"may hold bf16 weights resident -- the full model is "
            f"{BF16_27B_BYTES / GB:.1f} GB and neither VRAM (16 GB) nor RAM (32 GB) can "
            f"take it.\n  largest offenders:\n{top}\n"
            f"  Check that the 4-bit quantization config was actually applied "
            f"(bitsandbytes installed, load_in_4bit=True honoured by this transformers "
            f"version)."
        )
    logutil.event(
        log,
        "bf16 residency ok",
        resident_m=round(resident / 1e6, 1),
        ceiling_m=round(allow_params / 1e6, 1),
    )


# ---------------------------------------------------------------------------
# disk budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetItem:
    """One artefact the pipeline puts on disk."""

    rung: str
    what: str
    bytes_: int
    #: False for artefacts that can be deleted once the stage that produced them is done.
    persists: bool

    @property
    def gb(self) -> float:
        return self.bytes_ / GB


def disk_budget(rungs: list[tuple[str, float, int]]) -> list[BudgetItem]:
    """Itemise disk use for a ladder.

    ``rungs`` is ``[(name, params_b, teacher_tokens), ...]`` in ladder order. Sizes are derived
    from the parameter counts rather than transcribed, so the second rung is accounted for
    properly: its parent is the *merged* first-rung checkpoint, which is an additional
    full-size safetensors tree that has to coexist with everything else.
    """
    items: list[BudgetItem] = []
    parent_b = rungs[0][1] if rungs else 0.0

    # Rung 0: the original parent, plus the artefacts every later comparison depends on.
    items += [
        BudgetItem("parent", "bf16 safetensors", int(parent_b * 1e9 * 2), True),
        BudgetItem("parent", "bf16 GGUF (KL reference source)", int(parent_b * 1e9 * 2), False),
        BudgetItem("parent", "reference.kld", 2 * GB, True),
        BudgetItem("parent", "stage-0 quant candidates (8 x ~10 GB)", 80 * GB, False),
    ]

    for name, params_b, teacher_tokens in rungs[1:] if len(rungs) > 1 else []:
        w = int(params_b * 1e9 * 2)
        items += [
            BudgetItem(name, "unhealed safetensors", w, False),
            BudgetItem(
                name,
                f"teacher cache ({teacher_tokens / 1e6:.0f}M x top-16)",
                teacher_tokens * 100,
                False,
            ),
            BudgetItem(name, "LoRA checkpoints", 2 * GB, False),
            # This is the one the brief's 187 GB estimate missed: the merged checkpoint is
            # not a transient, it is the next rung's parent and must survive.
            BudgetItem(name, "merged safetensors (= next rung's parent)", w, True),
            BudgetItem(name, "bf16 GGUF", w, False),
            BudgetItem(name, "quant candidates (2 x ~10 GB)", 20 * GB, False),
        ]
    return items


def budget_totals(items: list[BudgetItem]) -> dict[str, float]:
    """Peak with and without cleaning transients, in GB."""
    persist = sum(i.bytes_ for i in items if i.persists)
    transient = [i.bytes_ for i in items if not i.persists]
    return {
        "total_if_nothing_deleted": (persist + sum(transient)) / GB,
        "persistent": persist / GB,
        # Cleaning as you go still needs room for the single largest transient alongside it.
        "peak_if_cleaned": (persist + max(transient, default=0)) / GB,
    }


def default_rungs() -> list[tuple[str, float, int]]:
    """The shipped ladder: 27B parent -> 22B -> 18B."""
    return [("parent 27B", 26.8961, 0), ("rung 1: 22B", 22.2968, 50_000_000),
            ("rung 2: 18B", 18.0807, 100_000_000)]


def check_bitsandbytes() -> None:
    """Fail early and specifically when NF4 is unavailable."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ResourceError(
            "bitsandbytes is not installed, so NF4 4-bit loading is unavailable. This stage "
            "cannot fall back to bf16: the full model is "
            f"{BF16_27B_BYTES / GB:.1f} GB against 16 GB of VRAM and 32 GB of RAM. "
            "Install it with: pip install 'marlowe[score]'"
        ) from exc
