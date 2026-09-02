"""Driver-level GPU memory, read from the adapter rather than from torch.

Why this module exists
----------------------

On Windows/WDDM, exceeding VRAM does not raise. The driver silently backs the excess with
host memory and the job keeps running, roughly an order of magnitude slower. Neither of the
instruments this project had could see that reliably:

* **torch accounting** counts the host-backed reservation as though it were resident. It
  does push the total past the card's capacity, so a large overshoot shows up as negative
  headroom -- but it cannot separate "reserved and resident" from "reserved and paged", and
  the allocator's slack (5.4 GB in the measured 22B case) sits inside the same number.
* **throughput** was believed to be the fit signal. It is not. A probe measured 74.5 tok/s
  at seq 2048 while 6.8 GB over the card; had throughput been the gate, that configuration
  would have been accepted. Paging cost varies with *which* pages get evicted, so a job can
  be badly over-committed and still look fast for a few steps.

The adapter's ``Shared Usage`` performance counter is a direct readout of bytes the driver
has spilled to host memory. Idle it sits around 90 MB; the 22B student at any sequence
length in the ladder drove it to 8.3 GB. That is the gate.

Sampling runs in a side process for the duration of the probe, so it adds nothing to the
measured step time -- a per-step counter read costs 1-2 s against a 10 s step and would
corrupt the throughput number it sits beside.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from marlowe import logutil

log = logutil.get("gpumem")

#: Bytes of driver spill, above the idle baseline, that count as paging.
#:
#: The idle floor is desktop compositor surfaces and moves around by a few tens of MB;
#: 250 MB clears that comfortably while being ~30x below the smallest real overshoot
#: observed (8.25 GB). There is no ambiguous middle in the measurements to date: a
#: configuration either sits at the floor or is gigabytes over it.
PAGING_THRESHOLD_BYTES = 250_000_000

_SHARED = r"\GPU Adapter Memory(*)\Shared Usage"
_DEDICATED = r"\GPU Adapter Memory(*)\Dedicated Usage"


@dataclass
class PagingReport:
    """What the driver did with memory while the probe ran."""

    #: Peak host-backed bytes on the busiest adapter, minus that adapter's own floor.
    paged_bytes: int = 0
    shared_peak_bytes: int = 0
    shared_floor_bytes: int = 0
    dedicated_peak_bytes: int = 0
    samples: int = 0
    #: True only when the counter was actually read. Defaults to False so that a report
    #: nobody populated cannot silently assert "the driver saw no paging" -- with a True
    #: default, every probe built without a sampler would pass the fit gate.
    available: bool = False

    @property
    def paging(self) -> bool:
        return self.available and self.paged_bytes > PAGING_THRESHOLD_BYTES

    def render(self) -> str:
        if not self.available:
            return "  driver paging  UNAVAILABLE -- fit falls back to torch accounting"
        verdict = "PAGING" if self.paging else "resident"
        return (
            f"  driver paging  {self.paged_bytes / 1e9:8.2f} GB spilled to host  [{verdict}]"
            f"\n                 (shared {self.shared_peak_bytes / 1e9:.2f} GB peak over a "
            f"{self.shared_floor_bytes / 1e9:.2f} GB floor, dedicated "
            f"{self.dedicated_peak_bytes / 1e9:.2f} GB peak, {self.samples} samples)"
        )


class PagingSampler:
    """Sample the adapter counters in a side process for the life of a probe.

    Start it *before* the model loads: the earliest samples are taken while this process
    holds nothing on the device, which is what makes the floor self-calibrating. That also
    removes the need to identify the discrete adapter up front -- whichever instance shows
    the largest dedicated usage over the run is the one doing the work.

    **Reads typeperf's stdout, not its ``-o`` file.** The file variant never produced a single
    sample: typeperf buffers into the output file and ``terminate()`` kills it before any
    flush, so the file was still zero bytes after a five-second run. Every probe therefore
    reported ``available=False`` and fell back to torch accounting -- the instrument this
    module exists to replace. stdout is line-buffered and drained by a reader thread, so the
    samples already collected survive termination.
    """

    def __init__(self, interval_s: int = 2) -> None:
        self._interval = interval_s
        self._proc: subprocess.Popen[str] | None = None
        self._lines: list[str] = []
        self._thread: threading.Thread | None = None

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.append(line)

    def start(self) -> None:
        if sys.platform != "win32":
            return
        try:
            self._proc = subprocess.Popen(
                ["typeperf", _SHARED, _DEDICATED, "-si", str(self._interval), "-sc", "100000"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except OSError:
            self._proc = None
            return
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def stop(self) -> PagingReport:
        if self._proc is None:
            return PagingReport(available=False)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return _parse_rows(self._lines)


def _parse_rows(lines: list[str]) -> PagingReport:
    """Reduce typeperf CSV lines to one adapter's peak spill above its own floor."""
    rows = [r for r in csv.reader(lines) if r]
    if len(rows) < 2:
        return PagingReport(available=False)

    header = rows[0]
    # typeperf emits one column per (counter, instance) pair. Pair them up by instance.
    shared_cols: dict[str, int] = {}
    dedicated_cols: dict[str, int] = {}
    for i, col in enumerate(header[1:], start=1):
        inst = col.split("(")[-1].split(")")[0] if "(" in col else col
        if col.rstrip('"').endswith("Shared Usage"):
            shared_cols[inst] = i
        elif col.rstrip('"').endswith("Dedicated Usage"):
            dedicated_cols[inst] = i

    def series(idx: int) -> list[float]:
        out = []
        for row in rows[1:]:
            if len(row) > idx:
                try:
                    out.append(float(row[idx]))
                except ValueError:
                    continue
        return out

    best: PagingReport | None = None
    for inst, d_idx in dedicated_cols.items():
        ded = series(d_idx)
        if not ded:
            continue
        s_idx = shared_cols.get(inst)
        sh = series(s_idx) if s_idx is not None else []
        if not sh:
            continue
        # The floor is this adapter's own quietest sample, which is the pre-load state
        # because the sampler starts before the model is loaded.
        floor = min(sh)
        cand = PagingReport(
            paged_bytes=int(max(sh) - floor),
            shared_peak_bytes=int(max(sh)),
            shared_floor_bytes=int(floor),
            dedicated_peak_bytes=int(max(ded)),
            samples=len(sh),
            available=True,
        )
        # The adapter under test is the one that actually held the model.
        if best is None or cand.dedicated_peak_bytes > best.dedicated_peak_bytes:
            best = cand
    return best or PagingReport(available=False)




# ---------------------------------------------------------------------------
# proving the instrument works, in this session, before trusting its verdict
# ---------------------------------------------------------------------------

#: Torch device headroom, in GB, below which a configuration is *reported* as tight.
#:
#: **Advisory. Not a gate.** It replaced a 1.0 GB hard rule that had to go: torch accounting
#: reported 0.11 GB of "headroom" for rank32-seq1024, a configuration the driver counter
#: showed spilling 7.51 GB to host at 216 tok/s -- the fastest rung measured. Headroom and
#: paging can disagree completely, and only one of them saw that.
ADVISORY_HEADROOM_GB = 0.3

#: Set by :func:`validate_sampler` when the counter has demonstrably detected a real spill in
#: this process. Deliberately session-scoped and not persisted: a sampler that worked
#: yesterday, on a different driver or a different desktop composition, is not evidence.
_VALIDATION: PagingReport | None = None


class SamplerNotValidated(RuntimeError):
    """The paging gate was asked for a verdict before the counter proved it can see."""


def sampler_validation() -> PagingReport | None:
    return _VALIDATION


def validate_sampler(over_commit_bytes: int = 2_000_000_000, settle_s: float = 6.0) -> PagingReport:
    """Force a real spill and confirm the counter reports it. Returns the proof.

    A detector only ever run against configurations that do not page has not been shown to
    detect anything, and this one has twice been believed working while returning nothing.
    So the gate requires evidence from the current session rather than from the code's
    reputation.

    The known-positive is deliberate over-commitment: on WDDM, allocating past VRAM does not
    raise -- the driver backs the excess with host memory, which is precisely the condition
    being detected. Costs seconds and needs no model, unlike the other known-positive
    (``MARLOWE_CPU_GATHER_EMBED=0``, which restages a 2.54 GB embedding table every forward
    and reported 11.585 GB spilled across 103 samples).
    """
    global _VALIDATION
    import time

    import torch

    sampler = PagingSampler(interval_s=1)
    sampler.start()
    time.sleep(settle_s / 3)
    free, _ = torch.cuda.mem_get_info(0)
    n = (free + over_commit_bytes) // 2
    hog = None
    try:
        hog = torch.empty(int(n), dtype=torch.float16, device="cuda")
        hog.fill_(0)
        torch.cuda.synchronize()
        time.sleep(settle_s)
    except (RuntimeError, torch.OutOfMemoryError):
        # The allocator refused instead of the driver spilling -- which is what the cap is
        # for, and means this box does not reproduce the condition by over-commitment.
        pass
    finally:
        del hog
        torch.cuda.empty_cache()
    report = sampler.stop()

    if not report.available:
        raise SamplerNotValidated(
            "the paging counter returned no samples during validation, so it cannot be "
            "trusted to report a clean run either. Check that typeperf runs and that "
            r"'\GPU Adapter Memory(*)\Shared Usage' exists on this machine."
        )
    if not report.paging:
        raise SamplerNotValidated(
            f"validation over-committed VRAM by {over_commit_bytes / 1e9:.1f} GB and the "
            f"counter reported only {report.paged_bytes / 1e6:.0f} MB spilled. A counter that "
            f"cannot see a deliberate spill cannot certify that a training config is "
            f"resident. Verdicts from it would be no-verdicts wearing a pass."
        )
    _VALIDATION = report
    logutil.event(
        log,
        "paging sampler validated",
        paged_gb=round(report.paged_bytes / 1e9, 2),
        samples=report.samples,
    )
    return report


#: Cap torch's allocator at the VRAM that physically exists, so over-commitment raises
#: instead of silently paging.
#:
#: On WDDM the driver never refuses: it backs the excess with host memory and the job runs
#: ~10x slow. The caching allocator therefore never sees an allocation fail, never flushes
#: its cache, and never retries -- which is why the reserved pool grows to 18.57 GB while
#: live tensors peak at 15.62 GB and stay there. On Linux that same workload would OOM,
#: empty_cache, retry, and compact.
#:
#: The fraction is derived, not guessed. torch's cap applies to its own allocator, while
#: the CUDA context and other processes sit outside it, so the usable fraction is
#: (total - context) / total. A hardcoded 0.88 would cap at 15.11 GB -- below the measured
#: 15.62 GB of live tensors -- and OOM on a configuration that fits.
_MEMORY_FRACTION_ENV = "MARLOWE_CUDA_MEMORY_FRACTION"


def cap_process_memory(context_bytes: int | None = None) -> float | None:
    """Limit torch to the free VRAM. Returns the fraction applied, or None if disabled.

    Set ``MARLOWE_CUDA_MEMORY_FRACTION=0`` to disable, or to a float to override.
    """
    import torch

    # Opt-OUT, not opt-in.
    #
    # It was opt-in, which meant the safety net existed and was not deployed: every probe in
    # this project ran with MARLOWE_CUDA_MEMORY_FRACTION unset, so cap_process_memory returned
    # None and never applied. rank32-seq1024 then paged 7.51 GB while torch reported 0.11 GB
    # of headroom. A mechanism that is present but inactive by default is the same defect
    # class as a guard that fails open.
    #
    # Cap and counter are belt and braces, and they see different things: the cap makes torch
    # over-commitment raise instead of silently paging, and the counter catches the spill the
    # cap cannot see -- host-backed memory the driver hands out below torch's own ceiling.
    raw = os.environ.get(_MEMORY_FRACTION_ENV)
    if raw is not None:
        try:
            fraction = float(raw)
        except ValueError:
            log.warning("%s=%r is not a float; ignoring", _MEMORY_FRACTION_ENV, raw)
            return None
        if fraction <= 0:
            log.warning("%s=%s disables the allocator cap explicitly", _MEMORY_FRACTION_ENV, raw)
            return None
    else:
        total = torch.cuda.get_device_properties(0).total_memory
        if context_bytes is None:
            # Nobody measured the context, so derive it: whatever is already gone from a card
            # this process has not allocated on is context plus other tenants.
            free_now, _ = torch.cuda.mem_get_info(0)
            context_bytes = max(0, total - free_now - torch.cuda.memory_reserved(0))
        fraction = max(0.5, (total - context_bytes) / total)

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    ceiling = fraction * total_bytes
    reserved = torch.cuda.memory_reserved(0)
    if reserved > ceiling:
        # Fail here, where the cause is legible. A cap set below what the allocator already
        # holds does not shrink it -- the next allocation that needs to grow simply fails,
        # arbitrarily far away, and reads as the training configuration not fitting.
        raise RuntimeError(
            f"cannot cap the allocator at {ceiling / 1e9:.2f} GB: it already holds "
            f"{reserved / 1e9:.2f} GB reserved. Call torch.cuda.empty_cache() first, or cap "
            f"before the allocation that grew it. Capping now would OOM on the next growth "
            f"allocation rather than here."
        )
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    total_gb = total_bytes / 1e9
    logutil.event(
        log,
        "allocator capped",
        fraction=round(fraction, 3),
        ceiling_gb=round(fraction * total_gb, 2),
    )
    return fraction
