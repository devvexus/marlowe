"""Marlowe: structure-aware depth pruning and distillation for hybrid attention stacks."""

from __future__ import annotations

import os
import sys

# Set before torch is imported anywhere, or the allocator is already configured.
#
# expandable_segments lets the caching allocator grow a segment in place instead of
# reserving fixed-size blocks it can never merge, which would reduce fragmentation at the
# source.
#
# **It does nothing on this machine.** Windows/WDDM builds reject it in torch's C++ config
# parser -- every run prints "expandable_segments not supported on this platform" and
# torch.cuda.get_allocator_backend() reports "native" (checked on torch 2.5.1+cu121). It is
# left set because it is correct on Linux and costs nothing, but no reasoning about
# fragmentation here may assume it is active: the measured 2.95 GB of fragmentation during
# training accrued with this flag set. Fragmentation is controlled instead by
# gpumem.cap_process_memory, which forces the allocator to flush and retry rather than grow
# -- on WDDM the driver never refuses an allocation, so without a cap the allocator never
# learns it should compact.
#
# The two settings that ARE supported here target the observed symptom directly. The native
# allocator refused a 2 MB allocation while holding 1.42 GB reserved-but-unallocated: free
# space trapped inside partially-used segments, which it can only release when a segment is
# entirely free, and which it never compacts.
#
#   max_split_size_mb:512          stop splitting cached blocks larger than this to serve
#                                  small requests -- that splitting is how large blocks decay
#                                  into unusable fragments over 52 layers of churn.
#   garbage_collection_threshold   reclaim unused cached blocks once reserved passes this
#                                  fraction of the cap, rather than waiting for a failure.
_ALLOC_CONF = ["max_split_size_mb:512", "garbage_collection_threshold:0.8"]
if sys.platform != "win32":
    _ALLOC_CONF.insert(0, "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ",".join(_ALLOC_CONF))

__version__ = "0.1.0"
