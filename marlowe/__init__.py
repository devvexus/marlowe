"""Marlowe: structure-aware depth pruning and distillation for hybrid attention stacks."""

from __future__ import annotations

import os

# Set before torch is imported anywhere, or the allocator is already configured.
#
# expandable_segments lets the caching allocator grow a segment in place instead of
# reserving fixed-size blocks it can never merge. Without it, a configuration that fits
# comfortably at step 10 can OOM at step 40,000 purely from fragmentation -- which is the
# failure this pipeline is least able to absorb, three days into an unattended run. It
# costs nothing and matters most in exactly the tight-margin case the memory search is
# there to find.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
