#!/usr/bin/env python3
"""Thin wrapper so stages can be run without installing the package.

    python scripts/run_stage.py stage1-smoke --config configs/marlowe-22b.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marlowe.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
