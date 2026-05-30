#!/usr/bin/env python3
"""Compatibility wrapper for scripts/normal_state/inspect_normal_state_blocks.py."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shim import export_impl, load_impl

_IMPL = load_impl(__file__, "normal_state/inspect_normal_state_blocks.py")
export_impl(_IMPL, globals())

if __name__ == "__main__":
    _IMPL.main()
