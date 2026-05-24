#!/usr/bin/env python3
"""Compatibility wrapper for scripts/normal_state/inspect_band_structure.py."""

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "normal_state" / "inspect_band_structure.py"), run_name="__main__")
