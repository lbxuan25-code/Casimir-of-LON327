#!/usr/bin/env python3
"""Compatibility wrapper for scripts/normal_state/compute_normal_state_conductivity_real.py."""

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "normal_state" / "compute_normal_state_conductivity_real.py"),
    run_name="__main__",
)
