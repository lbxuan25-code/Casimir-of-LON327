"""Isolated fixed-grid reference chain.

This module exists only for regression and historical fixed-chain comparisons.  It is
not the main Casimir calculation route and is intentionally not re-exported from the
package root.  New calculations must use ``lno327.casimir.run_full_casimir``.
"""
from __future__ import annotations

from .fixed_chain import (
    FixedCasimirConfig,
    FixedCasimirExecutionError,
    FixedCasimirResult,
    run_casimir as run_fixed_reference_casimir,
)

__all__ = [
    "FixedCasimirConfig",
    "FixedCasimirExecutionError",
    "FixedCasimirResult",
    "run_fixed_reference_casimir",
]
