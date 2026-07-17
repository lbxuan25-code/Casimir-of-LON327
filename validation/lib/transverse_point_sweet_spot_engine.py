"""Validation facade for the production fixed point engine."""
from __future__ import annotations

from typing import Sequence

from lno327.casimir import fixed_transverse_point_engine as _production

for _name in dir(_production):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_production, _name))


def _parse_args(argv: Sequence[str] | None = None):
    """Delegate parsing while preserving the historical monkeypatch surface.

    Validation tests and downstream diagnostics have historically monkeypatched
    ``validation.lib.transverse_point_sweet_spot_engine.affinity_cpu_count``.
    Temporarily synchronize that injected dependency into the production module;
    all parsing and worker-budget logic still executes in production, and the
    production global is restored before returning.
    """

    original = _production.affinity_cpu_count
    _production.affinity_cpu_count = affinity_cpu_count
    try:
        return _production._parse_args(argv)
    finally:
        _production.affinity_cpu_count = original
