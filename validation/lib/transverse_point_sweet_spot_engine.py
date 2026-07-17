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
    Synchronize that injected dependency into the production module before parsing;
    all numerical parsing and worker-budget logic still executes in production.
    """

    _production.affinity_cpu_count = affinity_cpu_count
    return _production._parse_args(argv)
