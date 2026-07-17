"""Validation facade for the production fixed point engine."""
from __future__ import annotations

from lno327.casimir import fixed_transverse_point_engine as _production

for _name in dir(_production):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_production, _name))
