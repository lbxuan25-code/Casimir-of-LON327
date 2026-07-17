"""Validation facade for production point certification."""
from __future__ import annotations

from typing import Sequence

from lno327.casimir import fixed_transverse_point_certification as _production

for _name in dir(_production):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_production, _name))


def main(argv: Sequence[str] | None = None) -> None:
    # Preserve the historical validation monkeypatch surface while the
    # implementation and all numerical state live in production.
    _production.assess_frequency_level = assess_frequency_level
    _production.assess_oscillatory_envelope = assess_oscillatory_envelope
    _production.main(argv)
