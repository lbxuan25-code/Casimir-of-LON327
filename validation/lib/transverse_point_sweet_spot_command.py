"""Validation facade for production point certification."""
from __future__ import annotations

from typing import Sequence

from lno327.casimir import fixed_transverse_point_certification as _production

for _name in dir(_production):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_production, _name))


def main(argv: Sequence[str] | None = None) -> None:
    """Run production certification through the historical validation API."""

    original_level = _production.assess_frequency_level
    original_envelope = _production.assess_oscillatory_envelope
    _production.assess_frequency_level = assess_frequency_level
    _production.assess_oscillatory_envelope = assess_oscillatory_envelope
    try:
        _production.main(argv)
    finally:
        _production.assess_frequency_level = original_level
        _production.assess_oscillatory_envelope = original_envelope
