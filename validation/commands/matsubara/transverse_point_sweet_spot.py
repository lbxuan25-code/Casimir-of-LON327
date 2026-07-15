"""Sole public CLI entry for universal transverse-point convergence."""
from __future__ import annotations

from typing import Sequence

from validation.lib import transverse_point_sweet_spot_command as _command
from validation.lib.transverse_point_sweet_spot_engine import (
    DEFAULT_OUTPUT,
    DEFAULT_SHIFTS,
)

DEFAULT_LOGDET_ATOL = _command.DEFAULT_LOGDET_ATOL
ENVELOPE_LEVELS = _command.ENVELOPE_LEVELS
assess_frequency_level = _command.assess_frequency_level
assess_oscillatory_envelope = _command.assess_oscillatory_envelope


def main(argv: Sequence[str] | None = None) -> None:
    # Preserve the public monkeypatch/test surface while keeping the numerical
    # implementation in non-runnable internal library modules.
    _command.assess_frequency_level = assess_frequency_level
    _command.assess_oscillatory_envelope = assess_oscillatory_envelope
    _command.main(argv)


__all__ = [
    "DEFAULT_LOGDET_ATOL",
    "DEFAULT_OUTPUT",
    "DEFAULT_SHIFTS",
    "ENVELOPE_LEVELS",
    "assess_frequency_level",
    "assess_oscillatory_envelope",
    "main",
]


if __name__ == "__main__":
    main()
