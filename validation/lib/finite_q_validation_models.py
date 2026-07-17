"""Shared finite-q model selection for independent validation diagnostics.

This module owns no microscopic implementation. It gives Ward/response validation
code stable validation-oriented names while delegating model construction entirely
to :mod:`lno327.casimir.microscopic_model`. The production Casimir chain does not
import this module.
"""
from __future__ import annotations

from lno327.casimir.microscopic_model import (
    FiniteQMicroscopicModel,
    available_finite_q_microscopic_models,
    get_finite_q_microscopic_model,
)

FiniteQValidationModel = FiniteQMicroscopicModel


def available_finite_q_validation_models() -> tuple[str, ...]:
    """Return model families available to finite-q validation diagnostics."""

    return available_finite_q_microscopic_models()


def get_finite_q_validation_model(name: str) -> FiniteQValidationModel:
    """Return the production-owned model adapter for a validation diagnostic."""

    try:
        return get_finite_q_microscopic_model(name)
    except ValueError as exc:
        choices = ", ".join(available_finite_q_validation_models())
        raise ValueError(
            f"unknown finite-q validation model {name!r}; choices: {choices}"
        ) from exc


__all__ = [
    "FiniteQValidationModel",
    "available_finite_q_validation_models",
    "get_finite_q_validation_model",
]
