"""Validation compatibility facade for the production finite-q model adapter."""
from __future__ import annotations

from lno327.casimir.microscopic_model import (
    FiniteQMicroscopicModel,
    available_finite_q_microscopic_models,
    get_finite_q_microscopic_model,
)

FiniteQValidationModel = FiniteQMicroscopicModel


def available_finite_q_validation_models() -> tuple[str, ...]:
    """Return model names through the historical validation API."""

    return available_finite_q_microscopic_models()


def get_finite_q_validation_model(name: str) -> FiniteQValidationModel:
    """Return the production model through the historical validation API."""

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
