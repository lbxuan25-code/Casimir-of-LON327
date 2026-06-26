"""Pure Ward-identity validation helpers for response tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ward_response import physical_ward_residuals


@dataclass(frozen=True)
class WardValidationReport:
    left_residual: np.ndarray
    right_residual: np.ndarray
    left_norm: float
    right_norm: float
    passed: bool
    tolerance: float
    ward_vectors: dict[str, Any]
    notes: tuple[str, ...] = ()


def validate_physical_ward_identity(
    response: np.ndarray,
    omega_eV: float,
    q_model: np.ndarray,
    *,
    tolerance: float = 1e-8,
    notes: tuple[str, ...] = (),
) -> WardValidationReport:
    """Evaluate Ward residuals without mutating or repairing ``response``."""

    matrix = np.asarray(response)
    q = np.asarray(q_model, dtype=float)
    left, right = physical_ward_residuals(matrix, omega_eV, q)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    return WardValidationReport(
        left_residual=left.copy(),
        right_residual=right.copy(),
        left_norm=left_norm,
        right_norm=right_norm,
        passed=bool(max(left_norm, right_norm) <= tolerance),
        tolerance=float(tolerance),
        ward_vectors={
            "left_convention": "[i*omega, qx, qy]",
            "right_convention": "physical_ward_residuals convention",
            "omega_eV": float(omega_eV),
            "q_model": q.tolist(),
        },
        notes=tuple(notes),
    )
