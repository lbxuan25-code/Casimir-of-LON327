"""Collective-channel Schur corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SchurResult:
    corrected_response: np.ndarray
    condition_number: float | None
    inverse_method: Literal["inv", "pinv_diagnostic", "not_used"]
    status: str
    warning: str | None = None


class BdGPhaseCorrectionError(RuntimeError):
    """Raised when the global phase channel is singular."""


def apply_phase_only_schur(
    bare_response: np.ndarray,
    phase_left: np.ndarray,
    phase_phase_total: complex,
    phase_right: np.ndarray,
    *,
    sign: Literal["minus", "plus"] = "minus",
) -> SchurResult:
    """Apply the existing one-channel phase Schur correction."""

    bare = np.asarray(bare_response, dtype=complex)
    left = np.asarray(phase_left, dtype=complex)
    right = np.asarray(phase_right, dtype=complex)
    kernel = complex(phase_phase_total)
    if sign not in {"minus", "plus"}:
        raise ValueError("sign must be 'minus' or 'plus'")
    if abs(kernel) <= 0.0:
        return SchurResult(
            corrected_response=bare.copy(),
            condition_number=None,
            inverse_method="not_used",
            status="skipped_zero_phase_kernel",
            warning="phase_phase_total is zero; phase-only Schur correction was skipped",
        )
    schur_term = np.outer(left, right) / kernel
    corrected = bare - schur_term if sign == "minus" else bare + schur_term
    return SchurResult(
        corrected_response=corrected,
        condition_number=None,
        inverse_method="not_used",
        status=f"{sign}_phase_schur_applied",
    )


def apply_amplitude_phase_schur(
    bare_response: np.ndarray,
    em_collective_left: np.ndarray,
    collective_total: np.ndarray,
    collective_em_right: np.ndarray,
    *,
    condition_threshold: float = 1e12,
) -> SchurResult:
    """Apply K_AA - K_Aeta inv(K_etaeta) K_etaA with diagnostic pinv fallback."""

    bare = np.asarray(bare_response, dtype=complex)
    left = np.asarray(em_collective_left, dtype=complex)
    kernel = np.asarray(collective_total, dtype=complex)
    right = np.asarray(collective_em_right, dtype=complex)
    condition = float(np.linalg.cond(kernel))
    if not np.isfinite(condition) or condition > condition_threshold:
        kernel_inv = np.linalg.pinv(kernel)
        inverse_method: Literal["inv", "pinv_diagnostic", "not_used"] = "pinv_diagnostic"
        status = "applied_with_pinv_diagnostic"
        warning = f"collective_total condition number {condition:.3e} exceeds threshold {condition_threshold:.3e}"
    else:
        kernel_inv = np.linalg.inv(kernel)
        inverse_method = "inv"
        status = "applied"
        warning = None
    return SchurResult(
        corrected_response=bare - left @ kernel_inv @ right,
        condition_number=condition,
        inverse_method=inverse_method,
        status=status,
        warning=warning,
    )
