"""Pure linear algebra Schur complement helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetSchurResult:
    effective: np.ndarray
    correction: np.ndarray
    etaeta_condition_number: float
    solve_method: str


def schur_effective(
    k_ss: np.ndarray,
    k_seta: np.ndarray,
    k_etaeta: np.ndarray,
    k_etas: np.ndarray,
    *,
    condition_threshold: float = 1e12,
) -> TargetSchurResult:
    """Return K_SS - K_Seta solve(K_etaeta, K_etaS)."""

    ss = np.asarray(k_ss, dtype=complex)
    seta = np.asarray(k_seta, dtype=complex)
    etaeta = np.asarray(k_etaeta, dtype=complex)
    etas = np.asarray(k_etas, dtype=complex)
    condition = float(np.linalg.cond(etaeta))
    if not np.isfinite(condition) or condition > float(condition_threshold):
        solved = np.linalg.pinv(etaeta) @ etas
        method = "pinv_diagnostic"
    else:
        solved = np.linalg.solve(etaeta, etas)
        method = "solve"
    correction = seta @ solved
    return TargetSchurResult(effective=ss - correction, correction=correction, etaeta_condition_number=condition, solve_method=method)

