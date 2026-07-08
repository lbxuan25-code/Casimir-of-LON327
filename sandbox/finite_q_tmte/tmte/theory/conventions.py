"""Central finite-q TM/TE sandbox conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASIS_NORMALIZATION = "unnormalized_gauge_orthogonal_tm_te"
SOURCE_ORDER_DIAGNOSTIC = ("G", "TM", "TE")
SOURCE_ORDER_PHYSICAL = ("TM", "TE")
XI_OMEGA_TOL = 1e-14


@dataclass(frozen=True)
class FiniteQConventions:
    q: np.ndarray
    q_norm: float
    qhat: np.ndarray
    that: np.ndarray
    xi: float
    g0: float
    gL: float
    basis_normalization: str = BASIS_NORMALIZATION


def finite_q_conventions(q_model: np.ndarray | tuple[float, float], xi: float, *, q_tol: float = 1e-14) -> FiniteQConventions:
    """Return q direction and sandbox v1 gauge coefficients."""

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_model must have shape (2,)")
    q_norm = float(np.linalg.norm(q))
    if q_norm < float(q_tol):
        raise ValueError("finite-q TM/TE target basis is undefined for q < q_tol")
    qhat = q / q_norm
    that = np.asarray([-qhat[1], qhat[0]], dtype=float)
    g0 = float(xi)
    gL = q_norm
    return FiniteQConventions(q=q, q_norm=q_norm, qhat=qhat, that=that, xi=float(xi), g0=g0, gL=gL)


def require_xi_matches_omega(xi: float, omega_eV: float, *, tol: float = XI_OMEGA_TOL) -> None:
    """Enforce sandbox v1 convention that target-basis xi equals Kubo omega."""

    if abs(float(omega_eV) - float(xi)) > float(tol):
        raise ValueError("sandbox v1 requires xi == omega_eV")


def require_diagnostic_source_order(source_order: tuple[str, ...]) -> None:
    """Require the v1 diagnostic order used by gauge diagnostics."""

    if tuple(source_order) != SOURCE_ORDER_DIAGNOSTIC:
        raise ValueError("sandbox v1 requires diagnostic source order ('G', 'TM', 'TE')")
