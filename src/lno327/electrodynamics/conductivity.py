"""Conductivity tensor tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConductivityTensor:
    """2D optical conductivity tensor at a fixed imaginary frequency."""

    xx: complex
    yy: complex
    xy: complex = 0.0
    yx: complex = 0.0

    def matrix(self) -> np.ndarray:
        return np.array([[self.xx, self.xy], [self.yx, self.yy]], dtype=complex)


def rotate_conductivity(conductivity: ConductivityTensor, phi: float) -> ConductivityTensor:
    """Rotate a 2D conductivity tensor counter-clockwise by phi."""

    c = np.cos(phi)
    s = np.sin(phi)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    rotated = rotation @ conductivity.matrix() @ rotation.T
    return ConductivityTensor(rotated[0, 0], rotated[1, 1], rotated[0, 1], rotated[1, 0])


def anisotropy_delta(conductivity: ConductivityTensor) -> complex:
    """Return delta = (sigma_xx - sigma_yy)/(sigma_xx + sigma_yy)."""

    denom = conductivity.xx + conductivity.yy
    if np.isclose(denom, 0.0):
        raise ValueError("sigma_xx + sigma_yy must be nonzero")
    return (conductivity.xx - conductivity.yy) / denom


def anisotropy_summary(conductivity: ConductivityTensor) -> dict[str, complex]:
    """Return compact diagnostics for a 2D conductivity tensor."""

    trace = conductivity.xx + conductivity.yy
    delta = anisotropy_delta(conductivity)
    hall_symmetric = 0.5 * (conductivity.xy + conductivity.yx)
    hall_antisymmetric = 0.5 * (conductivity.xy - conductivity.yx)
    return {
        "sigma_trace": trace,
        "delta": delta,
        "sigma_xy": conductivity.xy,
        "sigma_yx": conductivity.yx,
        "offdiag_symmetric": hall_symmetric,
        "offdiag_antisymmetric": hall_antisymmetric,
    }


def conductivity_matrix_diagnostics(conductivity: ConductivityTensor) -> dict[str, np.ndarray | complex | float]:
    """Return matrix diagnostics for a 2D conductivity tensor."""

    sigma_matrix = conductivity.matrix()
    eigenvalues, eigenvectors = np.linalg.eig(sigma_matrix)
    offdiag_norm = float(np.linalg.norm([sigma_matrix[0, 1], sigma_matrix[1, 0]]))
    relative_xx_yy_error = 0.0
    diagonal_scale = 0.5 * (abs(conductivity.xx) + abs(conductivity.yy))
    if not np.isclose(diagonal_scale, 0.0):
        relative_xx_yy_error = float(abs(conductivity.xx - conductivity.yy) / diagonal_scale)

    return {
        "sigma_matrix": sigma_matrix,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "anisotropy_delta": anisotropy_delta(conductivity),
        "offdiag_norm": offdiag_norm,
        "relative_xx_yy_error": relative_xx_yy_error,
    }
