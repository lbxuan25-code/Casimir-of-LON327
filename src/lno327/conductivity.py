"""Conductivity helpers for future Kubo and Casimir calculations."""

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


def kubo_placeholder(*_: object, **__: object) -> None:
    """Mark the future Kubo-response insertion point.

    Numerical conductivity evaluation is intentionally absent in this foundation
    layer because the theory corrections are still pending.
    """

    raise NotImplementedError("Kubo conductivity evaluation will be added after theory corrections.")
