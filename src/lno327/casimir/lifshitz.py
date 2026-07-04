"""Trace-log Lifshitz energy integrand."""

from __future__ import annotations

import numpy as np

from lno327.constants import C0
from lno327.electrodynamics.conductivity import ConductivityTensor

from .reflection import reflection_matrix_weak_2d
from .setup import CasimirSetup


def casimir_energy_integrand(
    setup: CasimirSetup,
    xi: float,
    k_parallel: float,
    phi: float,
    theta: float,
    left: ConductivityTensor,
    right: ConductivityTensor,
) -> complex:
    """Return the k-space integrand of the two-plate Casimir energy."""

    kappa = np.sqrt((xi / C0) ** 2 + k_parallel**2)
    r1 = reflection_matrix_weak_2d(xi, k_parallel, phi, left)
    r2 = reflection_matrix_weak_2d(xi, k_parallel, phi + theta, right)
    matrix = np.eye(2, dtype=complex) - r1 @ r2 * np.exp(-2.0 * kappa * setup.distance)
    return k_parallel * np.log(np.linalg.det(matrix))
