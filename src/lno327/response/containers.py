"""Small containers for model-independent response algebra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandBasisEigensystem:
    energies: np.ndarray
    states: np.ndarray
    occupations: np.ndarray
    negative_fermi_derivative: np.ndarray


@dataclass(frozen=True)
class KernelComponents:
    paramagnetic: np.ndarray
    diamagnetic: np.ndarray
    total: np.ndarray
