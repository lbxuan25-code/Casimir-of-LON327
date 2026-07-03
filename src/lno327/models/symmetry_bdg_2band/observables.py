"""Lightweight data helpers for the symmetry-focused two-band model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lno327.models.projection import band_project_pairing
from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec


def _spec_or_default(spec: SymmetryBdG2BandSpec | None) -> SymmetryBdG2BandSpec:
    return spec or SymmetryBdG2BandSpec()


def normal_band_energies(kx: float, ky: float, spec: SymmetryBdG2BandSpec | None = None) -> np.ndarray:
    model = _spec_or_default(spec)
    return np.linalg.eigvalsh(model.normal_hamiltonian(kx, ky))


def band_energies_on_path(
    spec: SymmetryBdG2BandSpec,
    k_path: Sequence[tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    points = np.asarray(k_path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("k_path must have shape (n, 2)")
    return np.asarray([normal_band_energies(float(kx), float(ky), spec) for kx, ky in points])


def bdg_energies(
    kx: float,
    ky: float,
    channel: str,
    spec: SymmetryBdG2BandSpec | None = None,
) -> np.ndarray:
    model = _spec_or_default(spec)
    return np.linalg.eigvalsh(model.bdg_hamiltonian(kx, ky, channel))


def min_positive_bdg_energy(
    kx: float,
    ky: float,
    channel: str,
    spec: SymmetryBdG2BandSpec | None = None,
) -> float:
    energies = bdg_energies(kx, ky, channel, spec)
    positive = energies[energies > 1e-12]
    if positive.size == 0:
        return 0.0
    return float(np.min(positive))


def gap_value(
    kx: float,
    ky: float,
    channel: str,
    spec: SymmetryBdG2BandSpec | None = None,
) -> np.ndarray:
    model = _spec_or_default(spec)
    return model.pairing_matrix(kx, ky, channel)


def band_projected_gap(
    kx: float,
    ky: float,
    channel: str,
    spec: SymmetryBdG2BandSpec | None = None,
    *,
    gauge: str = "anchor",
) -> np.ndarray:
    model = _spec_or_default(spec)
    delta = model.pairing_matrix(kx, ky, channel)
    _, states_k = np.linalg.eigh(model.normal_hamiltonian(kx, ky))
    _, states_minus_k = np.linalg.eigh(model.normal_hamiltonian(-kx, -ky))
    return band_project_pairing(delta, states_k, states_minus_k, gauge=gauge)
