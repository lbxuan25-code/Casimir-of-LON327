"""Lightweight data helpers for the LNO327 four-orbital model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec


def _spec_or_default(spec: LNO327FourOrbitalSpec | None) -> LNO327FourOrbitalSpec:
    return spec or LNO327FourOrbitalSpec()


def normal_band_energies(kx: float, ky: float, spec: LNO327FourOrbitalSpec | None = None) -> np.ndarray:
    model = _spec_or_default(spec)
    return np.linalg.eigvalsh(model.normal_hamiltonian(kx, ky))


def bdg_energies(
    kx: float,
    ky: float,
    channel: str,
    spec: LNO327FourOrbitalSpec | None = None,
) -> np.ndarray:
    model = _spec_or_default(spec)
    return np.linalg.eigvalsh(model.bdg_hamiltonian(kx, ky, channel))


def min_positive_bdg_energy(
    kx: float,
    ky: float,
    channel: str,
    spec: LNO327FourOrbitalSpec | None = None,
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
    spec: LNO327FourOrbitalSpec | None = None,
) -> np.ndarray:
    model = _spec_or_default(spec)
    return model.pairing_matrix(kx, ky, channel)


def band_projected_gap(
    kx: float,
    ky: float,
    channel: str,
    spec: LNO327FourOrbitalSpec | None = None,
) -> np.ndarray:
    model = _spec_or_default(spec)
    delta = model.pairing_matrix(kx, ky, channel)
    _, states_k = np.linalg.eigh(model.normal_hamiltonian(kx, ky))
    _, states_minus_k = np.linalg.eigh(model.normal_hamiltonian(-kx, -ky))
    return np.asarray(
        [
            states_k[:, band].conjugate().T @ delta @ states_minus_k[:, band].conjugate()
            for band in range(states_k.shape[1])
        ],
        dtype=complex,
    )


def band_energies_on_path(
    spec: LNO327FourOrbitalSpec,
    k_path: Sequence[tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    points = np.asarray(k_path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("k_path must have shape (n, 2)")
    return np.asarray([normal_band_energies(float(kx), float(ky), spec) for kx, ky in points])
