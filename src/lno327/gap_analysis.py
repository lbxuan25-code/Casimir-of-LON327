"""BdG gap-structure diagnostics for minimal pairing channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conductivity import uniform_bz_mesh
from .model import normal_state_hamiltonian
from .pairing import PairingAmplitudes, PairingKind, pairing_matrix


@dataclass(frozen=True)
class FermiSurfacePoints:
    """Approximate normal-state Fermi-surface points from a uniform mesh."""

    kx: np.ndarray
    ky: np.ndarray
    band_index: np.ndarray
    energy_eV: np.ndarray

    def __len__(self) -> int:
        return self.kx.size


@dataclass(frozen=True)
class GapStatistics:
    """Projected gap data and compact statistics on approximate FS points."""

    kx: np.ndarray
    ky: np.ndarray
    band_index: np.ndarray
    energy_eV: np.ndarray
    gap_complex: np.ndarray
    gap_abs: np.ndarray
    gap_sign: np.ndarray
    node_tolerance_eV: float
    min_abs_gap: float
    max_abs_gap: float
    mean_abs_gap: float
    sign_changes: bool
    approximate_nodes: int
    relative_node_fraction: float


def band_gap_projection(
    kx: float,
    ky: float,
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
) -> np.ndarray:
    """Project Delta(k) into the normal-state band basis.

    Returns one complex projected gap per normal-state band:
    Delta_n(k) = u_n(k)^dagger Delta(k) u_n(-k)^*.
    """

    _, states_k = np.linalg.eigh(normal_state_hamiltonian(kx, ky))
    _, states_minus_k = np.linalg.eigh(normal_state_hamiltonian(-kx, -ky))
    delta = pairing_matrix(pairing_kind, kx, ky, pairing_params)

    gaps = np.empty(states_k.shape[1], dtype=complex)
    for band in range(states_k.shape[1]):
        u_k = states_k[:, band]
        u_minus_k = states_minus_k[:, band]
        gaps[band] = u_k.conjugate().T @ delta @ u_minus_k.conjugate()
    return gaps


def fermi_surface_points(
    nk: int,
    energy_tolerance_eV: float,
    band_index: int | None = None,
) -> FermiSurfacePoints:
    """Return mesh points whose normal-state band energy lies near zero."""

    if energy_tolerance_eV <= 0.0:
        raise ValueError("energy_tolerance_eV must be positive")
    if band_index is not None and band_index not in range(4):
        raise ValueError("band_index must be between 0 and 3")

    kx_values: list[float] = []
    ky_values: list[float] = []
    bands: list[int] = []
    energies: list[float] = []

    for kx, ky in uniform_bz_mesh(nk):
        eigenvalues = np.linalg.eigvalsh(normal_state_hamiltonian(float(kx), float(ky)))
        candidate_bands = range(eigenvalues.size) if band_index is None else (band_index,)
        for band in candidate_bands:
            energy = float(eigenvalues[band])
            if abs(energy) <= energy_tolerance_eV:
                kx_values.append(float(kx))
                ky_values.append(float(ky))
                bands.append(int(band))
                energies.append(energy)

    return FermiSurfacePoints(
        kx=np.array(kx_values, dtype=float),
        ky=np.array(ky_values, dtype=float),
        band_index=np.array(bands, dtype=int),
        energy_eV=np.array(energies, dtype=float),
    )


def _gap_sign(gaps: np.ndarray, zero_tolerance_eV: float) -> np.ndarray:
    signs = np.zeros(gaps.shape, dtype=int)
    real_gaps = gaps.real
    signs[real_gaps > zero_tolerance_eV] = 1
    signs[real_gaps < -zero_tolerance_eV] = -1
    return signs


def gap_statistics_on_fermi_surface(
    pairing_kind: PairingKind,
    pairing_params: PairingAmplitudes | None = None,
    nk: int = 48,
    energy_tolerance_eV: float = 0.05,
    band_index: int | None = None,
    node_tolerance_eV: float = 1e-3,
) -> GapStatistics:
    """Return projected gap data and diagnostics on normal-state FS points."""

    if node_tolerance_eV < 0.0:
        raise ValueError("node_tolerance_eV must be non-negative")

    fs = fermi_surface_points(nk, energy_tolerance_eV, band_index)
    gap_values = np.empty(len(fs), dtype=complex)
    for index, (kx, ky, band) in enumerate(zip(fs.kx, fs.ky, fs.band_index, strict=True)):
        projected = band_gap_projection(float(kx), float(ky), pairing_kind, pairing_params)
        gap_values[index] = projected[int(band)]

    gap_abs = np.abs(gap_values)
    gap_sign = _gap_sign(gap_values, node_tolerance_eV)
    nonzero_signs = gap_sign[gap_sign != 0]
    sign_changes = bool(np.unique(nonzero_signs).size > 1)
    approximate_nodes = int(np.count_nonzero(gap_abs <= node_tolerance_eV))
    relative_node_fraction = float(approximate_nodes / gap_abs.size) if gap_abs.size else float("nan")

    if gap_abs.size == 0:
        min_abs_gap = max_abs_gap = mean_abs_gap = float("nan")
    else:
        min_abs_gap = float(np.min(gap_abs))
        max_abs_gap = float(np.max(gap_abs))
        mean_abs_gap = float(np.mean(gap_abs))

    return GapStatistics(
        kx=fs.kx,
        ky=fs.ky,
        band_index=fs.band_index,
        energy_eV=fs.energy_eV,
        gap_complex=gap_values,
        gap_abs=gap_abs,
        gap_sign=gap_sign,
        node_tolerance_eV=node_tolerance_eV,
        min_abs_gap=min_abs_gap,
        max_abs_gap=max_abs_gap,
        mean_abs_gap=mean_abs_gap,
        sign_changes=sign_changes,
        approximate_nodes=approximate_nodes,
        relative_node_fraction=relative_node_fraction,
    )


def gap_statistics_by_band(stats: GapStatistics) -> dict[int, dict[str, float | int]]:
    """Return gap statistics grouped by normal-state band index."""

    summary: dict[int, dict[str, float | int]] = {}
    for band in np.unique(stats.band_index):
        mask = stats.band_index == band
        band_gaps = stats.gap_abs[mask]
        approximate_nodes = int(np.count_nonzero(band_gaps <= stats.node_tolerance_eV))
        summary[int(band)] = {
            "count": int(np.count_nonzero(mask)),
            "min_abs_gap": float(np.min(band_gaps)) if band_gaps.size else float("nan"),
            "mean_abs_gap": float(np.mean(band_gaps)) if band_gaps.size else float("nan"),
            "max_abs_gap": float(np.max(band_gaps)) if band_gaps.size else float("nan"),
            "approximate_nodes": approximate_nodes,
        }
    return summary
