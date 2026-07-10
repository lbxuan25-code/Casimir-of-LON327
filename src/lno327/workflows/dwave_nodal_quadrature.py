"""Model-aware nodal quadrature for exact-static two-band d-wave response.

The uniform midpoint grid is a poor error controller for the exact-static
``d``-wave response because narrow nodal quasiparticle regions and shifted
near-degenerate transitions can move through the tensor grid as ``nk`` changes.
This module refines Brillouin-zone cells before any response block or Schur
complement is formed.

The returned points and normalized weights are intended to be passed directly
to the existing finite-q material workspace.  Consequently the primitive EM,
EM-collective, collective and Goldstone-counterterm integrals are merged over
one common quadrature before the single amplitude/phase Schur complement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327.response.finite_q_bdg import bdg_eigensystem_from_model_pairing


@dataclass(frozen=True)
class DWaveNodalQuadratureOptions:
    """Controls for recursive two-band d-wave nodal refinement."""

    coarse_grid: int = 20
    adaptive_level: int = 3
    gauss_order: int = 3
    sample_order: int = 3
    quasiparticle_window_eV: float = 0.02
    normal_window_eV: float = 0.08
    gap_window_eV: float = 0.02
    transition_window_eV: float = 0.01
    transition_shell_eV: float = 0.08
    include_transition_indicator: bool = True
    fermi_level_eV: float = 0.0
    max_quadrature_points: int = 400_000


@dataclass(frozen=True)
class _PointSpectrum:
    normal_energies: np.ndarray
    bdg_energies: np.ndarray
    gap_scalar: float
    gap_norm: float


@dataclass(frozen=True)
class _CellIndicator:
    flagged: bool
    quasiparticle: bool
    fermi_node_crossing: bool
    shifted_transition: bool
    minimum_bdg_abs_eV: float
    minimum_normal_abs_eV: float
    minimum_gap_norm_eV: float
    minimum_transition_abs_eV: float


def _validate_options(options: DWaveNodalQuadratureOptions) -> None:
    if int(options.coarse_grid) <= 0:
        raise ValueError("coarse_grid must be positive")
    if int(options.adaptive_level) < 0:
        raise ValueError("adaptive_level must be non-negative")
    if int(options.gauss_order) <= 0:
        raise ValueError("gauss_order must be positive")
    if int(options.sample_order) < 2:
        raise ValueError("sample_order must be at least two")
    for name in (
        "quasiparticle_window_eV",
        "normal_window_eV",
        "gap_window_eV",
        "transition_window_eV",
        "transition_shell_eV",
    ):
        value = float(getattr(options, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(float(options.fermi_level_eV)):
        raise ValueError("fermi_level_eV must be finite")
    if int(options.max_quadrature_points) <= 0:
        raise ValueError("max_quadrature_points must be positive")


def _wrap_bz(value: float) -> float:
    wrapped = (float(value) + np.pi) % (2.0 * np.pi) - np.pi
    # Keep the half-open convention even under roundoff at +pi.
    return float(-np.pi if wrapped >= np.pi else wrapped)


def _sample_points(
    cell: tuple[float, float, float, float],
    order: int,
) -> tuple[tuple[float, float], ...]:
    x0, x1, y0, y1 = cell
    xs = np.linspace(x0, x1, int(order), dtype=float)
    ys = np.linspace(y0, y1, int(order), dtype=float)
    return tuple((float(x), float(y)) for x in xs for y in ys)


def _subdivide(
    cell: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    x0, x1, y0, y1 = cell
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    return (
        (x0, xm, y0, ym),
        (x0, xm, ym, y1),
        (xm, x1, y0, ym),
        (xm, x1, ym, y1),
    )


class _SpectrumCache:
    def __init__(self, spec: object, ansatz: object, pairing_params: object) -> None:
        self.spec = spec
        self.ansatz = ansatz
        self.pairing_params = pairing_params
        self.values: dict[tuple[float, float], _PointSpectrum] = {}

    def at(self, kx: float, ky: float) -> _PointSpectrum:
        x = _wrap_bz(kx)
        y = _wrap_bz(ky)
        key = (round(x, 14), round(y, 14))
        cached = self.values.get(key)
        if cached is not None:
            return cached

        normal = np.linalg.eigvalsh(np.asarray(self.spec.normal_hamiltonian(x, y), dtype=complex))
        pairing = np.asarray(self.ansatz.mean_pairing(x, y, self.pairing_params), dtype=complex)
        if pairing.ndim != 2 or pairing.shape[0] != pairing.shape[1]:
            raise ValueError("mean_pairing must return a square matrix")
        bands = bdg_eigensystem_from_model_pairing(self.spec, x, y, pairing)
        gap_scalar_complex = np.trace(pairing) / float(pairing.shape[0])
        imaginary_scale = max(abs(gap_scalar_complex.real), 1.0)
        if abs(gap_scalar_complex.imag) > 1e-10 * imaginary_scale:
            raise ValueError(
                "d-wave nodal classifier requires an effectively real scalar pairing trace"
            )
        spectrum = _PointSpectrum(
            normal_energies=np.asarray(normal, dtype=float),
            bdg_energies=np.asarray(bands.energies, dtype=float),
            gap_scalar=float(gap_scalar_complex.real),
            gap_norm=float(np.linalg.norm(pairing, ord=2)),
        )
        self.values[key] = spectrum
        return spectrum


def _cell_indicator(
    cell: tuple[float, float, float, float],
    q_model: np.ndarray,
    cache: _SpectrumCache,
    options: DWaveNodalQuadratureOptions,
) -> _CellIndicator:
    samples = _sample_points(cell, int(options.sample_order))
    qx, qy = float(q_model[0]), float(q_model[1])
    shifts = ((0.0, 0.0), (-0.5 * qx, -0.5 * qy), (0.5 * qx, 0.5 * qy))
    fermi = float(options.fermi_level_eV)

    quasiparticle = False
    fermi_node = False
    shifted_transition = False
    min_bdg = float("inf")
    min_normal = float("inf")
    min_gap = float("inf")
    min_transition = float("inf")

    spectra_by_shift: list[list[_PointSpectrum]] = []
    for sx, sy in shifts:
        spectra = [cache.at(kx + sx, ky + sy) for kx, ky in samples]
        spectra_by_shift.append(spectra)
        normal = np.stack([value.normal_energies for value in spectra], axis=0)
        bdg = np.stack([value.bdg_energies for value in spectra], axis=0)
        gap_scalar = np.asarray([value.gap_scalar for value in spectra], dtype=float)
        gap_norm = np.asarray([value.gap_norm for value in spectra], dtype=float)

        min_bdg = min(min_bdg, float(np.min(np.abs(bdg))))
        min_normal = min(min_normal, float(np.min(np.abs(normal - fermi))))
        min_gap = min(min_gap, float(np.min(gap_norm)))
        if min_bdg <= float(options.quasiparticle_window_eV):
            quasiparticle = True

        band_crossing = bool(
            np.any(
                (np.min(normal, axis=0) <= fermi + float(options.normal_window_eV))
                & (np.max(normal, axis=0) >= fermi - float(options.normal_window_eV))
            )
        )
        gap_crossing = bool(np.min(gap_scalar) <= 0.0 <= np.max(gap_scalar))
        gap_small = bool(np.min(gap_norm) <= float(options.gap_window_eV))
        if band_crossing and (gap_crossing or gap_small):
            fermi_node = True

    if bool(options.include_transition_indicator):
        minus_values = spectra_by_shift[1]
        plus_values = spectra_by_shift[2]
        shell = float(options.transition_shell_eV)
        for minus, plus in zip(minus_values, plus_values, strict=True):
            em = minus.bdg_energies[:, None]
            en = plus.bdg_energies[None, :]
            relevant = (np.abs(em) <= shell) & (np.abs(en) <= shell)
            if np.any(relevant):
                differences = np.abs(em - en)
                local_min = float(np.min(differences[relevant]))
                min_transition = min(min_transition, local_min)
                if local_min <= float(options.transition_window_eV):
                    shifted_transition = True

    flagged = bool(quasiparticle or fermi_node or shifted_transition)
    return _CellIndicator(
        flagged=flagged,
        quasiparticle=quasiparticle,
        fermi_node_crossing=fermi_node,
        shifted_transition=shifted_transition,
        minimum_bdg_abs_eV=min_bdg,
        minimum_normal_abs_eV=min_normal,
        minimum_gap_norm_eV=min_gap,
        minimum_transition_abs_eV=min_transition,
    )


def _indicator_summary(indicators: list[_CellIndicator]) -> dict[str, Any]:
    finite_transition = [
        item.minimum_transition_abs_eV
        for item in indicators
        if np.isfinite(item.minimum_transition_abs_eV)
    ]
    return {
        "num_cells_evaluated": len(indicators),
        "num_cells_flagged": sum(item.flagged for item in indicators),
        "num_quasiparticle_flagged": sum(item.quasiparticle for item in indicators),
        "num_fermi_node_flagged": sum(item.fermi_node_crossing for item in indicators),
        "num_shifted_transition_flagged": sum(item.shifted_transition for item in indicators),
        "minimum_bdg_abs_eV": min(item.minimum_bdg_abs_eV for item in indicators),
        "minimum_normal_abs_eV": min(item.minimum_normal_abs_eV for item in indicators),
        "minimum_gap_norm_eV": min(item.minimum_gap_norm_eV for item in indicators),
        "minimum_transition_abs_eV": min(finite_transition, default=float("inf")),
    }


def _quadrature_for_cells(
    cells: list[tuple[float, float, float, float]],
    gauss_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, node_weights = np.polynomial.legendre.leggauss(int(gauss_order))
    points: list[tuple[float, float]] = []
    weights: list[float] = []
    bz_area = (2.0 * np.pi) ** 2
    for x0, x1, y0, y1 in cells:
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        xh = 0.5 * (x1 - x0)
        yh = 0.5 * (y1 - y0)
        for ix, nx in enumerate(nodes):
            for iy, ny in enumerate(nodes):
                points.append((_wrap_bz(xm + xh * nx), _wrap_bz(ym + yh * ny)))
                weights.append(
                    float(node_weights[ix] * node_weights[iy] * xh * yh / bz_area)
                )
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def build_dwave_nodal_quadrature(
    spec: object,
    ansatz: object,
    pairing_params: object,
    q_model: np.ndarray,
    options: DWaveNodalQuadratureOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a recursive node-aware quadrature for one finite-q d-wave response."""

    _validate_options(options)
    if str(getattr(ansatz, "name", "")) != "dwave":
        raise ValueError("d-wave nodal quadrature requires ansatz.name == 'dwave'")
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")

    edges = np.linspace(-np.pi, np.pi, int(options.coarse_grid) + 1, dtype=float)
    cells = [
        (float(edges[ix]), float(edges[ix + 1]), float(edges[iy]), float(edges[iy + 1]))
        for ix in range(int(options.coarse_grid))
        for iy in range(int(options.coarse_grid))
    ]
    cache = _SpectrumCache(spec, ansatz, pairing_params)
    history: list[dict[str, Any]] = []

    for level in range(int(options.adaptive_level)):
        indicators = [_cell_indicator(cell, q, cache, options) for cell in cells]
        summary = _indicator_summary(indicators)
        next_cells: list[tuple[float, float, float, float]] = []
        for cell, indicator in zip(cells, indicators, strict=True):
            if indicator.flagged:
                next_cells.extend(_subdivide(cell))
            else:
                next_cells.append(cell)
        history.append(
            {
                "level": level,
                "num_cells_before": len(cells),
                "num_cells_after": len(next_cells),
                **summary,
            }
        )
        cells = next_cells
        projected_points = len(cells) * int(options.gauss_order) ** 2
        if projected_points > int(options.max_quadrature_points):
            raise RuntimeError(
                "nodal adaptive quadrature exceeded max_quadrature_points: "
                f"projected={projected_points}, maximum={options.max_quadrature_points}"
            )
        if summary["num_cells_flagged"] == 0:
            break

    final_indicators = [_cell_indicator(cell, q, cache, options) for cell in cells]
    final_summary = _indicator_summary(final_indicators)
    points, weights = _quadrature_for_cells(cells, int(options.gauss_order))
    if points.shape != (len(cells) * int(options.gauss_order) ** 2, 2):
        raise RuntimeError("unexpected nodal quadrature point shape")
    if weights.shape != (len(points),) or not np.all(weights > 0.0):
        raise RuntimeError("nodal quadrature weights must be positive and match points")
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) >= 2e-12:
        raise RuntimeError("nodal quadrature weights do not sum to one")

    metadata = {
        "integration_strategy": "two_band_dwave_nodal_adaptive",
        "model_name": getattr(spec.metadata(), "name", type(spec).__name__),
        "pairing_name": "dwave",
        "q_model": [float(q[0]), float(q[1])],
        "coarse_grid": int(options.coarse_grid),
        "requested_adaptive_level": int(options.adaptive_level),
        "completed_adaptive_levels": len(history),
        "gauss_order": int(options.gauss_order),
        "sample_order": int(options.sample_order),
        "quasiparticle_window_eV": float(options.quasiparticle_window_eV),
        "normal_window_eV": float(options.normal_window_eV),
        "gap_window_eV": float(options.gap_window_eV),
        "transition_window_eV": float(options.transition_window_eV),
        "transition_shell_eV": float(options.transition_shell_eV),
        "include_transition_indicator": bool(options.include_transition_indicator),
        "fermi_level_eV": float(options.fermi_level_eV),
        "num_base_cells": int(options.coarse_grid) ** 2,
        "num_final_cells": len(cells),
        "num_quadrature_points": len(points),
        "num_cached_spectrum_points": len(cache.values),
        "weight_sum": weight_sum,
        "parent_child_double_counting": False,
        "primitive_merge_before_schur_required": True,
        "cell_indicator": (
            "low BdG quasiparticle energy OR normal-FS/gap-node crossing OR "
            "low-energy shifted k-q/2 to k+q/2 transition"
        ),
        "refinement_history": history,
        "final_cell_indicator_summary": final_summary,
    }
    return points, weights, metadata


__all__ = [
    "DWaveNodalQuadratureOptions",
    "build_dwave_nodal_quadrature",
]
