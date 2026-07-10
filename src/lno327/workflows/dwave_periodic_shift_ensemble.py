"""Nested periodic-shift ensemble for exact-static two-band d-wave response.

Every shift contributes one complete periodic ``base_nk x base_nk`` lattice.
The shifts are deterministic, nested, and grouped into four-point C4/antithetic
orbits.  Per-shift primitive response components may therefore be cached once
and reused for cumulative 4/8/16-shift estimates.  Primitive blocks and the
Ward RHS are averaged before the single amplitude/phase Schur complement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.finite_q_bdg import _finalize_components
from lno327.response.finite_q_optimized import FiniteQQWorkspace
from lno327.response.ward_validation import PrimitiveWardRHS


@dataclass(frozen=True)
class DWavePeriodicShiftEnsembleOptions:
    """Controls for a nested complete-lattice shift ensemble."""

    base_nk: int = 56
    max_shifts: int = 16
    max_quadrature_points: int = 400_000


def _validate_options(options: DWavePeriodicShiftEnsembleOptions) -> None:
    if int(options.base_nk) <= 0:
        raise ValueError("base_nk must be positive")
    if int(options.max_shifts) <= 0 or int(options.max_shifts) % 4 != 0:
        raise ValueError("max_shifts must be a positive multiple of four")
    if int(options.max_quadrature_points) <= 0:
        raise ValueError("max_quadrature_points must be positive")
    requested = int(options.base_nk) ** 2 * int(options.max_shifts)
    if requested > int(options.max_quadrature_points):
        raise RuntimeError(
            "periodic shift ensemble exceeded max_quadrature_points: "
            f"requested={requested}, maximum={options.max_quadrature_points}"
        )


def _radical_inverse(index: int, base: int) -> float:
    if int(index) <= 0 or int(base) <= 1:
        raise ValueError("radical inverse requires index > 0 and base > 1")
    value = 0.0
    factor = 1.0 / float(base)
    integer = int(index)
    while integer:
        integer, digit = divmod(integer, int(base))
        value += float(digit) * factor
        factor /= float(base)
    return float(value)


def nested_c4_antithetic_shifts(max_shifts: int) -> np.ndarray:
    """Return a deterministic nested Halton/C4/antithetic shift sequence.

    The first 4, 8, 16, ... entries are unions of complete four-point orbits
    ``(x,y)``, ``(y,x)``, ``(1-x,1-y)``, ``(1-y,1-x)``.  This preserves the
    nesting required by incremental batch diagnostics while reducing finite
    sample inversion and x/y-exchange bias.
    """

    count = int(max_shifts)
    if count <= 0 or count % 4 != 0:
        raise ValueError("max_shifts must be a positive multiple of four")
    shifts: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    index = 1
    while len(shifts) < count:
        x = _radical_inverse(index, 2)
        y = _radical_inverse(index, 3)
        orbit = (
            (x, y),
            (y, x),
            ((1.0 - x) % 1.0, (1.0 - y) % 1.0),
            ((1.0 - y) % 1.0, (1.0 - x) % 1.0),
        )
        keys = tuple((round(a, 15), round(b, 15)) for a, b in orbit)
        if len(set(keys)) == 4 and not any(key in seen for key in keys):
            shifts.extend((float(a), float(b)) for a, b in orbit)
            seen.update(keys)
        index += 1
    array = np.asarray(shifts[:count], dtype=float)
    if array.shape != (count, 2) or np.any(array < 0.0) or np.any(array >= 1.0):
        raise RuntimeError("invalid nested periodic shifts")
    return array


def periodic_shift_mesh(base_nk: int, shift: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return one equally weighted complete periodic tensor lattice."""

    nk = int(base_nk)
    vector = np.asarray(shift, dtype=float)
    if nk <= 0:
        raise ValueError("base_nk must be positive")
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError("shift must be a finite vector with shape (2,)")
    if np.any(vector < 0.0) or np.any(vector >= 1.0):
        raise ValueError("shift coordinates must lie in [0, 1)")
    step = 2.0 * np.pi / float(nk)
    kx = -np.pi + (np.arange(nk, dtype=float) + float(vector[0])) * step
    ky = -np.pi + (np.arange(nk, dtype=float) + float(vector[1])) * step
    gx, gy = np.meshgrid(kx, ky, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel()])
    weights = np.full(nk * nk, 1.0 / float(nk * nk), dtype=float)
    return points, weights


def build_dwave_periodic_shift_ensemble(
    q_model: np.ndarray,
    options: DWavePeriodicShiftEnsembleOptions,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return nested shifts and audit metadata without expanding all lattices."""

    _validate_options(options)
    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_model must be a finite vector with shape (2,)")
    shifts = nested_c4_antithetic_shifts(int(options.max_shifts))
    metadata = {
        "integration_strategy": "dwave_periodic_nested_shift_ensemble",
        "q_model": [float(q[0]), float(q[1])],
        "base_nk": int(options.base_nk),
        "max_shifts": int(options.max_shifts),
        "num_points_per_shift": int(options.base_nk) ** 2,
        "max_quadrature_points": int(options.base_nk) ** 2 * int(options.max_shifts),
        "shift_sequence": shifts.tolist(),
        "shift_sequence_kind": "halton_bases_2_3_with_c4_antithetic_orbits",
        "nested_prefixes": [value for value in (4, 8, 16, 32, 64) if value <= int(options.max_shifts)],
        "full_periodic_lattice_per_shift": True,
        "primitive_merge_before_schur_required": True,
        "per_shift_cache_reused_across_prefixes": True,
        "local_cell_refinement": False,
    }
    return shifts, metadata


def _weighted_array(values: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([np.asarray(value, dtype=complex) for value in values], axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def _weighted_complex(values: Sequence[complex], weights: np.ndarray) -> complex:
    return complex(np.dot(weights, np.asarray(values, dtype=complex)))


def merge_shift_components_before_schur(
    components: Sequence[BdGFiniteQResponseComponents],
    rhs_values: Sequence[PrimitiveWardRHS],
    weights: Sequence[float],
    template_workspace: FiniteQQWorkspace,
    *,
    omega_eV: float = 0.0,
) -> tuple[BdGFiniteQResponseComponents, PrimitiveWardRHS]:
    """Average cached shift primitives and perform exactly one Schur complement."""

    if not components or len(components) != len(rhs_values):
        raise ValueError("components and rhs_values must be nonempty and have equal length")
    normalized = np.asarray(weights, dtype=float)
    if normalized.shape != (len(components),) or not np.isfinite(normalized).all():
        raise ValueError("weights must be a finite vector matching components")
    if np.any(normalized < 0.0) or float(np.sum(normalized)) <= 0.0:
        raise ValueError("weights must be non-negative with positive sum")
    normalized = normalized / float(np.sum(normalized))

    q = np.asarray(template_workspace.q_model, dtype=float)
    for rhs in rhs_values:
        if not np.allclose(rhs.q_model, q, rtol=0.0, atol=1e-14):
            raise ValueError("all Ward RHS values must share q_model")
    material = template_workspace.material
    config = replace(material.config, omega_eV=float(omega_eV))
    delta0 = float(material.pairing_params.delta0_eV)

    bubble = _weighted_array([item.bare_bubble for item in components], normalized)
    direct = _weighted_array([item.direct for item in components], normalized)
    collective_bubble = _weighted_array(
        [item.collective_bubble for item in components], normalized
    )
    collective_counterterm = _weighted_array(
        [item.collective_counterterm for item in components], normalized
    )
    em_collective_left = _weighted_array(
        [item.em_collective_left for item in components], normalized
    )
    collective_em_right = _weighted_array(
        [item.collective_em_right for item in components], normalized
    )
    phase_left = delta0 * em_collective_left[:, 1]
    phase_right = delta0 * collective_em_right[1, :]
    phase_bubble = np.asarray(
        [[delta0 * delta0 * collective_bubble[1, 1]]], dtype=complex
    )
    phase_direct_plus = _weighted_complex(
        [complex(item.metadata["phase_phase_direct_plus_convention"]) for item in components],
        normalized,
    )
    phase_direct_minus = _weighted_complex(
        [complex(item.metadata["phase_phase_direct_minus_convention"]) for item in components],
        normalized,
    )

    merged = _finalize_components(
        ansatz=material.ansatz,
        opts=material.options,
        shared_eigenbasis_q0=template_workspace.shared_eigenbasis_q0,
        shared_eigenbasis_q0_tolerance=1e-14,
        collective_mode=material.collective_mode,
        collective_mode_disabled_reason=material.collective_mode_disabled_reason,
        bubble=bubble,
        direct=direct,
        phase_left=phase_left,
        phase_right=phase_right,
        phase_phase_bubble_matrix=phase_bubble,
        phase_phase_direct_plus=phase_direct_plus,
        phase_phase_direct_minus=phase_direct_minus,
        collective_bubble=collective_bubble,
        collective_counterterm_matrix=collective_counterterm,
        em_collective_left=em_collective_left,
        collective_em_right=collective_em_right,
        config=config,
        q=q,
        workspace_evaluation=True,
    )
    merged_metadata = dict(merged.metadata)
    merged_metadata.update(
        {
            "shift_ensemble_merged_before_schur": True,
            "num_shift_components": len(components),
            "shift_component_weights": normalized.tolist(),
            "per_shift_schur_results_discarded": True,
        }
    )
    merged = replace(merged, metadata=merged_metadata)

    left = _weighted_array([item.left for item in rhs_values], normalized)
    right = _weighted_array([item.right for item in rhs_values], normalized)
    rhs_metadata = {
        "convention": rhs_values[0].metadata.get("convention"),
        "basis": rhs_values[0].metadata.get("basis"),
        "source": "weighted cached complete-periodic-shift Ward RHS values",
        "shift_ensemble_merged_before_ward_validation": True,
        "num_shift_components": len(rhs_values),
        "shift_component_weights": normalized.tolist(),
    }
    merged_rhs = PrimitiveWardRHS(
        left=left,
        right=right,
        q_model=q,
        xi_eV=float(omega_eV),
        delta0_eV=delta0,
        metadata=rhs_metadata,
    )
    return merged, merged_rhs


__all__ = [
    "DWavePeriodicShiftEnsembleOptions",
    "build_dwave_periodic_shift_ensemble",
    "merge_shift_components_before_schur",
    "nested_c4_antithetic_shifts",
    "periodic_shift_mesh",
]
