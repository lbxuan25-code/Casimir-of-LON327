"""Signed actual-shift reconstruction diagnostics for d-wave static sensitivity.

This module corrects two limitations of the earlier band-pair diagnostic:

* normal-state Fermi-surface masks are evaluated at every *actual shifted-grid
  point*, not at common base-cell centers;
* BdG pair contributions retain their complex sign and phase.  Pair-selected
  bubble corrections can therefore be compared directly with the true signed
  primitive difference instead of using a positive norm proxy.

For an independent Ward test, spatial masks are also applied to the complete
46-component pointwise primitive vector.  Those masked sums include direct,
collective-counterterm, phase-direct, and full Ward-RHS terms.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from validation.lib.dwave_shift_bandpair import pointwise_bandpair_data
from validation.lib.dwave_shift_spatial import PRIMITIVE_SLICES, PRIMITIVE_VECTOR_SIZE
from lno327.response.finite_q_optimized import FiniteQQWorkspace, _vectorized_kubo_factors


SIGNED_PAIR_CLASSES = (
    "all_pairs",
    "same_normal_fs_crossing",
    "normal_band_0_fs_crossing",
    "normal_band_1_fs_crossing",
    "sorted_1_to_2_fs_crossing",
    "sorted_2_to_1_fs_crossing",
)


def actual_shift_normal_fs_data(workspace: FiniteQQWorkspace) -> dict[str, np.ndarray]:
    """Normal-band energies and crossing masks at every actual shift-grid point."""

    material = workspace.material
    spec = material.spec
    qx, qy = map(float, workspace.q_model)
    fermi = float(material.config.fermi_level_eV)
    points = np.asarray(material.k_points, dtype=float)
    minus = np.stack(
        [
            np.linalg.eigvalsh(
                np.asarray(spec.normal_hamiltonian(float(kx - 0.5 * qx), float(ky - 0.5 * qy)))
            )
            - fermi
            for kx, ky in points
        ],
        axis=0,
    )
    plus = np.stack(
        [
            np.linalg.eigvalsh(
                np.asarray(spec.normal_hamiltonian(float(kx + 0.5 * qx), float(ky + 0.5 * qy)))
            )
            - fermi
            for kx, ky in points
        ],
        axis=0,
    )
    if minus.shape != plus.shape or minus.ndim != 2:
        raise RuntimeError("normal-state shifted energies have an invalid shape")
    crossing = minus * plus < 0.0
    minimum_abs = np.minimum(np.abs(minus), np.abs(plus))
    return {
        "normal_minus_eV": np.asarray(minus, dtype=float),
        "normal_plus_eV": np.asarray(plus, dtype=float),
        "crossing_by_band": np.asarray(crossing, dtype=bool),
        "minimum_shifted_abs_eV_by_band": np.asarray(minimum_abs, dtype=float),
        "any_crossing": np.any(crossing, axis=1),
        "minimum_shifted_abs_eV": np.min(minimum_abs, axis=1),
    }


def signed_pair_primitive_vectors(workspace: FiniteQQWorkspace) -> np.ndarray:
    """Return signed complex bubble contributions for every ``(k,m,n)`` pair.

    The returned array has shape ``(nk, nb, nb, 46)``.  Only the bubble slices
    are populated; direct/contact, Goldstone counterterm, phase-direct, and Ward
    RHS pieces are deliberately left zero because they do not admit the same
    shifted-BdG-pair decomposition.
    """

    material = workspace.material
    nk, nb = workspace.nk, workspace.nb
    factors = np.asarray(
        _vectorized_kubo_factors(workspace, np.asarray([0.0], dtype=float))[0],
        dtype=complex,
    )
    weighted = 0.5 * np.asarray(material.k_weights, dtype=float)[:, None, None] * factors
    left = np.transpose(np.asarray(workspace.left_vertices_band, dtype=complex), (0, 2, 3, 1))
    right = np.transpose(np.asarray(workspace.right_vertices_band, dtype=complex), (0, 2, 3, 1))
    unified = weighted[..., None, None] * left[..., :, None] * np.conjugate(right[..., None, :])
    vectors = np.zeros((nk, nb, nb, PRIMITIVE_VECTOR_SIZE), dtype=complex)
    vectors[..., PRIMITIVE_SLICES["bare_bubble"]] = unified[..., :3, :3].reshape(nk, nb, nb, 9)
    vectors[..., PRIMITIVE_SLICES["em_collective_left"]] = unified[..., :3, 3:5].reshape(
        nk, nb, nb, 6
    )
    vectors[..., PRIMITIVE_SLICES["collective_em_right"]] = unified[..., 3:5, :3].reshape(
        nk, nb, nb, 6
    )
    vectors[..., PRIMITIVE_SLICES["collective_bubble"]] = unified[..., 3:5, 3:5].reshape(
        nk, nb, nb, 4
    )
    return vectors


def _pair_class_masks(
    workspace: FiniteQQWorkspace,
    bandpair: Mapping[str, Any],
    fs_data: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    nk, nb = workspace.nk, workspace.nb
    same_normal = np.asarray(bandpair["pair_same_normal_band"], dtype=bool)
    normal_minus = np.asarray(bandpair["pair_normal_band_minus"], dtype=int)
    normal_plus = np.asarray(bandpair["pair_normal_band_plus"], dtype=int)
    crossing_by_band = np.asarray(fs_data["crossing_by_band"], dtype=bool)
    rows = np.arange(nk)[:, None, None]
    pair_actual_crossing = same_normal & crossing_by_band[rows, normal_minus]
    m_grid = np.broadcast_to(np.arange(nb)[None, :, None], (nk, nb, nb))
    n_grid = np.broadcast_to(np.arange(nb)[None, None, :], (nk, nb, nb))
    return {
        "all_pairs": np.ones((nk, nb, nb), dtype=bool),
        "same_normal_fs_crossing": pair_actual_crossing,
        "normal_band_0_fs_crossing": pair_actual_crossing & (normal_minus == 0) & (normal_plus == 0),
        "normal_band_1_fs_crossing": pair_actual_crossing & (normal_minus == 1) & (normal_plus == 1),
        "sorted_1_to_2_fs_crossing": pair_actual_crossing & (m_grid == 1) & (n_grid == 2),
        "sorted_2_to_1_fs_crossing": pair_actual_crossing & (m_grid == 2) & (n_grid == 1),
    }


def summarize_shift_signed_reconstruction(
    workspace: FiniteQQWorkspace,
    pointwise_primitive_vectors: np.ndarray,
    *,
    shell_multiples_T: Sequence[float] = (2.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Build portable signed spatial and pair sums for one complete shift grid."""

    vectors = np.asarray(pointwise_primitive_vectors, dtype=complex)
    if vectors.shape != (workspace.nk, PRIMITIVE_VECTOR_SIZE):
        raise ValueError("pointwise primitive vectors have an invalid shape")
    fs_data = actual_shift_normal_fs_data(workspace)
    crossing = np.asarray(fs_data["crossing_by_band"], dtype=bool)
    minimum_abs = np.asarray(fs_data["minimum_shifted_abs_eV_by_band"], dtype=float)
    spatial_masks: dict[str, np.ndarray] = {
        "all_points": np.ones(workspace.nk, dtype=bool),
        "any_normal_fs_crossing": np.any(crossing, axis=1),
        "normal_band_0_fs_crossing": crossing[:, 0],
        "normal_band_1_fs_crossing": crossing[:, 1],
    }
    temperature = float(workspace.material.config.temperature_eV)
    for multiple in shell_multiples_T:
        value = float(multiple)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError("shell multiples must be positive finite values")
        label = f"any_normal_fs_shell_{value:g}T"
        spatial_masks[label] = np.any(minimum_abs <= value * temperature, axis=1)

    spatial_sums = {
        name: np.sum(vectors[mask], axis=0) for name, mask in spatial_masks.items()
    }
    spatial_point_fractions = {
        name: float(np.mean(mask)) for name, mask in spatial_masks.items()
    }

    pair_data = pointwise_bandpair_data(workspace)
    pair_vectors = signed_pair_primitive_vectors(workspace)
    pair_masks = _pair_class_masks(workspace, pair_data, fs_data)
    pair_sums = {
        name: np.sum(pair_vectors[mask], axis=0) for name, mask in pair_masks.items()
    }
    pair_event_fractions = {
        name: float(np.mean(mask)) for name, mask in pair_masks.items()
    }
    return {
        "spatial_sums": spatial_sums,
        "spatial_point_fractions": spatial_point_fractions,
        "pair_sums": pair_sums,
        "pair_event_fractions": pair_event_fractions,
    }


def aggregate_rule_signed_summaries(
    shifts: Sequence[np.ndarray],
    weights: Sequence[float],
    cache: Mapping[tuple[float, float], Mapping[str, Any]],
    *,
    key_function,
) -> dict[str, Any]:
    """Weighted average of portable per-shift signed summaries."""

    shift_values = list(shifts)
    weight_array = np.asarray(weights, dtype=float)
    if not shift_values or weight_array.shape != (len(shift_values),):
        raise ValueError("shifts and weights must be nonempty and aligned")
    weight_array = weight_array / float(np.sum(weight_array))
    first = cache[key_function(shift_values[0])]
    result: dict[str, Any] = {}
    for group in ("spatial_sums", "pair_sums"):
        names = set(first[group])
        if any(set(cache[key_function(shift)][group]) != names for shift in shift_values):
            raise ValueError(f"inconsistent {group} keys across shifts")
        result[group] = {
            name: np.tensordot(
                weight_array,
                np.stack(
                    [
                        np.asarray(cache[key_function(shift)][group][name], dtype=complex)
                        for shift in shift_values
                    ],
                    axis=0,
                ),
                axes=(0, 0),
            )
            for name in sorted(names)
        }
    for group in ("spatial_point_fractions", "pair_event_fractions"):
        names = set(first[group])
        result[group] = {
            name: float(
                np.dot(
                    weight_array,
                    [float(cache[key_function(shift)][group][name]) for shift in shift_values],
                )
            )
            for name in sorted(names)
        }
    return result


def primitive_block_arrays(vector: np.ndarray) -> dict[str, np.ndarray]:
    """Return block-level arrays used for signed reconstruction residuals."""

    value = np.asarray(vector, dtype=complex)
    if value.shape != (PRIMITIVE_VECTOR_SIZE,):
        raise ValueError("primitive vector has an invalid shape")
    k_ss = (
        value[PRIMITIVE_SLICES["bare_bubble"]]
        + value[PRIMITIVE_SLICES["direct"]]
    )
    k_etaeta = np.concatenate(
        [
            value[PRIMITIVE_SLICES["collective_bubble"]]
            + value[PRIMITIVE_SLICES["collective_counterterm"]],
            value[PRIMITIVE_SLICES["phase_direct"]],
        ]
    )
    return {
        "k_ss": k_ss,
        "k_seta": value[PRIMITIVE_SLICES["em_collective_left"]],
        "k_etas": value[PRIMITIVE_SLICES["collective_em_right"]],
        "k_etaeta": k_etaeta,
        "ward_rhs": np.concatenate(
            [
                value[PRIMITIVE_SLICES["rhs_left"]],
                value[PRIMITIVE_SLICES["rhs_right"]],
            ]
        ),
    }


def signed_reconstruction_residuals(
    target_delta: np.ndarray,
    selected_delta: np.ndarray,
) -> dict[str, float]:
    """Relative block residual after retaining one signed correction class."""

    target = primitive_block_arrays(target_delta)
    selected = primitive_block_arrays(selected_delta)
    result: dict[str, float] = {}
    for name in target:
        denominator = float(np.linalg.norm(target[name]))
        numerator = float(np.linalg.norm(target[name] - selected[name]))
        result[name] = numerator / max(denominator, 1e-30)
    return result


def bubble_block_arrays(vector: np.ndarray) -> dict[str, np.ndarray]:
    """Bubble-only blocks for exact signed pair-reconstruction tests."""

    value = np.asarray(vector, dtype=complex)
    return {
        "k_ss_bubble": value[PRIMITIVE_SLICES["bare_bubble"]],
        "k_seta_bubble": value[PRIMITIVE_SLICES["em_collective_left"]],
        "k_etas_bubble": value[PRIMITIVE_SLICES["collective_em_right"]],
        "k_etaeta_bubble": value[PRIMITIVE_SLICES["collective_bubble"]],
    }


def signed_pair_reconstruction_residuals(
    target_pair_delta: np.ndarray,
    selected_pair_delta: np.ndarray,
) -> dict[str, float]:
    target = bubble_block_arrays(target_pair_delta)
    selected = bubble_block_arrays(selected_pair_delta)
    return {
        name: float(np.linalg.norm(target[name] - selected[name]))
        / max(float(np.linalg.norm(target[name])), 1e-30)
        for name in target
    }


__all__ = [
    "SIGNED_PAIR_CLASSES",
    "actual_shift_normal_fs_data",
    "aggregate_rule_signed_summaries",
    "bubble_block_arrays",
    "primitive_block_arrays",
    "signed_pair_primitive_vectors",
    "signed_pair_reconstruction_residuals",
    "signed_reconstruction_residuals",
    "summarize_shift_signed_reconstruction",
]
