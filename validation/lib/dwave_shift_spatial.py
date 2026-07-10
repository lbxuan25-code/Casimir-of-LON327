"""Spatial localization diagnostics for exact-static d-wave shift sensitivity.

The diagnostic compares two weighted ensembles of complete periodic lattices with
the same ``base_nk``.  Lattice points with the same tensor index belong to the
same underlying base cell, so the difference of their pointwise primitive
contributions defines a common Brillouin-zone cell map.  No locally refined
quadrature is used and every rule-level total remains a weighted average of
complete periodic lattices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq

from lno327.bdg.finite_q import density_vertex, phase_phase_direct_vertex
from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.finite_q_bdg import (
    _finalize_components,
    bdg_contact_vertex_from_spec,
    bdg_vector_vertex_from_spec,
)
from lno327.response.finite_q_optimized import (
    FiniteQQWorkspace,
    _thermal_expectation_from_bands,
    _vectorized_kubo_factors,
)
from lno327.response.ward_validation import PrimitiveWardRHS
from lno327.workflows.dwave_periodic_shift_ensemble import (
    nested_c4_antithetic_shifts,
    periodic_shift_mesh,
)
from lno327.workflows.dwave_periodic_multishift_quadrature import _gauss_shifts
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327 import KuboConfig
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


PRIMITIVE_SLICES = {
    "bare_bubble": slice(0, 9),
    "direct": slice(9, 18),
    "collective_bubble": slice(18, 22),
    "collective_counterterm": slice(22, 26),
    "em_collective_left": slice(26, 32),
    "collective_em_right": slice(32, 38),
    "phase_direct": slice(38, 40),
    "rhs_left": slice(40, 43),
    "rhs_right": slice(43, 46),
}
PRIMITIVE_VECTOR_SIZE = 46


@dataclass(frozen=True)
class SpatialDiagnosticConfig:
    base_nk: int
    qx: float
    qy: float
    temperature_K: float
    delta0_eV: float
    eta_eV: float

    @property
    def q(self) -> np.ndarray:
        return np.asarray([self.qx, self.qy], dtype=float)


def shift_rule(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic complete-lattice shifts and normalized weights."""

    key = str(name).strip().lower()
    if key == "midpoint":
        shifts = np.asarray([[0.5, 0.5]], dtype=float)
        weights = np.asarray([1.0], dtype=float)
    elif key == "gauss2":
        nodes, one_d = _gauss_shifts(2)
        shifts = np.asarray([[x, y] for x in nodes for y in nodes], dtype=float)
        weights = np.asarray([wx * wy for wx in one_d for wy in one_d], dtype=float)
    elif key == "halton4":
        shifts = nested_c4_antithetic_shifts(4)
        weights = np.full(4, 0.25, dtype=float)
    else:
        raise ValueError("shift rule must be 'midpoint', 'gauss2', or 'halton4'")
    weights = weights / float(np.sum(weights))
    return shifts, weights


def _static_factor_matrix(energies: np.ndarray, occupations: np.ndarray, config) -> np.ndarray:
    em = np.asarray(energies, dtype=float)[:, None]
    en = np.asarray(energies, dtype=float)[None, :]
    fm = np.asarray(occupations, dtype=float)[:, None]
    fn = np.asarray(occupations, dtype=float)[None, :]
    delta = em - en
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = (fm - fn) / delta
    degenerate = np.abs(delta) < float(config.eta_eV)
    if np.any(degenerate):
        midpoint = 0.5 * (em + en)
        shifted = midpoint - float(config.fermi_level_eV)
        if config.temperature_eV <= 0.0:
            width = max(float(config.eta_eV), 1e-12)
            derivative = -width / (np.pi * (shifted * shifted + width * width))
        else:
            x = np.clip(shifted / (2.0 * float(config.temperature_eV)), -350.0, 350.0)
            derivative = -1.0 / (4.0 * float(config.temperature_eV) * np.cosh(x) ** 2)
        factors = np.where(degenerate, derivative, factors)
    return np.asarray(factors, dtype=complex)


def pointwise_primitive_vectors(workspace: FiniteQQWorkspace) -> np.ndarray:
    """Return one complete linear primitive vector for every periodic-grid point."""

    material = workspace.material
    nk = material.nk
    raw = _vectorized_kubo_factors(workspace, np.asarray([0.0], dtype=float))[0]
    weighted = 0.5 * material.k_weights[:, None, None] * raw
    unified = np.einsum(
        "kmn,kamn,kbmn->kab",
        weighted,
        workspace.left_vertices_band,
        np.conjugate(workspace.right_vertices_band),
        optimize=True,
    )
    vectors = np.zeros((nk, PRIMITIVE_VECTOR_SIZE), dtype=complex)
    vectors[:, PRIMITIVE_SLICES["bare_bubble"]] = unified[:, :3, :3].reshape(nk, 9)
    vectors[:, PRIMITIVE_SLICES["em_collective_left"]] = unified[:, :3, 3:5].reshape(nk, 6)
    vectors[:, PRIMITIVE_SLICES["collective_em_right"]] = unified[:, 3:5, :3].reshape(nk, 6)
    vectors[:, PRIMITIVE_SLICES["collective_bubble"]] = unified[:, 3:5, 3:5].reshape(nk, 4)

    spec, ansatz = material.spec, material.ansatz
    amp, opts = material.pairing_params, material.options
    qx, qy = float(workspace.q_model[0]), float(workspace.q_model[1])
    dim = np.asarray(spec.normal_hamiltonian(*material.k_points[0])).shape[0]
    rho = density_vertex(int(dim))
    direct_rows = np.zeros((nk, 3, 3), dtype=complex)
    counter_rows = np.zeros((nk, 2, 2), dtype=complex)
    phase_rows = np.zeros((nk, 2), dtype=complex)
    rhs_rows = np.zeros((nk, 3), dtype=complex)

    for index, (weight, point) in enumerate(zip(material.k_weights, material.k_points, strict=True)):
        weight_f = float(weight)
        kx, ky = float(point[0]), float(point[1])
        states_mid = material.midpoint_states[index]
        energies_mid = material.midpoint_energies[index]
        occupations_mid = material.midpoint_occupations[index]
        source_band = workspace.right_vertices_band[index, :3]
        occupations_minus = workspace.occupations_minus[index]
        occupations_plus = workspace.occupations_plus[index]
        occupation_difference = occupations_minus[:, None] - occupations_plus[None, :]
        equal_forward = 0.5 * weight_f * np.einsum(
            "mn,mn,jmn->j",
            occupation_difference,
            source_band[0],
            np.conjugate(source_band),
            optimize=True,
        )

        q_contact = np.zeros(3, dtype=complex)
        for i, direction_i in enumerate(("x", "y")):
            qi = qx if direction_i == "x" else qy
            for j, direction_j in enumerate(("x", "y")):
                contact = bdg_contact_vertex_from_spec(
                    spec, kx, ky, qx, qy, direction_i, direction_j, opts.current_vertex
                )
                direct = -weight_f * _thermal_expectation_from_bands(
                    states_mid, occupations_mid, contact
                )
                direct_rows[index, 1 + i, 1 + j] = direct
                q_contact[1 + j] += qi * direct

        delta_v = np.zeros(3, dtype=complex)
        for j, direction in enumerate(("x", "y"), start=1):
            vertex_plus = bdg_vector_vertex_from_spec(
                spec, kx + 0.5 * qx, ky + 0.5 * qy, qx, qy, direction, opts.current_vertex
            )
            vertex_minus = bdg_vector_vertex_from_spec(
                spec, kx - 0.5 * qx, ky - 0.5 * qy, qx, qy, direction, opts.current_vertex
            )
            delta_v[j] = weight_f * _thermal_expectation_from_bands(
                states_mid, occupations_mid, vertex_plus - vertex_minus
            )
        rhs_rows[index] = equal_forward - delta_v + q_contact

        delta_theta = ansatz.phase_pairing_matrix(kx, ky, qx, qy, amp)
        phase_value = weight_f * _thermal_expectation_from_bands(
            states_mid, occupations_mid, phase_phase_direct_vertex(delta_theta)
        )
        phase_rows[index] = np.asarray([phase_value, -phase_value], dtype=complex)

        eta_vertices = tuple(ansatz.collective_vertices(kx, ky, 0.0, 0.0, amp))
        eta2_band = states_mid.conjugate().T @ eta_vertices[1] @ states_mid
        factors_mid = _static_factor_matrix(energies_mid, occupations_mid, material.config)
        eta2_bubble = 0.5 * weight_f * np.sum(
            factors_mid * eta2_band * np.conjugate(eta2_band)
        )
        counter_rows[index] = -complex(eta2_bubble) * np.eye(2, dtype=complex)

    vectors[:, PRIMITIVE_SLICES["direct"]] = direct_rows.reshape(nk, 9)
    vectors[:, PRIMITIVE_SLICES["collective_counterterm"]] = counter_rows.reshape(nk, 4)
    vectors[:, PRIMITIVE_SLICES["phase_direct"]] = phase_rows
    vectors[:, PRIMITIVE_SLICES["rhs_left"]] = rhs_rows
    vectors[:, PRIMITIVE_SLICES["rhs_right"]] = rhs_rows
    return vectors


def evaluate_shift_spatial(
    config: SpatialDiagnosticConfig,
    shift: Sequence[float],
    *,
    keep_workspace: bool = False,
) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(config.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=config.temperature_K,
        eta_eV=config.eta_eV,
        output_si=False,
    )
    points, weights = periodic_shift_mesh(config.base_nk, shift)
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec, ansatz, points, weights, kubo, pairing, FiniteQEngineOptions()
    )
    workspace = precompute_finite_q_q_workspace(material, config.q)
    vectors = pointwise_primitive_vectors(workspace)
    return {
        "shift": np.asarray(shift, dtype=float),
        "vectors": vectors,
        "workspace": workspace if keep_workspace else None,
        "components": finite_q_bdg_response_from_q_workspace(workspace, 0.0),
        "rhs": primitive_ward_rhs_from_q_workspace(workspace, 0.0),
    }


def _unpack(vector: np.ndarray, name: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(vector[PRIMITIVE_SLICES[name]], dtype=complex).reshape(shape)


def components_from_primitive_vector(
    vector: np.ndarray,
    template_workspace: FiniteQQWorkspace,
) -> tuple[BdGFiniteQResponseComponents, PrimitiveWardRHS]:
    """Rebuild one response from a summed linear primitive vector."""

    value = np.asarray(vector, dtype=complex)
    if value.shape != (PRIMITIVE_VECTOR_SIZE,):
        raise ValueError(f"primitive vector must have shape ({PRIMITIVE_VECTOR_SIZE},)")
    material = template_workspace.material
    delta0 = float(material.pairing_params.delta0_eV)
    bubble = _unpack(value, "bare_bubble", (3, 3))
    direct = _unpack(value, "direct", (3, 3))
    collective_bubble = _unpack(value, "collective_bubble", (2, 2))
    counterterm = _unpack(value, "collective_counterterm", (2, 2))
    em_left = _unpack(value, "em_collective_left", (3, 2))
    collective_right = _unpack(value, "collective_em_right", (2, 3))
    phase_direct = _unpack(value, "phase_direct", (2,))
    config = replace(material.config, omega_eV=0.0)
    merged = _finalize_components(
        ansatz=material.ansatz,
        opts=material.options,
        shared_eigenbasis_q0=template_workspace.shared_eigenbasis_q0,
        shared_eigenbasis_q0_tolerance=1e-14,
        collective_mode=material.collective_mode,
        collective_mode_disabled_reason=material.collective_mode_disabled_reason,
        bubble=bubble,
        direct=direct,
        phase_left=delta0 * em_left[:, 1],
        phase_right=delta0 * collective_right[1, :],
        phase_phase_bubble_matrix=np.asarray(
            [[delta0 * delta0 * collective_bubble[1, 1]]], dtype=complex
        ),
        phase_phase_direct_plus=complex(phase_direct[0]),
        phase_phase_direct_minus=complex(phase_direct[1]),
        collective_bubble=collective_bubble,
        collective_counterterm_matrix=counterterm,
        em_collective_left=em_left,
        collective_em_right=collective_right,
        config=config,
        q=template_workspace.q_model,
        workspace_evaluation=True,
    )
    metadata = dict(merged.metadata)
    metadata["shift_spatial_primitive_reconstruction"] = True
    merged = replace(merged, metadata=metadata)
    rhs = PrimitiveWardRHS(
        left=_unpack(value, "rhs_left", (3,)),
        right=_unpack(value, "rhs_right", (3,)),
        q_model=template_workspace.q_model,
        xi_eV=0.0,
        delta0_eV=delta0,
        metadata={"source": "summed pointwise primitive spatial diagnostic"},
    )
    return merged, rhs


def block_mass_table(delta_cells: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return per-cell block norms and an equal-block normalized ranking score."""

    delta = np.asarray(delta_cells, dtype=complex)
    if delta.ndim != 2 or delta.shape[1] != PRIMITIVE_VECTOR_SIZE:
        raise ValueError("delta_cells has an invalid shape")
    block_arrays = {
        "k_ss": delta[:, PRIMITIVE_SLICES["bare_bubble"]]
        + delta[:, PRIMITIVE_SLICES["direct"]],
        "k_seta": delta[:, PRIMITIVE_SLICES["em_collective_left"]],
        "k_etas": delta[:, PRIMITIVE_SLICES["collective_em_right"]],
        "k_etaeta": delta[:, PRIMITIVE_SLICES["collective_bubble"]]
        + delta[:, PRIMITIVE_SLICES["collective_counterterm"]],
        "ward_rhs": np.concatenate(
            [delta[:, PRIMITIVE_SLICES["rhs_left"]], delta[:, PRIMITIVE_SLICES["rhs_right"]]],
            axis=1,
        ),
    }
    masses = {name: np.linalg.norm(values, axis=1) for name, values in block_arrays.items()}
    score = np.zeros(delta.shape[0], dtype=float)
    active = 0
    for values in masses.values():
        total = float(np.sum(values))
        if total > 0.0:
            score += values / total
            active += 1
    if active:
        score /= float(active)
    return masses, score


def concentration_area(mass: np.ndarray, target: float) -> float:
    values = np.asarray(mass, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return float("nan")
    ordered = np.sort(values)[::-1]
    count = int(np.searchsorted(np.cumsum(ordered), float(target) * total, side="left") + 1)
    return float(count / len(values))


def estimate_dwave_nodes(
    spec: object,
    *,
    fermi_level_eV: float = 0.0,
    samples: int = 2049,
) -> np.ndarray:
    """Locate normal-FS intersections with the two d-wave nodal diagonals."""

    count = max(int(samples), 65)
    ts = np.linspace(-np.pi, np.pi, count, dtype=float)
    nodes: list[tuple[float, float]] = []
    for sign in (1.0, -1.0):
        values = np.stack(
            [
                np.linalg.eigvalsh(np.asarray(spec.normal_hamiltonian(float(t), float(sign * t))))
                - float(fermi_level_eV)
                for t in ts
            ],
            axis=0,
        )
        for band in range(values.shape[1]):
            for index in range(count - 1):
                left, right = float(values[index, band]), float(values[index + 1, band])
                if left == 0.0:
                    root = float(ts[index])
                elif left * right > 0.0:
                    continue
                else:
                    root = float(
                        brentq(
                            lambda t: float(
                                np.linalg.eigvalsh(
                                    np.asarray(spec.normal_hamiltonian(float(t), float(sign * t)))
                                )[band]
                                - float(fermi_level_eV)
                            ),
                            float(ts[index]),
                            float(ts[index + 1]),
                        )
                    )
                point = (
                    float((root + np.pi) % (2.0 * np.pi) - np.pi),
                    float((sign * root + np.pi) % (2.0 * np.pi) - np.pi),
                )
                if not any(
                    np.linalg.norm(
                        ((np.asarray(point) - np.asarray(old) + np.pi) % (2.0 * np.pi)) - np.pi
                    )
                    < 1e-7
                    for old in nodes
                ):
                    nodes.append(point)
    return np.asarray(nodes, dtype=float).reshape(-1, 2)


def periodic_node_distances(points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    node_values = np.asarray(nodes, dtype=float)
    if node_values.size == 0:
        return np.full(values.shape[0], np.nan, dtype=float)
    delta = ((values[:, None, :] - node_values[None, :, :] + np.pi) % (2.0 * np.pi)) - np.pi
    return np.min(np.linalg.norm(delta, axis=2), axis=1)


__all__ = [
    "PRIMITIVE_SLICES",
    "PRIMITIVE_VECTOR_SIZE",
    "SpatialDiagnosticConfig",
    "block_mass_table",
    "components_from_primitive_vector",
    "concentration_area",
    "estimate_dwave_nodes",
    "evaluate_shift_spatial",
    "periodic_node_distances",
    "pointwise_primitive_vectors",
    "shift_rule",
]
