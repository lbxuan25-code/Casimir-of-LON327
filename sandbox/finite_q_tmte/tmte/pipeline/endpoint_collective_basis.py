"""Debug-only endpoint symmetric/antisymmetric collective basis diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.response.finite_q import add_bubble, thermal_expectation_bdg_from_hamiltonian
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

from ..adapters.bubble_adapter import TargetBareBlocks, _bands_for_point
from ..adapters.collective_adapter import collective_counterterm
from ..adapters.model_adapter import build_model_scan_inputs, shifted_uniform_bz_mesh, weights_for_points
from ..adapters.primitive_vertices_adapter import primitive_observable_vertices, primitive_source_vertices, primitive_spatial_contact_vertices
from ..io.writers import write_json
from ..theory.contacts import project_spatial_contact
from ..theory.conventions import SOURCE_ORDER_DIAGNOSTIC, finite_q_conventions, require_diagnostic_source_order, require_xi_matches_omega
from ..theory.frequency import frequency_payload, matsubara_xi_eV
from ..theory.vertices import target_vertices
from .contact_ablation import _shifted_payload, scaled_contact_blocks
from .eta_channel_ablation import _entries
from .nk_sweep import RATIO_EPS
from .shifted_average import average_bare_blocks_then_schur, shift_pairs_from_fractions
from .signed_decomposition import ENTRY_SPECS, decomposition_ratios

SCHEMA_VERSION = "finite_q_tmte_endpoint_collective_basis_v1"
DEFAULT_BASIS_MODES = ("current_2ch", "s_only_2ch", "a_only_2ch", "phase_s_only", "phase_a_only", "expanded_4ch")
SUMMARY_BASIS_ORDER = ("current_2ch", "s_only_2ch", "phase_s_only", "phase_a_only", "a_only_2ch", "expanded_4ch")
DEFAULT_COUNTERTERM_POLICY = "embed_existing_s_counterterm_zero_a_counterterm"


def endpoint_form_factors(ansatz: object, kx: float, ky: float, qx: float, qy: float, pairing_params: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return phi_minus, phi_plus, phi_s, phi_a for debug endpoint decomposition."""

    delta0 = float(pairing_params.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("endpoint form factors are undefined for delta0=0")
    phi_minus = np.asarray(ansatz.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, pairing_params), dtype=complex) / delta0
    phi_plus = np.asarray(ansatz.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, pairing_params), dtype=complex) / delta0
    phi_s = 0.5 * (phi_plus + phi_minus)
    phi_a = 0.5 * (phi_plus - phi_minus)
    return phi_minus, phi_plus, phi_s, phi_a


def amplitude_s_vertex(phi_s: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_s)
    return np.block([[zero, phi_s], [phi_s.conjugate().T, zero]]).astype(complex)


def phase_s_vertex(phi_s: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_s)
    return np.block([[zero, 1j * phi_s], [-1j * phi_s.conjugate().T, zero]]).astype(complex)


def amplitude_a_vertex(phi_a: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_a)
    return np.block([[zero, phi_a], [-phi_a.conjugate().T, zero]]).astype(complex)


def phase_a_vertex(phi_a: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_a)
    return np.block([[zero, 1j * phi_a], [1j * phi_a.conjugate().T, zero]]).astype(complex)


def amplitude_endpoint_vertex(phi_minus: np.ndarray, phi_plus: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_plus)
    return np.block([[zero, phi_plus], [phi_minus.conjugate().T, zero]]).astype(complex)


def phase_endpoint_vertex(phi_minus: np.ndarray, phi_plus: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_plus)
    return np.block([[zero, 1j * phi_plus], [-1j * phi_minus.conjugate().T, zero]]).astype(complex)


def endpoint_collective_vertices(
    *,
    ansatz: object,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    pairing_params: object,
    basis_mode: str,
) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
    """Return debug collective vertices and labels for one endpoint basis mode."""

    _, _, phi_s, phi_a = endpoint_form_factors(ansatz, kx, ky, qx, qy, pairing_params)
    vertices_by_name = {
        "amplitude_s": amplitude_s_vertex(phi_s),
        "phase_s": phase_s_vertex(phi_s),
        "amplitude_a": amplitude_a_vertex(phi_a),
        "phase_a": phase_a_vertex(phi_a),
    }
    if basis_mode == "current_2ch":
        return tuple(np.asarray(vertex, dtype=complex) for vertex in ansatz.collective_vertices(kx, ky, qx, qy, pairing_params)), ("amplitude_eta1", "phase_eta2")
    if basis_mode == "s_only_2ch":
        names = ("amplitude_s", "phase_s")
    elif basis_mode == "a_only_2ch":
        names = ("amplitude_a", "phase_a")
    elif basis_mode == "phase_s_only":
        names = ("phase_s",)
    elif basis_mode == "phase_a_only":
        names = ("phase_a",)
    elif basis_mode == "expanded_4ch":
        names = ("amplitude_s", "phase_s", "amplitude_a", "phase_a")
    else:
        raise ValueError(f"unknown endpoint collective basis mode: {basis_mode}")
    return tuple(vertices_by_name[name] for name in names), names


def embedded_counterterm(
    existing_counterterm: np.ndarray,
    collective_order: tuple[str, ...],
    *,
    policy: str = DEFAULT_COUNTERTERM_POLICY,
) -> np.ndarray:
    """Embed the existing symmetric 2x2 counterterm into a debug collective basis."""

    existing = np.asarray(existing_counterterm, dtype=complex)
    if policy == "zero_all_counterterms":
        return np.zeros((len(collective_order), len(collective_order)), dtype=complex)
    result = np.zeros((len(collective_order), len(collective_order)), dtype=complex)
    if policy == "embed_existing_s_counterterm_zero_a_counterterm":
        if collective_order in {("amplitude_eta1", "phase_eta2"), ("amplitude_s", "phase_s")}:
            result[:, :] = existing[np.ix_([0, 1], [0, 1])]
        else:
            source_index = {"amplitude_s": 0, "phase_s": 1}
            for out_i, name_i in enumerate(collective_order):
                if name_i not in source_index:
                    continue
                for out_j, name_j in enumerate(collective_order):
                    if name_j not in source_index:
                        continue
                    result[out_i, out_j] = existing[source_index[name_i], source_index[name_j]]
        return result
    if policy == "diagonal_regularizer_for_a_channels":
        result = embedded_counterterm(existing, collective_order, policy=DEFAULT_COUNTERTERM_POLICY)
        scale = float(np.max(np.abs(existing))) if existing.size else 0.0
        for idx, name in enumerate(collective_order):
            if name in {"amplitude_a", "phase_a"}:
                result[idx, idx] = scale
        return result
    raise ValueError(f"unknown endpoint collective counterterm policy: {policy}")


def compute_endpoint_basis_bare_blocks(
    *,
    spec: object,
    ansatz: object,
    q_model: np.ndarray,
    xi_eV: float,
    k_points: np.ndarray,
    weights: np.ndarray,
    config: object,
    pairing_params: object,
    basis_mode: str,
    counterterm_policy: str = DEFAULT_COUNTERTERM_POLICY,
    options: FiniteQEngineOptions | None = None,
    source_order: tuple[str, ...] = SOURCE_ORDER_DIAGNOSTIC,
) -> TargetBareBlocks:
    """Compute debug endpoint-basis bare blocks through the target-basis finite-q path."""

    opts = options or FiniteQEngineOptions(include_phase_correction=False, collective_mode="amplitude_phase", collective_counterterm="goldstone_gap_equation")
    require_diagnostic_source_order(source_order)
    require_xi_matches_omega(xi_eV, config.omega_eV)
    q, points, mesh_weights = validate_finite_q_inputs(q_model, k_points, weights, config)
    conventions = finite_q_conventions(q, xi_eV)
    qx, qy = float(q[0]), float(q[1])
    if np.linalg.norm(q) <= 1e-14:
        raise ValueError("endpoint collective basis debug path requires q > q_tol")

    probe_vertices, collective_order = endpoint_collective_vertices(ansatz=ansatz, kx=float(points[0, 0]), ky=float(points[0, 1]), qx=qx, qy=qy, pairing_params=pairing_params, basis_mode=basis_mode)
    n_source = len(source_order)
    n_collective = len(probe_vertices)
    k_ss_bubble = np.zeros((n_source, n_source), dtype=complex)
    k_ss_contact = np.zeros((n_source, n_source), dtype=complex)
    k_seta = np.zeros((n_source, n_collective), dtype=complex)
    k_etas = np.zeros((n_collective, n_source), dtype=complex)
    k_etaeta_bubble = np.zeros((n_collective, n_collective), dtype=complex)

    for weight, (kx_value, ky_value) in zip(mesh_weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        bands_minus, bands_plus, delta_mid = _bands_for_point(spec, ansatz, pairing_params, kx, ky, qx, qy, shared=False)
        occ_minus = fermi_function(bands_minus.energies, config.fermi_level_eV, config.temperature_eV)
        occ_plus = fermi_function(bands_plus.energies, config.fermi_level_eV, config.temperature_eV)

        left_primitive = primitive_observable_vertices(spec, kx, ky, qx, qy, current_vertex=opts.current_vertex)
        right_primitive = primitive_source_vertices(spec, kx, ky, qx, qy, current_vertex=opts.current_vertex)
        left_targets = target_vertices(*left_primitive, conventions, source_order=source_order)
        right_targets = target_vertices(*right_primitive, conventions, source_order=source_order)
        add_bubble(k_ss_bubble, left_targets, right_targets, bands_minus.energies, bands_minus.states, occ_minus, bands_plus.energies, bands_plus.states, occ_plus, config.omega_eV, float(weight), config=config, static_limit=False)

        coll, _ = endpoint_collective_vertices(ansatz=ansatz, kx=kx, ky=ky, qx=qx, qy=qy, pairing_params=pairing_params, basis_mode=basis_mode)
        add_bubble(k_seta, left_targets, coll, bands_minus.energies, bands_minus.states, occ_minus, bands_plus.energies, bands_plus.states, occ_plus, config.omega_eV, float(weight), config=config, static_limit=False)
        add_bubble(k_etas, coll, right_targets, bands_minus.energies, bands_minus.states, occ_minus, bands_plus.energies, bands_plus.states, occ_plus, config.omega_eV, float(weight), config=config, static_limit=False)
        add_bubble(k_etaeta_bubble, coll, coll, bands_minus.energies, bands_minus.states, occ_minus, bands_plus.energies, bands_plus.states, occ_plus, config.omega_eV, float(weight), config=config, static_limit=False)

        h_mid = bdg_hamiltonian_from_model_pairing(spec, kx, ky, delta_mid)
        spatial = np.zeros((2, 2), dtype=complex)
        contact_vertices = primitive_spatial_contact_vertices(spec, kx, ky, qx, qy, current_vertex=opts.current_vertex)
        for i, di in enumerate(("x", "y")):
            for j, dj in enumerate(("x", "y")):
                spatial[i, j] += -float(weight) * thermal_expectation_bdg_from_hamiltonian(h_mid, contact_vertices[(di, dj)], config)
        k_ss_contact += project_spatial_contact(spatial, conventions, source_order=source_order)

    existing_counterterm = collective_counterterm(ansatz, config, points, mesh_weights, pairing_params)
    counterterm = embedded_counterterm(existing_counterterm, collective_order, policy=counterterm_policy)
    k_ss = k_ss_bubble + k_ss_contact
    k_etaeta = k_etaeta_bubble + counterterm
    return TargetBareBlocks(
        source_order=source_order,
        conventions=conventions,
        k_ss_bubble=k_ss_bubble,
        k_ss_contact=k_ss_contact,
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta_bubble=k_etaeta_bubble,
        k_etaeta_counterterm=counterterm,
        k_etaeta=k_etaeta,
        metadata={
            "adapter": "debug_endpoint_collective_basis_v1",
            "basis_mode": basis_mode,
            "collective_order": list(collective_order),
            "counterterm_policy": counterterm_policy,
            "valid_for_casimir_input": False,
        },
    )


def endpoint_basis_result(
    *,
    basis_mode: str,
    response: object,
    counterterm_policy: str,
    current_diagnostics: dict[str, Any] | None = None,
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    blocks = response.bare_blocks
    ratios = decomposition_ratios(response.schur.effective, eps=ratio_eps, source_order=blocks.source_order)
    diagnostics = {
        "gauge_row_norm": ratios["gauge_row_norm"],
        "gauge_col_norm": ratios["gauge_col_norm"],
        "gauge_gg_norm": ratios["gauge_gg_norm"],
        "physical_matrix_norm": ratios["physical_matrix_norm"],
        "gauge_over_physical": ratios["gauge_over_physical"],
        "gauge_over_tm_abs": ratios["gauge_over_tm_abs"],
        "gauge_gg_over_tm_abs": ratios["gauge_gg_over_tm_abs"],
        "ratio_eps": ratios["ratio_eps"],
        "etaeta_condition_number": response.schur.etaeta_condition_number,
        "schur_solve_method": response.schur.solve_method,
        "schur_numerically_suspect": response.schur.numerically_suspect,
        "valid_for_casimir_input": False,
    }
    if current_diagnostics is not None:
        diagnostics.update(
            {
                "delta_gauge_over_tm_abs_vs_current_2ch": diagnostics["gauge_over_tm_abs"] - current_diagnostics["gauge_over_tm_abs"],
                "delta_gauge_gg_over_tm_abs_vs_current_2ch": diagnostics["gauge_gg_over_tm_abs"] - current_diagnostics["gauge_gg_over_tm_abs"],
                "delta_gauge_row_norm_vs_current_2ch": diagnostics["gauge_row_norm"] - current_diagnostics["gauge_row_norm"],
                "delta_gauge_gg_norm_vs_current_2ch": diagnostics["gauge_gg_norm"] - current_diagnostics["gauge_gg_norm"],
            }
        )
    return {
        "basis_mode": basis_mode,
        "collective_order": list(blocks.metadata["collective_order"]),
        "counterterm_policy": counterterm_policy,
        "Schur_correction_entries": _entries(response.schur.correction, tuple(name for name, _, _ in ENTRY_SPECS), source_order=blocks.source_order),
        "K_eff_entries": _entries(response.schur.effective, tuple(name for name, _, _ in ENTRY_SPECS), source_order=blocks.source_order),
        "diagnostics": diagnostics,
        "schur": {
            "solve_method": response.schur.solve_method,
            "etaeta_condition_number": response.schur.etaeta_condition_number,
            "condition_threshold": response.schur.condition_threshold,
            "numerically_suspect": response.schur.numerically_suspect,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def endpoint_collective_basis_payload(
    *,
    model_name: str,
    pairing_name: str,
    frequency: dict[str, Any],
    debug_parameters: dict[str, Any],
    basis_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "valid_for_casimir_input": False,
            "reason": "endpoint_collective_basis_debug_not_casimir_input",
        },
        "model": {"name": model_name, "pairing": pairing_name, "valid_for_casimir_input": False},
        "frequency": frequency,
        "debug_parameters": {
            **debug_parameters,
            "debug_only_endpoint_collective_basis": True,
            "average_order": "average_blocks_then_schur",
            "valid_for_casimir_input": False,
        },
        "source_order_diagnostic": list(SOURCE_ORDER_DIAGNOSTIC),
        "basis_results": list(basis_results),
        "valid_for_casimir_input": False,
    }


def run_endpoint_collective_basis(
    *,
    model_name: str,
    pairing_name: str,
    matsubara_index: int,
    temperature_K: float,
    q_value: float,
    nk: int,
    delta0_eV: float | None = None,
    eta_eV: float = 1e-8,
    shift_fractions: Sequence[float] = (0.0,),
    contact_scale: float = 1.0,
    basis_modes: Sequence[str] = DEFAULT_BASIS_MODES,
    counterterm_policy: str = DEFAULT_COUNTERTERM_POLICY,
    ratio_eps: float = RATIO_EPS,
) -> dict[str, Any]:
    xi_eV = matsubara_xi_eV(matsubara_index, temperature_K)
    inputs = build_model_scan_inputs(
        model_name=model_name,
        pairing_name=pairing_name,
        xi_eV=xi_eV,
        nk=nk,
        delta0_eV=delta0_eV,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
    )
    q_model = np.asarray([float(q_value), 0.0], dtype=float)
    shifts = shift_pairs_from_fractions(shift_fractions)
    responses_by_mode: dict[str, Any] = {}
    for mode in basis_modes:
        scaled_blocks: list[TargetBareBlocks] = []
        for sx, sy in shifts:
            points = shifted_uniform_bz_mesh(nk, sx, sy)
            weights = weights_for_points(points)
            blocks = compute_endpoint_basis_bare_blocks(
                spec=inputs.spec,
                ansatz=inputs.ansatz,
                q_model=q_model,
                xi_eV=xi_eV,
                k_points=points,
                weights=weights,
                config=inputs.config,
                pairing_params=inputs.pairing_params,
                basis_mode=str(mode),
                counterterm_policy=counterterm_policy,
            )
            scaled_blocks.append(scaled_contact_blocks(blocks, contact_scale))
        responses_by_mode[str(mode)] = average_bare_blocks_then_schur(scaled_blocks)

    current_response = responses_by_mode["current_2ch"]
    current_ratios = decomposition_ratios(current_response.schur.effective, eps=ratio_eps, source_order=current_response.bare_blocks.source_order)
    current_diagnostics = {
        "gauge_over_tm_abs": current_ratios["gauge_over_tm_abs"],
        "gauge_gg_over_tm_abs": current_ratios["gauge_gg_over_tm_abs"],
        "gauge_row_norm": current_ratios["gauge_row_norm"],
        "gauge_gg_norm": current_ratios["gauge_gg_norm"],
    }
    basis_results = []
    for mode in basis_modes:
        result = endpoint_basis_result(
            basis_mode=str(mode),
            response=responses_by_mode[str(mode)],
            counterterm_policy=counterterm_policy,
            current_diagnostics=current_diagnostics,
            ratio_eps=ratio_eps,
        )
        if str(mode) == "current_2ch":
            result["diagnostics"].update(
                {
                    "delta_gauge_over_tm_abs_vs_current_2ch": 0.0,
                    "delta_gauge_gg_over_tm_abs_vs_current_2ch": 0.0,
                    "delta_gauge_row_norm_vs_current_2ch": 0.0,
                    "delta_gauge_gg_norm_vs_current_2ch": 0.0,
                }
            )
        basis_results.append(result)

    return endpoint_collective_basis_payload(
        model_name=model_name,
        pairing_name=pairing_name,
        frequency=frequency_payload(matsubara_index, temperature_K),
        debug_parameters={
            "debug_only_endpoint_collective_basis": True,
            "q_model_convention": "q_along_x_only",
            "q_value": float(q_value),
            "nk": int(nk),
            "shift_fractions": [float(value) for value in shift_fractions],
            "num_shifted_meshes": len(shifts),
            "contact_scale": float(contact_scale),
            "basis_modes": [str(mode) for mode in basis_modes],
            "counterterm_policy": counterterm_policy,
            "ratio_eps": float(ratio_eps),
            "average_order": "average_blocks_then_schur",
            "shifted_mesh_average": _shifted_payload(shift_fractions, shifts),
            "valid_for_casimir_input": False,
        },
        basis_results=basis_results,
    )


def run_and_write_endpoint_collective_basis(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_endpoint_collective_basis(**kwargs)
    write_json(Path(output_dir) / "endpoint_collective_basis.json", payload)
    return payload
