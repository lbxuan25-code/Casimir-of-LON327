#!/usr/bin/env python3
"""Compact superconducting BdG finite-q Ward residual audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from lno327.conductivity import KuboConfig, k_weights  # noqa: E402
from lno327.finite_q_engine import FiniteQEngineOptions, finite_q_bdg_response_from_ansatz  # noqa: E402
from lno327.pairing import PairingAmplitudes, build_pairing_ansatz  # noqa: E402
from lno327.ward_response import normal_physical_density_current_response_components_imag_axis  # noqa: E402
from normal_finite_q_ward_audit import (  # noqa: E402
    DIRECTION_VECTORS,
    _print_progress,
    actual_twist_offsets,
    uniform_bz_mesh_twisted,
)

WARD_COMPONENT_LABELS = ("density", "current_x", "current_y")
MAX_JSON_SIZE_MB = 50.0


def _physical_ward_residuals(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    response = np.asarray(matrix, dtype=complex)
    qx, qy = float(q[0]), float(q[1])
    left = 1j * omega_eV * response[0, :] + qx * response[1, :] + qy * response[2, :]
    right = 1j * omega_eV * response[:, 0] - qx * response[:, 1] - qy * response[:, 2]
    return left, right


def _nested_twist_mesh(nk: int, actual_twist_count: int, twist_mode: str) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    offsets = actual_twist_offsets(actual_twist_count, twist_mode)
    meshes = [uniform_bz_mesh_twisted(nk, offset) for offset in offsets]
    points = np.vstack(meshes)
    weights = k_weights(points)
    weight_sum = float(np.sum(weights))
    if abs(weight_sum - 1.0) >= 1e-12:
        raise ValueError(f"twist quadrature weights sum to {weight_sum}, not 1")
    return points, weights, [[float(x), float(y)] for x, y in offsets]


def _norm(value: np.ndarray | complex) -> float:
    return float(np.linalg.norm(value))


def _response_norms(matrix: np.ndarray) -> dict[str, float]:
    response = np.asarray(matrix, dtype=complex)
    return {
        "total_response_norm": _norm(response),
        "density_density_norm": float(abs(response[0, 0])),
        "density_current_block_norm": _norm(response[0:1, 1:3]),
        "current_density_block_norm": _norm(response[1:3, 0:1]),
        "current_current_block_norm": _norm(response[1:3, 1:3]),
    }


def _ward_norms(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, float]:
    left, right = _physical_ward_residuals(matrix, omega_eV, q)
    q_norm = float(np.linalg.norm(q))
    return {
        "left_ward_residual_norm": _norm(left),
        "right_ward_residual_norm": _norm(right),
        "left_ward_residual_over_q_norm": float(_norm(left) / q_norm),
        "right_ward_residual_over_q_norm": float(_norm(right) / q_norm),
        "density_column_residual_norm": float(abs(right[0])),
        "current_x_column_residual_norm": float(abs(right[1])),
        "current_y_column_residual_norm": float(abs(right[2])),
        "density_row_residual_norm": float(abs(left[0])),
        "current_x_row_residual_norm": float(abs(left[1])),
        "current_y_row_residual_norm": float(abs(left[2])),
    }


def _normal_like_em_ward_diagnostic(matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    values = _ward_norms(matrix, omega_eV, q)
    return {
        "normal_like_em_left_residual_norm": values["left_ward_residual_norm"],
        "normal_like_em_right_residual_norm": values["right_ward_residual_norm"],
        "normal_like_em_left_residual_over_q_norm": values["left_ward_residual_over_q_norm"],
        "normal_like_em_right_residual_over_q_norm": values["right_ward_residual_over_q_norm"],
        "normal_like_em_residual_interpretation": (
            "bare_em_block_only_not_expected_to_close_in_superconducting_state; "
            "normal-state electromagnetic Ward contraction is diagnostic-only for BdG"
        ),
    }


def _collective_channel_count(response: Any, variant: str) -> int:
    if variant == "phase_schur":
        return 1
    if variant == "amplitude_phase_schur":
        return int(np.asarray(response.collective_total).shape[0])
    return 0


def _mixed_norms(response: Any, variant: str) -> tuple[float, float, float]:
    if variant == "phase_schur":
        left = np.asarray(response.phase_coupling_left, dtype=complex).reshape(3, 1)
        right = np.asarray(response.phase_coupling_right, dtype=complex).reshape(1, 3)
        kernel = np.asarray([[response.phase_phase_total]], dtype=complex)
    elif variant == "amplitude_phase_schur":
        left = np.asarray(response.em_collective_left, dtype=complex)
        right = np.asarray(response.collective_em_right, dtype=complex)
        kernel = np.asarray(response.collective_total, dtype=complex)
    else:
        left = np.zeros((3, 0), dtype=complex)
        right = np.zeros((0, 3), dtype=complex)
        kernel = np.zeros((0, 0), dtype=complex)
    return _norm(left), _norm(right), _norm(kernel)


def _extended_ward_diagnostic(response: Any, variant: str, matrix: np.ndarray, omega_eV: float, q: np.ndarray) -> dict[str, Any]:
    gi_left, gi_right = _physical_ward_residuals(matrix, omega_eV, q)
    q_norm = float(np.linalg.norm(q))
    mixed_left_norm, mixed_right_norm, kernel_norm = _mixed_norms(response, variant)
    tangent_available = variant in {"phase_schur", "amplitude_phase_schur"}
    if variant == "bare_bdg":
        unsupported_reason = "bare electromagnetic block alone is not a closed superconducting BdG Ward object"
    else:
        unsupported_reason = (
            "order-parameter tangent basis is provided by finite_q_engine collective vertices, "
            "but explicit W_eta coefficients are not separately exposed in this compact audit"
        )
    return {
        "extended_left_residual_norm": None,
        "extended_right_residual_norm": None,
        "extended_left_residual_over_q_norm": None,
        "extended_right_residual_over_q_norm": None,
        "em_collective_mixed_left_norm": mixed_left_norm,
        "em_collective_mixed_right_norm": mixed_right_norm,
        "collective_kernel_ward_norm": kernel_norm,
        "schur_gauge_invariant_left_residual_norm": _norm(gi_left),
        "schur_gauge_invariant_right_residual_norm": _norm(gi_right),
        "schur_gauge_invariant_left_residual_over_q_norm": float(_norm(gi_left) / q_norm),
        "schur_gauge_invariant_right_residual_over_q_norm": float(_norm(gi_right) / q_norm),
        "extended_ward_implemented": False,
        "order_parameter_tangent_available": bool(tangent_available),
        "order_parameter_tangent_rule": (
            "reuses finite_q_engine pairing ansatz collective_vertices / phase_coupling / em_collective blocks"
            if tangent_available
            else "not_applicable_for_bare_em_block"
        ),
        "phase_tangent_included": variant in {"phase_schur", "amplitude_phase_schur"},
        "amplitude_tangent_included": variant == "amplitude_phase_schur",
        "dwave_form_factor_tangent_used": "pairing ansatz dependent; true for dwave via collective_vertices when variant uses collective sector",
        "unsupported_reason": unsupported_reason,
    }


def _collective_metadata(response: Any, variant: str, matrix: np.ndarray) -> dict[str, Any]:
    metadata = response.metadata
    if variant == "bare_bdg":
        correction = np.zeros_like(matrix)
        return {
            "collective_mode": "none",
            "schur_applied": False,
            "schur_sign_convention": "not_applied",
            "schur_denominator_norm": 0.0,
            "schur_condition_estimate": None,
            "collective_block_norm": _norm(response.collective_total),
            "collective_correction_norm": _norm(correction),
        }
    if variant == "phase_schur":
        correction = response.bare_total - response.minus_schur
        return {
            "collective_mode": "phase_only",
            "schur_applied": metadata.get("phase_only_schur_status") not in {None, "skipped_zero_phase_kernel"},
            "schur_sign_convention": "minus",
            "schur_denominator_norm": float(abs(response.phase_phase_total)),
            "schur_condition_estimate": None,
            "collective_block_norm": float(abs(response.phase_phase_total)),
            "collective_correction_norm": _norm(correction),
        }
    correction = response.bare_total - response.amplitude_phase_schur
    return {
        "collective_mode": "amplitude_phase",
        "schur_applied": str(metadata.get("amplitude_phase_schur_status")) not in {"not_used", "skipped"},
        "schur_sign_convention": "minus_matrix_schur",
        "schur_denominator_norm": _norm(response.collective_total),
        "schur_condition_estimate": metadata.get("collective_total_condition_number"),
        "collective_block_norm": _norm(response.collective_total),
        "collective_correction_norm": _norm(correction),
    }


def _variant_matrix(response: Any, variant: str) -> tuple[np.ndarray | None, bool, str | None]:
    if variant == "bare_bdg":
        return response.bare_total, True, None
    if variant == "phase_schur":
        return response.minus_schur, True, None
    if variant == "amplitude_phase_schur":
        if response.amplitude_phase_schur is None:
            return None, False, "amplitude_phase_schur_missing"
        return response.amplitude_phase_schur, True, None
    return None, False, f"unsupported response variant: {variant}"


def _case_worker(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        pairing_name,
        nk,
        actual_twist_count,
        twist_mode,
        q_direction_name,
        q_value,
        direction,
        response_variants,
        omega_eV,
        temperature_K,
        eta_eV,
        delta0_eV,
    ) = args
    started = time.perf_counter()
    q = float(q_value) * np.asarray(direction, dtype=float) / float(np.linalg.norm(direction))
    points, weights, offsets = _nested_twist_mesh(int(nk), int(actual_twist_count), str(twist_mode))
    config = KuboConfig.from_kelvin(
        omega_eV=float(omega_eV),
        temperature_K=float(temperature_K),
        eta_eV=float(eta_eV),
        output_si=False,
    )
    ansatz = build_pairing_ansatz(str(pairing_name), phase_vertex="bond_endpoint_gauge")
    pairing_params = PairingAmplitudes(delta0_eV=float(delta0_eV))
    options = FiniteQEngineOptions(
        current_vertex="peierls",
        collective_mode="amplitude_phase",
        collective_counterterm="goldstone_gap_equation",
        include_phase_phase_direct=True,
    )
    response = finite_q_bdg_response_from_ansatz(
        ansatz,
        float(omega_eV),
        q,
        points,
        weights,
        config,
        pairing_params,
        options,
    )
    rows: list[dict[str, Any]] = []
    matrices_by_variant: dict[str, np.ndarray] = {}
    for variant in response_variants:
        matrix, supported, reason = _variant_matrix(response, str(variant))
        if matrix is None:
            rows.append(
                {
                    "pairing_name": str(pairing_name),
                    "response_variant": str(variant),
                    "supported": False,
                    "unsupported_reason": reason,
                    "valid_for_casimir_input": False,
                }
            )
            continue
        matrices_by_variant[str(variant)] = matrix
        row = {
            "pairing_name": str(pairing_name),
            "delta0_eV": float(delta0_eV),
            "temperature_K": float(temperature_K),
            "omega_eV": float(omega_eV),
            "eta_eV": float(eta_eV),
            "nk": int(nk),
            "q_direction": str(q_direction_name),
            "q_norm": float(np.linalg.norm(q)),
            "actual_twist_count": int(actual_twist_count),
            "twist_mode": str(twist_mode),
            "adaptive_mode": "none",
            "response_variant": str(variant),
            "supported": True,
            "effective_total_nodes": int(points.shape[0]),
            "weight_sum": float(np.sum(weights)),
            "abs_weight_sum_minus_one": float(abs(np.sum(weights) - 1.0)),
            "twist_offset_rule": (
                "q-independent nested symmetry-preserving equal-weight twist quadrature; "
                "offsets are not fitted to Ward residuals"
            ),
            "twist_nested_family": "halton_orbit_prefix_24_32_48",
            "twist_q_independent": True,
            "twist_equal_weight": True,
            "twist_residual_fitted": False,
            "twist_symmetry_inversion": True,
            "twist_symmetry_xy_exchange": True,
            "valid_for_casimir_input": False,
        }
        normal_like = _ward_norms(matrix, float(omega_eV), q)
        row.update(_normal_like_em_ward_diagnostic(matrix, float(omega_eV), q))
        row.update(_extended_ward_diagnostic(response, str(variant), matrix, float(omega_eV), q))
        row.update(
            {
                "density_column_residual_norm": normal_like["density_column_residual_norm"],
                "current_x_column_residual_norm": normal_like["current_x_column_residual_norm"],
                "current_y_column_residual_norm": normal_like["current_y_column_residual_norm"],
                "density_row_residual_norm": normal_like["density_row_residual_norm"],
                "current_x_row_residual_norm": normal_like["current_x_row_residual_norm"],
                "current_y_row_residual_norm": normal_like["current_y_row_residual_norm"],
            }
        )
        row.update(_response_norms(matrix))
        row.update(_collective_metadata(response, str(variant), matrix))
        rows.append(row)
    normal_components = normal_physical_density_current_response_components_imag_axis(points, config, q, weights)
    normal_total = normal_components["total"]
    normal_left, normal_right = _physical_ward_residuals(normal_total, float(omega_eV), q)
    comparison_rows = []
    for corrected_variant in ("bare_bdg", "phase_schur", "amplitude_phase_schur"):
        if corrected_variant not in matrices_by_variant:
            continue
        sc_matrix = matrices_by_variant[corrected_variant]
        sc_left, sc_right = _physical_ward_residuals(sc_matrix, float(omega_eV), q)
        comparison_rows.append(
            {
                "pairing_name": str(pairing_name),
                "response_variant": corrected_variant,
                "nk": int(nk),
                "q_direction": str(q_direction_name),
                "q_norm": float(np.linalg.norm(q)),
                "actual_twist_count": int(actual_twist_count),
                "normal_current_current_block_norm": _norm(normal_total[1:3, 1:3]),
                "superconducting_current_current_block_norm": _norm(sc_matrix[1:3, 1:3]),
                "normal_ward_residual_norm": float(max(_norm(normal_left), _norm(normal_right))),
                "superconducting_ward_residual_norm": float(max(_norm(sc_left), _norm(sc_right))),
                "valid_for_casimir_input": False,
            }
        )
    return {
        "rows": rows,
        "comparison_rows": comparison_rows,
        "convergence_items": [
            {
                "pairing_name": str(pairing_name),
                "response_variant": variant,
                "q_direction": str(q_direction_name),
                "q_norm": float(np.linalg.norm(q)),
                "temperature_K": float(temperature_K),
                "nk": int(nk),
                "actual_twist_count": int(actual_twist_count),
                "effective_total_nodes": int(points.shape[0]),
                "matrix": matrix,
                "current_current_block": matrix[1:3, 1:3],
                "ward_residual_norm": float(max(_ward_norms(matrix, float(omega_eV), q)["left_ward_residual_norm"], _ward_norms(matrix, float(omega_eV), q)["right_ward_residual_norm"])),
            }
            for variant, matrix in matrices_by_variant.items()
        ],
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _convergence_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for item in items:
        key = (
            item["pairing_name"],
            item["response_variant"],
            item["q_direction"],
            f"{float(item['q_norm']):.16g}",
            float(item["temperature_K"]),
        )
        grouped.setdefault(key, []).append(item)
    rows = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda row: (int(row["nk"]), int(row["actual_twist_count"])))
        for level_a, level_b in zip(ordered, ordered[1:], strict=False):
            response_b_norm = max(_norm(level_b["matrix"]), 1e-300)
            block_b_norm = max(_norm(level_b["current_current_block"]), 1e-300)
            rows.append(
                {
                    "diagnostic_only": True,
                    "valid_for_casimir_input": False,
                    "level_a": {
                        "nk": int(level_a["nk"]),
                        "actual_twist_count": int(level_a["actual_twist_count"]),
                    },
                    "level_b": {
                        "nk": int(level_b["nk"]),
                        "actual_twist_count": int(level_b["actual_twist_count"]),
                    },
                    "pairing_name": level_b["pairing_name"],
                    "response_variant": level_b["response_variant"],
                    "q_direction": level_b["q_direction"],
                    "q_norm": float(level_b["q_norm"]),
                    "response_relative_change_norm": float(_norm(level_b["matrix"] - level_a["matrix"]) / response_b_norm),
                    "current_current_block_relative_change_norm": float(
                        _norm(level_b["current_current_block"] - level_a["current_current_block"]) / block_b_norm
                    ),
                    "ward_residual_change_norm": float(abs(level_b["ward_residual_norm"] - level_a["ward_residual_norm"])),
                    "cost_ratio_effective_nodes": float(level_b["effective_total_nodes"] / max(level_a["effective_total_nodes"], 1)),
                }
            )
    return rows


def run_bdg_finite_q_ward_audit(
    *,
    pairings: tuple[str, ...],
    response_variants: tuple[str, ...],
    omega_eV: float,
    temperature_K: float,
    eta_eV: float,
    delta0_eV: float,
    nk_values: tuple[int, ...],
    actual_twist_counts: tuple[int, ...],
    twist_mode: str,
    q_values: tuple[float, ...],
    directions: tuple[str, ...],
    workers: int,
    progress_enabled: bool,
) -> dict[str, Any]:
    if twist_mode != "nested_symmetric":
        raise ValueError("BdG compact Ward audit currently supports --twist-mode nested_symmetric")
    tasks = [
        (
            pairing,
            nk,
            actual_twist_count,
            twist_mode,
            direction_name,
            q_value,
            DIRECTION_VECTORS[direction_name],
            response_variants,
            omega_eV,
            temperature_K,
            eta_eV,
            delta0_eV,
        )
        for pairing in pairings
        for nk in nk_values
        for actual_twist_count in actual_twist_counts
        for direction_name in directions
        for q_value in q_values
    ]
    progress_enabled = bool(progress_enabled and sys.stdout.isatty())
    _print_progress(0, len(tasks), enabled=progress_enabled)
    started = time.perf_counter()
    if workers > 1:
        results = []
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_case_worker, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                _print_progress(completed, len(tasks), enabled=progress_enabled)
        backend = "concurrent.futures.ProcessPoolExecutor"
    else:
        results = []
        for completed, task in enumerate(tasks, start=1):
            results.append(_case_worker(task))
            _print_progress(completed, len(tasks), enabled=progress_enabled)
        backend = "sequential"
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    convergence_items: list[dict[str, Any]] = []
    worker_runtime = 0.0
    for result in results:
        rows.extend(result["rows"])
        comparison_rows.extend(result["comparison_rows"])
        convergence_items.extend(result["convergence_items"])
        worker_runtime += float(result["runtime_seconds"])
    return {
        "audit_name": "bdg_finite_q_ward_audit",
        "scope": "diagnostic_only_superconducting_bdg_finite_q_ward_residual_summary",
        "ward_formula_scope": (
            "superconducting BdG Ward closure requires extended electromagnetic + order-parameter collective "
            "kernel diagnostics; bare electromagnetic normal-like residual is not a closure criterion"
        ),
        "pairing_names": list(pairings),
        "response_variants": list(response_variants),
        "omega_eV": float(omega_eV),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "delta0_eV": float(delta0_eV),
        "nk_values": [int(value) for value in nk_values],
        "actual_twist_counts": [int(value) for value in actual_twist_counts],
        "twist_mode": twist_mode,
        "adaptive_mode": "none",
        "component_labels": list(WARD_COMPONENT_LABELS),
        "bdg_extended_ward_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "normal_like_em_ward_residual": (
                "retained as electromagnetic-block diagnostic only; not used as superconducting BdG closure"
            ),
            "rows": rows,
        },
        "bdg_ward_convergence_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "rows": _convergence_rows(convergence_items),
        },
        "normal_reference_comparison_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "interpretation": (
                "normal-state closure formula is only for normal response; this block is an integration-error "
                "reference and is not used to judge superconducting BdG Ward closure"
            ),
            "rows": comparison_rows,
        },
        "runtime_profile_summary": {
            "diagnostic_only": True,
            "valid_for_casimir_input": False,
            "total_runtime_seconds": float(time.perf_counter() - started),
            "worker_runtime_seconds_sum": float(worker_runtime),
            "workers": int(workers),
            "parallel_backend": backend,
        },
        "output_format": {
            "summary_only": True,
            "removed_large_fields": [
                "per_k_residuals",
                "full_response_matrices",
                "band_basis_matrix_dumps",
                "4x4_matrix_entries",
                "eigenvectors",
                "full_bdg_spectrum_dump",
                "full_collective_matrices_per_k",
            ],
            "max_expected_file_size_mb": 10.0,
            "github_safe_output": True,
        },
        "ward_identity_closed": False,
        "valid_for_casimir_input": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    if size_mb > MAX_JSON_SIZE_MB:
        raise RuntimeError(f"BdG Ward audit JSON is {size_mb:.2f} MB, above {MAX_JSON_SIZE_MB:.1f} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 compact superconducting BdG finite-q Ward residual audit。")
    parser.add_argument("--temperature-K", type=float, default=30.0)
    parser.add_argument("--omega-eV", type=float, default=0.01)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--delta0-eV", type=float, default=0.04)
    parser.add_argument("--nk-values", nargs="+", type=int, default=[41])
    parser.add_argument("--actual-twist-counts", nargs="+", type=int, default=[32])
    parser.add_argument("--twist-mode", choices=("nested_symmetric",), default="nested_symmetric")
    parser.add_argument("--adaptive-mode", choices=("none",), default="none")
    parser.add_argument("--pairings", nargs="+", choices=("onsite_s", "spm", "dwave"), default=["onsite_s", "spm", "dwave"])
    parser.add_argument("--response-variants", nargs="+", choices=("bare_bdg", "phase_schur", "amplitude_phase_schur"), default=["bare_bdg", "phase_schur", "amplitude_phase_schur"])
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--directions", nargs="+", choices=tuple(DIRECTION_VECTORS), default=["x", "diagonal"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    payload = run_bdg_finite_q_ward_audit(
        pairings=tuple(args.pairings),
        response_variants=tuple(args.response_variants),
        omega_eV=args.omega_eV,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        delta0_eV=args.delta0_eV,
        nk_values=tuple(args.nk_values),
        actual_twist_counts=tuple(args.actual_twist_counts),
        twist_mode=args.twist_mode,
        q_values=tuple(args.q_values),
        directions=tuple(args.directions),
        workers=max(1, int(args.workers)),
        progress_enabled=not bool(args.no_progress),
    )
    if args.json_output is not None:
        _write_json(args.json_output, payload)
    print(
        "BdG finite-q Ward audit prepared: "
        f"pairings={payload['pairing_names']}, actual_twist_counts={payload['actual_twist_counts']}, "
        f"valid_for_casimir_input={payload['valid_for_casimir_input']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
