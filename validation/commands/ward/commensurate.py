"""Test the d-wave exact-static gauge contract on a q-commensurate periodic grid.

The external momentum is not entered as an approximate decimal. It is constructed
from integer grid shifts,

    q = (2 pi / nk) * (mx, my),

so translation by q is an exact permutation of the complete ``nk x nk`` tensor
lattice. All 48 complex primitive quantities share the same points and weights,
the full periodic average is formed before one Schur complement, and no static
longitudinal projection is applied.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.commensurate_periodic import (
    CommensuratePeriodicGrid,
    integrate_commensurate_periodic_vector,
)
from validation.lib.dwave_iterated_adaptive import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.static_ward_component_sources import (
    audit_static_ward_contract_with_components,
)


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw/"
    "dwave_commensurate_n628_m3_2_T10.csv"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        z = complex(value)
        return {"real": float(z.real), "imag": float(z.imag)}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _max_side_phase_abs_over_q(audit: dict[str, Any], source_name: str) -> float:
    q_norm = float(audit["q_norm"])
    values = []
    for side in ("left", "right"):
        vector = np.asarray(
            audit["component_sources"][side]["collective_defect_parts"][source_name],
            dtype=complex,
        )
        values.append(abs(complex(vector[1])) / q_norm)
    return max(values)


def _max_side_scalar(audit: dict[str, Any], path: tuple[str, ...]) -> float:
    values = []
    for side in ("left", "right"):
        value: Any = audit
        for key in (side, *path):
            value = value[key]
        values.append(float(value))
    return max(values)


def _result_row(
    *,
    grid: CommensuratePeriodicGrid,
    integral: Any,
    sheet: Any,
    ward: Any,
    audit: dict[str, Any],
) -> dict[str, Any]:
    q = np.asarray(grid.q_model, dtype=float)
    entries = np.asarray(audit["longitudinal_entries"], dtype=complex)
    k_etaeta = np.linalg.inv(np.asarray(audit["k_etaeta_inverse"], dtype=complex))
    singular_values = np.linalg.svd(k_etaeta, compute_uv=False)
    left_defect = np.asarray(audit["left"]["collective_defect"], dtype=complex)
    right_defect = np.asarray(audit["right"]["collective_defect"], dtype=complex)
    q_norm = float(audit["q_norm"])

    return {
        "nk": int(grid.nk),
        "mx": int(grid.mx),
        "my": int(grid.my),
        "shift_x": float(grid.shift_x),
        "shift_y": float(grid.shift_y),
        "grid_step": float(grid.step),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_norm": q_norm,
        "qx_over_step_minus_mx": float(q[0] / grid.step - grid.mx),
        "qy_over_step_minus_my": float(q[1] / grid.step - grid.my),
        "translation_permutation_exact": bool(grid.translation_permutation_exact),
        "point_evaluations": int(integral.point_evaluations),
        "chunks": int(integral.chunks),
        "chunk_size": int(integral.chunk_size),
        "summation_method": str(integral.summation_method),
        "integration_wall_seconds": float(integral.wall_seconds),
        "chi_bar": float(sheet.chi_bar),
        "dbar_t": float(sheet.dbar_t),
        "raw_longitudinal": float(audit["relative_longitudinal_gauge_residual"]),
        "ward_passed": bool(ward.passed),
        "ward_primitive_mixed_ratio_max": max(
            float(ward.left.primitive_mixed_ratio),
            float(ward.right.primitive_mixed_ratio),
        ),
        "external_rhs_over_q_max": float(audit["max_external_rhs_over_q"]),
        "collective_defect_over_q_max": float(audit["max_collective_defect_over_q"]),
        "collective_phase_defect_over_q_left_abs": float(abs(left_defect[1]) / q_norm),
        "collective_phase_defect_over_q_right_abs": float(abs(right_defect[1]) / q_norm),
        "collective_amplitude_defect_over_q_left_abs": float(abs(left_defect[0]) / q_norm),
        "collective_amplitude_defect_over_q_right_abs": float(abs(right_defect[0]) / q_norm),
        "collective_em_phase_over_q_max": _max_side_phase_abs_over_q(
            audit, "em_collective_contraction"
        ),
        "collective_bubble_phase_over_q_max": _max_side_phase_abs_over_q(
            audit, "phase_rotation_bubble"
        ),
        "collective_counterterm_phase_over_q_max": _max_side_phase_abs_over_q(
            audit, "phase_rotation_counterterm"
        ),
        "collective_projection_over_q_max": float(
            audit["max_collective_projection_over_q"]
        ),
        "effective_predicted_over_q_max": float(
            audit["max_effective_predicted_over_q"]
        ),
        "primitive_residual_over_q_max": float(
            audit["max_primitive_residual_over_q"]
        ),
        "effective_direct_over_q_max": float(audit["max_effective_direct_over_q"]),
        "primitive_bubble_translation_residual_over_q_max": _max_side_scalar(
            audit, ("component_sources",)
        ) if False else max(
            float(
                audit["component_sources"]["left"][
                    "bubble_translation_residual_norm_over_q"
                ]
            ),
            float(
                audit["component_sources"]["right"][
                    "bubble_translation_residual_norm_over_q"
                ]
            ),
        ),
        "primitive_contact_residual_over_q_max": max(
            float(
                audit["component_sources"]["left"]["contact_residual_norm_over_q"]
            ),
            float(
                audit["component_sources"]["right"]["contact_residual_norm_over_q"]
            ),
        ),
        "K_etaeta_singular_value_max": float(singular_values[0]),
        "K_etaeta_singular_value_min": float(singular_values[-1]),
        "K_etaeta_condition_number": float(singular_values[0] / singular_values[-1]),
        "K_0L_abs": float(abs(entries[0])),
        "K_L0_abs": float(abs(entries[1])),
        "K_LL_abs": float(abs(entries[2])),
        "K_LT_abs": float(abs(entries[3])),
        "K_TL_abs": float(abs(entries[4])),
        "rhs_metadata_error_norm": float(audit["rhs_metadata_error_norm"]),
        "lt_mapping_error_norm": float(audit["lt_contraction_mapping_error_norm"]),
        "component_source_consistency_max": float(
            audit["component_source_consistency_max"]
        ),
        "diagnostic_only": True,
        "projection_applied": False,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }


def _summary_text(args: argparse.Namespace, row: dict[str, Any], audit: dict[str, Any]) -> str:
    q = np.asarray(audit["q_model"], dtype=float)
    lines = [
        "d-wave exact-static commensurate periodic Ward audit",
        "=" * 56,
        f"grid = {row['nk']} x {row['nk']}; integer shift = ({row['mx']}, {row['my']})",
        f"q = ({q[0]:.12g}, {q[1]:.12g}); |q| = {row['q_norm']:.12g}",
        f"grid origin shift = ({row['shift_x']:.8g}, {row['shift_y']:.8g})",
        f"points = {row['point_evaluations']}; chunks = {row['chunks']}; "
        f"wall time = {row['integration_wall_seconds']:.3f} s",
        "",
        "Discrete translation contract",
        "-----------------------------",
        f"q_x / step - m_x = {row['qx_over_step_minus_mx']:.3e}",
        f"q_y / step - m_y = {row['qy_over_step_minus_my']:.3e}",
        f"translation_permutation_exact = {row['translation_permutation_exact']}",
        "",
        "Residual decomposition",
        "----------------------",
        f"raw_longitudinal                     = {row['raw_longitudinal']:.6e}",
        f"external_rhs_over_q_max              = {row['external_rhs_over_q_max']:.6e}",
        f"collective_phase_defect_over_q_left  = {row['collective_phase_defect_over_q_left_abs']:.6e}",
        f"collective_phase_defect_over_q_right = {row['collective_phase_defect_over_q_right_abs']:.6e}",
        f"collective_projection_over_q_max     = {row['collective_projection_over_q_max']:.6e}",
        f"effective_predicted_over_q_max       = {row['effective_predicted_over_q_max']:.6e}",
        f"primitive_residual_over_q_max        = {row['primitive_residual_over_q_max']:.6e}",
        f"effective_direct_over_q_max          = {row['effective_direct_over_q_max']:.6e}",
        "",
        "Collective phase-column pieces",
        "-------------------------------",
        f"EM-collective / q   = {row['collective_em_phase_over_q_max']:.12e}",
        f"bubble rotation / q = {row['collective_bubble_phase_over_q_max']:.12e}",
        f"counterterm / q     = {row['collective_counterterm_phase_over_q_max']:.12e}",
        "",
        "Physical channels",
        "-----------------",
        f"chi_bar = {row['chi_bar']:.10f}",
        f"Dbar_T  = {row['dbar_t']:.10f}",
        "",
        "Decision rule",
        "-------------",
        "collective phase defect / |q| near machine precision supports a pure "
        "non-translation-invariant quadrature explanation. A stable 1e-4--1e-3 "
        "defect on this commensurate complete lattice requires a phase-Hessian / "
        "counterterm convention audit.",
        "",
        "Fail-closed status",
        "------------------",
        "diagnostic_only = True",
        "projection_applied = False",
        "production_reference_established = False",
        "valid_for_casimir_input = False",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=628)
    parser.add_argument("--mx", type=int, default=3)
    parser.add_argument("--my", type=int, default=2)
    parser.add_argument("--shift-x", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    grid = CommensuratePeriodicGrid(
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        max_points=args.max_points,
    )
    q = grid.q_model
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(),
    )

    print(
        "starting commensurate periodic Ward audit: "
        f"nk={grid.nk}, m=({grid.mx},{grid.my}), "
        f"q=({q[0]:.12g},{q[1]:.12g}), points={grid.num_points}",
        flush=True,
    )
    integral = integrate_commensurate_periodic_vector(
        grid, context.evaluate_complex, chunk_size=args.chunk_size
    )
    components, rhs, primitive_metadata = assemble_dwave_static_primitives(
        context,
        integral.value,
        metadata={
            "integration_strategy": "commensurate_complete_periodic_tensor_grid",
            "nk": int(grid.nk),
            "integer_q_shift": (int(grid.mx), int(grid.my)),
            "grid_origin_shift": (float(grid.shift_x), float(grid.shift_y)),
            "q_constructed_from_integer_grid_shift": True,
            "translation_by_q_is_exact_index_permutation": True,
            "all_primitive_channels_share_periodic_nodes": True,
            "primitive_vector_integrated_before_schur": True,
            "summation_method": integral.summation_method,
        },
    )
    kernel = effective_em_kernel_from_components(components, q_model=q, xi_eV=0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=args.ward_tolerance,
        absolute_residual_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
    )
    sheet = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=1.0,
        mixing_tolerance=1.0,
        reality_tolerance=1.0,
        passivity_tolerance=1.0,
    )
    audit = audit_static_ward_contract_with_components(kernel, rhs, components)
    row = _result_row(grid=grid, integral=integral, sheet=sheet, ward=ward, audit=audit)

    output = args.output
    _write_csv(output, row)
    summary_path = output.with_suffix(".summary.txt")
    json_path = output.with_suffix(".json")
    summary_path.write_text(_summary_text(args, row, audit), encoding="utf-8")
    payload = {
        "schema": "dwave_static_commensurate_periodic_ward_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": vars(args),
        "grid": {
            "nk": grid.nk,
            "mx": grid.mx,
            "my": grid.my,
            "step": grid.step,
            "q_model": grid.q_model,
            "shift": (grid.shift_x, grid.shift_y),
            "num_points": grid.num_points,
            "translation_permutation_exact": True,
        },
        "integral": {
            "point_evaluations": integral.point_evaluations,
            "chunks": integral.chunks,
            "chunk_size": integral.chunk_size,
            "wall_seconds": integral.wall_seconds,
            "summation_method": integral.summation_method,
        },
        "row": row,
        "audit": audit,
        "primitive_metadata": primitive_metadata,
        "status": {
            "diagnostic_run_completed": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
        "total_wall_seconds": float(time.perf_counter() - started),
        "output_csv": str(output),
        "output_summary": str(summary_path),
        "output_json": str(json_path),
    }
    json_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )

    print(summary_path.read_text(encoding="utf-8"), end="")
    print(f"CSV:     {output}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
