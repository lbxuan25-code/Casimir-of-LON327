"""Validate deterministic nested panel adaptation for complete d-wave q orbits."""

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

from lno327.constants import KB_EV_PER_K
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_positive_matsubara_pipeline,
    matrix_fields,
    mixed_matrix_gate,
    mixed_scalar_gate,
)
from validation.lib.dwave_positive_orbit_panel_adaptive import (
    integrate_dwave_positive_orbit_panel_adaptive,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/positive_matsubara/dwave_orbit_panel_adaptive/raw/"
    "dwave_positive_orbit_panel_adaptive.csv"
)


def matsubara_energy_eV(index: int, temperature_K: float) -> float:
    n = int(index)
    temperature = float(temperature_K)
    if n <= 0 or not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("positive Matsubara index and temperature are required")
    return float(2.0 * np.pi * n * KB_EV_PER_K * temperature)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=6)
    parser.add_argument("--my", type=int, default=4)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--max-transverse-evaluations", type=int, default=256)
    parser.add_argument("--epsabs", type=float, default=2e-5)
    parser.add_argument("--epsrel", type=float, default=2e-3)
    parser.add_argument("--audit-tolerance-factor", type=float, default=0.25)
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--scale-floor-relative", type=float, default=1e-8)
    parser.add_argument("--scale-floor-absolute", type=float, default=1e-14)
    parser.add_argument("--audit-matrix-rtol", type=float, default=1e-3)
    parser.add_argument("--audit-matrix-atol", type=float, default=1e-12)
    parser.add_argument("--audit-logdet-rtol", type=float, default=1e-3)
    parser.add_argument("--audit-logdet-atol", type=float, default=1e-12)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.nk <= 0 or args.max_transverse_evaluations <= 0:
        parser.error("--nk and --max-transverse-evaluations must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if any(index <= 0 for index in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if not 0.0 < args.audit_tolerance_factor < 1.0:
        parser.error("--audit-tolerance-factor must lie strictly between zero and one")
    return args


def _snapshot_payload(snapshot: object | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "success": bool(snapshot.success),
        "error_ratio": float(snapshot.integral_error_ratio),
        "tolerance_factor": float(snapshot.tolerance_factor),
        "unique_evaluations": int(snapshot.unique_evaluations),
        "cache_hits": int(snapshot.cache_hits),
        "panel_count": int(snapshot.panel_count),
        "maximum_depth": int(snapshot.maximum_depth),
        "refinement_steps": int(snapshot.refinement_steps),
        "group_errors": tuple(float(value) for value in snapshot.group_errors),
        "group_tolerances": tuple(float(value) for value in snapshot.group_tolerances),
        "group_ratios": tuple(float(value) for value in snapshot.group_ratios),
        "group_scales": tuple(float(value) for value in snapshot.group_scales),
        "worst_group_index": int(snapshot.worst_group_index),
        "worst_group_name": str(snapshot.worst_group_name),
        "worst_panel": tuple(float(value) for value in snapshot.worst_panel),
        "worst_panel_order": int(snapshot.worst_panel_order),
        "worst_local_ratio": float(snapshot.worst_local_ratio),
        "message": str(snapshot.message),
    }


def _trace_payload(entry: object) -> dict[str, Any]:
    return {
        "step": int(entry.step),
        "stage": str(entry.stage),
        "panel": tuple(float(value) for value in entry.panel),
        "old_order": int(entry.old_order),
        "operation": str(entry.operation),
        "required_new_nodes": int(entry.required_new_nodes),
        "unique_evaluations_after": int(entry.unique_evaluations_after),
        "worst_group_before": str(entry.worst_group_before),
        "worst_group_after": str(entry.worst_group_after),
        "global_error_ratio_before": float(entry.global_error_ratio_before),
        "global_error_ratio_after": float(entry.global_error_ratio_after),
    }


def _summary(rows: list[dict[str, Any]], shared: dict[str, Any]) -> str:
    primary = shared["primary"]
    audit = shared["audit"]
    trace = shared["refinement_trace"]
    lines = [
        "positive-Matsubara d-wave deterministic panel-adaptive validation",
        "=" * 82,
        f"grid q = (2 pi/{shared['nk']}) ({shared['mx']}, {shared['my']})",
        f"orbit origins = {shared['orbit_origins']}",
        f"strategy = {shared['strategy']}; quadrature = {shared['quadrature']}",
        f"full-period start = {shared['integration_start']:.12f}; "
        f"initial panels = {shared['initial_panel_count']}; pilots = {shared['pilot_count']}",
        f"symmetry reduction = {shared['symmetry_reduction_applied']}; "
        f"q-direction special case = {shared['q_direction_special_case']}",
        f"primary error ratio = {primary['error_ratio']:.6e}; success = {primary['success']}",
        (
            "audit error ratio = unavailable; success = False"
            if audit is None
            else f"audit error ratio = {audit['error_ratio']:.6e}; success = {audit['success']}"
        ),
        f"primary worst = {primary['worst_group_name']} on {primary['worst_panel']} "
        f"CC{primary['worst_panel_order']}",
        (
            "audit worst = unavailable"
            if audit is None
            else f"audit worst = {audit['worst_group_name']} on {audit['worst_panel']} "
            f"CC{audit['worst_panel_order']}"
        ),
        f"refinement trace steps = {len(trace)}",
        f"primitive group audit passed = {shared['primitive_group_agreement_passed']}",
        f"unique transverse evaluations = {shared['transverse_evaluations_unique']} / "
        f"{shared['max_transverse_evaluations']}; cache hits = {shared['cache_hits']}",
        f"microscopic point evaluations = {shared['point_evaluations']}",
        f"quadrature wall = {shared['quadrature_wall_seconds']:.3f} s; "
        f"evaluator wall = {shared['evaluator_wall_seconds']:.3f} s",
        "",
    ]
    if trace:
        last = trace[-1]
        lines.extend(
            [
                "last refinement:",
                f"  step={last['step']} stage={last['stage']} operation={last['operation']} "
                f"panel={last['panel']} CC{last['old_order']} "
                f"new_nodes={last['required_new_nodes']}",
                f"  ratio {last['global_error_ratio_before']:.6e} -> "
                f"{last['global_error_ratio_after']:.6e}; "
                f"worst {last['worst_group_before']} -> {last['worst_group_after']}",
                "",
            ]
        )
    if rows:
        lines.extend(
            [
                " n       xi[eV]  sigma-audit  R-audit  logdet-audit  primary  audit  point",
                "-" * 88,
            ]
        )
        for row in rows:
            lines.append(
                f"{int(row['matsubara_index']):2d} "
                f"{float(row['xi_eV']):12.5e} "
                f"{float(row['sigma_audit_ratio']):11.3e} "
                f"{float(row['reflection_audit_ratio']):8.3e} "
                f"{float(row['logdet_audit_ratio']):12.3e} "
                f"{str(bool(row['primary_physical_passed'])):>8s} "
                f"{str(bool(row['audit_physical_passed'])):>6s} "
                f"{str(bool(row['point_pipeline_passed'])):>6s}"
            )
    else:
        lines.append("no complete primary primitive integral was available")
    lines.extend(
        [
            "",
            f"quadrature success = {shared['adaptive_success']}",
            f"failure reason = {shared['failure_reason'] or 'none'}",
            f"all point pipelines passed = {bool(rows) and all(bool(row['point_pipeline_passed']) for row in rows)}",
            "diagnostic_only = True",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    indices = tuple(sorted(set(int(value) for value in args.matsubara_indices)))
    xi_values = np.asarray(
        [matsubara_energy_eV(index, args.temperature_K) for index in indices],
        dtype=float,
    )
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    physics = OrbitAcceptancePhysicsConfig(
        degeneracy=args.degeneracy,
        separation_nm=args.separation_nm,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
    )

    print(
        "starting deterministic complete-orbit panel adaptation: "
        f"nk={args.nk}, m=({args.mx},{args.my}), indices={indices}",
        flush=True,
    )
    started = time.perf_counter()
    integrated = integrate_dwave_positive_orbit_panel_adaptive(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_values,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        shift_s=args.shift_s,
        subgrid_average=args.subgrid_average,
        max_unique_transverse_evaluations=args.max_transverse_evaluations,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        audit_tolerance_factor=args.audit_tolerance_factor,
        norm=args.norm,
        scale_floor_relative=args.scale_floor_relative,
        scale_floor_absolute=args.scale_floor_absolute,
    )
    quadrature = integrated.quadrature
    q_model = np.asarray(quadrature.q_model, dtype=float)
    rows: list[dict[str, Any]] = []

    for position, (index, xi) in enumerate(zip(indices, xi_values, strict=True)):
        primary = None
        audit = None
        if integrated.primary_components:
            primary = evaluate_positive_matsubara_pipeline(
                components=integrated.primary_components[position],
                rhs=integrated.primary_rhs[position],
                q_model=q_model,
                xi_eV=float(xi),
                config=physics,
            )
        if integrated.audit_components:
            audit = evaluate_positive_matsubara_pipeline(
                components=integrated.audit_components[position],
                rhs=integrated.audit_rhs[position],
                q_model=q_model,
                xi_eV=float(xi),
                config=physics,
            )
        if primary is None:
            continue

        unavailable = (float("inf"), float("inf"), float("inf"), False)
        sigma_gate = unavailable
        reflection_gate = unavailable
        logdet_gate = unavailable
        if audit is not None:
            sigma_gate = mixed_matrix_gate(
                primary["sigma"],
                audit["sigma"],
                atol=args.audit_matrix_atol,
                rtol=args.audit_matrix_rtol,
            )
            reflection_gate = mixed_matrix_gate(
                primary["reflection"],
                audit["reflection"],
                atol=args.audit_matrix_atol,
                rtol=args.audit_matrix_rtol,
            )
            logdet_gate = mixed_scalar_gate(
                primary["logdet"],
                audit["logdet"],
                atol=args.audit_logdet_atol,
                rtol=args.audit_logdet_rtol,
            )

        row: dict[str, Any] = {
            "nk": int(args.nk),
            "mx": int(args.mx),
            "my": int(args.my),
            "qx": float(q_model[0]),
            "qy": float(q_model[1]),
            "q_norm": float(np.linalg.norm(q_model)),
            "matsubara_index": int(index),
            "xi_eV": float(xi),
            "primary_physical_passed": bool(primary["physical_passed"]),
            "audit_physical_passed": bool(audit and audit["physical_passed"]),
            "primary_ward_passed": bool(primary["ward_passed"]),
            "audit_ward_passed": bool(audit and audit["ward_passed"]),
            "primary_ward_effective_mixed_ratio_max": float(
                primary["ward_effective_mixed_ratio_max"]
            ),
            "audit_ward_effective_mixed_ratio_max": float(
                audit["ward_effective_mixed_ratio_max"] if audit else np.nan
            ),
            "primary_schur_condition_number": float(primary["schur_condition_number"]),
            "audit_schur_condition_number": float(
                audit["schur_condition_number"] if audit else np.nan
            ),
            "sigma_audit_absolute": sigma_gate[0],
            "sigma_audit_relative": sigma_gate[1],
            "sigma_audit_ratio": sigma_gate[2],
            "sigma_audit_passed": sigma_gate[3],
            "reflection_audit_absolute": reflection_gate[0],
            "reflection_audit_relative": reflection_gate[1],
            "reflection_audit_ratio": reflection_gate[2],
            "reflection_audit_passed": reflection_gate[3],
            "logdet_audit_absolute": logdet_gate[0],
            "logdet_audit_relative": logdet_gate[1],
            "logdet_audit_ratio": logdet_gate[2],
            "logdet_audit_passed": logdet_gate[3],
            "primary_logdet": float(primary["logdet"]),
            "audit_logdet": float(audit["logdet"] if audit else np.nan),
            "point_pipeline_passed": bool(
                quadrature.success
                and primary["physical_passed"]
                and audit is not None
                and audit["physical_passed"]
                and sigma_gate[3]
                and reflection_gate[3]
                and logdet_gate[3]
            ),
            "primary_error": str(primary["error"]),
            "audit_error": str(audit["error"] if audit else "audit unavailable"),
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        }
        row.update(matrix_fields("primary_sigma_tilde", primary["sigma"]))
        row.update(
            matrix_fields(
                "audit_sigma_tilde",
                audit["sigma"] if audit else np.full((2, 2), np.nan + 1j * np.nan),
            )
        )
        row.update(matrix_fields("primary_reflection", primary["reflection"]))
        row.update(
            matrix_fields(
                "audit_reflection",
                audit["reflection"] if audit else np.full((2, 2), np.nan + 1j * np.nan),
            )
        )
        rows.append(row)

    total_wall = time.perf_counter() - started
    primary_payload = _snapshot_payload(quadrature.primary)
    audit_payload = _snapshot_payload(quadrature.audit)
    assert primary_payload is not None
    trace_payload = tuple(_trace_payload(entry) for entry in quadrature.refinement_trace)
    shared = {
        "strategy": quadrature.strategy,
        "quadrature": quadrature.quadrature,
        "nk": int(args.nk),
        "mx": int(args.mx),
        "my": int(args.my),
        "qx": float(q_model[0]),
        "qy": float(q_model[1]),
        "q_norm": float(np.linalg.norm(q_model)),
        "primitive_direction": tuple(int(value) for value in quadrature.primitive_direction),
        "transverse_direction": tuple(int(value) for value in quadrature.transverse_direction),
        "orbit_origins": quadrature.orbit_origins,
        "epsabs": quadrature.epsabs,
        "epsrel": quadrature.epsrel,
        "audit_tolerance_factor": quadrature.audit_tolerance_factor,
        "norm": quadrature.norm,
        "integration_start": quadrature.integration_start,
        "initial_panel_count": quadrature.initial_panel_count,
        "pilot_count": quadrature.pilot_count,
        "full_transverse_period_integrated": quadrature.full_transverse_period_integrated,
        "symmetry_reduction_applied": quadrature.symmetry_reduction_applied,
        "q_direction_special_case": quadrature.q_direction_special_case,
        "max_transverse_evaluations": quadrature.max_unique_transverse_evaluations,
        "transverse_evaluations_unique": quadrature.transverse_evaluations,
        "cache_hits": quadrature.cache_hits,
        "point_evaluations": quadrature.point_evaluations,
        "primary": primary_payload,
        "audit": audit_payload,
        "group_names": quadrature.group_names,
        "control_group_names": quadrature.control_group_names,
        "monitor_group_names": quadrature.monitor_group_names,
        "primary_audit_group_ratios": quadrature.primary_audit_group_ratios,
        "primitive_group_agreement_passed": quadrature.primitive_group_agreement_passed,
        "refinement_trace": trace_payload,
        "geometry_wall_seconds": quadrature.geometry_wall_seconds,
        "evaluator_wall_seconds": quadrature.evaluator_wall_seconds,
        "quadrature_wall_seconds": quadrature.wall_seconds,
        "total_wall_seconds": float(total_wall),
        "evaluator_profile": integrated.evaluator_profile.as_dict(),
        "adaptive_success": quadrature.success,
        "adaptive_status": quadrature.status,
        "adaptive_message": quadrature.message,
        "failure_reason": quadrature.failure_reason,
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output.write_text("", encoding="utf-8")
    summary = _summary(rows, shared)
    output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_positive_commensurate_orbit_panel_adaptive_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "quadrature": shared,
        "rows": rows,
        "status": {
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(summary)
    print(f"CSV:     {output}")
    print(f"Summary: {output.with_suffix('.summary.txt')}")
    print(f"JSON:    {output.with_suffix('.json')}")

    all_points = bool(rows) and all(bool(row["point_pipeline_passed"]) for row in rows)
    if args.require_pass and not (quadrature.success and all_points):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
