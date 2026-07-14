"""Cross-check positive spm/d-wave complete-orbit results with one Gauss method."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Sequence

import numpy as np

from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.dwave_commensurate_orbit_gauss import OrbitEvaluationBudgetExceeded
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_positive_matsubara_pipeline,
    matrix_fields,
    mixed_matrix_gate,
    mixed_scalar_gate,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_gauss import integrate_positive_orbit_gauss

DEFAULT_OUTPUT = Path(
    "validation/outputs/positive_matsubara/positive_orbit_gauss_crosscheck/raw/"
    "positive_orbit_gauss_crosscheck.csv"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="dwave")
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=1)
    parser.add_argument("--my", type=int, default=1)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--gauss-orders", nargs="+", type=int, default=[160, 192])
    parser.add_argument("--panel-count", type=int, default=16)
    parser.add_argument(
        "--integration-start",
        action="append",
        type=float,
        dest="integration_starts",
        help="repeatable periodic cut; default is -pi",
    )
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--max-point-evaluations", type=int, default=1_100_000)
    parser.add_argument("--transverse-workers", type=int, default=1)
    parser.add_argument("--transverse-task-size", type=int, default=4)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--reference-matrix-rtol", type=float, default=1e-3)
    parser.add_argument("--reference-matrix-atol", type=float, default=1e-12)
    parser.add_argument("--reference-logdet-rtol", type=float, default=1e-3)
    parser.add_argument("--reference-logdet-atol", type=float, default=1e-12)
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    positive = (
        "nk",
        "panel_count",
        "max_point_evaluations",
        "transverse_workers",
        "transverse_task_size",
    )
    if any(int(getattr(args, name)) <= 0 for name in positive):
        parser.error("nk, panel count, budget, workers and task size must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if any(index <= 0 for index in args.matsubara_indices):
        parser.error("positive-Matsubara indices must be positive; exact n=0 is separate")
    if any(order <= 0 or order % args.panel_count != 0 for order in args.gauss_orders):
        parser.error("every Gauss order must be positive and divisible by panel count")
    starts = args.integration_starts or [-np.pi]
    if not np.isfinite(np.asarray(starts, dtype=float)).all():
        parser.error("integration starts must be finite")
    args.integration_starts = tuple(float(value) for value in starts)
    if args.reference_csv is not None and not args.reference_csv.is_file():
        parser.error(f"reference CSV does not exist: {args.reference_csv}")
    for name in (
        "reference_matrix_rtol",
        "reference_matrix_atol",
        "reference_logdet_rtol",
        "reference_logdet_atol",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def _load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["matsubara_index"]): row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"reference CSV contains no rows: {path}")
    return rows


def _matrix_from_row(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(float(row[f"{prefix}_xx_real"]), float(row[f"{prefix}_xx_imag"])),
                complex(float(row[f"{prefix}_xy_real"]), float(row[f"{prefix}_xy_imag"])),
            ],
            [
                complex(float(row[f"{prefix}_yx_real"]), float(row[f"{prefix}_yx_imag"])),
                complex(float(row[f"{prefix}_yy_real"]), float(row[f"{prefix}_yy_imag"])),
            ],
        ],
        dtype=complex,
    )


def _unavailable() -> tuple[float, float, float, bool]:
    return float("inf"), float("inf"), float("inf"), False


def _format_ratio(value: float, available: bool) -> str:
    return f"{value:11.3e}" if available else f"{'n/a':>11s}"


def _summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "positive-Matsubara complete-orbit fixed/composite-Gauss cross-check",
        "=" * 110,
        f"pairing={args.pairing}; nk={args.nk}; m=({args.mx},{args.my})",
        f"orders={tuple(sorted(set(args.gauss_orders)))}; panels={args.panel_count}; cuts={args.integration_starts}",
        f"workers/task={args.transverse_workers}/{args.transverse_task_size}",
        "",
        " cut total local  n   sigma-ref  sigma-prev  sigma-cut   Ward  physical  wall(s)",
        "-" * 110,
    ]
    for row in rows:
        lines.append(
            f"{int(row['cut_index']):4d} {int(row['gauss_order']):5d} "
            f"{int(row['panel_order']):5d} {int(row['matsubara_index']):2d} "
            f"{_format_ratio(float(row['reference_sigma_matrix_ratio']), bool(row['reference_available']))} "
            f"{_format_ratio(float(row['previous_gauss_sigma_matrix_ratio']), bool(row['previous_order_available']))} "
            f"{_format_ratio(float(row['baseline_cut_sigma_matrix_ratio']), bool(row['cut_comparison_available']))} "
            f"{str(bool(row['ward_passed'])):>6s} "
            f"{str(bool(row['point_pipeline_passed'])):>9s} "
            f"{float(row['quadrature_wall_seconds']):8.3f}"
        )
    references = [row for row in rows if bool(row["reference_available"])]
    previous = [row for row in rows if bool(row["previous_order_available"])]
    cuts = [row for row in rows if bool(row["cut_comparison_available"])]
    lines.extend(
        [
            "",
            f"all physical pipelines passed = {all(bool(row['point_pipeline_passed']) for row in rows)}",
            f"all available reference checks passed = {bool(references) and all(bool(row['crosscheck_passed']) for row in references)}",
            f"all consecutive-order checks passed = {bool(previous) and all(bool(row['previous_gauss_comparison_passed']) for row in previous)}",
            f"all periodic-cut checks passed = {(len(args.integration_starts) == 1) or (bool(cuts) and all(bool(row['cut_comparison_passed']) for row in cuts))}",
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
    orders = tuple(sorted(set(int(value) for value in args.gauss_orders)))
    starts = tuple(args.integration_starts)
    reference_rows = _load_rows(args.reference_csv) if args.reference_csv else None
    if reference_rows is not None:
        missing = [index for index in indices if index not in reference_rows]
        if missing:
            raise SystemExit(f"reference CSV is missing Matsubara indices: {missing}")

    xi_values = np.asarray(
        [matsubara_energy_eV(index, args.temperature_K) for index in indices],
        dtype=float,
    )
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(args.pairing, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    physics_config = OrbitAcceptancePhysicsConfig(
        degeneracy=args.degeneracy,
        separation_nm=args.separation_nm,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
    )

    output_rows: list[dict[str, Any]] = []
    baseline_by_order_index: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    started_all = time.perf_counter()

    for cut_index, integration_start in enumerate(starts):
        previous_by_index: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for order in orders:
            print(
                f"starting {args.pairing} fixed-Gauss cross-check: nk={args.nk}, "
                f"m=({args.mx},{args.my}), order={order}, panels={args.panel_count}, "
                f"cut={integration_start:.12f}, indices={indices}, "
                f"workers={args.transverse_workers}",
                flush=True,
            )
            try:
                integrated = integrate_positive_orbit_gauss(
                    spec=model.spec,
                    ansatz=ansatz,
                    pairing=pairing,
                    xi_eV_values=xi_values,
                    temperature_K=args.temperature_K,
                    eta_eV=args.eta_eV,
                    nk=args.nk,
                    mx=args.mx,
                    my=args.my,
                    transverse_order=order,
                    panel_count=args.panel_count,
                    integration_start=integration_start,
                    shift_s=args.shift_s,
                    subgrid_average=args.subgrid_average,
                    max_point_evaluations=args.max_point_evaluations,
                    transverse_workers=args.transverse_workers,
                    transverse_task_size=args.transverse_task_size,
                )
            except OrbitEvaluationBudgetExceeded as exc:
                raise SystemExit(str(exc)) from exc

            quadrature = integrated.quadrature
            profile = integrated.evaluator_profile
            q = np.asarray(quadrature.q_model, dtype=float)
            for index, xi, components, rhs in zip(
                indices,
                integrated.xi_eV_values,
                integrated.components,
                integrated.rhs,
                strict=True,
            ):
                physical = evaluate_positive_matsubara_pipeline(
                    components=components,
                    rhs=rhs,
                    q_model=q,
                    xi_eV=float(xi),
                    config=physics_config,
                )
                sigma = np.asarray(physical["sigma"], dtype=complex)
                reflection = np.asarray(physical["reflection"], dtype=complex)
                logdet = float(physical["logdet"])

                if reference_rows is None:
                    sigma_reference = reflection_reference = logdet_reference = _unavailable()
                    reference_available = False
                else:
                    reference = reference_rows[index]
                    sigma_reference = mixed_matrix_gate(
                        _matrix_from_row(reference, "sigma_tilde"), sigma,
                        atol=args.reference_matrix_atol, rtol=args.reference_matrix_rtol,
                    )
                    reflection_reference = mixed_matrix_gate(
                        _matrix_from_row(reference, "reflection"), reflection,
                        atol=args.reference_matrix_atol, rtol=args.reference_matrix_rtol,
                    )
                    logdet_reference = mixed_scalar_gate(
                        float(reference["logdet"]), logdet,
                        atol=args.reference_logdet_atol, rtol=args.reference_logdet_rtol,
                    )
                    reference_available = True

                previous_value = previous_by_index.get(index)
                previous_available = previous_value is not None
                if previous_value is None:
                    sigma_previous = reflection_previous = logdet_previous = _unavailable()
                else:
                    sigma_previous = mixed_matrix_gate(
                        previous_value[0], sigma,
                        atol=args.reference_matrix_atol, rtol=args.reference_matrix_rtol,
                    )
                    reflection_previous = mixed_matrix_gate(
                        previous_value[1], reflection,
                        atol=args.reference_matrix_atol, rtol=args.reference_matrix_rtol,
                    )
                    logdet_previous = mixed_scalar_gate(
                        previous_value[2], logdet,
                        atol=args.reference_logdet_atol, rtol=args.reference_logdet_rtol,
                    )
                previous_by_index[index] = (sigma.copy(), reflection.copy(), logdet)

                baseline_key = (order, index)
                baseline = baseline_by_order_index.get(baseline_key)
                cut_available = cut_index > 0 and baseline is not None
                if cut_index == 0:
                    baseline_by_order_index[baseline_key] = (
                        sigma.copy(), reflection.copy(), logdet
                    )
                if not cut_available:
                    sigma_cut = reflection_cut = logdet_cut = _unavailable()
                else:
                    assert baseline is not None
                    sigma_cut = mixed_matrix_gate(
                        baseline[0], sigma,
                        atol=args.reference_matrix_atol, rtol=args.reference_matrix_rtol,
                    )
                    reflection_cut = mixed_matrix_gate(
                        baseline[1], reflection,
                        atol=args.reference_matrix_atol, rtol=args.reference_matrix_rtol,
                    )
                    logdet_cut = mixed_scalar_gate(
                        baseline[2], logdet,
                        atol=args.reference_logdet_atol, rtol=args.reference_logdet_rtol,
                    )

                point_passed = bool(quadrature.success and physical["physical_passed"])
                crosscheck_passed = bool(
                    reference_available and point_passed
                    and sigma_reference[3] and reflection_reference[3] and logdet_reference[3]
                )
                previous_passed = bool(
                    previous_available and point_passed
                    and sigma_previous[3] and reflection_previous[3] and logdet_previous[3]
                )
                cut_passed = bool(
                    cut_available and point_passed
                    and sigma_cut[3] and reflection_cut[3] and logdet_cut[3]
                )

                row: dict[str, Any] = {
                    "pairing": args.pairing,
                    "phase_hessian_policy": integrated.phase_hessian_policy,
                    "nk": args.nk,
                    "mx": args.mx,
                    "my": args.my,
                    "qx": float(q[0]),
                    "qy": float(q[1]),
                    "q_norm": float(np.linalg.norm(q)),
                    "temperature_K": args.temperature_K,
                    "matsubara_index": index,
                    "xi_eV": float(xi),
                    "gauss_order": order,
                    "panel_count": int(quadrature.panel_count),
                    "panel_order": int(quadrature.panel_order),
                    "cut_index": cut_index,
                    "integration_start": float(quadrature.integration_start),
                    "quadrature": str(quadrature.quadrature),
                    "transverse_workers": int(quadrature.transverse_workers),
                    "transverse_task_size": int(quadrature.transverse_task_size),
                    "transverse_task_count": int(quadrature.transverse_task_count),
                    "execution_strategy": str(quadrature.execution_strategy),
                    "material_workspace_implementation": profile.material_workspace_implementation,
                    "q_workspace_implementation": profile.q_workspace_implementation,
                    "evaluator_worker_seconds": float(profile.total_seconds),
                    "evaluator_seconds_per_callback": float(profile.seconds_per_callback),
                    "full_transverse_period_integrated": bool(quadrature.full_transverse_period_integrated),
                    "symmetry_reduction_applied": bool(quadrature.symmetry_reduction_applied),
                    "q_direction_special_case": bool(quadrature.q_direction_special_case),
                    "point_evaluations": int(quadrature.point_evaluations),
                    "quadrature_wall_seconds": float(quadrature.wall_seconds),
                    "ward_passed": bool(physical["ward_passed"]),
                    "ward_effective_mixed_ratio_max": float(physical["ward_effective_mixed_ratio_max"]),
                    "schur_condition_number": float(physical["schur_condition_number"]),
                    "sheet_validation_passed": bool(physical["sheet_validation_passed"]),
                    "reflection_constructed": bool(physical["reflection_constructed"]),
                    "logdet_passed": bool(physical["logdet_passed"]),
                    "logdet": logdet,
                    "point_pipeline_passed": point_passed,
                    "reference_available": reference_available,
                    "reference_sigma_matrix_absolute": sigma_reference[0],
                    "reference_sigma_matrix_relative": sigma_reference[1],
                    "reference_sigma_matrix_ratio": sigma_reference[2],
                    "reference_sigma_matrix_passed": sigma_reference[3],
                    "reference_reflection_matrix_absolute": reflection_reference[0],
                    "reference_reflection_matrix_relative": reflection_reference[1],
                    "reference_reflection_matrix_ratio": reflection_reference[2],
                    "reference_reflection_matrix_passed": reflection_reference[3],
                    "reference_logdet_absolute": logdet_reference[0],
                    "reference_logdet_relative": logdet_reference[1],
                    "reference_logdet_ratio": logdet_reference[2],
                    "reference_logdet_passed": logdet_reference[3],
                    "previous_order_available": previous_available,
                    "previous_gauss_sigma_matrix_absolute": sigma_previous[0],
                    "previous_gauss_sigma_matrix_relative": sigma_previous[1],
                    "previous_gauss_sigma_matrix_ratio": sigma_previous[2],
                    "previous_gauss_reflection_matrix_absolute": reflection_previous[0],
                    "previous_gauss_reflection_matrix_relative": reflection_previous[1],
                    "previous_gauss_reflection_matrix_ratio": reflection_previous[2],
                    "previous_gauss_logdet_absolute": logdet_previous[0],
                    "previous_gauss_logdet_relative": logdet_previous[1],
                    "previous_gauss_logdet_ratio": logdet_previous[2],
                    "previous_gauss_comparison_passed": previous_passed,
                    "cut_comparison_available": cut_available,
                    "baseline_cut_sigma_matrix_absolute": sigma_cut[0],
                    "baseline_cut_sigma_matrix_relative": sigma_cut[1],
                    "baseline_cut_sigma_matrix_ratio": sigma_cut[2],
                    "baseline_cut_reflection_matrix_absolute": reflection_cut[0],
                    "baseline_cut_reflection_matrix_relative": reflection_cut[1],
                    "baseline_cut_reflection_matrix_ratio": reflection_cut[2],
                    "baseline_cut_logdet_absolute": logdet_cut[0],
                    "baseline_cut_logdet_relative": logdet_cut[1],
                    "baseline_cut_logdet_ratio": logdet_cut[2],
                    "cut_comparison_passed": cut_passed,
                    "crosscheck_passed": crosscheck_passed,
                    "physical_error": str(physical["error"]),
                    "diagnostic_only": True,
                    "production_reference_established": False,
                    "valid_for_casimir_input": False,
                }
                row.update(matrix_fields("sigma_tilde", sigma))
                row.update(matrix_fields("reflection", reflection))
                output_rows.append(row)

    if not output_rows:
        raise RuntimeError("fixed-Gauss crosscheck produced no rows")
    total_wall = float(time.perf_counter() - started_all)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    references = [row for row in output_rows if bool(row["reference_available"])]
    previous = [row for row in output_rows if bool(row["previous_order_available"])]
    cuts = [row for row in output_rows if bool(row["cut_comparison_available"])]
    summary = _summary(output_rows, args)
    output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    payload = {
        "schema": "positive_commensurate_orbit_fixed_gauss_crosscheck_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "total_wall_seconds": total_wall,
        "rows": output_rows,
        "status": {
            "all_physical_pipelines_passed": all(bool(row["point_pipeline_passed"]) for row in output_rows),
            "external_reference_provided": args.reference_csv is not None,
            "all_reference_crosschecks_passed": bool(references) and all(bool(row["crosscheck_passed"]) for row in references),
            "all_consecutive_order_checks_passed": bool(previous) and all(bool(row["previous_gauss_comparison_passed"]) for row in previous),
            "all_periodic_cut_checks_passed": len(starts) == 1 or (bool(cuts) and all(bool(row["cut_comparison_passed"]) for row in cuts)),
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary)
    print(f"CSV:     {output}")
    print(f"Summary: {output.with_suffix('.summary.txt')}")
    print(f"JSON:    {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
