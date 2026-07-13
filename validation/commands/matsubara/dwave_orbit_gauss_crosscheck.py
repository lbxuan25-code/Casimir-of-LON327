"""Cross-check positive d-wave periodic-orbit results with fixed Gauss quadrature.

The command recomputes the same primitive response payload on nonuniform fixed
Gauss-Legendre transverse nodes and compares the resulting sheet response,
reflection matrix, and passive logdet with an existing periodic-reference CSV.
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

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.conventions import (
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import (
    positive_matsubara_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from validation.commands.matsubara.dwave_orbit_adaptive import matsubara_energy_eV
from validation.lib.dwave_commensurate_orbit_gauss import OrbitEvaluationBudgetExceeded
from validation.lib.dwave_positive_orbit_gauss import (
    integrate_dwave_positive_orbit_gauss,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_REFERENCE = Path(
    "validation/outputs/positive_matsubara/dwave_orbit_adaptive/raw/"
    "dwave_positive_orbit_nested_n1_2_4_8_tight_t512.csv"
)
DEFAULT_OUTPUT = Path(
    "validation/outputs/positive_matsubara/dwave_orbit_gauss_crosscheck/raw/"
    "dwave_positive_orbit_gauss_crosscheck.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=6)
    parser.add_argument("--my", type=int, default=4)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--gauss-orders", nargs="+", type=int, default=[160, 192, 224])
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--max-point-evaluations", type=int, default=500_000)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--reference-matrix-rtol", type=float, default=1e-3)
    parser.add_argument("--reference-logdet-rtol", type=float, default=1e-3)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.nk <= 0 or args.max_point_evaluations <= 0:
        parser.error("--nk and --max-point-evaluations must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if any(index <= 0 for index in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if any(order <= 0 for order in args.gauss_orders):
        parser.error("all Gauss orders must be positive")
    if args.reference_matrix_rtol < 0.0 or args.reference_logdet_rtol < 0.0:
        parser.error("reference tolerances must be non-negative")
    if not args.reference_csv.is_file():
        parser.error(f"reference CSV does not exist: {args.reference_csv}")
    return args


def _load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {
            int(row["matsubara_index"]): row
            for row in csv.DictReader(handle)
        }
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


def _matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=complex)
    fields = {f"{prefix}_frobenius_norm": float(np.linalg.norm(value))}
    for label, row, col in (
        ("xx", 0, 0),
        ("xy", 0, 1),
        ("yx", 1, 0),
        ("yy", 1, 1),
    ):
        scalar = complex(value[row, col])
        fields[f"{prefix}_{label}_real"] = float(scalar.real)
        fields[f"{prefix}_{label}_imag"] = float(scalar.imag)
    return fields


def _matrix_relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=complex)
    right = np.asarray(b, dtype=complex)
    denominator = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-30)
    return float(np.linalg.norm(left - right) / denominator)


def _scalar_relative_difference(a: float, b: float) -> float:
    left, right = float(a), float(b)
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


def _summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "positive-Matsubara d-wave fixed-Gauss cross-check",
        "=" * 78,
        f"reference = {args.reference_csv}",
        f"grid: nk={args.nk}, m=({args.mx},{args.my})",
        f"Gauss orders = {tuple(sorted(set(args.gauss_orders)))}",
        f"matrix tolerance = {args.reference_matrix_rtol:.3e}; "
        f"logdet tolerance = {args.reference_logdet_rtol:.3e}",
        "",
        " order  n    sigma-ref       R-ref       logdet-ref    Ward   point   cross",
        "-" * 88,
    ]
    for row in rows:
        lines.append(
            f"{int(row['gauss_order']):6d} "
            f"{int(row['matsubara_index']):2d} "
            f"{float(row['reference_sigma_matrix_relative']):12.3e} "
            f"{float(row['reference_reflection_matrix_relative']):12.3e} "
            f"{float(row['reference_logdet_relative']):14.3e} "
            f"{str(bool(row['ward_passed'])):>7s} "
            f"{str(bool(row['point_pipeline_passed'])):>7s} "
            f"{str(bool(row['crosscheck_passed'])):>7s}"
        )
    lines.extend(
        [
            "",
            f"all fixed-Gauss physical pipelines passed = "
            f"{all(bool(row['point_pipeline_passed']) for row in rows)}",
            f"all reference cross-checks passed = "
            f"{all(bool(row['crosscheck_passed']) for row in rows)}",
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
    reference_rows = _load_rows(args.reference_csv)
    missing = [index for index in indices if index not in reference_rows]
    if missing:
        raise SystemExit(f"reference CSV is missing Matsubara indices: {missing}")

    xi_values = np.asarray(
        [matsubara_energy_eV(index, args.temperature_K) for index in indices],
        dtype=float,
    )
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    lattice = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    separation_m = float(args.separation_nm) * 1e-9

    output_rows: list[dict[str, Any]] = []
    previous_by_index: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    started_all = time.perf_counter()

    for order in orders:
        print(
            "starting positive d-wave fixed-Gauss cross-check: "
            f"nk={args.nk}, m=({args.mx},{args.my}), order={order}, indices={indices}",
            flush=True,
        )
        try:
            integrated = integrate_dwave_positive_orbit_gauss(
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
                shift_s=args.shift_s,
                subgrid_average=args.subgrid_average,
                max_point_evaluations=args.max_point_evaluations,
            )
        except OrbitEvaluationBudgetExceeded as exc:
            raise SystemExit(str(exc)) from exc

        quadrature = integrated.quadrature
        q = np.asarray(quadrature.q_model, dtype=float)
        for index, xi, components, rhs in zip(
            indices,
            integrated.xi_eV_values,
            integrated.components,
            integrated.rhs,
            strict=True,
        ):
            kernel = effective_em_kernel_from_components(
                components,
                q_model=q,
                xi_eV=float(xi),
            )
            ward = validate_effective_ward_xy(
                kernel,
                rhs,
                residual_tolerance=args.ward_tolerance,
                absolute_residual_tolerance=args.ward_absolute_tolerance,
                condition_max=args.condition_max,
            )
            sheet = positive_matsubara_kernel_to_sheet_response(
                kernel,
                degeneracy=args.degeneracy,
            )
            sheet_validation = validate_positive_matsubara_sheet_response(sheet)
            reflection_constructed = False
            logdet_passed = False
            reflection_error = ""
            logdet_error = ""
            reflection_matrix = np.full((2, 2), np.nan + 1j * np.nan, dtype=complex)
            logdet = float("nan")
            try:
                reflection = positive_matsubara_sheet_response_to_reflection(
                    sheet,
                    q_lab_model=q,
                    theta_rad=0.0,
                    lattice_constant_m=lattice,
                    require_physical=True,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                reflection_error = str(exc)
            else:
                reflection_constructed = True
                reflection_matrix = np.asarray(reflection.matrix_lt, dtype=complex)
                try:
                    point = passive_sheet_logdet(
                        reflection,
                        reflection,
                        separation_m=separation_m,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logdet_error = str(exc)
                else:
                    logdet_passed = True
                    logdet = float(point.logdet)

            sigma_matrix = np.asarray(sheet.matrix_tilde, dtype=complex)
            reference = reference_rows[index]
            reference_sigma = _matrix_from_row(reference, "sigma_tilde")
            reference_reflection = _matrix_from_row(reference, "reflection")
            reference_logdet = float(reference["logdet"])
            sigma_reference_relative = _matrix_relative_difference(
                sigma_matrix,
                reference_sigma,
            )
            reflection_reference_relative = _matrix_relative_difference(
                reflection_matrix,
                reference_reflection,
            )
            logdet_reference_relative = _scalar_relative_difference(
                logdet,
                reference_logdet,
            )

            previous = previous_by_index.get(index)
            if previous is None:
                previous_sigma_relative = float("nan")
                previous_reflection_relative = float("nan")
                previous_logdet_relative = float("nan")
            else:
                previous_sigma_relative = _matrix_relative_difference(
                    sigma_matrix,
                    previous[0],
                )
                previous_reflection_relative = _matrix_relative_difference(
                    reflection_matrix,
                    previous[1],
                )
                previous_logdet_relative = _scalar_relative_difference(
                    logdet,
                    previous[2],
                )
            previous_by_index[index] = (
                sigma_matrix.copy(),
                reflection_matrix.copy(),
                float(logdet),
            )

            point_pipeline_passed = bool(
                quadrature.success
                and ward.passed
                and sheet_validation.passed
                and reflection_constructed
                and logdet_passed
            )
            crosscheck_passed = bool(
                point_pipeline_passed
                and sigma_reference_relative <= args.reference_matrix_rtol
                and reflection_reference_relative <= args.reference_matrix_rtol
                and logdet_reference_relative <= args.reference_logdet_rtol
            )
            row: dict[str, Any] = {
                "nk": int(args.nk),
                "mx": int(args.mx),
                "my": int(args.my),
                "qx": float(q[0]),
                "qy": float(q[1]),
                "q_norm": float(np.linalg.norm(q)),
                "temperature_K": float(args.temperature_K),
                "matsubara_index": int(index),
                "xi_eV": float(xi),
                "gauss_order": int(order),
                "point_evaluations": int(quadrature.point_evaluations),
                "quadrature_wall_seconds": float(quadrature.wall_seconds),
                "ward_passed": bool(ward.passed),
                "ward_effective_mixed_ratio_max": max(
                    ward.left.effective_mixed_ratio,
                    ward.right.effective_mixed_ratio,
                ),
                "schur_condition_number": float(ward.schur_condition_number),
                "sheet_validation_passed": bool(sheet_validation.passed),
                "reflection_constructed": reflection_constructed,
                "logdet_passed": logdet_passed,
                "logdet": logdet,
                "point_pipeline_passed": point_pipeline_passed,
                "reference_sigma_matrix_relative": sigma_reference_relative,
                "reference_reflection_matrix_relative": reflection_reference_relative,
                "reference_logdet_relative": logdet_reference_relative,
                "previous_gauss_sigma_matrix_relative": previous_sigma_relative,
                "previous_gauss_reflection_matrix_relative": previous_reflection_relative,
                "previous_gauss_logdet_relative": previous_logdet_relative,
                "crosscheck_passed": crosscheck_passed,
                "reflection_error": reflection_error,
                "logdet_error": logdet_error,
                "diagnostic_only": True,
                "production_reference_established": False,
                "valid_for_casimir_input": False,
            }
            row.update(_matrix_fields("sigma_tilde", sigma_matrix))
            row.update(_matrix_fields("reflection", reflection_matrix))
            output_rows.append(row)

    total_wall = float(time.perf_counter() - started_all)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = _summary(output_rows, args)
    output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_positive_commensurate_orbit_fixed_gauss_crosscheck_v1",
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
            "all_physical_pipelines_passed": all(
                bool(row["point_pipeline_passed"]) for row in output_rows
            ),
            "all_crosschecks_passed": all(
                bool(row["crosscheck_passed"]) for row in output_rows
            ),
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


if __name__ == "__main__":
    main()
