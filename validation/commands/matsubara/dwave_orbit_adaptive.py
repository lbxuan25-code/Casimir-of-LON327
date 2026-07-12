"""Positive-Matsubara d-wave validation on an exact q orbit and adaptive transverse rule.

All requested Matsubara indices share one adaptive transverse mesh.  Every mesh
sample contains a complete commensurate q orbit.  Primitive response blocks and
the analytic Ward RHS are integrated before the nearest-neighbour bond metric and
the amplitude/phase Schur complement are applied.
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
from lno327.constants import KB_EV_PER_K
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
from validation.lib.dwave_commensurate_orbit_gauss import OrbitEvaluationBudgetExceeded
from validation.lib.dwave_positive_orbit_adaptive import (
    integrate_dwave_positive_orbit_adaptive,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/positive_matsubara/dwave_orbit_adaptive/raw/"
    "dwave_positive_orbit_adaptive.csv"
)


def matsubara_energy_eV(index: int, temperature_K: float) -> float:
    n = int(index)
    temperature = float(temperature_K)
    if n <= 0 or not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("positive Matsubara index and temperature are required")
    return float(2.0 * np.pi * n * KB_EV_PER_K * temperature)


def _matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=complex)
    result = {f"{prefix}_frobenius_norm": float(np.linalg.norm(value))}
    for label, row, col in (
        ("xx", 0, 0),
        ("xy", 0, 1),
        ("yx", 1, 0),
        ("yy", 1, 1),
    ):
        scalar = complex(value[row, col])
        result[f"{prefix}_{label}_real"] = float(scalar.real)
        result[f"{prefix}_{label}_imag"] = float(scalar.imag)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=6)
    parser.add_argument("--my", type=int, default=4)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument("--subgrid-average", choices=("auto", "none"), default="auto")
    parser.add_argument("--pilot-order", type=int, default=16)
    parser.add_argument("--epsabs", type=float, default=2e-5)
    parser.add_argument("--epsrel", type=float, default=2e-3)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--quadrature", choices=("gk15", "gk21", "trapezoid"), default="gk15")
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--scale-floor-relative", type=float, default=1e-8)
    parser.add_argument("--scale-floor-absolute", type=float, default=1e-14)
    parser.add_argument("--max-point-evaluations", type=int, default=500_000)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.nk <= 0 or args.pilot_order <= 0 or args.limit <= 0:
        parser.error("--nk, --pilot-order, and --limit must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if any(index <= 0 for index in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if args.max_point_evaluations <= 0:
        parser.error("--max-point-evaluations must be positive")
    return args


def _summary(rows: list[dict[str, Any]], shared: dict[str, Any]) -> str:
    lines = [
        "positive-Matsubara d-wave commensurate-orbit transverse-adaptive validation",
        "=" * 78,
        f"grid q = (2 pi/{shared['nk']}) ({shared['mx']}, {shared['my']})",
        f"q = ({shared['qx']:.12g}, {shared['qy']:.12g}); |q| = {shared['q_norm']:.12g}",
        f"orbit basis p = {shared['primitive_direction']}; transverse n = {shared['transverse_direction']}",
        f"orbit origins = {shared['orbit_origins']}",
        f"adaptive: pilot={shared['pilot_order']}, epsabs={shared['epsabs']:.3e}, "
        f"epsrel={shared['epsrel']:.3e}, limit={shared['limit']}, "
        f"rule={shared['quadrature']}, norm={shared['norm']}",
        f"transverse evaluations = {shared['transverse_evaluations']}; "
        f"point evaluations = {shared['point_evaluations']}",
        f"scaled error estimate = {shared['scaled_error_estimate']:.6e}; "
        f"adaptive success = {shared['adaptive_success']}",
        "",
        " n       xi[eV]    Ward-mixed    condition   sheet  reflection  logdet  point",
        "-" * 88,
    ]
    for row in rows:
        lines.append(
            f"{int(row['matsubara_index']):2d} "
            f"{float(row['xi_eV']):12.5e} "
            f"{float(row['ward_effective_mixed_ratio_max']):12.3e} "
            f"{float(row['schur_condition_number']):11.3e} "
            f"{str(bool(row['sheet_validation_passed'])):>6s} "
            f"{str(bool(row['reflection_constructed'])):>10s} "
            f"{str(bool(row['logdet_passed'])):>7s} "
            f"{str(bool(row['point_pipeline_passed'])):>6s}"
        )
    lines.extend(
        [
            "",
            f"all point pipelines passed = {all(bool(row['point_pipeline_passed']) for row in rows)}",
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

    print(
        "starting positive d-wave orbit-adaptive integration: "
        f"nk={args.nk}, m=({args.mx},{args.my}), indices={indices}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        integrated = integrate_dwave_positive_orbit_adaptive(
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
            max_point_evaluations=args.max_point_evaluations,
            pilot_order=args.pilot_order,
            epsabs=args.epsabs,
            epsrel=args.epsrel,
            limit=args.limit,
            quadrature=args.quadrature,
            norm=args.norm,
            scale_floor_relative=args.scale_floor_relative,
            scale_floor_absolute=args.scale_floor_absolute,
        )
    except OrbitEvaluationBudgetExceeded as exc:
        raise SystemExit(str(exc)) from exc

    quadrature = integrated.quadrature
    q = np.asarray(quadrature.q_model, dtype=float)
    lattice = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    separation_m = float(args.separation_nm) * 1e-9
    rows: list[dict[str, Any]] = []

    for index, xi, components, rhs in zip(
        indices,
        integrated.xi_eV_values,
        integrated.components,
        integrated.rhs,
        strict=True,
    ):
        kernel = effective_em_kernel_from_components(
            components, q_model=q, xi_eV=float(xi)
        )
        ward = validate_effective_ward_xy(
            kernel,
            rhs,
            residual_tolerance=args.ward_tolerance,
            absolute_residual_tolerance=args.ward_absolute_tolerance,
            condition_max=args.condition_max,
        )
        sheet = positive_matsubara_kernel_to_sheet_response(
            kernel, degeneracy=args.degeneracy
        )
        sheet_validation = validate_positive_matsubara_sheet_response(sheet)

        reflection_constructed = False
        reflection_error = ""
        logdet_passed = False
        logdet_error = ""
        reflection_matrix = np.full((2, 2), np.nan + 1j * np.nan, dtype=complex)
        reflection_spectral_radius = float("nan")
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
            reflection_spectral_radius = float(
                np.max(np.abs(np.linalg.eigvals(reflection_matrix)))
            )
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
            "delta0_eV": float(args.delta0_eV),
            "eta_eV": float(args.eta_eV),
            "ward_passed": bool(ward.passed),
            "ward_condition_ok": bool(ward.condition_ok),
            "ward_primitive_mixed_ratio_max": max(
                ward.left.primitive_mixed_ratio, ward.right.primitive_mixed_ratio
            ),
            "ward_effective_mixed_ratio_max": max(
                ward.left.effective_mixed_ratio, ward.right.effective_mixed_ratio
            ),
            "ward_primitive_absolute_max": max(
                ward.left.primitive_absolute_residual,
                ward.right.primitive_absolute_residual,
            ),
            "ward_effective_absolute_max": max(
                ward.left.effective_absolute_residual,
                ward.right.effective_absolute_residual,
            ),
            "schur_condition_number": float(ward.schur_condition_number),
            "schur_inverse_method": str(ward.schur_inverse_method),
            "sheet_validation_passed": bool(sheet_validation.passed),
            "sheet_relative_imaginary_norm": float(
                sheet_validation.relative_imaginary_norm
            ),
            "sheet_relative_symmetry_residual": float(
                sheet_validation.relative_symmetry_residual
            ),
            "sheet_minimum_symmetric_eigenvalue": float(
                sheet_validation.minimum_symmetric_eigenvalue
            ),
            "reflection_constructed": reflection_constructed,
            "reflection_spectral_radius": reflection_spectral_radius,
            "logdet_passed": logdet_passed,
            "logdet": logdet,
            "point_pipeline_passed": bool(
                quadrature.success
                and ward.passed
                and sheet_validation.passed
                and reflection_constructed
                and logdet_passed
            ),
            "reflection_error": reflection_error,
            "logdet_error": logdet_error,
            "integration_strategy": "commensurate_q_orbit_transverse_adaptive",
            "adaptive_success": bool(quadrature.success),
            "adaptive_status": int(quadrature.status),
            "adaptive_message": str(quadrature.message),
            "scaled_error_estimate": float(quadrature.scaled_error_estimate),
            "transverse_evaluations": int(quadrature.transverse_evaluations),
            "point_evaluations": int(quadrature.point_evaluations),
            "phase_hessian_policy": str(
                components.metadata["phase_hessian_policy"]
            ),
            "phase_hessian_multiplier": float(
                components.metadata["phase_hessian_multiplier"]
            ),
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        }
        row.update(_matrix_fields("sigma_tilde", np.asarray(sheet.matrix_tilde)))
        row.update(_matrix_fields("reflection", reflection_matrix))
        rows.append(row)

    total_wall = time.perf_counter() - started
    shared = {
        "nk": int(args.nk),
        "mx": int(args.mx),
        "my": int(args.my),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_norm": float(np.linalg.norm(q)),
        "primitive_direction": tuple(int(v) for v in quadrature.primitive_direction),
        "transverse_direction": tuple(int(v) for v in quadrature.transverse_direction),
        "orbit_origins": tuple(float(v) for v in quadrature.orbit_origins),
        "pilot_order": int(quadrature.pilot_order),
        "epsabs": float(quadrature.epsabs),
        "epsrel": float(quadrature.epsrel),
        "limit": int(quadrature.limit),
        "quadrature": str(quadrature.quadrature),
        "norm": str(quadrature.norm),
        "transverse_evaluations": int(quadrature.transverse_evaluations),
        "point_evaluations": int(quadrature.point_evaluations),
        "scaled_error_estimate": float(quadrature.scaled_error_estimate),
        "adaptive_success": bool(quadrature.success),
        "adaptive_status": int(quadrature.status),
        "adaptive_message": str(quadrature.message),
        "quadrature_wall_seconds": float(quadrature.wall_seconds),
        "total_wall_seconds": float(total_wall),
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, shared)
    output.with_suffix(".summary.txt").write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_positive_commensurate_orbit_adaptive_v1",
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
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(summary)
    print(f"CSV:     {output}")
    print(f"Summary: {output.with_suffix('.summary.txt')}")
    print(f"JSON:    {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
