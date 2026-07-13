"""Validate the production-candidate complete-orbit adaptive GK21 integrator.

The command performs one primary GK21 primitive integral and one tightened
same-rule audit on a shared complete-orbit cache.  Metric application, the
amplitude/phase Schur complement, Ward validation, sheet construction,
reflection, and logdet are evaluated only after each complete global primitive
integral.
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
from validation.lib.dwave_positive_orbit_adaptive_gk21 import (
    integrate_dwave_positive_orbit_adaptive_gk21,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/positive_matsubara/dwave_orbit_adaptive_gk21/raw/"
    "dwave_positive_orbit_adaptive_gk21.csv"
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


def _mixed_matrix_gate(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float, bool]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    delta = float(np.linalg.norm(b - a))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    tolerance = float(atol) + float(rtol) * scale
    ratio = delta / max(tolerance, np.finfo(float).tiny)
    relative = delta / max(scale, np.finfo(float).tiny)
    return delta, relative, ratio, bool(np.isfinite(ratio) and ratio <= 1.0)


def _mixed_scalar_gate(
    left: float,
    right: float,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float, bool]:
    delta = abs(float(right) - float(left))
    scale = max(abs(float(left)), abs(float(right)))
    tolerance = float(atol) + float(rtol) * scale
    ratio = delta / max(tolerance, np.finfo(float).tiny)
    relative = delta / max(scale, np.finfo(float).tiny)
    return delta, relative, ratio, bool(np.isfinite(ratio) and ratio <= 1.0)


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
    parser.add_argument("--limit", type=int, default=60)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.nk <= 0 or args.limit <= 0 or args.max_transverse_evaluations <= 0:
        parser.error("--nk, --limit, and --max-transverse-evaluations must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if any(index <= 0 for index in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if not 0.0 < args.audit_tolerance_factor < 1.0:
        parser.error("--audit-tolerance-factor must lie strictly between zero and one")
    for name in (
        "audit_matrix_rtol",
        "audit_matrix_atol",
        "audit_logdet_rtol",
        "audit_logdet_atol",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def _pipeline(
    *,
    components,
    rhs,
    q: np.ndarray,
    xi: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "physical_passed": False,
        "ward_passed": False,
        "sheet_validation_passed": False,
        "reflection_constructed": False,
        "logdet_passed": False,
        "error": "",
        "sigma": np.full((2, 2), np.nan + 1j * np.nan, dtype=complex),
        "reflection": np.full((2, 2), np.nan + 1j * np.nan, dtype=complex),
        "logdet": float("nan"),
        "ward_effective_mixed_ratio_max": float("nan"),
        "schur_condition_number": float("nan"),
    }
    try:
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
        reflection = positive_matsubara_sheet_response_to_reflection(
            sheet,
            q_lab_model=q,
            theta_rad=0.0,
            lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
            require_physical=True,
        )
        point = passive_sheet_logdet(
            reflection,
            reflection,
            separation_m=float(args.separation_nm) * 1e-9,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        result["error"] = str(exc)
        return result

    result.update(
        {
            "ward_passed": bool(ward.passed),
            "sheet_validation_passed": bool(sheet_validation.passed),
            "reflection_constructed": True,
            "logdet_passed": True,
            "sigma": np.asarray(sheet.matrix_tilde, dtype=complex),
            "reflection": np.asarray(reflection.matrix_lt, dtype=complex),
            "logdet": float(point.logdet),
            "ward_effective_mixed_ratio_max": max(
                ward.left.effective_mixed_ratio,
                ward.right.effective_mixed_ratio,
            ),
            "schur_condition_number": float(ward.schur_condition_number),
        }
    )
    result["physical_passed"] = bool(
        result["ward_passed"]
        and result["sheet_validation_passed"]
        and result["reflection_constructed"]
        and result["logdet_passed"]
    )
    return result


def _summary(rows: list[dict[str, Any]], shared: dict[str, Any]) -> str:
    lines = [
        "positive-Matsubara d-wave complete-orbit adaptive GK21 validation",
        "=" * 78,
        f"grid q = (2 pi/{shared['nk']}) ({shared['mx']}, {shared['my']})",
        f"orbit origins = {shared['orbit_origins']}",
        f"strategy = {shared['strategy']}; scipy = {shared['scipy_version']}",
        f"primary error ratio = {shared['primary_integral_error_ratio']:.6e}; "
        f"success = {shared['primary_success']}",
        f"audit error ratio = {shared['audit_integral_error_ratio']:.6e}; "
        f"success = {shared['audit_success']}",
        f"primitive group audit passed = {shared['primitive_group_agreement_passed']}",
        f"unique transverse evaluations = {shared['transverse_evaluations_unique']} / "
        f"{shared['max_transverse_evaluations']}; cache hits = {shared['cache_hits']}",
        f"microscopic point evaluations = {shared['point_evaluations']}",
        f"quadrature wall = {shared['quadrature_wall_seconds']:.3f} s; "
        f"evaluator wall = {shared['evaluator_wall_seconds']:.3f} s",
        "",
    ]
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
            f"all point pipelines passed = {bool(rows) and all(bool(r['point_pipeline_passed']) for r in rows)}",
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
        "starting complete-orbit adaptive GK21 integration: "
        f"nk={args.nk}, m=({args.mx},{args.my}), indices={indices}",
        flush=True,
    )
    started = time.perf_counter()
    integrated = integrate_dwave_positive_orbit_adaptive_gk21(
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
        limit=args.limit,
        norm=args.norm,
        scale_floor_relative=args.scale_floor_relative,
        scale_floor_absolute=args.scale_floor_absolute,
    )
    quadrature = integrated.quadrature
    q = np.asarray(quadrature.q_model, dtype=float)
    rows: list[dict[str, Any]] = []

    for position, (index, xi) in enumerate(zip(indices, xi_values, strict=True)):
        primary = None
        audit = None
        if integrated.primary_components:
            primary = _pipeline(
                components=integrated.primary_components[position],
                rhs=integrated.primary_rhs[position],
                q=q,
                xi=float(xi),
                args=args,
            )
        if integrated.audit_components:
            audit = _pipeline(
                components=integrated.audit_components[position],
                rhs=integrated.audit_rhs[position],
                q=q,
                xi=float(xi),
                args=args,
            )
        if primary is None:
            continue

        sigma_gate = (float("inf"), float("inf"), float("inf"), False)
        reflection_gate = sigma_gate
        logdet_gate = sigma_gate
        if audit is not None:
            sigma_gate = _mixed_matrix_gate(
                primary["sigma"],
                audit["sigma"],
                atol=args.audit_matrix_atol,
                rtol=args.audit_matrix_rtol,
            )
            reflection_gate = _mixed_matrix_gate(
                primary["reflection"],
                audit["reflection"],
                atol=args.audit_matrix_atol,
                rtol=args.audit_matrix_rtol,
            )
            logdet_gate = _mixed_scalar_gate(
                primary["logdet"],
                audit["logdet"],
                atol=args.audit_logdet_atol,
                rtol=args.audit_logdet_rtol,
            )

        row: dict[str, Any] = {
            "nk": int(args.nk),
            "mx": int(args.mx),
            "my": int(args.my),
            "qx": float(q[0]),
            "qy": float(q[1]),
            "q_norm": float(np.linalg.norm(q)),
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
            "primary_schur_condition_number": float(
                primary["schur_condition_number"]
            ),
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
        row.update(_matrix_fields("primary_sigma_tilde", primary["sigma"]))
        row.update(
            _matrix_fields(
                "audit_sigma_tilde",
                audit["sigma"] if audit else np.full((2, 2), np.nan + 1j * np.nan),
            )
        )
        row.update(_matrix_fields("primary_reflection", primary["reflection"]))
        row.update(
            _matrix_fields(
                "audit_reflection",
                audit["reflection"] if audit else np.full((2, 2), np.nan + 1j * np.nan),
            )
        )
        rows.append(row)

    audit_pass = quadrature.audit
    total_wall = time.perf_counter() - started
    shared = {
        "strategy": quadrature.strategy,
        "scipy_version": quadrature.scipy_version,
        "nk": int(args.nk),
        "mx": int(args.mx),
        "my": int(args.my),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "q_norm": float(np.linalg.norm(q)),
        "primitive_direction": tuple(int(v) for v in quadrature.primitive_direction),
        "transverse_direction": tuple(int(v) for v in quadrature.transverse_direction),
        "orbit_origins": quadrature.orbit_origins,
        "epsabs": quadrature.epsabs,
        "epsrel": quadrature.epsrel,
        "audit_tolerance_factor": quadrature.audit_tolerance_factor,
        "limit": quadrature.limit,
        "norm": quadrature.norm,
        "max_transverse_evaluations": quadrature.max_unique_transverse_evaluations,
        "transverse_evaluations_unique": quadrature.transverse_evaluations,
        "cache_hits": quadrature.cache_hits,
        "point_evaluations": quadrature.point_evaluations,
        "primary_integral_error_estimate": quadrature.primary.integral_error_estimate,
        "primary_integral_tolerance": quadrature.primary.integral_tolerance,
        "primary_integral_error_ratio": quadrature.primary.integral_error_ratio,
        "primary_success": quadrature.primary.success,
        "primary_subinterval_count": quadrature.primary.subinterval_count,
        "primary_unique_evaluations_added": (
            quadrature.primary.unique_evaluations_added
        ),
        "primary_cache_hits_added": quadrature.primary.cache_hits_added,
        "primary_worst_intervals": quadrature.primary.worst_intervals,
        "audit_integral_error_estimate": (
            audit_pass.integral_error_estimate if audit_pass else float("inf")
        ),
        "audit_integral_tolerance": (
            audit_pass.integral_tolerance if audit_pass else float("nan")
        ),
        "audit_integral_error_ratio": (
            audit_pass.integral_error_ratio if audit_pass else float("inf")
        ),
        "audit_success": bool(audit_pass and audit_pass.success),
        "audit_subinterval_count": audit_pass.subinterval_count if audit_pass else 0,
        "audit_unique_evaluations_added": (
            audit_pass.unique_evaluations_added if audit_pass else 0
        ),
        "audit_cache_hits_added": audit_pass.cache_hits_added if audit_pass else 0,
        "audit_worst_intervals": audit_pass.worst_intervals if audit_pass else (),
        "group_names": quadrature.group_names,
        "control_group_names": quadrature.control_group_names,
        "monitor_group_names": quadrature.monitor_group_names,
        "frozen_group_scales": tuple(
            float(value) for value in quadrature.frozen_group_scales
        ),
        "primary_audit_group_ratios": quadrature.primary_audit_group_ratios,
        "primitive_group_agreement_passed": quadrature.primitive_group_agreement_passed,
        "observed_to_frozen_scale_ratios": quadrature.observed_to_frozen_scale_ratios,
        "geometry_wall_seconds": quadrature.geometry_wall_seconds,
        "evaluator_wall_seconds": quadrature.evaluator_wall_seconds,
        "quadrature_wall_seconds": quadrature.wall_seconds,
        "total_wall_seconds": float(total_wall),
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
        "schema": "dwave_positive_commensurate_orbit_adaptive_gk21_v1",
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
