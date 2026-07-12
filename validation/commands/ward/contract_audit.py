"""Locate the source of the exact-static d-wave longitudinal residual.

The runner uses the bounded-cost fixed-Gauss-outer/adaptive-inner quadrature and
integrates all primitive response, collective, counterterm, and Ward-RHS terms on
shared nodes. It audits, on both sides,

    u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS + r_primitive.

It additionally splits ``r_primitive`` into bubble/translation and direct/contact
parts, and splits the collective defect into EM-mixed, fermionic collective-bubble,
and Goldstone-counterterm sources. No projection is applied and no run can
establish a production reference.
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
from validation.lib.dwave_iterated_adaptive import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
    unpack_complex_vector,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.gauss_outer_adaptive import gauss_outer_adaptive_integral
from validation.lib.iterated_adaptive import EvaluationBudgetExceeded, IteratedAdaptiveOptions
from validation.lib.static_ward_component_sources import (
    audit_static_ward_contract_with_components,
)


_BZ_NORMALIZATION = 1.0 / (2.0 * np.pi) ** 2
DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw/"
    "dwave_q003_002_T10_ward_contract_audit.csv"
)


def _relative_difference(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-30)


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _max_side(audit: dict[str, Any], section: str, key: str) -> float:
    return max(
        float(audit["component_sources"]["left"][section][key]),
        float(audit["component_sources"]["right"][section][key]),
    )


def _max_component_norm_over_q(audit: dict[str, Any], section: str, key: str) -> float:
    return max(
        float(
            audit["component_sources"]["left"][section][key]
        ),
        float(
            audit["component_sources"]["right"][section][key]
        ),
    )


def _rhs_piece_over_q(audit: dict[str, Any], name: str) -> float:
    q_norm = float(audit["q_norm"])
    value = np.asarray(audit["left"]["rhs_pieces"][name], dtype=complex)
    return float(np.linalg.norm(value)) / q_norm


def _projection_channel_over_q(audit: dict[str, Any], channel: int) -> float:
    q_norm = float(audit["q_norm"])
    left = np.asarray(audit["left"]["collective_projection_by_channel"], dtype=complex)
    right = np.asarray(audit["right"]["collective_projection_by_channel"], dtype=complex)
    return max(float(np.linalg.norm(left[channel])), float(np.linalg.norm(right[channel]))) / q_norm


def _row_from_result(
    *,
    order: str,
    outer_order: int,
    quadrature: Any,
    sheet: Any,
    ward: Any,
    audit: dict[str, Any],
) -> dict[str, Any]:
    entries = np.asarray(audit["longitudinal_entries"], dtype=complex)
    left_components = audit["component_sources"]["left"]
    right_components = audit["component_sources"]["right"]
    left_effective_sources = left_components["effective_predicted_sources"]
    right_effective_sources = right_components["effective_predicted_sources"]

    return {
        "order": order,
        "outer_order": int(outer_order),
        "point_evaluations": int(quadrature.point_evaluations),
        "outer_evaluations": int(quadrature.outer_evaluations),
        "inner_integrals": int(quadrature.inner_integrals),
        "adaptive_error_estimate": float(quadrature.error_estimate),
        "adaptive_success": bool(quadrature.success),
        "wall_seconds": float(quadrature.wall_seconds),
        "chi_bar": float(sheet.chi_bar),
        "dbar_t": float(sheet.dbar_t),
        "raw_longitudinal": float(audit["relative_longitudinal_gauge_residual"]),
        "ward_passed": bool(ward.passed),
        "ward_primitive_mixed_ratio_max": max(
            float(ward.left.primitive_mixed_ratio),
            float(ward.right.primitive_mixed_ratio),
        ),
        "ward_effective_mixed_ratio_max": max(
            float(ward.left.effective_mixed_ratio),
            float(ward.right.effective_mixed_ratio),
        ),
        "schur_condition_number": float(audit["schur_condition_number"]),
        "schur_inverse_method": str(audit["schur_inverse_method"]),
        "external_rhs_over_q_max": float(audit["max_external_rhs_over_q"]),
        "rhs_equal_over_q": _rhs_piece_over_q(audit, "equal_forward"),
        "rhs_minus_delta_v_over_q": _rhs_piece_over_q(audit, "minus_delta_v_mid"),
        "rhs_qM_over_q": _rhs_piece_over_q(audit, "qM_mid"),
        "collective_defect_over_q_max": float(audit["max_collective_defect_over_q"]),
        "collective_projection_over_q_max": float(
            audit["max_collective_projection_over_q"]
        ),
        "collective_projection_amplitude_over_q_max": _projection_channel_over_q(
            audit, 0
        ),
        "collective_projection_phase_over_q_max": _projection_channel_over_q(audit, 1),
        "effective_predicted_over_q_max": float(
            audit["max_effective_predicted_over_q"]
        ),
        "primitive_residual_over_q_max": float(
            audit["max_primitive_residual_over_q"]
        ),
        "effective_direct_over_q_max": float(audit["max_effective_direct_over_q"]),
        "primitive_bubble_translation_residual_over_q_max": max(
            float(left_components["bubble_translation_residual_norm_over_q"]),
            float(right_components["bubble_translation_residual_norm_over_q"]),
        ),
        "primitive_contact_residual_over_q_max": max(
            float(left_components["contact_residual_norm_over_q"]),
            float(right_components["contact_residual_norm_over_q"]),
        ),
        "collective_em_contraction_over_q_max": _max_component_norm_over_q(
            audit,
            "collective_defect_part_norms_over_q",
            "em_collective_contraction",
        ),
        "collective_rotation_bubble_over_q_max": _max_component_norm_over_q(
            audit,
            "collective_defect_part_norms_over_q",
            "phase_rotation_bubble",
        ),
        "collective_rotation_counterterm_over_q_max": _max_component_norm_over_q(
            audit,
            "collective_defect_part_norms_over_q",
            "phase_rotation_counterterm",
        ),
        "projection_from_em_contraction_over_q_max": _max_component_norm_over_q(
            audit,
            "collective_projection_part_norms_over_q",
            "em_collective_contraction",
        ),
        "projection_from_phase_bubble_over_q_max": _max_component_norm_over_q(
            audit,
            "collective_projection_part_norms_over_q",
            "phase_rotation_bubble",
        ),
        "projection_from_phase_counterterm_over_q_max": _max_component_norm_over_q(
            audit,
            "collective_projection_part_norms_over_q",
            "phase_rotation_counterterm",
        ),
        "left_largest_effective_source": str(left_effective_sources["largest_source"]),
        "right_largest_effective_source": str(right_effective_sources["largest_source"]),
        "left_effective_source_cancellation_ratio": float(
            left_effective_sources["cancellation_ratio"]
        ),
        "right_effective_source_cancellation_ratio": float(
            right_effective_sources["cancellation_ratio"]
        ),
        "left_source_classification": str(audit["left"]["source_classification"]),
        "right_source_classification": str(audit["right"]["source_classification"]),
        "left_external_collective_classification": str(
            audit["left"]["external_collective_classification"]
        ),
        "right_external_collective_classification": str(
            audit["right"]["external_collective_classification"]
        ),
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
        "left_reconstruction_error_norm": float(
            audit["left"]["norms"]["reconstruction_error"]
        ),
        "right_reconstruction_error_norm": float(
            audit["right"]["norms"]["reconstruction_error"]
        ),
        "production_reference_established": False,
        "projection_eligible": False,
        "valid_for_casimir_input": False,
    }


def _failed_row(order: str, outer_order: int, message: str, wall_seconds: float) -> dict[str, Any]:
    return {
        "order": order,
        "outer_order": int(outer_order),
        "point_evaluations": 0,
        "outer_evaluations": 0,
        "inner_integrals": 0,
        "adaptive_error_estimate": float("nan"),
        "adaptive_success": False,
        "wall_seconds": float(wall_seconds),
        "failure_message": message,
        "production_reference_established": False,
        "projection_eligible": False,
        "valid_for_casimir_input": False,
    }


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if bool(row.get("adaptive_success", False))]
    orientation: list[dict[str, Any]] = []
    outer_step: list[dict[str, Any]] = []

    for outer in sorted({int(row["outer_order"]) for row in successful}):
        by_order = {
            str(row["order"]): row
            for row in successful
            if int(row["outer_order"]) == outer
        }
        if "xy" in by_order and "yx" in by_order:
            xy, yx = by_order["xy"], by_order["yx"]
            orientation.append(
                {
                    "outer_order": outer,
                    "relative_chi": _relative_difference(xy["chi_bar"], yx["chi_bar"]),
                    "relative_dbar_t": _relative_difference(
                        xy["dbar_t"], yx["dbar_t"]
                    ),
                    "relative_raw_longitudinal": _relative_difference(
                        xy["raw_longitudinal"], yx["raw_longitudinal"]
                    ),
                    "relative_effective_predicted_over_q": _relative_difference(
                        xy["effective_predicted_over_q_max"],
                        yx["effective_predicted_over_q_max"],
                    ),
                    "relative_effective_direct_over_q": _relative_difference(
                        xy["effective_direct_over_q_max"],
                        yx["effective_direct_over_q_max"],
                    ),
                }
            )

    tracked = (
        "external_rhs_over_q_max",
        "collective_projection_over_q_max",
        "effective_predicted_over_q_max",
        "primitive_residual_over_q_max",
        "effective_direct_over_q_max",
        "raw_longitudinal",
        "chi_bar",
        "dbar_t",
    )
    for order in ("xy", "yx"):
        order_rows = sorted(
            [row for row in successful if str(row["order"]) == order],
            key=lambda row: int(row["outer_order"]),
        )
        for previous, current in zip(order_rows, order_rows[1:], strict=False):
            item: dict[str, Any] = {
                "order": order,
                "outer_from": int(previous["outer_order"]),
                "outer_to": int(current["outer_order"]),
            }
            for name in tracked:
                item[f"relative_{name}"] = _relative_difference(
                    previous[name], current[name]
                )
            outer_step.append(item)
    return {"orientation": orientation, "outer_step": outer_step}


def _summary_text(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    comparisons: dict[str, Any],
    total_wall_seconds: float,
) -> str:
    lines = [
        "d-wave exact-static Ward contract residual-source audit",
        "=" * 58,
        f"q = ({args.qx:.8g}, {args.qy:.8g}); T = {args.temperature_K:.8g} K; "
        f"delta0 = {args.delta0_eV:.8g} eV",
        f"outer orders = {list(args.outer_orders)}; integration orders = {list(args.orders)}",
        f"epsabs = {args.epsabs:.3e}; epsrel = {args.epsrel:.3e}; "
        f"quadrature = {args.quadrature}",
        f"total wall time = {total_wall_seconds:.3f} s",
        "",
        "Identity audited",
        "----------------",
        "u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS + r_primitive",
        "The exact continuum target is zero. No projection is applied.",
        "",
        "Per-run scalar diagnostics",
        "--------------------------",
        " order outer  points      chi_bar       Dbar_T   raw-long      |R|/q  "
        "|C K^-1 K|/q |R_eff|/q |r_prim|/q |uKeff|/q",
    ]
    for row in rows:
        if not bool(row.get("adaptive_success", False)):
            lines.append(
                f" {str(row['order']):>3s} {int(row['outer_order']):5d}  FAILED: "
                f"{row.get('failure_message', 'unknown failure')}"
            )
            continue
        lines.append(
            f" {str(row['order']):>3s} {int(row['outer_order']):5d} "
            f"{int(row['point_evaluations']):7d} "
            f"{float(row['chi_bar']):12.8f} {float(row['dbar_t']):12.8f} "
            f"{float(row['raw_longitudinal']):10.3e} "
            f"{float(row['external_rhs_over_q_max']):10.3e} "
            f"{float(row['collective_projection_over_q_max']):13.3e} "
            f"{float(row['effective_predicted_over_q_max']):10.3e} "
            f"{float(row['primitive_residual_over_q_max']):10.3e} "
            f"{float(row['effective_direct_over_q_max']):10.3e}"
        )
        lines.append(
            "      primitive split: bubble/translation="
            f"{float(row['primitive_bubble_translation_residual_over_q_max']):.3e}, "
            f"direct/contact={float(row['primitive_contact_residual_over_q_max']):.3e}"
        )
        lines.append(
            "      collective projection sources: EM="
            f"{float(row['projection_from_em_contraction_over_q_max']):.3e}, "
            f"bubble={float(row['projection_from_phase_bubble_over_q_max']):.3e}, "
            f"counterterm={float(row['projection_from_phase_counterterm_over_q_max']):.3e}"
        )
        lines.append(
            "      largest effective sources: "
            f"L={row['left_largest_effective_source']}, "
            f"R={row['right_largest_effective_source']}"
        )

    lines.extend(["", "Orientation comparisons", "-----------------------"])
    if comparisons["orientation"]:
        for item in comparisons["orientation"]:
            lines.append(
                f" outer={item['outer_order']}: chi={item['relative_chi']:.3e}, "
                f"D_T={item['relative_dbar_t']:.3e}, "
                f"raw-long={item['relative_raw_longitudinal']:.3e}, "
                f"R_eff/q={item['relative_effective_predicted_over_q']:.3e}, "
                f"uKeff/q={item['relative_effective_direct_over_q']:.3e}"
            )
    else:
        lines.append(" unavailable")

    lines.extend(
        [
            "",
            "Fail-closed status",
            "------------------",
            "diagnostic_only = True",
            "projection_applied = False",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
            "Interpretation: R_eff isolates the mismatch between the external "
            "translation/contact defect and the Schur-projected collective defect. "
            "r_primitive is split into bubble/translation and direct/contact closure "
            "errors. The JSON additionally records every complex source vector in "
            "xy and LT/q form.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", choices=("xy", "yx"), default=["xy", "yx"])
    parser.add_argument("--outer-orders", nargs="+", type=int, default=[96])
    parser.add_argument("--epsabs", type=float, default=2e-4)
    parser.add_argument("--epsrel", type=float, default=2e-2)
    parser.add_argument("--inner-limit", type=int, default=60)
    parser.add_argument("--max-point-evaluations", type=int, default=100_000)
    parser.add_argument("--cache-size-bytes", type=int, default=64_000_000)
    parser.add_argument("--quadrature", choices=("gk15", "gk21", "trapezoid"), default="gk15")
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--split-points", type=float, nargs="*", default=[0.0])
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if len(set(args.orders)) != len(args.orders):
        parser.error("--orders entries must be distinct")
    if len(set(args.outer_orders)) != len(args.outer_orders):
        parser.error("--outer-orders entries must be distinct")
    if not args.outer_orders or any(value <= 0 for value in args.outer_orders):
        parser.error("--outer-orders must contain positive integers")
    if args.max_point_evaluations <= 0:
        parser.error("--max-point-evaluations must be positive")
    if float(np.hypot(args.qx, args.qy)) == 0.0:
        parser.error("q must be nonzero")
    return args


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        output_si=False,
    )
    q = np.asarray([args.qx, args.qy], dtype=float)
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(),
    )
    options = IteratedAdaptiveOptions(
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        inner_limit=args.inner_limit,
        outer_limit=1,
        max_point_evaluations=args.max_point_evaluations,
        cache_size_bytes=args.cache_size_bytes,
        quadrature=args.quadrature,
        norm=args.norm,
        split_points=tuple(args.split_points),
    )

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for outer_order in args.outer_orders:
        for order in args.orders:
            print(
                f"starting Ward contract audit order={order}, outer_order={outer_order}",
                flush=True,
            )
            run_started = time.perf_counter()
            try:
                quadrature = gauss_outer_adaptive_integral(
                    lambda kx, ky: _BZ_NORMALIZATION * context.evaluate_real(kx, ky),
                    order=order,
                    outer_order=int(outer_order),
                    options=options,
                )
            except EvaluationBudgetExceeded as exc:
                row = _failed_row(
                    order,
                    int(outer_order),
                    str(exc),
                    time.perf_counter() - run_started,
                )
                rows.append(row)
                results.append({"row": row, "audit": None})
                print(f"budget exceeded: {exc}", flush=True)
                continue

            primitive_vector = unpack_complex_vector(quadrature.value)
            components, rhs, primitive_metadata = assemble_dwave_static_primitives(
                context,
                primitive_vector,
                metadata={
                    "integration_strategy": "fixed_gauss_outer_adaptive_inner",
                    "integration_order": order,
                    "outer_order": int(outer_order),
                    "inner_error_estimate": float(quadrature.error_estimate),
                    "outer_discretization_error_estimated": False,
                    "ward_contract_source_audit": True,
                },
            )
            kernel = effective_em_kernel_from_components(
                components, q_model=q, xi_eV=0.0
            )
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
            audit = audit_static_ward_contract_with_components(
                kernel, rhs, components
            )
            row = _row_from_result(
                order=order,
                outer_order=int(outer_order),
                quadrature=quadrature,
                sheet=sheet,
                ward=ward,
                audit=audit,
            )
            rows.append(row)
            results.append(
                {
                    "row": row,
                    "audit": audit,
                    "primitive_metadata": primitive_metadata,
                    "quadrature": {
                        "success": bool(quadrature.success),
                        "message": str(quadrature.message),
                        "error_estimate": float(quadrature.error_estimate),
                        "max_inner_error": float(quadrature.max_inner_error),
                        "sum_inner_error": float(quadrature.sum_inner_error),
                        "point_evaluations": int(quadrature.point_evaluations),
                        "outer_evaluations": int(quadrature.outer_evaluations),
                        "inner_integrals": int(quadrature.inner_integrals),
                        "wall_seconds": float(quadrature.wall_seconds),
                    },
                }
            )
            print(
                f"finished order={order}, outer={outer_order}: "
                f"points={quadrature.point_evaluations}, "
                f"raw-long={row['raw_longitudinal']:.3e}, "
                f"R/q={row['external_rhs_over_q_max']:.3e}, "
                f"SchurProj/q={row['collective_projection_over_q_max']:.3e}, "
                f"R_eff/q={row['effective_predicted_over_q_max']:.3e}, "
                f"r_prim/q={row['primitive_residual_over_q_max']:.3e}, "
                f"uKeff/q={row['effective_direct_over_q_max']:.3e}",
                flush=True,
            )

    comparisons = _comparisons(rows)
    total_wall = time.perf_counter() - started
    output = args.output
    _write_csv(output, rows)
    summary_path = output.with_suffix(".summary.txt")
    json_path = output.with_suffix(".json")
    summary_path.write_text(
        _summary_text(args, rows, comparisons, total_wall), encoding="utf-8"
    )
    payload = {
        "schema": "dwave_static_ward_contract_audit_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": vars(args),
        "analytic_contract": {
            "continuum_target": "u K_eff = 0 and K_eff u = 0",
            "finite_quadrature_identity": (
                "u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS + r_primitive"
            ),
            "projection_applied": False,
        },
        "rows": rows,
        "comparisons": comparisons,
        "results": results,
        "status": {
            "diagnostic_run_completed": bool(rows),
            "all_requested_runs_succeeded": bool(rows)
            and all(bool(row.get("adaptive_success", False)) for row in rows),
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
        "total_wall_seconds": float(total_wall),
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
