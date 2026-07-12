"""Bond-metric-aware global iterated-adaptive exact-static d-wave validation.

The full 48-component primitive vector and analytic Ward RHS share every adaptive
node.  The q=0 Goldstone counterterm is integrated as a primitive quantity, one
amplitude/phase response is assembled after the complete Brillouin-zone integral,
and the diagnosed nearest-neighbour bond phase Hessian is then applied exactly
once before strict static Ward validation.

Both nesting orders are retained as an independent ridge-resolution diagnostic.
No longitudinal projection is performed and all results remain fail-closed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_adaptive_bond_metric import (
    AdaptiveStaticValidationConfig,
    postprocess_adaptive_bond_metric_static,
)
from validation.lib.dwave_global_extrapolation import relative_difference
from validation.lib.dwave_iterated_adaptive import (
    build_dwave_static_integrand_context,
    integrate_dwave_static_order,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.iterated_adaptive import IteratedAdaptiveOptions


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_iterated_adaptive/raw/"
    "dwave_bond_metric_q003_002_T10_iterated_adaptive.csv"
)


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


def _summary_text(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    comparison: dict[str, Any],
    total_wall_seconds: float,
) -> str:
    lines = [
        "d-wave bond-metric iterated-adaptive exact-static validation",
        "=" * 64,
        f"q = ({args.qx:.10g}, {args.qy:.10g}); T = {args.temperature_K:.8g} K; "
        f"delta0 = {args.delta0_eV:.8g} eV",
        f"level = {args.level_label}; epsabs = {args.epsabs:.3e}; "
        f"epsrel = {args.epsrel:.3e}; quadrature = {args.quadrature}",
        f"point budget per order = {args.max_point_evaluations}; "
        f"total wall time = {total_wall_seconds:.3f} s",
        "",
        "Per-order results",
        "-----------------",
        " order  points   quad_err success   phase/q   eff-direct/q longitudinal "
        "strict      chi_bar       Dbar_T",
    ]
    for row in rows:
        lines.append(
            f" {str(row['order']):>3s} {int(row['point_evaluations']):8d} "
            f"{float(row['adaptive_error_estimate']):10.3e} "
            f"{str(bool(row['adaptive_success'])):>7s} "
            f"{float(row['phase_defect_over_q']):10.3e} "
            f"{float(row['effective_direct_over_q']):13.3e} "
            f"{float(row['relative_longitudinal_gauge_residual']):12.3e} "
            f"{str(bool(row['strict_gate_passed'])):>6s} "
            f"{float(row['chi_bar']):12.8f} {float(row['dbar_t']):12.8f}"
        )

    lines.extend(["", "Nesting-order comparison", "------------------------"])
    if comparison.get("available", False):
        lines.extend(
            [
                f"relative_chi_xy_yx = {float(comparison['relative_chi']):.3e}",
                f"relative_dbar_xy_yx = {float(comparison['relative_dbar']):.3e}",
                f"relative_longitudinal_xy_yx = "
                f"{float(comparison['relative_longitudinal']):.3e}",
                f"physical_order_disagreement = "
                f"{float(comparison['physical_order_disagreement']):.3e}",
                f"order_agreement_tolerance = {args.order_agreement_tolerance:.3e}",
                f"order_agreement_pass = {bool(comparison['order_agreement_pass'])}",
            ]
        )
    else:
        lines.append("order comparison unavailable: run both xy and yx")

    lines.extend(
        [
            "",
            "Fail-closed status",
            "------------------",
            f"all_adaptive_orders_succeeded = "
            f"{all(bool(row['adaptive_success']) for row in rows)}",
            f"all_strict_static_gates_passed = "
            f"{all(bool(row['strict_gate_passed']) for row in rows)}",
            f"all_sheet_validations_passed = "
            f"{all(bool(row['sheet_validation_passed']) for row in rows)}",
            f"adaptive_feasibility_pass = {bool(comparison['adaptive_feasibility_pass'])}",
            "phase_hessian_policy = nearest_neighbor_bond_metric",
            "phase_hessian_application_stage = after_complete_primitive_integral",
            "projection_applied = False",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
            "A production reference still requires tolerance/budget convergence in both "
            "nesting orders and an independent numerical cross-check.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", choices=("xy", "yx"), default=["xy", "yx"])
    parser.add_argument("--level-label", default="single")
    parser.add_argument("--epsabs", type=float, default=1e-6)
    parser.add_argument("--epsrel", type=float, default=5e-4)
    parser.add_argument("--inner-limit", type=int, default=160)
    parser.add_argument("--outer-limit", type=int, default=160)
    parser.add_argument("--max-point-evaluations", type=int, default=200_000)
    parser.add_argument("--cache-size-bytes", type=int, default=64_000_000)
    parser.add_argument(
        "--quadrature", choices=("gk15", "gk21", "trapezoid"), default="gk15"
    )
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--split-points", type=float, nargs="*", default=[0.0])
    parser.add_argument("--order-agreement-tolerance", type=float, default=5e-3)
    parser.add_argument("--qx", type=float, default=0.0300152164356)
    parser.add_argument("--qy", type=float, default=0.0200101442904)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--mixed-ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--mixed-ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--primitive-tolerance", type=float, default=1e-6)
    parser.add_argument("--amplitude-tolerance", type=float, default=1e-6)
    parser.add_argument("--phase-tolerance", type=float, default=1e-6)
    parser.add_argument("--effective-direct-tolerance", type=float, default=1e-6)
    parser.add_argument("--effective-residual-tolerance", type=float, default=1e-6)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-6)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--reality-tolerance", type=float, default=1e-8)
    parser.add_argument("--mixing-tolerance", type=float, default=1e-6)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--energy-scale-eV", type=float, default=1.0)
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if len(set(args.orders)) != len(args.orders):
        parser.error("--orders entries must be distinct")
    if args.max_point_evaluations <= 0:
        parser.error("--max-point-evaluations must be positive")
    if args.inner_limit <= 0 or args.outer_limit <= 0:
        parser.error("adaptive subdivision limits must be positive")
    if args.order_agreement_tolerance <= 0.0:
        parser.error("--order-agreement-tolerance must be positive")
    if not np.isfinite([args.qx, args.qy]).all() or float(
        np.hypot(args.qx, args.qy)
    ) == 0.0:
        parser.error("q must be finite and nonzero")
    return args


def _validation_config(args: argparse.Namespace) -> AdaptiveStaticValidationConfig:
    return AdaptiveStaticValidationConfig(
        mixed_ward_tolerance=args.mixed_ward_tolerance,
        mixed_ward_absolute_tolerance=args.mixed_ward_absolute_tolerance,
        primitive_tolerance=args.primitive_tolerance,
        amplitude_tolerance=args.amplitude_tolerance,
        phase_tolerance=args.phase_tolerance,
        effective_direct_tolerance=args.effective_direct_tolerance,
        effective_residual_tolerance=args.effective_residual_tolerance,
        longitudinal_tolerance=args.longitudinal_tolerance,
        condition_max=args.condition_max,
        reality_tolerance=args.reality_tolerance,
        mixing_tolerance=args.mixing_tolerance,
        passivity_tolerance=args.passivity_tolerance,
        energy_scale_eV=args.energy_scale_eV,
        degeneracy=args.degeneracy,
    )


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

    # The q-dependent metric is deliberately not inserted pointwise.  The adaptive
    # integral first assembles the common q=0 Goldstone primitive counterterm.
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(phase_hessian_policy="q_independent"),
    )
    adaptive_options = IteratedAdaptiveOptions(
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        inner_limit=args.inner_limit,
        outer_limit=args.outer_limit,
        max_point_evaluations=args.max_point_evaluations,
        cache_size_bytes=args.cache_size_bytes,
        quadrature=args.quadrature,
        norm=args.norm,
        split_points=tuple(args.split_points),
    )
    validation_config = _validation_config(args)

    rows: list[dict[str, Any]] = []
    for order in args.orders:
        print(
            f"starting bond-metric iterated-adaptive order={order}, "
            f"level={args.level_label}",
            flush=True,
        )
        integrated = integrate_dwave_static_order(
            context,
            order=order,
            options=adaptive_options,
        )
        processed = postprocess_adaptive_bond_metric_static(
            integrated.components,
            integrated.rhs,
            ansatz=ansatz,
            q_model=q,
            config=validation_config,
        )
        quadrature = integrated.quadrature
        fields = processed.to_row_fields()
        row = {
            "level_label": args.level_label,
            "order": order,
            "point_evaluations": quadrature.point_evaluations,
            "outer_evaluations": quadrature.outer_evaluations,
            "inner_integrals": quadrature.inner_integrals,
            "adaptive_error_estimate": quadrature.error_estimate,
            "adaptive_max_inner_error": quadrature.max_inner_error,
            "adaptive_sum_inner_error": quadrature.sum_inner_error,
            "adaptive_success": quadrature.success,
            "adaptive_message": quadrature.message,
            "wall_seconds": quadrature.wall_seconds,
            "epsabs": args.epsabs,
            "epsrel": args.epsrel,
            "inner_limit": args.inner_limit,
            "outer_limit": args.outer_limit,
            "max_point_evaluations": args.max_point_evaluations,
            "qx": args.qx,
            "qy": args.qy,
            "temperature_K": args.temperature_K,
            "delta0_eV": args.delta0_eV,
            "eta_eV": args.eta_eV,
            **fields,
        }
        row["raw_longitudinal"] = row["relative_longitudinal_gauge_residual"]
        rows.append(row)
        print(
            f"finished order={order}: points={quadrature.point_evaluations}, "
            f"chi={row['chi_bar']:.10g}, D_T={row['dbar_t']:.10g}, "
            f"phase/q={row['phase_defect_over_q']:.3e}, "
            f"longitudinal={row['relative_longitudinal_gauge_residual']:.3e}, "
            f"strict={row['strict_gate_passed']}",
            flush=True,
        )

    by_order = {str(row["order"]): row for row in rows}
    comparison: dict[str, Any] = {"available": "xy" in by_order and "yx" in by_order}
    if comparison["available"]:
        xy, yx = by_order["xy"], by_order["yx"]
        relative_chi = relative_difference(yx["chi_bar"], xy["chi_bar"])
        relative_dbar = relative_difference(yx["dbar_t"], xy["dbar_t"])
        relative_longitudinal = relative_difference(
            yx["relative_longitudinal_gauge_residual"],
            xy["relative_longitudinal_gauge_residual"],
        )
        physical_disagreement = max(relative_chi, relative_dbar)
        comparison.update(
            {
                "relative_chi": relative_chi,
                "relative_dbar": relative_dbar,
                "relative_longitudinal": relative_longitudinal,
                "physical_order_disagreement": physical_disagreement,
                "order_agreement_pass": bool(
                    physical_disagreement <= args.order_agreement_tolerance
                ),
            }
        )
    else:
        comparison.update(
            {
                "relative_chi": float("nan"),
                "relative_dbar": float("nan"),
                "relative_longitudinal": float("nan"),
                "physical_order_disagreement": float("nan"),
                "order_agreement_pass": False,
            }
        )

    comparison["adaptive_feasibility_pass"] = bool(
        comparison["order_agreement_pass"]
        and all(bool(row["adaptive_success"]) for row in rows)
        and all(bool(row["strict_gate_passed"]) for row in rows)
        and all(bool(row["sheet_validation_passed"]) for row in rows)
        and all(str(row["schur_inverse_method"]) == "inv" for row in rows)
    )
    comparison.update(
        {
            "phase_hessian_policy": "nearest_neighbor_bond_metric",
            "phase_hessian_application_stage": "after_complete_primitive_integral",
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        }
    )

    total_wall = time.perf_counter() - started
    output = args.output
    _write_csv(output, rows)
    summary_path = output.with_suffix(".summary.txt")
    json_path = output.with_suffix(".json")
    summary = _summary_text(args, rows, comparison, total_wall)
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_bond_metric_iterated_adaptive_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "adaptive_options": asdict(adaptive_options),
        "validation_config": asdict(validation_config),
        "rows": rows,
        "comparison": comparison,
        "total_wall_seconds": total_wall,
        "status": {
            "diagnostic_only": True,
            "projection_applied": False,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(summary, end="")
    print(f"CSV:     {output}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
