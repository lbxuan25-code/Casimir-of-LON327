"""Canonical exact-static d-wave validation with fixed outer Gauss quadrature.

The outer Brillouin-zone coordinate uses a deterministic Gauss-Legendre rule and
the inner coordinate uses one shared vector-valued adaptive integral at each outer
node.  The complete 48-channel primitive response and analytic Ward RHS are
integrated before any nonlinear collective operation.  The nearest-neighbour bond
phase Hessian is then applied exactly once, followed by the amplitude/phase Schur,
strict static Ward validation, and raw static sheet extraction.

No longitudinal projection is performed.  Results remain diagnostic-only until
outer-order convergence and an independent numerical cross-check are established.
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
from validation.lib.dwave_static_primitives import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
    unpack_complex_vector,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.gauss_outer_adaptive import (
    EvaluationBudgetExceeded,
    GaussAdaptiveOptions,
    gauss_outer_adaptive_integral,
)

_BZ_NORMALIZATION = 1.0 / (2.0 * np.pi) ** 2
DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_static/raw/"
    "dwave_bond_metric_gauss_outer.csv"
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


def _failed_row(
    order: str,
    args: argparse.Namespace,
    exc: EvaluationBudgetExceeded,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "order": order,
        "outer_order": int(args.outer_order),
        "point_evaluations": int(exc.maximum),
        "attempted_point_evaluations": int(exc.attempted),
        "outer_evaluations": 0,
        "inner_integrals": 0,
        "adaptive_error_estimate": float("nan"),
        "adaptive_max_inner_error": float("nan"),
        "adaptive_sum_inner_error": float("nan"),
        "adaptive_success": False,
        "adaptive_message": str(exc),
        "failure_kind": "evaluation_budget_exceeded",
        "wall_seconds": float(wall_seconds),
        "qx": args.qx,
        "qy": args.qy,
        "temperature_K": args.temperature_K,
        "delta0_eV": args.delta0_eV,
        "eta_eV": args.eta_eV,
        "phase_hessian_policy": "nearest_neighbor_bond_metric",
        "ward_passed": False,
        "strict_gate_passed": False,
        "sheet_validation_passed": False,
        "phase_defect_over_q": float("nan"),
        "effective_direct_over_q": float("nan"),
        "effective_residual_over_q": float("nan"),
        "relative_longitudinal_gauge_residual": float("nan"),
        "schur_condition_number": float("nan"),
        "schur_inverse_method": "unavailable",
        "chi_bar": float("nan"),
        "dbar_t": float("nan"),
        "projection_applied": False,
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }


def _summary_text(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    comparison: dict[str, Any],
    total_wall_seconds: float,
) -> str:
    lines = [
        "d-wave bond-metric fixed-Gauss-outer adaptive static validation",
        "=" * 69,
        f"q = ({args.qx:.10g}, {args.qy:.10g}); T = {args.temperature_K:.8g} K; "
        f"delta0 = {args.delta0_eV:.8g} eV",
        f"outer_order = {args.outer_order}; inner epsabs = {args.epsabs:.3e}; "
        f"inner epsrel = {args.epsrel:.3e}; quadrature = {args.quadrature}",
        f"point budget per orientation = {args.max_point_evaluations}; "
        f"total wall time = {total_wall_seconds:.3f} s",
        "",
        "Per-orientation results",
        "-----------------------",
        " order  points outer   inner_err success   phase/q   eff-direct/q "
        "longitudinal strict      chi_bar       Dbar_T",
    ]
    for row in rows:
        lines.append(
            f" {str(row['order']):>3s} {int(row['point_evaluations']):8d} "
            f"{int(row['outer_evaluations']):5d} "
            f"{float(row['adaptive_error_estimate']):10.3e} "
            f"{str(bool(row['adaptive_success'])):>7s} "
            f"{float(row['phase_defect_over_q']):10.3e} "
            f"{float(row['effective_direct_over_q']):13.3e} "
            f"{float(row['relative_longitudinal_gauge_residual']):12.3e} "
            f"{str(bool(row['strict_gate_passed'])):>6s} "
            f"{float(row['chi_bar']):12.8f} {float(row['dbar_t']):12.8f}"
        )

    lines.extend(["", "Orientation comparison", "----------------------"])
    if comparison["available"]:
        lines.extend(
            [
                f"relative_chi_xy_yx = {comparison['relative_chi']:.3e}",
                f"relative_dbar_xy_yx = {comparison['relative_dbar']:.3e}",
                f"physical_orientation_disagreement = "
                f"{comparison['physical_orientation_disagreement']:.3e}",
                f"orientation_agreement_tolerance = "
                f"{args.orientation_agreement_tolerance:.3e}",
                f"orientation_agreement_pass = "
                f"{comparison['orientation_agreement_pass']}",
            ]
        )
    else:
        lines.append("orientation comparison unavailable: run both xy and yx")

    lines.extend(
        [
            "",
            "Fail-closed status",
            "------------------",
            f"all_adaptive_orientations_succeeded = "
            f"{all(bool(row['adaptive_success']) for row in rows)}",
            f"all_strict_static_gates_passed = "
            f"{all(bool(row['strict_gate_passed']) for row in rows)}",
            f"all_sheet_validations_passed = "
            f"{all(bool(row['sheet_validation_passed']) for row in rows)}",
            f"adaptive_feasibility_pass = {comparison['adaptive_feasibility_pass']}",
            "integration_strategy = fixed_gauss_outer_adaptive_inner",
            "phase_hessian_policy = nearest_neighbor_bond_metric",
            "phase_hessian_application_stage = after_complete_primitive_integral",
            "outer_discretization_error_estimated = False",
            "projection_applied = False",
            "production_reference_established = False",
            "valid_for_casimir_input = False",
            "",
            "A production reference requires convergence in outer_order and agreement "
            "between xy and yx at the final numerical target.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", choices=("xy", "yx"), default=["xy", "yx"])
    parser.add_argument("--outer-order", type=int, default=96)
    parser.add_argument("--epsabs", type=float, default=2e-4)
    parser.add_argument("--epsrel", type=float, default=2e-2)
    parser.add_argument("--inner-limit", type=int, default=60)
    parser.add_argument("--max-point-evaluations", type=int, default=100_000)
    parser.add_argument("--cache-size-bytes", type=int, default=64_000_000)
    parser.add_argument(
        "--quadrature", choices=("gk15", "gk21", "trapezoid"), default="gk15"
    )
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--split-points", type=float, nargs="*", default=[0.0])
    parser.add_argument("--orientation-agreement-tolerance", type=float, default=5e-3)
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
    if args.outer_order <= 0:
        parser.error("--outer-order must be positive")
    if args.inner_limit <= 0 or args.max_point_evaluations <= 0:
        parser.error("adaptive limits and point budget must be positive")
    if args.orientation_agreement_tolerance <= 0.0:
        parser.error("--orientation-agreement-tolerance must be positive")
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

    # The q-dependent bond metric is a collective action term and is therefore
    # applied only after the complete primitive BZ integral.
    context = build_dwave_static_integrand_context(
        model.spec,
        ansatz,
        q,
        kubo,
        pairing,
        FiniteQEngineOptions(phase_hessian_policy="q_independent"),
    )
    adaptive_options = GaussAdaptiveOptions(
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        inner_limit=args.inner_limit,
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
            f"starting d-wave static order={order}, outer_order={args.outer_order}",
            flush=True,
        )
        order_started = time.perf_counter()
        try:
            quadrature = gauss_outer_adaptive_integral(
                lambda kx, ky: _BZ_NORMALIZATION * context.evaluate_real(kx, ky),
                order=order,
                outer_order=args.outer_order,
                options=adaptive_options,
            )
        except EvaluationBudgetExceeded as exc:
            row = _failed_row(order, args, exc, time.perf_counter() - order_started)
            rows.append(row)
            print(f"budget exceeded for order={order}: {exc}", flush=True)
            continue

        primitive_vector = unpack_complex_vector(quadrature.value)
        components, rhs, _ = assemble_dwave_static_primitives(
            context,
            primitive_vector,
            metadata={
                "integration_strategy": "fixed_gauss_outer_adaptive_inner",
                "integration_order": order,
                "outer_order": int(args.outer_order),
                "inner_error_estimate": float(quadrature.error_estimate),
                "outer_discretization_error_estimated": False,
            },
        )
        processed = postprocess_adaptive_bond_metric_static(
            components,
            rhs,
            ansatz=ansatz,
            q_model=q,
            config=validation_config,
        )
        row = {
            "order": order,
            "outer_order": int(args.outer_order),
            "point_evaluations": quadrature.point_evaluations,
            "attempted_point_evaluations": quadrature.point_evaluations,
            "outer_evaluations": quadrature.outer_evaluations,
            "inner_integrals": quadrature.inner_integrals,
            "adaptive_error_estimate": quadrature.error_estimate,
            "adaptive_max_inner_error": quadrature.max_inner_error,
            "adaptive_sum_inner_error": quadrature.sum_inner_error,
            "adaptive_success": quadrature.success,
            "adaptive_message": quadrature.message,
            "failure_kind": "",
            "wall_seconds": quadrature.wall_seconds,
            "qx": args.qx,
            "qy": args.qy,
            "temperature_K": args.temperature_K,
            "delta0_eV": args.delta0_eV,
            "eta_eV": args.eta_eV,
            **processed.to_row_fields(),
        }
        rows.append(row)
        print(
            f"finished order={order}: points={quadrature.point_evaluations}, "
            f"chi={row['chi_bar']:.10g}, D_T={row['dbar_t']:.10g}, "
            f"phase/q={row['phase_defect_over_q']:.3e}, "
            f"longitudinal={row['relative_longitudinal_gauge_residual']:.3e}, "
            f"strict={row['strict_gate_passed']}",
            flush=True,
        )

    by_order = {
        str(row["order"]): row
        for row in rows
        if bool(row["adaptive_success"])
        and np.isfinite(float(row["chi_bar"]))
        and np.isfinite(float(row["dbar_t"]))
    }
    available = "xy" in by_order and "yx" in by_order
    comparison: dict[str, Any] = {"available": available}
    if available:
        xy, yx = by_order["xy"], by_order["yx"]
        relative_chi = relative_difference(yx["chi_bar"], xy["chi_bar"])
        relative_dbar = relative_difference(yx["dbar_t"], xy["dbar_t"])
        disagreement = max(relative_chi, relative_dbar)
        comparison.update(
            {
                "relative_chi": relative_chi,
                "relative_dbar": relative_dbar,
                "physical_orientation_disagreement": disagreement,
                "orientation_agreement_pass": bool(
                    disagreement <= args.orientation_agreement_tolerance
                ),
            }
        )
    else:
        comparison.update(
            {
                "relative_chi": float("nan"),
                "relative_dbar": float("nan"),
                "physical_orientation_disagreement": float("nan"),
                "orientation_agreement_pass": False,
            }
        )
    comparison["adaptive_feasibility_pass"] = bool(
        comparison["orientation_agreement_pass"]
        and len(by_order) == len(args.orders) == 2
        and all(bool(row["strict_gate_passed"]) for row in by_order.values())
        and all(bool(row["sheet_validation_passed"]) for row in by_order.values())
        and all(str(row["schur_inverse_method"]) == "inv" for row in by_order.values())
    )
    comparison["production_reference_established"] = False

    total_wall = time.perf_counter() - started
    output = args.output
    _write_csv(output, rows)
    summary_path = output.with_suffix(".summary.txt")
    json_path = output.with_suffix(".json")
    summary = _summary_text(args, rows, comparison, total_wall)
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "schema": "dwave_bond_metric_gauss_outer_static_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "adaptive_options": asdict(adaptive_options),
        "rows": rows,
        "comparison": comparison,
        "total_wall_seconds": total_wall,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(summary, end="")
    print(f"CSV:     {output}")
    print(f"Summary: {summary_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()
