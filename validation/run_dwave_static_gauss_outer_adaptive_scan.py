"""Bounded-cost global exact-static d-wave adaptive validation.

The outer BZ coordinate uses a fixed Gauss-Legendre rule and the inner
coordinate uses one vector-valued adaptive ``quad_vec`` integral at every outer
node.  All microscopic primitive blocks, counterterms and Ward-RHS terms share
the same inner nodes and weights.  One amplitude/phase Schur complement is
formed only after the full two-dimensional integral.

The reported quadrature error contains inner errors only.  Outer convergence
must be checked by changing ``--outer-order`` and by comparing ``xy`` with
``yx``.  This runner is fail-closed and never establishes a production
reference on its own.
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
from validation.lib.dwave_global_extrapolation import relative_difference
from validation.lib.dwave_iterated_adaptive import (
    assemble_dwave_static_primitives,
    build_dwave_static_integrand_context,
    unpack_complex_vector,
)
from validation.lib.dwave_shift_batch import ShiftBatchConfig, postprocess_merged
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.gauss_outer_adaptive import gauss_outer_adaptive_integral
from validation.lib.iterated_adaptive import EvaluationBudgetExceeded, IteratedAdaptiveOptions

_BZ_NORMALIZATION = 1.0 / (2.0 * np.pi) ** 2
DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_gauss_outer_adaptive/raw/"
    "dwave_q003_002_T10_gauss_outer_adaptive.csv"
)


def _physical_config(args: argparse.Namespace) -> ShiftBatchConfig:
    return ShiftBatchConfig(
        base_nk=1,
        qx=args.qx,
        qy=args.qy,
        temperature_K=args.temperature_K,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta_eV,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
        raw_longitudinal_ceiling=args.raw_longitudinal_ceiling,
        longitudinal_tolerance=args.longitudinal_tolerance,
        mixing_tolerance=args.mixing_tolerance,
        reality_tolerance=args.reality_tolerance,
        passivity_tolerance=args.passivity_tolerance,
        separation_nm=args.separation_nm,
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


def _failed_row(order: str, args: argparse.Namespace, message: str, wall_seconds: float) -> dict[str, Any]:
    return {
        "order": order,
        "outer_order": int(args.outer_order),
        "point_evaluations": int(args.max_point_evaluations),
        "outer_evaluations": 0,
        "inner_integrals": 0,
        "adaptive_error_estimate": float("nan"),
        "adaptive_max_inner_error": float("nan"),
        "adaptive_sum_inner_error": float("nan"),
        "adaptive_success": False,
        "adaptive_message": message,
        "wall_seconds": float(wall_seconds),
        "qx": args.qx,
        "qy": args.qy,
        "temperature_K": args.temperature_K,
        "delta0_eV": args.delta0_eV,
        "eta_eV": args.eta_eV,
        "chi_bar": float("nan"),
        "dbar_t": float("nan"),
        "ward_passed": False,
        "ward_primitive_mixed_ratio_max": float("nan"),
        "ward_effective_mixed_ratio_max": float("nan"),
        "schur_condition_number": float("nan"),
        "schur_inverse_method": "unavailable",
        "raw_longitudinal": float("nan"),
        "projection_eligible": False,
    }


def _summary_text(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    comparison: dict[str, Any],
    total_wall_seconds: float,
) -> str:
    lines = [
        "d-wave fixed-outer global adaptive exact-static scan",
        "=" * 55,
        f"q = ({args.qx:.8g}, {args.qy:.8g}); T = {args.temperature_K:.8g} K; "
        f"delta0 = {args.delta0_eV:.8g} eV",
        f"outer_order = {args.outer_order}; epsabs = {args.epsabs:.3e}; "
        f"epsrel = {args.epsrel:.3e}; quadrature = {args.quadrature}",
        f"point budget per orientation = {args.max_point_evaluations}; "
        f"total wall time = {total_wall_seconds:.3f} s",
        "",
        "Per-orientation results",
        "-----------------------",
        " order  points outer   inner_err success      chi_bar       Dbar_T      "
        "Ward-prim   raw-long  projection",
    ]
    for row in rows:
        lines.append(
            f" {str(row['order']):>3s} {int(row['point_evaluations']):8d} "
            f"{int(row['outer_evaluations']):5d} "
            f"{float(row['adaptive_error_estimate']):10.3e} "
            f"{str(bool(row['adaptive_success'])):>7s} "
            f"{float(row['chi_bar']):12.8f} {float(row['dbar_t']):12.8f} "
            f"{float(row['ward_primitive_mixed_ratio_max']):10.3e} "
            f"{float(row['raw_longitudinal']):10.3e} "
            f"{str(bool(row['projection_eligible'])):>10s}"
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
        lines.append("orientation comparison unavailable")
    lines.extend(
        [
            "",
            "Fail-closed status",
            "------------------",
            f"adaptive_feasibility_pass = {comparison['adaptive_feasibility_pass']}",
            "outer_discretization_error_estimated = False",
            "production_reference_established = False",
            "",
            "A successful run is only a bounded-cost feasibility result. Outer-order "
            "convergence and an independent complete-periodic benchmark remain required.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", choices=("xy", "yx"), default=["xy", "yx"])
    parser.add_argument("--outer-order", type=int, default=40)
    parser.add_argument("--epsabs", type=float, default=1e-4)
    parser.add_argument("--epsrel", type=float, default=1e-2)
    parser.add_argument("--inner-limit", type=int, default=80)
    parser.add_argument("--max-point-evaluations", type=int, default=100_000)
    parser.add_argument("--cache-size-bytes", type=int, default=64_000_000)
    parser.add_argument("--quadrature", choices=("gk15", "gk21", "trapezoid"), default="gk15")
    parser.add_argument("--norm", choices=("max", "2"), default="max")
    parser.add_argument("--split-points", type=float, nargs="*", default=[0.0])
    parser.add_argument("--orientation-agreement-tolerance", type=float, default=2e-2)
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--raw-longitudinal-ceiling", type=float, default=1e-3)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-7)
    parser.add_argument("--mixing-tolerance", type=float, default=1e-7)
    parser.add_argument("--reality-tolerance", type=float, default=1e-9)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if len(set(args.orders)) != len(args.orders):
        parser.error("--orders entries must be distinct")
    if args.outer_order <= 0:
        parser.error("--outer-order must be positive")
    if args.max_point_evaluations <= 0:
        parser.error("--max-point-evaluations must be positive")
    if args.orientation_agreement_tolerance <= 0.0:
        parser.error("--orientation-agreement-tolerance must be positive")
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
    adaptive_options = IteratedAdaptiveOptions(
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
    physical = _physical_config(args)

    rows: list[dict[str, Any]] = []
    for order in args.orders:
        print(
            f"starting gauss-outer adaptive order={order}, outer_order={args.outer_order}",
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
            row = _failed_row(order, args, str(exc), time.perf_counter() - order_started)
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
        processed = postprocess_merged(components, rhs, physical)
        row = {
            "order": order,
            "outer_order": int(args.outer_order),
            "point_evaluations": quadrature.point_evaluations,
            "outer_evaluations": quadrature.outer_evaluations,
            "inner_integrals": quadrature.inner_integrals,
            "adaptive_error_estimate": quadrature.error_estimate,
            "adaptive_max_inner_error": quadrature.max_inner_error,
            "adaptive_sum_inner_error": quadrature.sum_inner_error,
            "adaptive_success": quadrature.success,
            "adaptive_message": quadrature.message,
            "wall_seconds": quadrature.wall_seconds,
            "qx": args.qx,
            "qy": args.qy,
            "temperature_K": args.temperature_K,
            "delta0_eV": args.delta0_eV,
            "eta_eV": args.eta_eV,
            **processed,
        }
        rows.append(row)
        print(
            f"finished order={order}: points={quadrature.point_evaluations}, "
            f"chi={row['chi_bar']:.10g}, D_T={row['dbar_t']:.10g}, "
            f"Ward={row['ward_primitive_mixed_ratio_max']:.3e}",
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
        and all(bool(row["ward_passed"]) for row in by_order.values())
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
        "schema": "dwave_gauss_outer_adaptive_v1",
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
