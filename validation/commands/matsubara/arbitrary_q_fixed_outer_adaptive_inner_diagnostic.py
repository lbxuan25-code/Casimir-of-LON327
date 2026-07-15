"""Run the generalized fixed-outer/adaptive-inner d-wave reference workflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327.numerics.fixed_outer_adaptive_inner import FixedOuterAdaptiveInnerOptions
from lno327.workflows.arbitrary_q_fixed_outer_adaptive_inner import (
    integrate_arbitrary_q_fixed_outer_adaptive_inner,
)
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.finite_q_validation_models import get_finite_q_validation_model

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_fixed_outer_adaptive_inner/diagnostic.json"
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-model", nargs=2, type=float, default=(0.0300152, 0.0200101))
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1])
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
    parser.add_argument("--split-points", nargs="*", type=float, default=[0.0])
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--orientation-rtol", type=float, default=5e-3)
    parser.add_argument("--orientation-atol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.matsubara_indices = tuple(sorted(set(int(v) for v in args.matsubara_indices)))
    if not args.matsubara_indices or any(v < 0 for v in args.matsubara_indices):
        parser.error("--matsubara-indices must be nonempty and non-negative")
    if int(args.outer_order) <= 0:
        parser.error("--outer-order must be positive")
    return args


def _mixed(left: np.ndarray, right: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    absolute = float(np.linalg.norm(b - a))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    threshold = float(atol) + float(rtol) * scale
    ratio = absolute / max(threshold, np.finfo(float).tiny)
    return {
        "passed": bool(np.isfinite(ratio) and ratio <= 1.0),
        "absolute": absolute,
        "relative": absolute / max(scale, np.finfo(float).tiny),
        "scale": scale,
        "threshold": threshold,
        "mixed_ratio": ratio,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(args.delta0_eV)
    q = np.asarray(args.q_model, dtype=float)
    xi = np.asarray(
        [
            0.0 if n == 0 else matsubara_energy_eV(n, args.temperature_K)
            for n in args.matsubara_indices
        ],
        dtype=float,
    )
    result = integrate_arbitrary_q_fixed_outer_adaptive_inner(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=float(args.temperature_K),
        eta_eV=float(args.eta_eV),
        q_model=q,
        outer_order=int(args.outer_order),
        inner_options=FixedOuterAdaptiveInnerOptions(
            epsabs=float(args.epsabs),
            epsrel=float(args.epsrel),
            inner_limit=int(args.inner_limit),
            max_point_evaluations=int(args.max_point_evaluations),
            cache_size_bytes=int(args.cache_size_bytes),
            quadrature=str(args.quadrature),
            norm=str(args.norm),
            split_points=tuple(float(v) for v in args.split_points),
        ),
        orders=("xy", "yx"),
        primary_order="xy",
    )

    rows: list[dict[str, Any]] = []
    for orientation in result.orientations:
        rows.append(
            {
                "order": orientation.order,
                "packed_primitive_norm": float(
                    np.linalg.norm(orientation.packed_primitives)
                ),
                "component_count": len(orientation.components),
                "rhs_count": len(orientation.rhs),
                "quadrature": {
                    "success": bool(orientation.quadrature.success),
                    "message": str(orientation.quadrature.message),
                    "error_estimate": float(orientation.quadrature.error_estimate),
                    "point_evaluations": int(
                        orientation.quadrature.point_evaluations
                    ),
                    "outer_evaluations": int(
                        orientation.quadrature.outer_evaluations
                    ),
                    "inner_integrals": int(orientation.quadrature.inner_integrals),
                    "max_inner_error": float(
                        orientation.quadrature.max_inner_error
                    ),
                    "sum_inner_error": float(
                        orientation.quadrature.sum_inner_error
                    ),
                    "wall_seconds": float(orientation.quadrature.wall_seconds),
                },
                "pointwise_profile": orientation.pointwise_profile.as_dict(),
                "operator_ward": orientation.operator_ward.as_dict(),
                "metadata": orientation.metadata,
            }
        )

    xy, yx = result.orientations
    comparison = _mixed(
        xy.packed_primitives,
        yx.packed_primitives,
        rtol=float(args.orientation_rtol),
        atol=float(args.orientation_atol),
    )
    payload = {
        "schema": "arbitrary-q-fixed-outer-adaptive-inner-diagnostic-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": result.metadata["method_id"],
        "executor_id": result.metadata["executor_id"],
        "q_model": q.tolist(),
        "matsubara_indices": list(args.matsubara_indices),
        "xi_eV_values": xi.tolist(),
        "primary_order": result.primary_order,
        "orientation_primitive_comparison": comparison,
        "orientations": rows,
        "diagnostic_passed": bool(
            comparison["passed"]
            and all(row["quadrature"]["success"] for row in rows)
            and all(row["operator_ward"]["passed"] for row in rows)
        ),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
    _atomic_write(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "method_id": payload["method_id"],
                "diagnostic_passed": payload["diagnostic_passed"],
                "xy_points": rows[0]["quadrature"]["point_evaluations"],
                "yx_points": rows[1]["quadrature"]["point_evaluations"],
                "orientation_mixed_ratio": comparison["mixed_ratio"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
