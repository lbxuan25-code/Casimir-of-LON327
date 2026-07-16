"""Analytic preflight for the fixed outer-q Casimir quadrature contract.

This command validates the SI measure, dimensionless model-momentum mapping,
full-period angular rule, radial Gauss rule, and Matsubara prime weight without
calling the microscopic response.  Passing this preflight permits a later
microscopic outer-q smoke; it does not authorize a production Casimir result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lno327.casimir.outer_quadrature import (
    build_outer_q_polar_grid,
    free_energy_per_area_from_logdet,
    integrate_outer_q,
    matsubara_prime_weights,
)
from lno327.constants import KB
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


DEFAULT_OUTPUT = Path(
    "validation/outputs/casimir/outer_q_quadrature_preflight/preflight.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--u-max", type=float, default=24.0)
    parser.add_argument("--radial-order-low", type=int, default=16)
    parser.add_argument("--radial-order-high", type=int, default=32)
    parser.add_argument("--angular-order-low", type=int, default=16)
    parser.add_argument("--angular-order-high", type=int, default=32)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--comparison-rtol", type=float, default=5e-13)
    parser.add_argument("--comparison-atol", type=float, default=1e-18)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for name in ("separation_nm", "u_max", "temperature_K"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for low_name, high_name in (
        ("radial_order_low", "radial_order_high"),
        ("angular_order_low", "angular_order_high"),
    ):
        low = int(getattr(args, low_name))
        high = int(getattr(args, high_name))
        if low <= 0 or high <= low:
            parser.error(
                f"require positive --{low_name.replace('_', '-')} < "
                f"--{high_name.replace('_', '-')}"
            )
    if args.angular_order_low < 8 or args.angular_order_high < 8:
        parser.error("angular preflight orders must be at least eight")
    for name in ("comparison_rtol", "comparison_atol"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def _comparison(actual: float, expected: float, *, rtol: float, atol: float) -> dict[str, Any]:
    left = float(actual)
    right = float(expected)
    absolute = abs(left - right)
    scale = max(abs(left), abs(right))
    relative = absolute / max(scale, np.finfo(float).tiny)
    passed = bool(np.isclose(left, right, rtol=rtol, atol=atol))
    return {
        "actual": left,
        "expected": right,
        "absolute": absolute,
        "relative": relative,
        "passed": passed,
    }


def _grid(args: argparse.Namespace, *, radial: int, angular: int, offset: float):
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    return build_outer_q_polar_grid(
        separation_m=float(args.separation_nm) * 1e-9,
        lattice_a_x_m=material.lattice_a_x_m,
        lattice_a_y_m=material.lattice_a_y_m,
        u_max=float(args.u_max),
        radial_order=int(radial),
        angular_order=int(angular),
        angular_offset_fraction=float(offset),
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    rtol = float(args.comparison_rtol)
    atol = float(args.comparison_atol)
    low = _grid(
        args,
        radial=args.radial_order_low,
        angular=args.angular_order_low,
        offset=0.5,
    )
    high = _grid(
        args,
        radial=args.radial_order_high,
        angular=args.angular_order_high,
        offset=0.5,
    )
    cut_audit = _grid(
        args,
        radial=args.radial_order_high,
        angular=args.angular_order_high,
        offset=0.0,
    )

    disk_low = _comparison(
        integrate_outer_q(np.ones(low.node_count), low),
        low.disk_measure_m_inv2,
        rtol=rtol,
        atol=atol,
    )
    disk_high = _comparison(
        integrate_outer_q(np.ones(high.node_count), high),
        high.disk_measure_m_inv2,
        rtol=rtol,
        atol=atol,
    )

    radial_u2_expected = float(
        args.u_max**4 / (32.0 * np.pi * high.separation_m**2)
    )
    radial_u2 = _comparison(
        integrate_outer_q(high.u**2, high),
        radial_u2_expected,
        rtol=rtol,
        atol=atol,
    )

    anisotropic_values = 1.0 + 0.3 * np.cos(4.0 * high.phi_rad)
    anisotropic_cut_values = 1.0 + 0.3 * np.cos(4.0 * cut_audit.phi_rad)
    anisotropic_full_angle = _comparison(
        integrate_outer_q(anisotropic_values, high),
        high.disk_measure_m_inv2,
        rtol=rtol,
        atol=atol,
    )
    angular_cut_invariance = _comparison(
        integrate_outer_q(anisotropic_cut_values, cut_audit),
        integrate_outer_q(anisotropic_values, high),
        rtol=rtol,
        atol=atol,
    )

    q_model_roundtrip = np.column_stack(
        [
            high.q_model[:, 0] / high.lattice_a_x_m,
            high.q_model[:, 1] / high.lattice_a_y_m,
        ]
    )
    q_roundtrip_abs = float(np.max(np.abs(q_model_roundtrip - high.q_si_m_inv)))
    q_roundtrip_scale = max(float(np.max(np.abs(high.q_si_m_inv))), 1.0)
    q_roundtrip = {
        "maximum_absolute_m_inv": q_roundtrip_abs,
        "maximum_relative": q_roundtrip_abs / q_roundtrip_scale,
        "passed": bool(q_roundtrip_abs <= atol + rtol * q_roundtrip_scale),
    }

    prime = matsubara_prime_weights([0, 1, 2])
    prime_weight = {
        "actual": prime.tolist(),
        "expected": [0.5, 1.0, 1.0],
        "passed": bool(np.array_equal(prime, np.array([0.5, 1.0, 1.0]))),
    }

    constant_logdets = np.vstack(
        [
            np.full(high.node_count, -0.2),
            np.full(high.node_count, -0.1),
        ]
    )
    free_energy = free_energy_per_area_from_logdet(
        constant_logdets,
        matsubara_indices=[0, 1],
        temperature_K=float(args.temperature_K),
        grid=high,
    )
    free_energy_expected = float(
        KB
        * float(args.temperature_K)
        * high.disk_measure_m_inv2
        * (0.5 * -0.2 + -0.1)
    )
    free_energy_check = _comparison(
        free_energy.total_J_m2,
        free_energy_expected,
        rtol=rtol,
        atol=atol,
    )

    checks = {
        "constant_disk_measure_low": disk_low,
        "constant_disk_measure_high": disk_high,
        "radial_u_squared_measure": radial_u2,
        "full_angle_fourfold_harmonic": anisotropic_full_angle,
        "angular_cut_offset_invariance": angular_cut_invariance,
        "model_to_si_wavevector_roundtrip": q_roundtrip,
        "matsubara_prime_weight": prime_weight,
        "finite_partial_free_energy_units_and_weight": free_energy_check,
    }
    passed = all(bool(record["passed"]) for record in checks.values())
    return {
        "schema": "outer-q-quadrature-preflight-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "separation_nm": float(args.separation_nm),
            "u_max": float(args.u_max),
            "radial_order_low": int(args.radial_order_low),
            "radial_order_high": int(args.radial_order_high),
            "angular_order_low": int(args.angular_order_low),
            "angular_order_high": int(args.angular_order_high),
            "temperature_K": float(args.temperature_K),
            "comparison_rtol": rtol,
            "comparison_atol": atol,
        },
        "contract": {
            "radial_variable": "u = 2 Q d",
            "radial_domain": "[0, u_max]",
            "radial_rule": "Gauss-Legendre",
            "angular_domain": "[0, 2 pi)",
            "angular_rule": "full-period equal-weight trapezoidal",
            "angular_symmetry_reduction": False,
            "measure": "d^2Q/(2pi)^2 = u du dphi/(16 pi^2 d^2)",
            "model_momentum": "q_model = (a_x Q_x, a_y Q_y)",
            "zero_matsubara_prime_weight": 0.5,
            "q_zero_node_present": False,
            "tail_treatment": "finite u_max ladder required by microscopic preflight",
        },
        "high_grid": {
            "node_count": high.node_count,
            "q_max_m_inv": high.q_max_m_inv,
            "disk_measure_m_inv2": high.disk_measure_m_inv2,
            "max_abs_q_model_x": high.metadata["max_abs_q_model_x"],
            "max_abs_q_model_y": high.metadata["max_abs_q_model_y"],
        },
        "checks": checks,
        "status": {
            "passed": passed,
            "outer_q_measure_contract_fixed": passed,
            "microscopic_outer_q_preflight_allowed": passed,
            "production_casimir_allowed": False,
        },
        "diagnostic_only": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run_preflight(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": payload["status"]["passed"],
                "node_count": payload["high_grid"]["node_count"],
                "q_max_m_inv": payload["high_grid"]["q_max_m_inv"],
                "max_abs_q_model_x": payload["high_grid"]["max_abs_q_model_x"],
                "max_abs_q_model_y": payload["high_grid"]["max_abs_q_model_y"],
            },
            indent=2,
        )
    )
    if not payload["status"]["passed"]:
        raise SystemExit("outer-q quadrature preflight failed")


if __name__ == "__main__":
    main()
