#!/usr/bin/env python3
"""Stage 5.9 Casimir energy integration grid planning scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.casimir_grid import (  # noqa: E402
    casimir_grid_scaffold_metadata,
    kappa_si,
    material_response_grid_requirements,
    matsubara_prime_weights,
    matsubara_xi_grid,
    omega_eV_to_xi_si,
    polar_measure_weights,
    q_polar_grid,
    round_trip_factor_from_xi_Q_d,
    xi_si_to_omega_eV,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "casimir_grid"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "casimir_integrand" / "stage5_8_casimir_integrand_prototype.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_9_casimir_grid_planning_scaffold.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_9_casimir_grid_planning_scaffold.md"

Q0_WARNING = "Q=0 has undefined TE/TM in-plane direction and must be handled by symmetry/limit or excluded from angular-grid production runs."
RESPONSE_GRID_WARNING = "Existing 8 validation reflection cases are not a production integration grid."

BOUNDARY = {
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "no_heavy_response_run": True,
    "no_full_matsubara_sum": True,
    "no_full_Q_integral": True,
    "no_energy_output": True,
    "no_force_output": True,
    "no_torque_output": True,
    "not_casimir_ready_claim": True,
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, complex | np.complexfloating):
        return {"real": float(np.real(value)), "imag": float(np.imag(value)), "abs": float(abs(value))}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def run_checks(
    *,
    xi: np.ndarray,
    omega_eV: np.ndarray,
    weights: np.ndarray,
    grid: dict[str, np.ndarray],
    measure: np.ndarray,
    round_trip_factors: np.ndarray,
    warnings: list[str],
    n_max: int,
    n_q: int,
    n_phi: int,
) -> dict[str, str]:
    checks: dict[str, str] = {}
    checks["matsubara_grid"] = (
        "PASS"
        if len(xi) == n_max + 1
        and xi[0] == 0.0
        and np.allclose(weights, np.array([0.5] + [1.0] * n_max))
        and (n_max < 2 or np.allclose(np.diff(xi), np.diff(xi)[0]))
        else "FAIL"
    )
    xi_round_trip = omega_eV_to_xi_si(omega_eV)
    checks["omega_eV_round_trip"] = "PASS" if np.allclose(xi_round_trip, xi, rtol=1e-14, atol=1e-6) else "FAIL"
    phi = grid["phi_rad"]
    checks["phi_no_duplicate_endpoint"] = "PASS" if phi[0] == 0.0 and phi[-1] < 2.0 * np.pi else "FAIL"
    checks["q_grid_shape"] = (
        "PASS"
        if grid["Qx_m_inv"].shape == (n_q, n_phi)
        and grid["Qy_m_inv"].shape == (n_q, n_phi)
        and grid["Q_m_inv"].shape == (n_q,)
        and grid["phi_rad"].shape == (n_phi,)
        else "FAIL"
    )
    checks["polar_measure_nonnegative"] = "PASS" if np.all(measure >= 0.0) else "FAIL"
    checks["round_trip_factor_range"] = (
        "PASS" if np.all(round_trip_factors > 0.0) and np.all(round_trip_factors <= 1.0) else "FAIL"
    )
    checks["Q0_warning_present"] = "PASS" if Q0_WARNING in warnings else "FAIL"
    checks["response_grid_insufficiency_warning_present"] = "PASS" if RESPONSE_GRID_WARNING in warnings else "FAIL"
    return checks


def run_scaffold(
    input_json: Path,
    *,
    temperature_K: float,
    n_max: int,
    q_max_nm_inv: float,
    n_q: int,
    n_phi: int,
    separation_nm: float,
    require_stage5_8_passed: bool,
) -> dict[str, Any]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    status = data.get("diagnostic_status", {}).get("stage5_8_status")
    if require_stage5_8_passed and status != "STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED":
        raise ValueError("input must have STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED status")

    q_max_m_inv = float(q_max_nm_inv) * 1.0e9
    separation_m = float(separation_nm) * 1.0e-9
    xi = matsubara_xi_grid(temperature_K, n_max)
    omega_eV = xi_si_to_omega_eV(xi)
    prime_weights = matsubara_prime_weights(n_max)
    grid = q_polar_grid(q_max_m_inv, n_q, n_phi)
    measure = polar_measure_weights(grid["Q_m_inv"], grid["phi_rad"])
    kappa_values = kappa_si(grid["Q_m_inv"][None, :], xi[:, None])
    round_trip_factors = np.exp(-2.0 * kappa_values * separation_m)
    warnings = [
        Q0_WARNING,
        RESPONSE_GRID_WARNING,
        "This scaffold does not perform full Matsubara sum or full Q integration.",
        "No Casimir energy, force, or torque is output.",
    ]
    checks = run_checks(
        xi=xi,
        omega_eV=omega_eV,
        weights=prime_weights,
        grid=grid,
        measure=measure,
        round_trip_factors=round_trip_factors,
        warnings=warnings,
        n_max=n_max,
        n_q=n_q,
        n_phi=n_phi,
    )
    metadata = casimir_grid_scaffold_metadata()
    requirements = material_response_grid_requirements()
    has_zero_mode_q0 = bool(np.isclose(round_trip_factor_from_xi_Q_d(0.0, 0.0, separation_m), 1.0))
    return {
        "stage": "Stage 5.9",
        "purpose": "Casimir energy integration grid planning and scaffold",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": status,
        },
        "target_formula": {
            "free_energy_density": metadata["formula"],
            "kappa": "sqrt(Q^2 + xi_n^2/c^2)",
            "measure_polar": metadata["measure"],
        },
        "grid_parameters": {
            "temperature_K": float(temperature_K),
            "n_max": int(n_max),
            "q_max_m_inv": q_max_m_inv,
            "q_max_nm_inv": float(q_max_nm_inv),
            "n_q": int(n_q),
            "n_phi": int(n_phi),
            "separation_m": separation_m,
            "separation_nm": float(separation_nm),
        },
        "matsubara_grid": {
            "xi_si_s_inv": xi,
            "omega_eV": omega_eV,
            "prime_weights": prime_weights,
        },
        "q_phi_grid_summary": {
            "Q_m_inv": grid["Q_m_inv"],
            "Q_nm_inv": grid["Q_m_inv"] / 1.0e9,
            "phi_rad": grid["phi_rad"],
            "phi_deg": np.degrees(grid["phi_rad"]),
            "shape_Qx_Qy": list(grid["Qx_m_inv"].shape),
        },
        "measure_summary": {
            "min_weight": float(np.min(measure)),
            "max_weight": float(np.max(measure)),
            "sum_weight_scaffold": float(np.sum(measure)),
            "quadrature_scope": "scaffold only; not production convergence quadrature",
        },
        "round_trip_factor_summary": {
            "min": float(np.min(round_trip_factors)),
            "max": float(np.max(round_trip_factors)),
            "has_zero_mode_Q0_factor_one": has_zero_mode_q0,
        },
        "warnings": warnings,
        "material_response_grid_requirements": requirements,
        "checks": checks,
        "diagnostic_status": {
            "stage5_9_status": "STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED"
            if all(value == "PASS" for value in checks.values())
            else "STAGE5_9_CASIMIR_GRID_SCAFFOLD_FAILED",
            "recommended_next_action": "Proceed to toy-model full integration convergence audit before real material production energy.",
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "# Stage 5.9 Casimir grid planning scaffold",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Target energy formula\n\n"
            + data["target_formula"]["free_energy_density"]
            + "\n\n本阶段只规划变量和测度，不执行正式求和积分。",
            "## 4. Matsubara grid\n\n"
            + _table(("quantity", "value"), list(data["grid_parameters"].items())[:2])
            + "\n\nMatsubara prime weight 使用 `w0=1/2`，`n>0` 权重为 1。",
            "## 5. (Q, phi) grid\n\n"
            + _table(("quantity", "value"), list(data["q_phi_grid_summary"].items())),
            "## 6. Polar measure scaffold\n\n"
            + _table(("quantity", "value"), list(data["measure_summary"].items())),
            "## 7. Round-trip factor summary\n\n"
            + _table(("quantity", "value"), list(data["round_trip_factor_summary"].items())),
            "## 8. Warnings and limitations\n\n" + "\n".join(f"- {warning}" for warning in data["warnings"]),
            "## 9. Material response grid requirements\n\n"
            + _table(("requirement", "value"), list(data["material_response_grid_requirements"].items())),
            "## 10. Checks\n\n" + _table(("check", "status"), list(data["checks"].items())),
            "## 11. Diagnostic decision\n\n"
            + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 12. Recommended next step\n\n"
            + data["diagnostic_status"]["recommended_next_action"]
            + " 当前没有 full Matsubara sum，没有 full Q integral，没有输出 Casimir energy、force 或 torque。",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--q-max-nm-inv", type=float, default=0.5)
    parser.add_argument("--n-q", type=int, default=8)
    parser.add_argument("--n-phi", type=int, default=12)
    parser.add_argument("--separation-nm", type=float, default=100.0)
    parser.add_argument("--require-stage5-8-passed", action="store_true", default=True)
    parser.add_argument("--allow-non-passed-input", dest="require_stage5_8_passed", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_scaffold(
        args.input_json,
        temperature_K=args.temperature_K,
        n_max=args.n_max,
        q_max_nm_inv=args.q_max_nm_inv,
        n_q=args.n_q,
        n_phi=args.n_phi,
        separation_nm=args.separation_nm,
        require_stage5_8_passed=args.require_stage5_8_passed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
