#!/usr/bin/env python3
"""Stage 5.10 toy-model full Casimir integration convergence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.casimir_toy_integration import (  # noqa: E402
    convergence_scan_toy,
    integrate_toy_free_energy_density,
    toy_integration_metadata,
    xi_c_from_omega_eV,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "casimir_toy_integration"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "casimir_grid" / "stage5_9_casimir_grid_planning_scaffold.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_10_toy_casimir_integration_convergence_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_10_toy_casimir_integration_convergence_audit.md"

BOUNDARY = {
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "no_heavy_response_run": True,
    "toy_model_only": True,
    "no_real_material_response_grid": True,
    "no_real_LNO327_energy": True,
    "no_force_output": True,
    "no_torque_output": True,
    "not_casimir_ready_claim": True,
}

WARNINGS = [
    "Toy-model full integration is not a real material Casimir energy calculation.",
    "No LNO327 material response grid is used.",
    "Do not interpret toy energy density as physical prediction.",
    "Next real-material stage requires R_TE_TM(i*xi_n,Q,phi) or sigma_tilde(i*xi_n,Q,phi) on a production grid.",
]


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


def _finite_scan(scan: dict[str, Any]) -> bool:
    return all(np.all(np.isfinite(section["free_energy_density_J_m2"])) for section in scan.values())


def _imaginary_part_ok(results: list[dict[str, Any]], *, abs_tol: float = 1e-28, rel_tol: float = 1e-8) -> bool:
    for result in results:
        real = abs(float(result["free_energy_density_J_m2"]))
        imag = abs(float(result["imag_part_J_m2"]))
        if real == 0.0:
            if imag > abs_tol:
                return False
        elif imag / real > rel_tol:
            return False
    return True


def run_audit(
    input_json: Path,
    *,
    temperature_K: float,
    separation_nm: float,
    require_stage5_9_passed: bool,
) -> dict[str, Any]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    status = data.get("diagnostic_status", {}).get("stage5_9_status")
    if require_stage5_9_passed and status != "STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED":
        raise ValueError("input must have STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED status")

    separation_m = float(separation_nm) * 1.0e-9
    q_c_m_inv = 0.2e9
    xi_c_si = xi_c_from_omega_eV(0.05)
    params = {"Qc_m_inv": q_c_m_inv, "xi_c_si": xi_c_si}
    common = {
        "temperature_K": float(temperature_K),
        "n_max": 8,
        "Q_max_m_inv": 0.5e9,
        "n_Q": 12,
        "n_phi": 12,
        "separation_m": separation_m,
    }

    zero = integrate_toy_free_energy_density(theta_rad=0.0, model="zero", **common)
    iso_angles = {
        str(theta): integrate_toy_free_energy_density(theta_rad=theta, model="isotropic_identical", **common, **params)
        for theta in (0.0, np.pi / 6.0, np.pi / 4.0, np.pi / 2.0, np.pi)
    }
    anisotropic_angles = {
        str(theta): integrate_toy_free_energy_density(theta_rad=theta, model="anisotropic_relative_rotation", **common, **params)
        for theta in (0.0, np.pi / 4.0, np.pi)
    }
    distance_results = {
        "50_nm": integrate_toy_free_energy_density(theta_rad=0.0, model="isotropic_identical", separation_m=50.0e-9, **{k: v for k, v in common.items() if k != "separation_m"}, **params),
        "100_nm": integrate_toy_free_energy_density(theta_rad=0.0, model="isotropic_identical", **common, **params),
        "200_nm": integrate_toy_free_energy_density(theta_rad=0.0, model="isotropic_identical", separation_m=200.0e-9, **{k: v for k, v in common.items() if k != "separation_m"}, **params),
    }
    scans = convergence_scan_toy(
        temperature_K=float(temperature_K),
        n_max_values=[2, 4, 8],
        Q_max_values_m_inv=[0.25e9, 0.5e9, 0.75e9],
        n_Q_values=[8, 12, 16],
        n_phi_values=[8, 12, 16],
        separation_m=separation_m,
        theta_rad=0.0,
        model="isotropic_identical",
        **params,
    )

    zero_pass = abs(zero["free_energy_density_J_m2"]) < 1e-32 and abs(zero["imag_part_J_m2"]) < 1e-32
    iso_values = [result["free_energy_density_J_m2"] for result in iso_angles.values()]
    iso_pass = max(abs(value - iso_values[0]) for value in iso_values) < 1e-30
    aniso_zero = anisotropic_angles[str(0.0)]["free_energy_density_J_m2"]
    aniso_pi = anisotropic_angles[str(np.pi)]["free_energy_density_J_m2"]
    aniso_quarter = anisotropic_angles[str(np.pi / 4.0)]["free_energy_density_J_m2"]
    aniso_periodic_pass = abs(aniso_zero - aniso_pi) < 1e-30
    aniso_variation = abs(aniso_quarter - aniso_zero)
    distance_values = [distance_results[key]["free_energy_density_J_m2"] for key in ("50_nm", "100_nm", "200_nm")]
    distance_pass = (
        distance_values[0] < distance_values[1] < distance_values[2] < 0.0
        and abs(distance_values[0]) > abs(distance_values[1]) > abs(distance_values[2])
    )
    all_results = [zero, *iso_angles.values(), *anisotropic_angles.values(), *distance_results.values()]
    scan_finite = _finite_scan(scans)
    checks = {
        "zero_toy_integration": "PASS" if zero_pass else "FAIL",
        "isotropic_angle_independence": "PASS" if iso_pass else "FAIL",
        "anisotropic_toy_periodicity": "PASS" if aniso_periodic_pass and aniso_variation > 1e-35 else "MONITOR",
        "distance_dependence": "PASS" if distance_pass else "FAIL",
        "n_max_convergence": scans["n_max"]["status"],
        "Q_max_convergence": scans["Q_max"]["status"],
        "n_Q_convergence": scans["n_Q"]["status"],
        "n_phi_convergence": scans["n_phi"]["status"],
        "imaginary_part_sanity": "PASS" if _imaginary_part_ok(all_results) and scan_finite else "FAIL",
    }
    hard_fail = any(value == "FAIL" for value in checks.values())
    monitor = any(value == "MONITOR" for value in checks.values())
    if hard_fail:
        status_out = "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_FAILED"
    elif monitor:
        status_out = "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_MONITOR"
    else:
        status_out = "STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED"

    metadata = toy_integration_metadata()
    return {
        "stage": "Stage 5.10",
        "purpose": "Toy-model full Casimir integration convergence audit",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": status,
        },
        "scope": {
            "toy_model_only": metadata["toy_model_only"],
            "not_material_prediction": metadata["not_material_prediction"],
            "no_real_LNO327_energy": metadata["no_real_LNO327_energy"],
            "no_force": metadata["no_force"],
            "no_torque": metadata["no_torque"],
        },
        "toy_models": {
            "zero": {"R": "0"},
            "isotropic_identical": {"r_s": "-r0*f(xi,Q)", "r_p": "+r0*f(xi,Q)", "r0": 0.3},
            "anisotropic_relative_rotation": {"rs0": -0.25, "rp0": 0.35, "mixing0": 0.05, "toy_rotation_only": True},
        },
        "baseline_parameters": {
            "temperature_K": float(temperature_K),
            "separation_nm": float(separation_nm),
            "Qc_nm_inv": 0.2,
            "omega_c_eV": 0.05,
        },
        "checks": checks,
        "baseline_toy_results": {
            "zero": zero,
            "isotropic_by_theta": iso_angles,
            "anisotropic_by_theta": anisotropic_angles,
            "distance_dependence": distance_results,
            "anisotropic_angle_variation_abs_J_m2": aniso_variation,
        },
        "convergence_scans": scans,
        "warnings": WARNINGS,
        "diagnostic_status": {
            "stage5_10_status": status_out,
            "recommended_next_action": "Proceed to real-material response/reflection grid generation planning; still do not run production torque.",
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    checks = list(data["checks"].items())
    scan_rows = [
        (name, scan["values"], scan["free_energy_density_J_m2"], scan["relative_changes"], scan["status"])
        for name, scan in data["convergence_scans"].items()
    ]
    return "\n\n".join(
        [
            "# Stage 5.10 toy Casimir integration convergence audit",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Toy model definitions\n\n"
            + _table(("model", "definition"), [(name, details) for name, details in data["toy_models"].items()]),
            "## 4. Integration formula\n\n"
            "`F_toy/A = k_B*T*sum_n' integral Q dQ dphi/(2*pi)^2 logdet[I-exp(-2*kappa*d) R1_toy R2_toy]`。"
            "这个公式只用于 toy matrices。",
            "## 5. Baseline toy results\n\n"
            + _table(("quantity", "value"), list(data["baseline_parameters"].items())),
            "## 6. Zero and isotropic checks\n\n"
            + _table(("check", "status"), [row for row in checks if row[0] in ("zero_toy_integration", "isotropic_angle_independence")]),
            "## 7. Anisotropic toy angle checks\n\n"
            + _table(("check", "status"), [row for row in checks if row[0] == "anisotropic_toy_periodicity"]),
            "## 8. Distance dependence\n\n"
            + _table(("check", "status"), [row for row in checks if row[0] == "distance_dependence"]),
            "## 9. Convergence scans\n\n"
            + _table(("scan", "values", "F_toy/A", "relative_changes", "status"), scan_rows),
            "## 10. Imaginary-part sanity\n\n"
            + _table(("check", "status"), [row for row in checks if row[0] == "imaginary_part_sanity"]),
            "## 11. What this is not\n\n"
            + "\n".join(f"- {warning}" for warning in data["warnings"]),
            "## 12. Diagnostic decision\n\n"
            + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 13. Recommended next step\n\n" + data["diagnostic_status"]["recommended_next_action"],
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--separation-nm", type=float, default=100.0)
    parser.add_argument("--require-stage5-9-passed", action="store_true", default=True)
    parser.add_argument("--allow-non-passed-input", dest="require_stage5_9_passed", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_audit(
        args.input_json,
        temperature_K=args.temperature_K,
        separation_nm=args.separation_nm,
        require_stage5_9_passed=args.require_stage5_9_passed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
