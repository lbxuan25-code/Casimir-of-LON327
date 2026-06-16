#!/usr/bin/env python3
"""Stage 5.8 Casimir trace-log integrand prototype checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.casimir_integrand import (  # noqa: E402
    casimir_integrand_prototype_metadata,
    casimir_integrand_single_point,
    rotate_2x2_te_tm_toy_matrix,
    toy_anisotropic_symmetric_reflection,
    toy_isotropic_reflection,
    toy_zero_reflection,
)
from lno327.lifshitz_readiness import round_trip_factor, trace_log_matrix  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "casimir_integrand"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "reflection_input" / "stage5_6_te_tm_reflection_adapter.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_8_casimir_integrand_prototype.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_8_casimir_integrand_prototype.md"

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


def parse_complex_component(value: Any) -> complex:
    if isinstance(value, dict):
        return complex(float(value["real"]), float(value.get("imag", 0.0)))
    return complex(value)


def parse_complex_matrix(value: Any) -> np.ndarray:
    matrix = np.array([[parse_complex_component(item) for item in row] for row in value], dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    return matrix


def run_synthetic_checks() -> dict[str, str]:
    checks: dict[str, str] = {}
    kappa = 2.0e7
    d = 100.0e-9
    zero = toy_zero_reflection()
    checks["zero_reflection"] = "PASS" if abs(casimir_integrand_single_point(zero, zero, kappa, d)["logdet_integrand"]) < 1e-14 else "FAIL"

    nonzero = toy_isotropic_reflection(-0.05, 0.25)
    checks["one_zero_plate"] = "PASS" if abs(casimir_integrand_single_point(zero, nonzero, kappa, d)["logdet_integrand"]) < 1e-14 else "FAIL"

    bounded = toy_anisotropic_symmetric_reflection(-0.2, 0.3, 0.04)
    large_sep = casimir_integrand_single_point(bounded, bounded, kappa, 1.0)["logdet_integrand"]
    checks["large_separation"] = "PASS" if abs(large_sep) < 1e-12 else "FAIL"

    weak = toy_isotropic_reflection(-0.02, 0.03)
    near = casimir_integrand_single_point(weak, weak, kappa, 20.0e-9)["logdet_integrand"]
    far = casimir_integrand_single_point(weak, weak, kappa, 200.0e-9)["logdet_integrand"]
    checks["small_separation_magnitude"] = "PASS" if abs(near) > abs(far) and near.real < far.real < 0.0 else "FAIL"

    scalar = toy_isotropic_reflection(-0.07, 0.21)
    u2 = round_trip_factor(kappa, d)
    expected = np.log(1.0 - u2 * scalar[0, 0] ** 2) + np.log(1.0 - u2 * scalar[1, 1] ** 2)
    actual = casimir_integrand_single_point(scalar, scalar, kappa, d)["logdet_integrand"]
    checks["isotropic_identical_sheets_formula"] = "PASS" if abs(actual - expected) < 1e-14 else "FAIL"

    scalar_angle_invariant = toy_isotropic_reflection(0.12, 0.12)
    isotropic_angles = [
        casimir_integrand_single_point(
            scalar_angle_invariant,
            rotate_2x2_te_tm_toy_matrix(scalar_angle_invariant, theta),
            kappa,
            d,
        )["logdet_integrand"]
        for theta in (0.0, np.pi / 8.0, np.pi / 2.0)
    ]
    checks["isotropic_angle_independence"] = "PASS" if max(abs(value - isotropic_angles[0]) for value in isotropic_angles) < 1e-14 else "FAIL"

    anisotropic = toy_anisotropic_symmetric_reflection(-0.12, 0.28, 0.05)
    theta = np.pi / 5.0
    i_theta = casimir_integrand_single_point(anisotropic, rotate_2x2_te_tm_toy_matrix(anisotropic, theta), kappa, d)["logdet_integrand"]
    i_theta_pi = casimir_integrand_single_point(anisotropic, rotate_2x2_te_tm_toy_matrix(anisotropic, theta + np.pi), kappa, d)["logdet_integrand"]
    i_zero = casimir_integrand_single_point(anisotropic, rotate_2x2_te_tm_toy_matrix(anisotropic, 0.0), kappa, d)["logdet_integrand"]
    i_pi = casimir_integrand_single_point(anisotropic, rotate_2x2_te_tm_toy_matrix(anisotropic, np.pi), kappa, d)["logdet_integrand"]
    checks["anisotropic_toy_periodicity"] = "PASS" if abs(i_theta - i_theta_pi) < 1e-14 and abs(i_zero - i_pi) < 1e-14 else "FAIL"

    r1 = np.array([[0.2, 0.4], [0.0, -0.1]], dtype=complex)
    r2 = np.array([[0.3, 0.0], [0.5, 0.1]], dtype=complex)
    order_expected = np.eye(2, dtype=complex) - u2 * (r1 @ r2)
    reverse_order = np.eye(2, dtype=complex) - u2 * (r2 @ r1)
    order_actual = trace_log_matrix(r1, r2, kappa, d)
    checks["matrix_order"] = "PASS" if np.allclose(order_actual, order_expected) and not np.allclose(order_actual, reverse_order) else "FAIL"
    return checks


def representative_row(row: dict[str, Any], separation_m: float) -> dict[str, Any]:
    reflection = parse_complex_matrix(row["reflection_TE_TM"])
    kappa = float(row["kappa_m_inv"])
    package = casimir_integrand_single_point(reflection, reflection, kappa, separation_m)
    return {
        "q_case": row.get("q_case"),
        "matsubara_n": row.get("matsubara_n"),
        "q_scale": row.get("q_scale", 1.0),
        "kappa_m_inv": kappa,
        "separation_m": separation_m,
        "R1_R2_choice": "identical-sheet toy pair using this Stage 5.6 row for both R1 and R2",
        "round_trip_factor": package["round_trip_factor"],
        "trace_log_matrix_M": package["trace_log_matrix"],
        "logdet_integrand": package["logdet_integrand"],
        "label": "representative_integrand_level_check_not_physical_energy",
    }


def run_prototype(
    input_json: Path,
    *,
    separation_nm: float,
    max_representative_rows: int,
    require_stage5_6_passed: bool,
) -> dict[str, Any]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    status = data.get("diagnostic_status", {}).get("stage5_6_status")
    if require_stage5_6_passed and status != "STAGE5_6_TE_TM_ADAPTER_PASSED":
        raise ValueError("input must have STAGE5_6_TE_TM_ADAPTER_PASSED status")
    separation_m = float(separation_nm) * 1.0e-9
    if separation_m <= 0.0:
        raise ValueError("separation_nm must be positive")
    rows = data.get("converted_results", [])
    representatives = [representative_row(row, separation_m) for row in rows[:max_representative_rows]]
    checks = run_synthetic_checks()
    metadata = casimir_integrand_prototype_metadata()
    max_abs_logdet = max((float(abs(row["logdet_integrand"])) for row in representatives), default=0.0)
    max_abs_imag = max((float(abs(row["logdet_integrand"].imag)) for row in representatives), default=0.0)
    all_pass = all(value == "PASS" for value in checks.values())
    return {
        "stage": "Stage 5.8",
        "purpose": "Casimir trace-log integrand prototype with controlled synthetic and validation-point checks",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": status,
            "num_input_cases": len(rows),
        },
        "integrand_convention": {
            "basis": metadata["basis"],
            "ordering": metadata["ordering"],
            "rows": metadata["rows"],
            "columns": metadata["columns"],
            "M": metadata["matrix_formula"],
            "integrand": metadata["integrand_formula"],
            "round_trip_factor": metadata["round_trip_factor_formula"],
        },
        "prototype_scope": {
            "full_matsubara_sum": False,
            "full_Q_integral": False,
            "casimir_energy": False,
            "casimir_force": False,
            "casimir_torque": False,
            "production_run": False,
            "toy_rotation_only_not_physical_material_rotation": metadata["toy_rotation_only_not_physical_material_rotation"],
        },
        "synthetic_checks": checks,
        "representative_stage5_6_checks": representatives,
        "summary": {
            "separation_m": separation_m,
            "num_representative_rows": len(representatives),
            "max_abs_representative_logdet": max_abs_logdet,
            "max_abs_representative_imag_logdet": max_abs_imag,
            "synthetic_checks_all_pass": bool(all_pass),
        },
        "diagnostic_status": {
            "stage5_8_status": "STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED"
            if all_pass
            else "STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_FAILED",
            "recommended_next_action": "Proceed to material response grid planning for full Matsubara/Q integration; do not run production torque yet.",
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    reps = [
        (
            row["q_case"],
            row["matsubara_n"],
            row["round_trip_factor"],
            row["logdet_integrand"],
        )
        for row in data["representative_stage5_6_checks"]
    ]
    return "\n\n".join(
        [
            "# Stage 5.8 Casimir trace-log integrand prototype",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Integrand convention\n\n"
            "`M = I - exp(-2*kappa*d) * R1 @ R2`，prototype value 是 `log(det(M))`。"
            "`R1` 和 `R2` 使用 TE/TM amplitude basis，顺序为 `['s', 'p']`，行是 reflected polarization，列是 incident polarization。",
            "## 4. Prototype scope\n\n" + _table(("scope", "enabled"), list(data["prototype_scope"].items())),
            "## 5. Synthetic checks\n\n" + _table(("check", "status"), list(data["synthetic_checks"].items())),
            "## 6. Representative Stage 5.6 integrand-level values\n\n"
            + _table(("q", "n", "exp(-2*kappa*d)", "logdet"), reps)
            + "\n\n这些 representative material rows 只是 validation-point integrand-level checks，不是物理能量或力矩。",
            "## 7. What this is not\n\n"
            "本阶段没有 full Matsubara sum，没有 full Q integral，没有输出 Casimir energy、force 或 torque。"
            "anisotropic toy periodicity 只是 synthetic matrix check，不是 LNO327 物理 torque。",
            "## 8. Diagnostic decision\n\n"
            + _table(("quantity", "value"), list(data["diagnostic_status"].items()))
            + "\n\n"
            + _table(("summary", "value"), list(data["summary"].items())),
            "## 9. Recommended next step\n\n" + data["diagnostic_status"]["recommended_next_action"],
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--separation-nm", type=float, default=100.0)
    parser.add_argument("--max-representative-rows", type=int, default=6)
    parser.add_argument("--require-stage5-6-passed", action="store_true", default=True)
    parser.add_argument("--allow-non-passed-input", dest="require_stage5_6_passed", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_prototype(
        args.input_json,
        separation_nm=args.separation_nm,
        max_representative_rows=args.max_representative_rows,
        require_stage5_6_passed=args.require_stage5_6_passed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
