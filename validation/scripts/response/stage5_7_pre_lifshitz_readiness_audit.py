#!/usr/bin/env python3
"""Stage 5.7 pre-Lifshitz readiness audit for TE/TM reflection matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.casimir.readiness import (  # noqa: E402
    lab_q_to_crystal_q,
    pre_lifshitz_readiness_metadata,
    round_trip_factor,
    scalar_sheet_te_tm_reflection,
    trace_log_integrand,
    trace_log_matrix,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "lifshitz_readiness"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "reflection_input" / "stage5_6_te_tm_reflection_adapter.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_7_pre_lifshitz_readiness_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_7_pre_lifshitz_readiness_audit.md"

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
    "no_lifshitz_trace_log_production": True,
    "no_casimir_energy": True,
    "no_casimir_force": True,
    "no_casimir_torque": True,
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
    separation = 100.0e-9
    zero = np.zeros((2, 2), dtype=complex)
    zero_m = trace_log_matrix(zero, zero, kappa, separation)
    zero_i = trace_log_integrand(zero, zero, kappa, separation)
    checks["zero_reflection"] = "PASS" if np.allclose(zero_m, np.eye(2)) and abs(zero_i) < 1e-14 else "FAIL"

    bounded = np.array([[0.2, 0.03], [0.01, -0.1]], dtype=complex)
    large_sep_i = trace_log_integrand(bounded, bounded, kappa, 1.0)
    checks["large_separation"] = "PASS" if abs(large_sep_i) < 1e-12 else "FAIL"

    zero_sheet = scalar_sheet_te_tm_reflection(0.0, eta_L=0.4, eta_T=2.0)
    checks["zero_sheet"] = "PASS" if abs(trace_log_integrand(zero_sheet, zero_sheet, kappa, separation)) < 1e-14 else "FAIL"

    scalar = scalar_sheet_te_tm_reflection(0.25, eta_L=0.4, eta_T=2.0)
    u2 = round_trip_factor(kappa, separation)
    rss = scalar[0, 0]
    rpp = scalar[1, 1]
    expected = np.log(1.0 - u2 * rss**2) + np.log(1.0 - u2 * rpp**2)
    actual = trace_log_integrand(scalar, scalar, kappa, separation)
    checks["isotropic_identical_sheets"] = "PASS" if abs(actual - expected) < 1e-14 else "FAIL"

    isotropic_by_angle = []
    for theta in (0.0, np.pi / 7.0, np.pi / 2.0):
        _ = lab_q_to_crystal_q(np.array([1.0, 0.0]), theta)
        isotropic_by_angle.append(trace_log_integrand(scalar, scalar, kappa, separation))
    checks["isotropic_angle_independence"] = "PASS" if max(abs(x - isotropic_by_angle[0]) for x in isotropic_by_angle) < 1e-14 else "FAIL"

    q_crystal = lab_q_to_crystal_q(np.array([3.0, 0.0]), np.pi / 2.0)
    checks["rotation_convention"] = "PASS" if np.allclose(q_crystal, np.array([0.0, -3.0]), atol=1e-14) else "FAIL"

    r1 = np.array([[0.2, 0.4], [0.0, -0.1]], dtype=complex)
    r2 = np.array([[0.3, 0.0], [0.5, 0.1]], dtype=complex)
    order_expected = np.eye(2, dtype=complex) - u2 * (r1 @ r2)
    reverse_order = np.eye(2, dtype=complex) - u2 * (r2 @ r1)
    order_actual = trace_log_matrix(r1, r2, kappa, separation)
    checks["matrix_order"] = "PASS" if np.allclose(order_actual, order_expected) and not np.allclose(order_actual, reverse_order) else "FAIL"
    return checks


def representative_row(row: dict[str, Any], separation_m: float) -> dict[str, Any]:
    reflection = parse_complex_matrix(row["reflection_TE_TM"])
    kappa = float(row["kappa_m_inv"])
    u2 = round_trip_factor(kappa, separation_m)
    matrix = trace_log_matrix(reflection, reflection, kappa, separation_m)
    logdet = trace_log_integrand(reflection, reflection, kappa, separation_m)
    return {
        "q_case": row.get("q_case"),
        "matsubara_n": row.get("matsubara_n"),
        "q_scale": row.get("q_scale", 1.0),
        "kappa_m_inv": kappa,
        "separation_m": separation_m,
        "round_trip_factor": u2,
        "R1_R2_choice": "identical-sheet toy pair using this Stage 5.6 row for both R1 and R2",
        "trace_log_matrix_M": matrix,
        "logdet_integrand": logdet,
        "interpretation": "integrand-level formatting check only; not a Casimir energy, force, or torque",
    }


def run_audit(
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
    representatives = [
        representative_row(row, separation_m)
        for row in rows[:max_representative_rows]
    ]
    checks = run_synthetic_checks()
    max_abs_logdet = max((float(abs(row["logdet_integrand"])) for row in representatives), default=0.0)
    max_abs_imag = max((float(abs(row["logdet_integrand"].imag)) for row in representatives), default=0.0)
    metadata = pre_lifshitz_readiness_metadata()
    return {
        "stage": "Stage 5.7",
        "purpose": "Pre-Lifshitz readiness audit for TE/TM reflection matrices",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": status,
            "num_input_cases": len(rows),
        },
        "matrix_convention": {
            "basis": "TE_TM_amplitude_basis",
            "ordering": metadata["matrix_ordering"],
            "rows": metadata["rows"],
            "columns": metadata["columns"],
            "R_definition": metadata["R_definition"],
        },
        "trace_log_convention": {
            "M": metadata["trace_log_matrix_formula"],
            "integrand": metadata["integrand_formula"],
            "round_trip_factor": metadata["round_trip_factor_formula"],
            "R1_R2_basis_requirement": metadata["R1_R2_basis_requirement"],
            "plate_convention": metadata["plate_convention"],
        },
        "rotation_convention": metadata["rotation_convention"],
        "synthetic_checks": checks,
        "representative_real_stage5_6_integrand_checks": representatives,
        "summary": {
            "separation_m": separation_m,
            "num_representative_rows": len(representatives),
            "max_abs_representative_logdet": max_abs_logdet,
            "max_abs_representative_imag_logdet": max_abs_imag,
        },
        "diagnostic_status": {
            "stage5_7_status": "STAGE5_7_PRE_LIFSHITZ_READINESS_PASSED"
            if all(value == "PASS" for value in checks.values())
            else "STAGE5_7_PRE_LIFSHITZ_READINESS_FAILED",
            "recommended_next_action": "Proceed to Casimir integrand prototype with controlled synthetic/material-grid inputs; still do not run production torque.",
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
        for row in data["representative_real_stage5_6_integrand_checks"]
    ]
    return "\n\n".join(
        [
            "# Stage 5.7 pre-Lifshitz readiness audit",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Matrix convention\n\n"
            "`R^{TE/TM}` 使用 TE/TM amplitude basis，顺序为 `['s', 'p']`。行是 reflected polarization，列是 incident polarization，定义为 `E_ref = R E_inc`。",
            "## 4. Trace-log integrand convention\n\n"
            "`M = I - exp(-2*kappa*d) * R1 @ R2`，单点 integrand 是 `log(det(M))`。"
            "`R1` 和 `R2` 必须在同一个 lab-frame TE/TM basis 中表达。本阶段只检查 integrand-level object。",
            "## 5. Plate rotation convention\n\n"
            "材料旋转角 `theta` 是 plate 2 crystal axes 相对 plate 1/lab axes 的旋转角。"
            "在材料自己的 crystal frame 中，`Q_crystal = R(-theta) Q_lab`。最终 `R_TE_TM` 都必须回到共同 lab TE/TM basis。",
            "## 6. Synthetic checks\n\n" + _table(("check", "status"), list(data["synthetic_checks"].items())),
            "## 7. Representative real Stage 5.6 integrand-level checks\n\n"
            + _table(("q", "n", "exp(-2*kappa*d)", "logdet"), reps)
            + "\n\n这些数值只来自 identical-sheet toy pair，不是完整 Matsubara sum，也不是完整 d^2Q 积分。",
            "## 8. Diagnostic decision\n\n"
            + _table(("quantity", "value"), list(data["diagnostic_status"].items()))
            + "\n\n"
            + _table(("summary", "value"), list(data["summary"].items())),
            "## 9. Recommended next step\n\n"
            + data["diagnostic_status"]["recommended_next_action"]
            + " 当前没有输出 Casimir energy、force 或 torque，也没有声明 production-ready。",
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
    data = run_audit(
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
