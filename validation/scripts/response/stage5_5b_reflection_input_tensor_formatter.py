#!/usr/bin/env python3
"""Stage 5.5b formatter for LT-basis tangential-electric reflection input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from lno327.material_structure import LNO327_THIN_FILM_SLAO_IN_PLANE  # noqa: E402
from lno327.electrodynamics.reflection import (  # noqa: E402
    model_q_to_si_wavevector,
    omega_eV_to_xi_si,
    reflection_input_metadata,
    rotate_sigma_tilde_xy_to_lt,
    symmetric_antisymmetric_offdiag,
    tangential_electric_reflection_matrix_LT,
    vacuum_admittance_LT,
    vacuum_kappa,
    xy_to_lt_rotation,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "reflection_input"
DEFAULT_INPUT = ROOT / "validation" / "outputs" / "response" / "conductivity" / "stage5_4b_si_sheet_dimensionless_conductivity.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_5b_reflection_input_tensor.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_5b_reflection_input_tensor.md"

BOUNDARY = {
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "no_heavy_response_run": True,
    "no_lifshitz_trace_log": True,
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


def sigma_tilde_xy_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [parse_complex_component(row["sigma_tilde_xx"]), parse_complex_component(row["sigma_tilde_xy"])],
            [parse_complex_component(row["sigma_tilde_yx"]), parse_complex_component(row["sigma_tilde_yy"])],
        ],
        dtype=complex,
    )


def q_model_from_row(row: dict[str, Any]) -> tuple[float, float, str]:
    if "q_model_x" in row and "q_model_y" in row:
        return float(row["q_model_x"]), float(row["q_model_y"]), "explicit_q_model_x_y"
    if "q_model" in row:
        q = row["q_model"]
        if len(q) != 2:
            raise ValueError("q_model must contain two components")
        return float(q[0]), float(q[1]), "input_q_model_vector"
    raise ValueError("input row must contain q_model_x/q_model_y or q_model")


def format_row(row: dict[str, Any], *, allow_q_zero: bool) -> dict[str, Any]:
    qx_model, qy_model, q_source = q_model_from_row(row)
    qx_si, qy_si, q_si = model_q_to_si_wavevector(
        qx_model,
        qy_model,
        LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
    )
    xi_si = omega_eV_to_xi_si(float(row["omega_eV"]))
    kappa = vacuum_kappa(q_si, xi_si)
    sigma_xy = sigma_tilde_xy_from_row(row)
    rotation = xy_to_lt_rotation(qx_si, qy_si, allow_q_zero=allow_q_zero)
    sigma_lt = rotate_sigma_tilde_xy_to_lt(sigma_xy, qx_si, qy_si, allow_q_zero=allow_q_zero)
    y0 = vacuum_admittance_LT(xi_si, kappa)
    reflection = tangential_electric_reflection_matrix_LT(sigma_lt, xi_si, kappa)
    return {
        "q_case": row.get("q_case"),
        "matsubara_n": row.get("matsubara_index"),
        "q_scale": row.get("q_scale", 1.0),
        "omega_eV": float(row["omega_eV"]),
        "xi_si_s_inv": xi_si,
        "q_model_x": qx_model,
        "q_model_y": qy_model,
        "q_model_source": q_source,
        "Q_x_m_inv": qx_si,
        "Q_y_m_inv": qy_si,
        "Q_m_inv": q_si,
        "kappa_m_inv": kappa,
        "sigma_tilde_xy_matrix": sigma_xy,
        "xy_to_lt_rotation_matrix": rotation,
        "sigma_tilde_LT_matrix": sigma_lt,
        "vacuum_admittance_Y0_LT": y0,
        "reflection_tangential_E_LT": reflection,
        "sigma_tilde_LT_offdiag_diagnostics": symmetric_antisymmetric_offdiag(sigma_lt),
        "q_zero_basis_convention": "L=x, T=y" if q_si == 0.0 and allow_q_zero else None,
    }


def run_synthetic_checks() -> dict[str, str]:
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    qx_rot = xy_to_lt_rotation(2.0, 0.0)
    qx_pass = bool(np.allclose(qx_rot, np.eye(2)) and np.allclose(rotate_sigma_tilde_xy_to_lt(tensor, 2.0, 0.0), tensor))
    qy_expected = np.array([[4.0, -3.0], [-2.0, 1.0]], dtype=complex)
    qy_pass = bool(np.allclose(xy_to_lt_rotation(0.0, 2.0), np.array([[0.0, 1.0], [-1.0, 0.0]])) and np.allclose(rotate_sigma_tilde_xy_to_lt(tensor, 0.0, 2.0), qy_expected))
    sigma_scalar = 0.2 * np.eye(2, dtype=complex)
    sigma_scalar_lt = rotate_sigma_tilde_xy_to_lt(sigma_scalar, 1.0, 1.0)
    xi, kappa = 3.0e13, 2.0e6
    scalar_r = tangential_electric_reflection_matrix_LT(sigma_scalar_lt, xi, kappa)
    y0 = vacuum_admittance_LT(xi, kappa)
    scalar_expected = np.diag([-0.2 / (2.0 * y0[0, 0] + 0.2), -0.2 / (2.0 * y0[1, 1] + 0.2)])
    scalar_pass = bool(np.allclose(sigma_scalar_lt, sigma_scalar) and np.allclose(scalar_r, scalar_expected) and abs(scalar_r[0, 1]) < 1e-14)
    weak_sigma = 1e-6 * np.array([[1.0, 0.2], [0.2, 2.0]], dtype=complex)
    weak_r = tangential_electric_reflection_matrix_LT(weak_sigma, xi, kappa)
    weak_expected = -0.5 * np.linalg.solve(y0, weak_sigma)
    weak_pass = bool(np.linalg.norm(weak_r - weak_expected) / max(np.linalg.norm(weak_expected), 1e-300) < 1e-5)
    diag_r = tangential_electric_reflection_matrix_LT(np.diag([0.1, 0.2]).astype(complex), xi, kappa)
    diag_pass = bool(abs(diag_r[0, 1]) < 1e-14 and abs(diag_r[1, 0]) < 1e-14)
    mixing_r = tangential_electric_reflection_matrix_LT(np.array([[0.1, 0.03], [0.03, 0.2]], dtype=complex), xi, kappa)
    mixing_pass = bool(abs(mixing_r[0, 1]) > 0.0 or abs(mixing_r[1, 0]) > 0.0)
    hall_diag = symmetric_antisymmetric_offdiag(np.array([[0.1, 0.02], [-0.02, 0.1]], dtype=complex))
    hall_pass = bool(hall_diag["antisymmetric_offdiag_abs"] > hall_diag["symmetric_offdiag_abs"])
    return {
        "qx_basis_check": "PASS" if qx_pass else "FAIL",
        "qy_basis_sign_check": "PASS" if qy_pass else "FAIL",
        "isotropic_scalar_sheet_check": "PASS" if scalar_pass else "FAIL",
        "weak_sheet_limit": "PASS" if weak_pass else "FAIL",
        "diagonal_LT_no_mixing": "PASS" if diag_pass else "FAIL",
        "offdiag_LT_retains_mixing": "PASS" if mixing_pass else "FAIL",
        "hall_like_antisymmetric_marker": "PASS" if hall_pass else "FAIL",
    }


def run_formatter(input_json: Path, *, allow_q_zero: bool, require_stage5_4b_passed: bool) -> dict[str, Any]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    status = data.get("diagnostic_status", {}).get("stage5_4b_status")
    if require_stage5_4b_passed and status != "STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED":
        raise ValueError("input must have STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED status")
    rows = [format_row(row, allow_q_zero=allow_q_zero) for row in data.get("converted_results", [])]
    checks = run_synthetic_checks()
    max_abs_r = max((float(np.max(np.abs(row["reflection_tangential_E_LT"]))) for row in rows), default=0.0)
    max_abs_r_offdiag = max((float(np.linalg.norm([row["reflection_tangential_E_LT"][0, 1], row["reflection_tangential_E_LT"][1, 0]])) for row in rows), default=0.0)
    max_abs_sigma_offdiag = max((float(np.linalg.norm([row["sigma_tilde_LT_matrix"][0, 1], row["sigma_tilde_LT_matrix"][1, 0]])) for row in rows), default=0.0)
    all_pass = all(value == "PASS" for value in checks.values())
    metadata = reflection_input_metadata(q_zero_basis_convention="L=x, T=y" if allow_q_zero else None)
    return {
        "stage": "Stage 5.5b",
        "purpose": "Format sigma_tilde sheet conductivity into LT-basis tangential-electric reflection input",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": status,
            "num_input_cases": len(data.get("converted_results", [])),
        },
        "basis_convention": {
            "basis": metadata["basis"],
            "basis_order": metadata["basis_order"],
            "L_definition": metadata["L_definition"],
            "T_definition": metadata["T_definition"],
            "xy_to_lt_rotation": "v_LT = R_Q v_xy",
            "sigma_rotation": metadata["formula_sigma_LT"],
            "TE_TM_adapter_included": False,
        },
        "frequency_wavevector_convention": {
            "q_model_to_SI": "Q_x = q_model_x/a_x, Q_y = q_model_y/a_y",
            "omega_eV_to_xi": "xi = omega_eV * eV_J / hbar",
            "kappa": "sqrt(Q^2 + xi^2/c^2)",
        },
        "reflection_input_formula": {
            "Y0_LT": metadata["formula_Y0"],
            "R_E_LT": metadata["formula_R_E_LT"],
            "meaning": "E_ref_LT = R_E_LT E_inc_LT",
        },
        "reflection_input_results": rows,
        "synthetic_checks": checks,
        "summary": {
            "num_cases": len(rows),
            "max_abs_R_E_LT": max_abs_r,
            "max_abs_R_E_offdiag": max_abs_r_offdiag,
            "max_abs_sigma_tilde_LT_offdiag": max_abs_sigma_offdiag,
            "num_q_zero_cases": sum(row["Q_m_inv"] == 0.0 for row in rows),
            "all_finite_q_cases_formatted": bool(all(row["Q_m_inv"] > 0.0 for row in rows)),
        },
        "diagnostic_status": {
            "stage5_5b_status": "STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED" if all_pass else "STAGE5_5B_REFLECTION_INPUT_FORMATTER_FAILED",
            "recommended_next_action": "Proceed to TE/TM adapter convention audit; still do not compute Casimir energy/torque.",
        },
    }


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any], *, max_representative_rows: int) -> str:
    reps = [
        (
            row["q_case"],
            row["matsubara_n"],
            row["Q_m_inv"],
            row["kappa_m_inv"],
            row["reflection_tangential_E_LT"],
        )
        for row in data["reflection_input_results"][:max_representative_rows]
    ]
    return "\n\n".join(
        [
            "# Stage 5.5b reflection-input tensor formatter",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. (L/T) basis definition\n\n"
            "(L/T) 不是全局固定坐标，而是每个 $\\mathbf q$ 点上的局部坐标：$L\\parallel\\mathbf Q$，$T=\\hat z\\times L$。",
            "## 4. Frequency and wave-vector conversion\n\n" + _table(("quantity", "value"), list(data["frequency_wavevector_convention"].items())),
            "## 5. sigma_tilde_xy to sigma_tilde_LT\n\n" + data["basis_convention"]["sigma_rotation"],
            "## 6. Vacuum admittance Y0\n\n" + data["reflection_input_formula"]["Y0_LT"],
            "## 7. Tangential electric reflection-input matrix\n\n"
            + data["reflection_input_formula"]["R_E_LT"]
            + "。这是 tangential electric field reflection-input matrix，尚未转换成文献 TE/TM amplitude convention。",
            "## 8. Synthetic checks\n\n" + _table(("check", "status"), list(data["synthetic_checks"].items())),
            "## 9. Representative formatted rows\n\n" + _table(("q", "n", "Q", "kappa", "R_E_LT"), reps),
            "## 10. Diagnostic decision\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 11. Recommended next step\n\n"
            + data["diagnostic_status"]["recommended_next_action"]
            + " 尚未计算 Lifshitz trace-log，也尚未计算 Casimir energy/force/torque。",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--allow-q-zero", action="store_true")
    parser.add_argument("--max-representative-rows", type=int, default=6)
    parser.add_argument("--require-stage5-4b-passed", action="store_true", default=True)
    parser.add_argument("--allow-non-passed-input", dest="require_stage5_4b_passed", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_formatter(args.input_json, allow_q_zero=args.allow_q_zero, require_stage5_4b_passed=args.require_stage5_4b_passed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data, max_representative_rows=args.max_representative_rows), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
