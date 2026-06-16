#!/usr/bin/env python3
"""Stage 5.6 adapter from LT tangential-electric input to TE/TM amplitudes."""

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
from lno327.reflection_input import (  # noqa: E402
    sigma_tilde_xy_to_te_tm_reflection_matrix,
    symmetric_antisymmetric_offdiag,
    tangential_electric_LT_to_TE_TM,
    tangential_electric_reflection_matrix_LT,
    te_tm_adapter_metadata,
    vacuum_admittance_LT,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "reflection_input"
DEFAULT_INPUT = OUTPUT_DIR / "stage5_5b_reflection_input_tensor.json"
JSON_OUTPUT = OUTPUT_DIR / "stage5_6_te_tm_reflection_adapter.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_6_te_tm_reflection_adapter.md"

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


def parse_complex_matrix(value: Any) -> np.ndarray:
    matrix = np.array([[parse_complex_component(item) for item in row] for row in value], dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("matrix must have shape (2, 2)")
    return matrix


def q_model_from_row(row: dict[str, Any]) -> tuple[float, float, str]:
    if "q_model_x" in row and "q_model_y" in row:
        return float(row["q_model_x"]), float(row["q_model_y"]), row.get("q_model_source", "explicit_q_model_x_y")
    raise ValueError("Stage 5.5b row must contain q_model_x and q_model_y")


def classify_offdiag_marker(matrix: np.ndarray) -> str:
    diagnostics = symmetric_antisymmetric_offdiag(matrix)
    sym = diagnostics["symmetric_offdiag_abs"]
    anti = diagnostics["antisymmetric_offdiag_abs"]
    if anti > 10.0 * max(sym, 1e-300):
        return "antisymmetric_marker"
    if sym > 10.0 * max(anti, 1e-300):
        return "symmetric_finite_q_mixing"
    if max(sym, anti) == 0.0:
        return "no_offdiag_mixing"
    return "mixed_symmetric_and_antisymmetric"


def format_row(row: dict[str, Any], *, allow_q_zero: bool) -> dict[str, Any]:
    qx_model, qy_model, q_source = q_model_from_row(row)
    sigma_xy = parse_complex_matrix(row["sigma_tilde_xy_matrix"])
    direct = sigma_tilde_xy_to_te_tm_reflection_matrix(
        sigma_xy,
        qx_model,
        qy_model,
        float(row["omega_eV"]),
        LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
        LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m,
        allow_q_zero=allow_q_zero,
    )
    stored_lt = parse_complex_matrix(row["reflection_tangential_E_LT"])
    adapted_from_stored_lt = tangential_electric_LT_to_TE_TM(stored_lt)
    adapter_delta = direct["reflection_TE_TM"] - adapted_from_stored_lt
    sigma_lt = direct["sigma_tilde_LT_matrix"]
    return {
        "q_case": row.get("q_case"),
        "matsubara_n": row.get("matsubara_n"),
        "q_scale": row.get("q_scale", 1.0),
        "omega_eV": direct["omega_eV"],
        "xi_si_s_inv": direct["xi_si_s_inv"],
        "q_model_x": qx_model,
        "q_model_y": qy_model,
        "q_model_source": q_source,
        "Q_x_m_inv": direct["Q_x_m_inv"],
        "Q_y_m_inv": direct["Q_y_m_inv"],
        "Q_m_inv": direct["Q_m_inv"],
        "kappa_m_inv": direct["kappa_m_inv"],
        "sigma_tilde_xy_matrix": direct["sigma_tilde_xy_matrix"],
        "xy_to_lt_rotation_matrix": direct["xy_to_lt_rotation_matrix"],
        "sigma_tilde_LT_matrix": sigma_lt,
        "vacuum_admittance_Y0_LT": direct["vacuum_admittance_Y0_LT"],
        "reflection_tangential_E_LT": direct["reflection_tangential_E_LT"],
        "reflection_TE_TM": direct["reflection_TE_TM"],
        "reflection_TE_TM_from_stored_LT": adapted_from_stored_lt,
        "adapter_formula_abs_delta_max": float(np.max(np.abs(adapter_delta))),
        "sigma_tilde_LT_offdiag_diagnostics": symmetric_antisymmetric_offdiag(sigma_lt),
        "offdiag_marker": classify_offdiag_marker(sigma_lt),
        "basis_convention": direct["basis_convention"],
    }


def run_synthetic_checks() -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    sample = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    checks["adapter_index_check"] = {
        "status": "PASS" if np.allclose(tangential_electric_LT_to_TE_TM(sample), np.array([[4.0, 3.0], [-2.0, -1.0]])) else "FAIL"
    }

    qx, qy = 0.02, 0.0
    omega = 0.02
    ax = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m
    ay = LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_y_m
    zero = sigma_tilde_xy_to_te_tm_reflection_matrix(np.zeros((2, 2), dtype=complex), qx, qy, omega, ax, ay)
    checks["zero_sheet_check"] = {
        "status": "PASS"
        if np.allclose(zero["reflection_tangential_E_LT"], 0.0) and np.allclose(zero["reflection_TE_TM"], 0.0)
        else "FAIL"
    }

    scalar_sigma = 0.3
    scalar = sigma_tilde_xy_to_te_tm_reflection_matrix(scalar_sigma * np.eye(2), qx, qy, omega, ax, ay)
    y0 = scalar["vacuum_admittance_Y0_LT"]
    rss_expected = -scalar_sigma / (2.0 * y0[1, 1] + scalar_sigma)
    rpp_expected = scalar_sigma / (2.0 * y0[0, 0] + scalar_sigma)
    checks["isotropic_scalar_sheet_no_mixing"] = {
        "status": "PASS" if abs(scalar["reflection_TE_TM"][0, 1]) < 1e-14 and abs(scalar["reflection_TE_TM"][1, 0]) < 1e-14 else "FAIL"
    }
    checks["te_tm_scalar_limit_consistency"] = {
        "status": "PASS"
        if np.allclose(scalar["reflection_TE_TM"][0, 0], rss_expected) and np.allclose(scalar["reflection_TE_TM"][1, 1], rpp_expected)
        else "FAIL",
        "r_ss_formula": "-sigma_tilde/(2*eta_T + sigma_tilde)",
        "r_pp_formula": "+sigma_tilde/(2*eta_L + sigma_tilde)",
    }

    strong_sigma = 1.0e9
    strong = sigma_tilde_xy_to_te_tm_reflection_matrix(strong_sigma * np.eye(2), qx, qy, omega, ax, ay)
    checks["strong_sheet_limit"] = {
        "status": "PASS"
        if abs(strong["reflection_TE_TM"][0, 0] + 1.0) < 1e-5 and abs(strong["reflection_TE_TM"][1, 1] - 1.0) < 1e-5
        else "FAIL"
    }

    weak_sigma = 1.0e-9
    weak = sigma_tilde_xy_to_te_tm_reflection_matrix(weak_sigma * np.eye(2), qx, qy, omega, ax, ay)
    weak_y0 = weak["vacuum_admittance_Y0_LT"]
    weak_expected = np.array(
        [[-weak_sigma / (2.0 * weak_y0[1, 1]), 0.0], [0.0, weak_sigma / (2.0 * weak_y0[0, 0])]],
        dtype=complex,
    )
    checks["weak_sheet_limit"] = {
        "status": "PASS"
        if np.linalg.norm(weak["reflection_TE_TM"] - weak_expected) / max(np.linalg.norm(weak_expected), 1e-300) < 1e-6
        else "FAIL"
    }

    xi = scalar["xi_si_s_inv"]
    kappa = scalar["kappa_m_inv"]
    symmetric_sigma_lt = np.array([[0.1, 0.03], [0.03, 0.2]], dtype=complex)
    symmetric_r = tangential_electric_reflection_matrix_LT(symmetric_sigma_lt, xi, kappa)
    symmetric_te_tm = tangential_electric_LT_to_TE_TM(symmetric_r)
    checks["symmetric_offdiag_mixing"] = {
        "status": "PASS"
        if (abs(symmetric_te_tm[0, 1]) > 0.0 or abs(symmetric_te_tm[1, 0]) > 0.0)
        and classify_offdiag_marker(symmetric_sigma_lt) == "symmetric_finite_q_mixing"
        else "FAIL",
        "classification": classify_offdiag_marker(symmetric_sigma_lt),
    }

    hall_like_sigma_lt = np.array([[0.1, 0.02], [-0.02, 0.1]], dtype=complex)
    hall_like_r = tangential_electric_reflection_matrix_LT(hall_like_sigma_lt, xi, kappa)
    hall_like_te_tm = tangential_electric_LT_to_TE_TM(hall_like_r)
    checks["hall_like_antisymmetric_marker"] = {
        "status": "PASS"
        if classify_offdiag_marker(hall_like_sigma_lt) == "antisymmetric_marker" and np.isfinite(hall_like_te_tm).all()
        else "FAIL",
        "classification": classify_offdiag_marker(hall_like_sigma_lt),
        "not_applied_to_real_LNO327_as_physical_claim": True,
    }
    return checks


def q_sign_offdiag_consistency(rows: list[dict[str, Any]]) -> str:
    by_key = {
        (row.get("q_case"), row.get("matsubara_n"), row.get("q_scale")): row
        for row in rows
    }
    checked = 0
    for key, pos in by_key.items():
        q_case, n, q_scale = key
        if q_case != "q_diag_pos":
            continue
        neg = by_key.get(("q_diag_neg", n, q_scale))
        if neg is None:
            continue
        checked += 1
        pos_expected = tangential_electric_LT_to_TE_TM(pos["reflection_tangential_E_LT"])
        neg_expected = tangential_electric_LT_to_TE_TM(neg["reflection_tangential_E_LT"])
        if not np.allclose(pos["reflection_TE_TM"], pos_expected) or not np.allclose(neg["reflection_TE_TM"], neg_expected):
            return "FAIL_adapter_formula_for_q_sign_pair"
        if not (
            np.allclose(pos["reflection_TE_TM"][0, 1], -neg["reflection_TE_TM"][0, 1])
            and np.allclose(pos["reflection_TE_TM"][1, 0], -neg["reflection_TE_TM"][1, 0])
        ):
            return "PASS_adapter_formula; offdiag parity not reduced to a simple universal label"
    if checked == 0:
        return "NOT_APPLICABLE_no_q_diag_pos_neg_pair"
    return "PASS_adapter_formula_and_representative_q_diag_offdiag_flip"


def run_adapter(input_json: Path, *, allow_q_zero: bool, require_stage5_5b_passed: bool) -> dict[str, Any]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    status = data.get("diagnostic_status", {}).get("stage5_5b_status")
    if require_stage5_5b_passed and status != "STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED":
        raise ValueError("input must have STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED status")
    rows = [format_row(row, allow_q_zero=allow_q_zero) for row in data.get("reflection_input_results", [])]
    checks = run_synthetic_checks()
    all_checks_pass = all(check["status"] == "PASS" for check in checks.values())
    all_adapter_deltas_small = all(row["adapter_formula_abs_delta_max"] < 1e-12 for row in rows)
    max_abs_r = max((float(np.max(np.abs(row["reflection_TE_TM"]))) for row in rows), default=0.0)
    max_abs_offdiag = max((float(np.linalg.norm([row["reflection_TE_TM"][0, 1], row["reflection_TE_TM"][1, 0]])) for row in rows), default=0.0)
    metadata = te_tm_adapter_metadata()
    return {
        "stage": "Stage 5.6",
        "purpose": "Convert LT tangential-electric reflection matrix to TE/TM amplitude-basis reflection matrix",
        "boundary": dict(BOUNDARY),
        "input": {
            "input_json": str(input_json),
            "input_stage": data.get("stage"),
            "input_status": status,
            "num_input_cases": len(data.get("reflection_input_results", [])),
        },
        "adapter_convention": {
            "internal_basis": metadata["internal_basis"],
            "internal_ordering": metadata["internal_ordering"],
            "output_basis": metadata["output_basis"],
            "output_ordering": metadata["output_ordering"],
            "E_s": metadata["E_s"],
            "E_p_inc": metadata["E_p_inc"],
            "E_p_ref": metadata["E_p_ref"],
            "formula": metadata["formula"],
        },
        "converted_results": rows,
        "synthetic_checks": checks,
        "summary": {
            "num_cases": len(rows),
            "max_abs_R_TE_TM": max_abs_r,
            "max_abs_R_TE_TM_offdiag": max_abs_offdiag,
            "q_sign_offdiag_consistency": q_sign_offdiag_consistency(rows),
            "all_cases_converted": len(rows) == len(data.get("reflection_input_results", [])),
            "all_adapter_formula_deltas_small": bool(all_adapter_deltas_small),
        },
        "diagnostic_status": {
            "stage5_6_status": "STAGE5_6_TE_TM_ADAPTER_PASSED"
            if all_checks_pass and all_adapter_deltas_small
            else "STAGE5_6_TE_TM_ADAPTER_FAILED",
            "recommended_next_action": "Proceed to trace-log integrand convention audit; still do not compute full Casimir energy/torque.",
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
            row["reflection_TE_TM"],
            row["adapter_formula_abs_delta_max"],
        )
        for row in data["converted_results"][:max_representative_rows]
    ]
    checks = [(name, check["status"], check.get("classification", "")) for name, check in data["synthetic_checks"].items()]
    return "\n\n".join(
        [
            "# Stage 5.6 TE/TM reflection adapter",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Input source\n\n" + _table(("quantity", "value"), list(data["input"].items())),
            "## 3. Why TE/TM adapter is needed\n\n"
            "Stage 5.5b 的 `R_E^{LT}` 是内部 tangential electric basis 下的反射输入矩阵。"
            "标准 Lifshitz/Casimir trace-log 通常使用 TE/TM amplitude basis；本阶段只做基底适配，当前没有计算 trace-log。",
            "## 4. Internal (L/T) basis\n\n"
            "`L` 平行于面内 SI 波矢 `Q`，`T = z_hat cross L`。内部顺序是 `['L', 'T']`，并保留 `sigma_tilde_LT` 和 `R_E_LT` 作为审计量。",
            "## 5. TE/TM amplitude convention\n\n"
            "输出顺序是 `['s', 'p']`，其中 `s/TE` 对应 `T` 方向电场，`p/TM` 对应 `L-z` 平面内电场。"
            "本文采用 `E_s = E_T`、`E_p_inc = E_L_inc`、`E_p_ref = -E_L_ref`。"
            "`p` 反射振幅的负号是本 adapter convention 的一部分。",
            "## 6. Adapter formula\n\n"
            "`R_TE_TM = [[R_TT, R_TL], [-R_LT, -R_LL]]`。这里 `R_E_LT` 的行列顺序为 `(L,T)`，`R_TE_TM` 的行列顺序为 `(s,p)`。",
            "## 7. Synthetic checks\n\n" + _table(("check", "status", "classification"), checks),
            "## 8. Representative R_TE_TM rows\n\n" + _table(("q", "n", "R_TE_TM", "adapter_delta_max"), reps),
            "## 9. Diagnostic decision\n\n"
            + _table(("quantity", "value"), list(data["diagnostic_status"].items()))
            + "\n\n"
            + _table(("summary", "value"), list(data["summary"].items())),
            "## 10. Recommended next step\n\n"
            + data["diagnostic_status"]["recommended_next_action"]
            + " 本阶段没有修改 response、conductivity convention、bubble sign、direct contact、source/observable、Ward convention；"
            + "没有 fitted contact，没有 `E^{ET}`，没有 heavy response，没有 Lifshitz trace-log，没有 Casimir energy/force/torque。"
            + "文档和 metadata 始终使用 `sigma_tilde`，不把它另记为其他裸符号。",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--allow-q-zero", action="store_true")
    parser.add_argument("--max-representative-rows", type=int, default=6)
    parser.add_argument("--require-stage5-5b-passed", action="store_true", default=True)
    parser.add_argument("--allow-non-passed-input", dest="require_stage5_5b_passed", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_adapter(args.input_json, allow_q_zero=args.allow_q_zero, require_stage5_5b_passed=args.require_stage5_5b_passed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data, max_representative_rows=args.max_representative_rows), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
