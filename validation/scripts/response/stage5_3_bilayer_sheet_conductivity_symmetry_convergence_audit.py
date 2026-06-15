#!/usr/bin/env python3
"""Stage 5.3 bilayer sheet conductivity off-diagonal symmetry audit."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage5_2_bilayer_sheet_conductivity_sanity_scan import (  # noqa: E402
    BOUNDARY,
    DIAG_NEGATIVE_TOLERANCE,
    Q_CASES,
    WARD_CLOSED_THRESHOLD,
    _case_failure_row,
    run_case,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "conductivity"
JSON_OUTPUT = OUTPUT_DIR / "stage5_3_bilayer_sheet_conductivity_symmetry_convergence_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_3_bilayer_sheet_conductivity_symmetry_convergence_audit.md"

EPS = 1e-300
GEOMETRIC_REDUCTION_FACTOR = 0.35
Q_SIGN_TOLERANCE = 1e-5
CONVERGENCE_MONITOR_THRESHOLD = 0.2


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


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_q_cases(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(names) - set(Q_CASES))
    if unknown:
        raise ValueError(f"unknown q case(s): {unknown}")
    return names


def xy_to_lt_rotation(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q_norm = float(np.linalg.norm(q))
    if q.shape != (2,) or q_norm <= 0.0:
        raise ValueError("q must be a nonzero 2-vector")
    qhat = q / q_norm
    return np.array([[qhat[0], qhat[1]], [-qhat[1], qhat[0]]], dtype=float)


def xy_to_lt(sigma_xy: np.ndarray, q: np.ndarray) -> np.ndarray:
    rotation = xy_to_lt_rotation(q)
    return rotation @ np.asarray(sigma_xy, dtype=complex) @ rotation.T


def offdiag_decomposition(sigma_xy: np.ndarray) -> dict[str, float]:
    sigma = np.asarray(sigma_xy, dtype=complex)
    symmetric = 0.5 * (sigma[0, 1] + sigma[1, 0])
    antisymmetric = 0.5 * (sigma[0, 1] - sigma[1, 0])
    symmetric_abs = float(abs(symmetric))
    antisymmetric_abs = float(abs(antisymmetric))
    return {
        "symmetric_offdiag_abs": symmetric_abs,
        "antisymmetric_offdiag_abs": antisymmetric_abs,
        "relative_antisymmetric_to_symmetric": float(antisymmetric_abs / max(symmetric_abs, EPS)),
    }


def augment_row_with_symmetry(row: dict[str, Any]) -> dict[str, Any]:
    sigma_xy = np.array(
        [
            [complex(row["sigma_xx_model"]), complex(row["sigma_xy_model"])],
            [complex(row["sigma_yx_model"]), complex(row["sigma_yy_model"])],
        ],
        dtype=complex,
    )
    q = np.asarray(row["q_model"], dtype=float)
    try:
        sigma_lt = xy_to_lt(sigma_xy, q)
        sigma_ll, sigma_lt_off = complex(sigma_lt[0, 0]), complex(sigma_lt[0, 1])
        sigma_tl_off, sigma_tt = complex(sigma_lt[1, 0]), complex(sigma_lt[1, 1])
        relative_lt = float(
            np.sqrt(abs(sigma_lt_off) ** 2 + abs(sigma_tl_off) ** 2)
            / max(np.sqrt(abs(sigma_ll) ** 2 + abs(sigma_tt) ** 2), EPS)
        )
    except ValueError:
        sigma_ll = sigma_lt_off = sigma_tl_off = sigma_tt = complex(float("nan"), float("nan"))
        relative_lt = float("nan")
    return {
        **row,
        "sigma_LL_model": sigma_ll,
        "sigma_LT_model": sigma_lt_off,
        "sigma_TL_model": sigma_tl_off,
        "sigma_TT_model": sigma_tt,
        "relative_LT_offdiag_norm": relative_lt,
        **offdiag_decomposition(sigma_xy),
        "q_norm": float(np.linalg.norm(q)),
    }


def build_scan_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.quick:
        return {
            "temperature_K": 30.0,
            "matsubara_indices": [1],
            "q_cases": ["qx", "q_diag_pos", "q_diag_neg"],
            "q_scales": [1.0],
            "adaptive_levels": [1],
            "gauss_orders": [2],
            "fermi_windows_eV": [0.05],
            "coarse_grid": 8,
            "eta_eV": 1e-10,
            "output_si": False,
            "quick": True,
            "workers": int(args.workers),
            "dry_run": bool(args.dry_run),
        }
    return {
        "temperature_K": 30.0,
        "matsubara_indices": _parse_int_list(args.matsubara_indices),
        "q_cases": _parse_q_cases(args.q_cases),
        "q_scales": _parse_float_list(args.q_scales),
        "adaptive_levels": _parse_int_list(args.levels),
        "gauss_orders": _parse_int_list(args.gauss_orders),
        "fermi_windows_eV": _parse_float_list(args.fermi_windows),
        "coarse_grid": int(args.coarse_grid),
        "eta_eV": 1e-10,
        "output_si": False,
        "quick": False,
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
    }


def planned_cases(config: dict[str, Any], max_cases: int | None = None) -> list[dict[str, Any]]:
    if max_cases is not None and max_cases <= 0:
        return []
    cases: list[dict[str, Any]] = []
    for matsubara_index in config["matsubara_indices"]:
        for q_case in config["q_cases"]:
            for q_scale in config["q_scales"]:
                for adaptive_level in config["adaptive_levels"]:
                    for gauss_order in config["gauss_orders"]:
                        for fermi_window_eV in config["fermi_windows_eV"]:
                            cases.append(
                                {
                                    "case_index": len(cases),
                                    "temperature_K": float(config["temperature_K"]),
                                    "matsubara_index": int(matsubara_index),
                                    "q_case": q_case,
                                    "base_q_case": q_case,
                                    "q_scale": float(q_scale),
                                    "q_model": Q_CASES[q_case].copy() * float(q_scale),
                                    "adaptive_level": int(adaptive_level),
                                    "gauss_order": int(gauss_order),
                                    "fermi_window_eV": float(fermi_window_eV),
                                    "coarse_grid": int(config["coarse_grid"]),
                                }
                            )
                            if max_cases is not None and len(cases) >= max_cases:
                                return cases
    return cases


def _run_case_job(index: int, case: dict[str, Any], eta_eV: float) -> tuple[int, dict[str, Any]]:
    return index, augment_row_with_symmetry(run_case(case, eta_eV=eta_eV))


def run_cases_parallel(
    cases: list[dict[str, Any]],
    *,
    eta_eV: float,
    workers: int,
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not cases:
        return []
    total = len(cases)
    actual_workers = min(workers, total)
    print(f"Running {total} Stage 5.3 cases with {actual_workers} worker(s)...")
    if actual_workers <= 1:
        rows = []
        for done, case in enumerate(cases, start=1):
            row = augment_row_with_symmetry(run_case(case, eta_eV=eta_eV))
            rows.append(row)
            print(f"Completed {done}/{total}: {case['q_case']} scale={case['q_scale']} n={case['matsubara_index']} status={row['status']}")
        return rows

    indexed_rows: dict[int, dict[str, Any]] = {}
    with executor_factory(max_workers=actual_workers) as executor:
        future_to_case = {executor.submit(_run_case_job, index, case, eta_eV): (index, case) for index, case in enumerate(cases)}
        for done, future in enumerate(as_completed(future_to_case), start=1):
            index, case = future_to_case[future]
            try:
                _result_index, row = future.result()
            except Exception as exc:
                row = augment_row_with_symmetry(
                    _case_failure_row(case, ["EXECUTOR_CASE_EXCEPTION", type(exc).__name__, str(exc)])
                )
            indexed_rows[index] = row
            print(f"Completed {done}/{total}: {case['q_case']} scale={case['q_scale']} n={case['matsubara_index']} status={row['status']}")
    return [indexed_rows[index] for index in range(total)]


def _rel_error(a: complex | float, b: complex | float) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), EPS))


def q_sign_symmetry_pair(row_pos: dict[str, Any], row_neg: dict[str, Any]) -> dict[str, Any]:
    diag_even = max(
        _rel_error(complex(row_pos["sigma_xx_model"]), complex(row_neg["sigma_xx_model"])),
        _rel_error(complex(row_pos["sigma_yy_model"]), complex(row_neg["sigma_yy_model"])),
    )
    offdiag_odd = max(
        _rel_error(complex(row_pos["sigma_xy_model"]), -complex(row_neg["sigma_xy_model"])),
        _rel_error(complex(row_pos["sigma_yx_model"]), -complex(row_neg["sigma_yx_model"])),
    )
    status = "PASS" if diag_even < Q_SIGN_TOLERANCE and offdiag_odd < Q_SIGN_TOLERANCE else "MONITOR"
    return {
        "matsubara_index": row_pos["matsubara_index"],
        "q_scale": row_pos["q_scale"],
        "adaptive_level": row_pos["adaptive_level"],
        "gauss_order": row_pos["gauss_order"],
        "fermi_window_eV": row_pos["fermi_window_eV"],
        "q_sign_diag_even_error": diag_even,
        "q_sign_offdiag_odd_error": offdiag_odd,
        "q_sign_symmetry_status": status,
    }


def q_sign_symmetry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    pos = {
        (row["matsubara_index"], row["q_scale"], row["adaptive_level"], row["gauss_order"], row["fermi_window_eV"]): row
        for row in rows
        if row["base_q_case"] == "q_diag_pos"
    }
    for row in rows:
        if row["base_q_case"] != "q_diag_neg":
            continue
        key = (row["matsubara_index"], row["q_scale"], row["adaptive_level"], row["gauss_order"], row["fermi_window_eV"])
        if key in pos:
            pairs.append(q_sign_symmetry_pair(pos[key], row))
    return {
        "num_pairs": len(pairs),
        "pairs": pairs,
        "max_diag_even_error": max((item["q_sign_diag_even_error"] for item in pairs), default=None),
        "max_offdiag_odd_error": max((item["q_sign_offdiag_odd_error"] for item in pairs), default=None),
        "status": "PASS" if pairs and all(item["q_sign_symmetry_status"] == "PASS" for item in pairs) else "MONITOR",
    }


def lt_projection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["relative_LT_offdiag_norm"]) / max(float(row["relative_offdiag_norm"]), EPS)
        for row in rows
        if np.isfinite(float(row["relative_LT_offdiag_norm"])) and np.isfinite(float(row["relative_offdiag_norm"]))
    ]
    return {
        "max_relative_xy_offdiag_norm": max((float(row["relative_offdiag_norm"]) for row in rows), default=None),
        "max_relative_LT_offdiag_norm": max((float(row["relative_LT_offdiag_norm"]) for row in rows if np.isfinite(float(row["relative_LT_offdiag_norm"]))), default=None),
        "median_LT_to_xy_offdiag_ratio": float(np.median(ratios)) if ratios else None,
        "lt_projection_reduces_offdiag": bool(ratios and np.median(ratios) < GEOMETRIC_REDUCTION_FACTOR),
    }


def axial_vs_diagonal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    axial = [float(row["relative_offdiag_norm"]) for row in rows if row["base_q_case"] in {"qx", "qy"}]
    diagonal = [float(row["relative_offdiag_norm"]) for row in rows if row["base_q_case"] in {"q_diag_pos", "q_diag_neg"}]
    return {
        "max_axial_relative_offdiag_norm": max(axial, default=None),
        "max_diagonal_relative_offdiag_norm": max(diagonal, default=None),
        "axial_smaller_than_diagonal": bool(axial and diagonal and max(axial) < max(diagonal)),
    }


def q_scaling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trends = []
    keys = sorted({
        (row["base_q_case"], row["matsubara_index"], row["adaptive_level"], row["gauss_order"], row["fermi_window_eV"])
        for row in rows
        if row["base_q_case"] in {"q_diag_pos", "q_diag_neg"}
    })
    for key in keys:
        group = [
            row for row in rows
            if (row["base_q_case"], row["matsubara_index"], row["adaptive_level"], row["gauss_order"], row["fermi_window_eV"]) == key
        ]
        group.sort(key=lambda item: float(item["q_scale"]), reverse=True)
        if len(group) < 2:
            continue
        trends.append(
            {
                "base_q_case": key[0],
                "matsubara_index": key[1],
                "q_scales_desc": [row["q_scale"] for row in group],
                "relative_offdiag_norms": [row["relative_offdiag_norm"] for row in group],
                "relative_LT_offdiag_norms": [row["relative_LT_offdiag_norm"] for row in group],
            }
        )
    return {"num_trends": len(trends), "trends": trends}


def convergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [
        row for row in rows
        if row["adaptive_level"] == 4 and row["gauss_order"] == 5 and abs(float(row["fermi_window_eV"]) - 0.05) < 1e-12
    ]
    diffs = []
    by_key = {
        (row["base_q_case"], row["q_scale"], row["matsubara_index"]): row
        for row in baseline
    }
    for row in rows:
        key = (row["base_q_case"], row["q_scale"], row["matsubara_index"])
        ref = by_key.get(key)
        if ref is None or row is ref:
            continue
        diffs.append(
            {
                "base_q_case": row["base_q_case"],
                "q_scale": row["q_scale"],
                "matsubara_index": row["matsubara_index"],
                "adaptive_level": row["adaptive_level"],
                "gauss_order": row["gauss_order"],
                "fermi_window_eV": row["fermi_window_eV"],
                "sigma_xx_relative_difference": _rel_error(complex(row["sigma_xx_model"]), complex(ref["sigma_xx_model"])),
                "sigma_yy_relative_difference": _rel_error(complex(row["sigma_yy_model"]), complex(ref["sigma_yy_model"])),
                "xy_offdiag_relative_difference": _rel_error(float(row["relative_offdiag_norm"]), float(ref["relative_offdiag_norm"])),
                "lt_offdiag_relative_difference": _rel_error(float(row["relative_LT_offdiag_norm"]), float(ref["relative_LT_offdiag_norm"])),
            }
        )
    max_diff = max(
        (
            max(item["sigma_xx_relative_difference"], item["sigma_yy_relative_difference"], item["xy_offdiag_relative_difference"], item["lt_offdiag_relative_difference"])
            for item in diffs
        ),
        default=None,
    )
    return {
        "num_comparisons": len(diffs),
        "comparisons": diffs,
        "max_relative_difference": max_diff,
        "convergence_status": "PASS" if max_diff is None or max_diff < CONVERGENCE_MONITOR_THRESHOLD else "MONITOR",
    }


def diagnostic_status(rows: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> dict[str, str]:
    if any(float(row["ward_max_norm"]) >= WARD_CLOSED_THRESHOLD for row in rows):
        status = "CONDUCTIVITY_SYMMETRY_AUDIT_FAILED_WARD"
        action = "Do not proceed; diagnose failed Ward channel."
    elif any(float(row["sigma_diag_min_real"]) < DIAG_NEGATIVE_TOLERANCE for row in rows):
        status = "CONDUCTIVITY_SYMMETRY_AUDIT_FAILED_NEGATIVE_DIAGONAL"
        action = "Do not proceed; diagnose negative diagonal conductivity."
    else:
        symmetric_dominates = all(float(row["relative_antisymmetric_to_symmetric"]) < 0.1 for row in rows)
        geometric = bool(
            summaries["lt_projection_summary"].get("lt_projection_reduces_offdiag")
            and summaries["q_sign_symmetry_summary"].get("status") == "PASS"
            and symmetric_dominates
        )
        convergence_ok = summaries["convergence_summary"].get("convergence_status") == "PASS"
        axial_ok = summaries["axial_vs_diagonal_summary"].get("axial_smaller_than_diagonal", False)
        if geometric and convergence_ok and axial_ok:
            status = "CONDUCTIVITY_SYMMETRY_AUDIT_PASSED_GEOMETRIC_MIXING"
            action = "Proceed to Stage 5.4 SI sheet scaling / reflection-input preparation; still do not enter reflection/Casimir."
        elif geometric:
            status = "CONDUCTIVITY_SYMMETRY_AUDIT_MONITOR_CONVERGENCE"
            action = "Increase level/window or q-scaling scan before Stage 5.4."
        else:
            status = "CONDUCTIVITY_SYMMETRY_AUDIT_REQUIRES_FURTHER_SOURCE_SYMMETRY_AUDIT"
            action = "Do not proceed; audit source symmetry and finite-q tensor structure."
    return {"conductivity_symmetry_audit_status": status, "recommended_next_action": action}


def run_audit(config: dict[str, Any], *, max_cases: int | None = None) -> dict[str, Any]:
    cases = planned_cases(config, max_cases)
    rows = [] if config["dry_run"] else run_cases_parallel(cases, eta_eV=float(config["eta_eV"]), workers=int(config["workers"]))
    summaries = {
        "lt_projection_summary": lt_projection_summary(rows),
        "q_sign_symmetry_summary": q_sign_symmetry_summary(rows),
        "axial_vs_diagonal_summary": axial_vs_diagonal_summary(rows),
        "q_scaling_summary": q_scaling_summary(rows),
        "convergence_summary": convergence_summary(rows),
    }
    return {
        "stage": "Stage 5.3",
        "purpose": "Bilayer sheet conductivity off-diagonal symmetry and convergence audit",
        "boundary": dict(BOUNDARY),
        "conductivity_convention": {
            "formula": "sigma_model_ij(iOmega) = - response[1:3,1:3] / omega_eV",
            "normalization": "bilayer-normalized 2D sheet conductivity",
            "si_scaling_applied": False,
            "bulk_3d_conductivity": False,
            "single_layer_conductivity": False,
        },
        "config": {**config, "planned_num_cases": len(cases), "planned_cases": cases if config["dry_run"] else []},
        "scan_results": rows,
        **summaries,
        "diagnostic_status": diagnostic_status(rows, summaries),
    }


def _fmt_float(value: Any) -> str:
    return "None" if value is None else f"{float(value):.6e}"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    rows = data["scan_results"]
    boundary = "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items())
    convention = "\n".join(f"- {key}: {value}" for key, value in data["conductivity_convention"].items())
    xy_rows = [
        (row["base_q_case"], row["q_scale"], row["matsubara_index"], _fmt_float(row["relative_offdiag_norm"]), _fmt_float(row["relative_LT_offdiag_norm"]), _fmt_float(row["relative_antisymmetric_to_symmetric"]))
        for row in rows
    ]
    ward_rows = [(row["base_q_case"], row["q_scale"], row["matsubara_index"], _fmt_float(row["ward_max_norm"]), row["status"]) for row in rows]
    return "\n\n".join(
        [
            "# Stage 5.3 双层 sheet 电导 offdiag 对称性 / 收敛性审计",
            "## 1. 边界\n\n" + boundary,
            "## 2. 电导约定\n\n" + convention,
            "## 3. 为什么需要审计 Stage 5.2 offdiag\n\n"
            "offdiag 不是自动错误。斜向 finite-q 向量会把 longitudinal/transverse 响应投影到 x/y 坐标中，因此需要单独区分几何混合、Hall-like 反对称响应和数值误差。",
            "## 4. 扫描配置\n\n" + _table(("quantity", "value"), [(key, value) for key, value in data["config"].items() if key != "planned_cases"]),
            "## 5. (x,y) offdiag 汇总\n\n" + (_table(("q", "scale", "n", "rel xy offdiag", "rel LT offdiag", "A/S"), xy_rows) if xy_rows else "Dry run：未执行 response 积分。"),
            "## 6. (L/T) 投影汇总\n\n" + _table(("quantity", "value"), list(data["lt_projection_summary"].items())),
            "## 7. symmetric vs antisymmetric offdiag\n\n"
            "$\\sigma_{xy}\\approx\\sigma_{yx}$ 表示 symmetric mixing，区别于 Hall-like antisymmetric response。",
            "## 8. q-sign 对称性\n\n"
            "$q_y\\to -q_y$ 时 offdiag 变号支持 finite-q geometry interpretation。\n\n"
            + _table(("quantity", "value"), list(data["q_sign_symmetry_summary"].items())),
            "## 9. 轴向 q 与斜向 q 比较\n\n" + _table(("quantity", "value"), list(data["axial_vs_diagonal_summary"].items())),
            "## 10. q-scaling 趋势\n\n" + _table(("quantity", "value"), [("num_trends", data["q_scaling_summary"]["num_trends"])]),
            "## 11. 收敛趋势\n\n" + _table(("quantity", "value"), [(key, value) for key, value in data["convergence_summary"].items() if key != "comparisons"]),
            "## 12. Ward residual 诊断\n\n" + (_table(("q", "scale", "n", "ward max", "status"), ward_rows) if ward_rows else "Dry run：未计算 Ward residual。"),
            "## 13. 诊断结论\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 14. 推荐下一步\n\n" + data["diagnostic_status"]["recommended_next_action"] + " 本审计仍未进入 reflection/Casimir。",
        ]
    ) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matsubara-indices", default="1,2,4,8")
    parser.add_argument("--q-cases", default="qx,qy,q_diag_pos,q_diag_neg")
    parser.add_argument("--q-scales", default="1.0,0.5")
    parser.add_argument("--levels", default="4,5")
    parser.add_argument("--gauss-orders", default="5")
    parser.add_argument("--fermi-windows", default="0.05,0.08")
    parser.add_argument("--coarse-grid", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    data = run_audit(build_scan_config(args), max_cases=args.max_cases)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
