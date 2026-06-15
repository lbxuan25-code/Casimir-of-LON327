#!/usr/bin/env python3
"""Stage 5.3b targeted finite-q offdiag convergence audit."""

from __future__ import annotations

import argparse
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
from stage5_3_bilayer_sheet_conductivity_symmetry_convergence_audit import (  # noqa: E402
    augment_row_with_symmetry,
    offdiag_decomposition,
    q_sign_symmetry_pair,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "conductivity"
JSON_OUTPUT = OUTPUT_DIR / "stage5_3b_bilayer_sheet_conductivity_offdiag_convergence_audit.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_3b_bilayer_sheet_conductivity_offdiag_convergence_audit.md"

EPS = 1e-300
REL_CHANGE_PASS = 0.05
REL_CHANGE_MONITOR = 0.15
ABS_REL_OFFDIAG_CHANGE_PASS = 0.01
ABS_REL_OFFDIAG_CHANGE_MONITOR = 0.03
SYMMETRIC_RATIO_THRESHOLD = 1e-3

TARGETED_INTEGRATION_CONFIGS = [
    {"adaptive_level": 4, "gauss_order": 5, "fermi_window_eV": 0.05},
    {"adaptive_level": 5, "gauss_order": 5, "fermi_window_eV": 0.05},
    {"adaptive_level": 4, "gauss_order": 5, "fermi_window_eV": 0.08},
]
BASELINE_CONFIG = {"adaptive_level": 4, "gauss_order": 5, "fermi_window_eV": 0.05}


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


def _rel_change(value: complex | float, ref: complex | float) -> float:
    return float(abs(value - ref) / max(abs(value), abs(ref), EPS))


def _config_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (int(row["adaptive_level"]), int(row["gauss_order"]), float(row["fermi_window_eV"]))


def build_scan_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.quick:
        integration_configs = [{"adaptive_level": 1, "gauss_order": 2, "fermi_window_eV": 0.05}]
        return {
            "temperature_K": 30.0,
            "matsubara_indices": [1],
            "q_cases": ["q_diag_pos", "q_diag_neg"],
            "q_scales": [1.0, 0.5],
            "integration_configs": integration_configs,
            "coarse_grid": 8,
            "eta_eV": 1e-10,
            "output_si": False,
            "quick": True,
            "workers": int(args.workers),
            "dry_run": bool(args.dry_run),
            "targeted_configs": True,
        }
    if args.targeted_configs:
        integration_configs = list(TARGETED_INTEGRATION_CONFIGS)
    else:
        integration_configs = [
            {"adaptive_level": level, "gauss_order": order, "fermi_window_eV": window}
            for level in _parse_int_list(args.levels)
            for order in _parse_int_list(args.gauss_orders)
            for window in _parse_float_list(args.fermi_windows)
        ]
    return {
        "temperature_K": 30.0,
        "matsubara_indices": _parse_int_list(args.matsubara_indices),
        "q_cases": _parse_q_cases(args.q_cases),
        "q_scales": _parse_float_list(args.q_scales),
        "integration_configs": integration_configs,
        "coarse_grid": int(args.coarse_grid),
        "eta_eV": 1e-10,
        "output_si": False,
        "quick": False,
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "targeted_configs": bool(args.targeted_configs),
    }


def planned_cases(config: dict[str, Any], max_cases: int | None = None) -> list[dict[str, Any]]:
    if max_cases is not None and max_cases <= 0:
        return []
    cases: list[dict[str, Any]] = []
    for matsubara_index in config["matsubara_indices"]:
        for q_case in config["q_cases"]:
            for q_scale in config["q_scales"]:
                for integration in config["integration_configs"]:
                    cases.append(
                        {
                            "case_index": len(cases),
                            "temperature_K": float(config["temperature_K"]),
                            "matsubara_index": int(matsubara_index),
                            "q_case": q_case,
                            "base_q_case": q_case,
                            "q_scale": float(q_scale),
                            "q_model": Q_CASES[q_case].copy() * float(q_scale),
                            "adaptive_level": int(integration["adaptive_level"]),
                            "gauss_order": int(integration["gauss_order"]),
                            "fermi_window_eV": float(integration["fermi_window_eV"]),
                            "coarse_grid": int(config["coarse_grid"]),
                        }
                    )
                    if max_cases is not None and len(cases) >= max_cases:
                        return cases
    return cases


def _run_case_job(index: int, case: dict[str, Any], eta_eV: float) -> tuple[int, dict[str, Any]]:
    return index, augment_row_with_symmetry(run_case(case, eta_eV=eta_eV))


def run_cases_parallel(cases: list[dict[str, Any]], *, eta_eV: float, workers: int) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not cases:
        return []
    total = len(cases)
    actual_workers = min(workers, total)
    print(f"Running {total} Stage 5.3b cases with {actual_workers} worker(s)...")
    if actual_workers <= 1:
        rows = []
        for done, case in enumerate(cases, start=1):
            row = augment_row_with_symmetry(run_case(case, eta_eV=eta_eV))
            rows.append(row)
            print(f"Completed {done}/{total}: {case['q_case']} scale={case['q_scale']} n={case['matsubara_index']} status={row['status']}")
        return rows

    indexed_rows: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
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


def compare_to_baseline(baseline: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    rel_xx = _rel_change(complex(comparison["sigma_xx_model"]), complex(baseline["sigma_xx_model"]))
    rel_yy = _rel_change(complex(comparison["sigma_yy_model"]), complex(baseline["sigma_yy_model"]))
    rel_xy_offdiag = _rel_change(float(comparison["relative_offdiag_norm"]), float(baseline["relative_offdiag_norm"]))
    rel_lt_offdiag = _rel_change(float(comparison["relative_LT_offdiag_norm"]), float(baseline["relative_LT_offdiag_norm"]))
    rel_sym = _rel_change(float(comparison["symmetric_offdiag_abs"]), float(baseline["symmetric_offdiag_abs"]))
    rel_antisym = _rel_change(float(comparison["antisymmetric_offdiag_abs"]), float(baseline["antisymmetric_offdiag_abs"]))
    abs_xy = float(abs(float(comparison["relative_offdiag_norm"]) - float(baseline["relative_offdiag_norm"])))
    abs_lt = float(abs(float(comparison["relative_LT_offdiag_norm"]) - float(baseline["relative_LT_offdiag_norm"])))
    reasons = []
    if float(comparison["ward_max_norm"]) > WARD_CLOSED_THRESHOLD or float(baseline["ward_max_norm"]) > WARD_CLOSED_THRESHOLD:
        reasons.append("WARD_NOT_CLOSED")
    if float(comparison["sigma_diag_min_real"]) < DIAG_NEGATIVE_TOLERANCE or float(baseline["sigma_diag_min_real"]) < DIAG_NEGATIVE_TOLERANCE:
        reasons.append("NEGATIVE_DIAGONAL")
    max_rel = max(rel_xx, rel_yy, rel_xy_offdiag, rel_lt_offdiag, rel_sym)
    if max_rel > REL_CHANGE_MONITOR or abs_xy > ABS_REL_OFFDIAG_CHANGE_MONITOR or abs_lt > ABS_REL_OFFDIAG_CHANGE_MONITOR:
        reasons.append("OFFDIAG_CHANGE_EXCEEDS_MONITOR")
    elif max_rel > REL_CHANGE_PASS or abs_xy > ABS_REL_OFFDIAG_CHANGE_PASS or abs_lt > ABS_REL_OFFDIAG_CHANGE_PASS:
        reasons.append("OFFDIAG_CHANGE_MONITOR")
    if "WARD_NOT_CLOSED" in reasons or "NEGATIVE_DIAGONAL" in reasons or "OFFDIAG_CHANGE_EXCEEDS_MONITOR" in reasons:
        status = "FAIL"
    elif reasons:
        status = "MONITOR"
    else:
        status = "PASS"
    return {
        "baseline_config": {
            "adaptive_level": baseline["adaptive_level"],
            "gauss_order": baseline["gauss_order"],
            "fermi_window_eV": baseline["fermi_window_eV"],
        },
        "comparison_config": {
            "adaptive_level": comparison["adaptive_level"],
            "gauss_order": comparison["gauss_order"],
            "fermi_window_eV": comparison["fermi_window_eV"],
        },
        "matsubara_index": baseline["matsubara_index"],
        "q_case": baseline["q_case"],
        "q_scale": baseline["q_scale"],
        "relative_change_sigma_xx": rel_xx,
        "relative_change_sigma_yy": rel_yy,
        "relative_change_xy_offdiag_norm": rel_xy_offdiag,
        "relative_change_LT_offdiag_norm": rel_lt_offdiag,
        "relative_change_symmetric_offdiag": rel_sym,
        "relative_change_antisymmetric_offdiag": rel_antisym,
        "absolute_change_relative_offdiag_norm": abs_xy,
        "absolute_change_relative_LT_offdiag_norm": abs_lt,
        "comparison_status": status,
        "comparison_reasons": reasons,
    }


def convergence_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_key = (BASELINE_CONFIG["adaptive_level"], BASELINE_CONFIG["gauss_order"], BASELINE_CONFIG["fermi_window_eV"])
    baselines = {
        (row["matsubara_index"], row["q_case"], row["q_scale"]): row
        for row in rows
        if _config_key(row) == baseline_key
    }
    comparisons = []
    for row in rows:
        key = (row["matsubara_index"], row["q_case"], row["q_scale"])
        if _config_key(row) == baseline_key or key not in baselines:
            continue
        comparisons.append(compare_to_baseline(baselines[key], row))
    return comparisons


def q_sign_symmetry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    pos = {
        (row["matsubara_index"], row["q_scale"], _config_key(row)): row
        for row in rows
        if row["q_case"] == "q_diag_pos"
    }
    for row in rows:
        if row["q_case"] != "q_diag_neg":
            continue
        key = (row["matsubara_index"], row["q_scale"], _config_key(row))
        if key in pos:
            pairs.append(q_sign_symmetry_pair(pos[key], row))
    return {
        "num_pairs": len(pairs),
        "pairs": pairs,
        "all_pass": bool(pairs and all(item["q_sign_symmetry_status"] == "PASS" for item in pairs)),
        "max_diag_even_error": max((item["q_sign_diag_even_error"] for item in pairs), default=None),
        "max_offdiag_odd_error": max((item["q_sign_offdiag_odd_error"] for item in pairs), default=None),
    }


def q_scaling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    q1 = {
        (row["matsubara_index"], row["q_case"], _config_key(row)): row
        for row in rows
        if abs(float(row["q_scale"]) - 1.0) < 1e-12
    }
    for row in rows:
        if abs(float(row["q_scale"]) - 0.5) > 1e-12:
            continue
        key = (row["matsubara_index"], row["q_case"], _config_key(row))
        if key not in q1:
            continue
        base = q1[key]
        xy_decreases = float(row["relative_offdiag_norm"]) < float(base["relative_offdiag_norm"])
        lt_decreases = float(row["relative_LT_offdiag_norm"]) < float(base["relative_LT_offdiag_norm"])
        pairs.append(
            {
                "matsubara_index": row["matsubara_index"],
                "q_case": row["q_case"],
                "adaptive_level": row["adaptive_level"],
                "gauss_order": row["gauss_order"],
                "fermi_window_eV": row["fermi_window_eV"],
                "relative_offdiag_norm_at_q1": base["relative_offdiag_norm"],
                "relative_offdiag_norm_at_q05": row["relative_offdiag_norm"],
                "relative_LT_offdiag_norm_at_q1": base["relative_LT_offdiag_norm"],
                "relative_LT_offdiag_norm_at_q05": row["relative_LT_offdiag_norm"],
                "xy_offdiag_decreases_with_q": bool(xy_decreases),
                "LT_offdiag_decreases_with_q": bool(lt_decreases),
                "decrease_factor_xy": float(float(row["relative_offdiag_norm"]) / max(float(base["relative_offdiag_norm"]), EPS)),
                "decrease_factor_LT": float(float(row["relative_LT_offdiag_norm"]) / max(float(base["relative_LT_offdiag_norm"]), EPS)),
            }
        )
    return {"num_pairs": len(pairs), "pairs": pairs, "all_xy_decrease": bool(pairs and all(item["xy_offdiag_decreases_with_q"] for item in pairs))}


def symmetric_antisymmetric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [float(row["relative_antisymmetric_to_symmetric"]) for row in rows]
    max_ratio = max(ratios, default=None)
    return {
        "max_relative_antisymmetric_to_symmetric": max_ratio,
        "status": "SYMMETRIC_OFFDIAG_DOMINATES" if max_ratio is not None and max_ratio < SYMMETRIC_RATIO_THRESHOLD else "HALL_LIKE_OR_NUMERICALLY_AMBIGUOUS",
    }


def global_summary(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_cases": len(rows),
        "num_convergence_comparisons": len(comparisons),
        "max_ward_norm": max((float(row["ward_max_norm"]) for row in rows), default=None),
        "min_diag_real": min((float(row["sigma_diag_min_real"]) for row in rows), default=None),
        "num_comparison_pass": sum(item["comparison_status"] == "PASS" for item in comparisons),
        "num_comparison_monitor": sum(item["comparison_status"] == "MONITOR" for item in comparisons),
        "num_comparison_fail": sum(item["comparison_status"] == "FAIL" for item in comparisons),
    }


def diagnostic_status(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    q_sign: dict[str, Any],
    q_scaling: dict[str, Any],
    sym: dict[str, Any],
) -> dict[str, str]:
    if any(float(row["ward_max_norm"]) > WARD_CLOSED_THRESHOLD for row in rows):
        status = "STAGE5_3B_FAILED_WARD"
        action = "Do not proceed; diagnose Ward closure for targeted offdiag cases."
    elif any(float(row["sigma_diag_min_real"]) < DIAG_NEGATIVE_TOLERANCE for row in rows):
        status = "STAGE5_3B_FAILED_NEGATIVE_DIAGONAL"
        action = "Do not proceed; diagnose negative diagonal conductivity."
    elif any(item["comparison_status"] == "FAIL" for item in comparisons):
        status = "STAGE5_3B_FAILED_OFFDIAG_UNSTABLE"
        action = "Do not proceed; increase or diagnose integration convergence."
    elif not q_sign["all_pass"] or sym["status"] != "SYMMETRIC_OFFDIAG_DOMINATES":
        status = "STAGE5_3B_REQUIRES_SOURCE_PROJECTION_AUDIT"
        action = "Audit source/projection symmetry before Stage 5.4."
    elif any(item["comparison_status"] == "MONITOR" for item in comparisons):
        status = "STAGE5_3B_MONITOR_OFFDIAG_CONVERGENCE"
        action = "Consider one more targeted level/window check before Stage 5.4."
    elif q_scaling["all_xy_decrease"]:
        status = "STAGE5_3B_PASSED_STABLE_FINITE_Q_LATTICE_TENSOR_EFFECT"
        action = "Proceed to Stage 5.4 SI sheet scaling / reflection-input preparation; still do not enter reflection/Casimir."
    else:
        status = "STAGE5_3B_REQUIRES_SOURCE_PROJECTION_AUDIT"
        action = "Audit q-scaling/source projection before Stage 5.4."
    return {"stage5_3b_status": status, "recommended_next_action": action}


def run_audit(config: dict[str, Any], *, max_cases: int | None = None) -> dict[str, Any]:
    cases = planned_cases(config, max_cases)
    rows = [] if config["dry_run"] else run_cases_parallel(cases, eta_eV=float(config["eta_eV"]), workers=int(config["workers"]))
    comparisons = convergence_comparisons(rows)
    q_sign = q_sign_symmetry_summary(rows)
    q_scaling = q_scaling_summary(rows)
    sym = symmetric_antisymmetric_summary(rows)
    summary = global_summary(rows, comparisons)
    return {
        "stage": "Stage 5.3b",
        "purpose": "Targeted convergence audit for finite-q off-diagonal bilayer sheet conductivity",
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
        "convergence_comparisons": comparisons,
        "q_sign_symmetry_summary": q_sign,
        "q_scaling_summary": q_scaling,
        "symmetric_antisymmetric_summary": sym,
        "global_summary": summary,
        "diagnostic_status": diagnostic_status(rows, comparisons, q_sign, q_scaling, sym),
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
    offdiag_rows = [
        (
            row["q_case"],
            row["q_scale"],
            row["matsubara_index"],
            row["adaptive_level"],
            row["fermi_window_eV"],
            _fmt_float(row["relative_offdiag_norm"]),
            _fmt_float(row["relative_LT_offdiag_norm"]),
            _fmt_float(row["relative_antisymmetric_to_symmetric"]),
        )
        for row in rows
    ]
    comparison_rows = [
        (
            item["q_case"],
            item["q_scale"],
            item["matsubara_index"],
            item["comparison_config"],
            _fmt_float(item["absolute_change_relative_offdiag_norm"]),
            _fmt_float(item["absolute_change_relative_LT_offdiag_norm"]),
            item["comparison_status"],
        )
        for item in data["convergence_comparisons"]
    ]
    return "\n\n".join(
        [
            "# Stage 5.3b finite-q offdiag 收敛性审计",
            "## 1. Boundary\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items()),
            "## 2. Conductivity convention\n\n" + "\n".join(f"- {key}: {value}" for key, value in data["conductivity_convention"].items()),
            "## 3. 为什么需要 Stage 5.3b\n\nStage 5.3 已说明 offdiag 更像 finite-q tensor structure，但仍需要 level/window convergence 来排除积分噪声。",
            "## 4. Targeted scan configuration\n\n" + _table(("quantity", "value"), [(key, value) for key, value in data["config"].items() if key != "planned_cases"]),
            "## 5. Ward and diagonal positivity\n\n" + _table(("quantity", "value"), list(data["global_summary"].items())),
            "## 6. Offdiag values\n\n" + (_table(("q", "scale", "n", "level", "window", "rel xy", "rel LT", "A/S"), offdiag_rows) if offdiag_rows else "Dry run：未执行 response 积分。"),
            "## 7. Convergence comparison against baseline\n\n" + (_table(("q", "scale", "n", "comparison", "abs d xy", "abs d LT", "status"), comparison_rows) if comparison_rows else "quick/dry-run 没有 convergence comparison。"),
            "## 8. q-sign symmetry stability\n\n" + _table(("quantity", "value"), [(key, value) for key, value in data["q_sign_symmetry_summary"].items() if key != "pairs"]),
            "## 9. q-scaling stability\n\n" + _table(("quantity", "value"), [(key, value) for key, value in data["q_scaling_summary"].items() if key != "pairs"]),
            "## 10. Symmetric vs antisymmetric offdiag\n\n" + _table(("quantity", "value"), list(data["symmetric_antisymmetric_summary"].items())),
            "## 11. Interpretation\n\n若 full targeted run 中 Ward、q-sign、q-scaling 和 convergence 均稳定，则应表述为 stable finite-q lattice tensor effect，而不是 Hall response。即使 L/T 投影不能完全消掉 offdiag，也不自动构成失败；这可能说明 lattice tensor structure 不等价于 simple continuum LT decomposition。",
            "## 12. Diagnostic decision\n\n" + _table(("quantity", "value"), list(data["diagnostic_status"].items())),
            "## 13. Recommended next step\n\n" + data["diagnostic_status"]["recommended_next_action"] + " 本阶段仍未进入 reflection/Casimir，也仍未做 SI scaling。",
        ]
    ) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matsubara-indices", default="1,2")
    parser.add_argument("--q-cases", default="q_diag_pos,q_diag_neg")
    parser.add_argument("--q-scales", default="1.0,0.5")
    parser.add_argument("--levels", default="4,5")
    parser.add_argument("--gauss-orders", default="5")
    parser.add_argument("--fermi-windows", default="0.05,0.08")
    parser.add_argument("--coarse-grid", type=int, default=32)
    parser.add_argument("--targeted-configs", dest="targeted_configs", action="store_true", default=True)
    parser.add_argument("--cartesian-configs", dest="targeted_configs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
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
