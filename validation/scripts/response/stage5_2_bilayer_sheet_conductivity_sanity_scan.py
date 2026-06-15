#!/usr/bin/env python3
"""Stage 5.2 bilayer sheet conductivity numerical sanity scan."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lno327.conductivity import KuboConfig, bosonic_matsubara_energy_eV  # noqa: E402
from lno327.conductivity_conventions import (  # noqa: E402
    spatial_response_to_bilayer_sheet_conductivity_model,
)
from lno327.ward_response import physical_ward_residuals  # noqa: E402

from stage4_15_fermi_window_adaptive_quadrature import build_adaptive_cells, quadrature_points_for_cells  # noqa: E402
from stage4_16_full_response_adaptive_ward_diagnostic import integrate_physical_components_on_points  # noqa: E402

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "conductivity"
JSON_OUTPUT = OUTPUT_DIR / "stage5_2_bilayer_sheet_conductivity_sanity_scan.json"
MD_OUTPUT = OUTPUT_DIR / "stage5_2_bilayer_sheet_conductivity_sanity_scan.md"

EPS = 1e-300
WARD_CLOSED_THRESHOLD = 1e-6
DIAG_NEGATIVE_TOLERANCE = -1e-8
RELATIVE_OFFDIAG_CLOSED_THRESHOLD = 1e-3
RELATIVE_OFFDIAG_MONITOR_THRESHOLD = 1e-2
FREQUENCY_JUMP_MONITOR_THRESHOLD = 2.0

BOUNDARY = {
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "no_reflection_casimir": True,
    "not_casimir_ready_claim": True,
}

Q_CASES = {
    "q0": np.array([0.0, 0.0], dtype=float),
    "qx": np.array([0.02, 0.0], dtype=float),
    "qy": np.array([0.0, 0.02], dtype=float),
    "q_diag_pos": np.array([0.02, 0.013], dtype=float),
    "q_diag_neg": np.array([0.02, -0.013], dtype=float),
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


def build_scan_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.quick:
        return {
            "temperature_K": 30.0,
            "matsubara_indices": [1, 2],
            "q_cases": ["q_diag_pos"],
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
            for adaptive_level in config["adaptive_levels"]:
                for gauss_order in config["gauss_orders"]:
                    for fermi_window_eV in config["fermi_windows_eV"]:
                        cases.append(
                            {
                                "case_index": len(cases),
                                "temperature_K": float(config["temperature_K"]),
                                "matsubara_index": int(matsubara_index),
                                "q_case": q_case,
                                "q_model": Q_CASES[q_case].copy(),
                                "adaptive_level": int(adaptive_level),
                                "gauss_order": int(gauss_order),
                                "fermi_window_eV": float(fermi_window_eV),
                                "coarse_grid": int(config["coarse_grid"]),
                            }
                        )
                        if max_cases is not None and len(cases) >= max_cases:
                            return cases
    return cases


def conductivity_metrics(sigma: np.ndarray) -> dict[str, Any]:
    sigma_xx = complex(sigma[0, 0])
    sigma_xy = complex(sigma[0, 1])
    sigma_yx = complex(sigma[1, 0])
    sigma_yy = complex(sigma[1, 1])
    offdiag_norm = float(np.sqrt(abs(sigma_xy) ** 2 + abs(sigma_yx) ** 2))
    diag_norm = float(np.sqrt(abs(sigma_xx) ** 2 + abs(sigma_yy) ** 2))
    return {
        "sigma_xx_model": sigma_xx,
        "sigma_xy_model": sigma_xy,
        "sigma_yx_model": sigma_yx,
        "sigma_yy_model": sigma_yy,
        "sigma_trace_real": float((sigma_xx + sigma_yy).real),
        "sigma_diag_min_real": float(min(sigma_xx.real, sigma_yy.real)),
        "sigma_diag_positive": bool(sigma_xx.real >= DIAG_NEGATIVE_TOLERANCE and sigma_yy.real >= DIAG_NEGATIVE_TOLERANCE),
        "offdiag_norm": offdiag_norm,
        "diag_norm": diag_norm,
        "relative_offdiag_norm": float(offdiag_norm / max(diag_norm, EPS)),
        "xy_plus_yx_abs": float(abs(sigma_xy + sigma_yx)),
        "xy_minus_yx_abs": float(abs(sigma_xy - sigma_yx)),
        "xx_minus_yy_abs": float(abs(sigma_xx - sigma_yy)),
        "relative_xx_yy_anisotropy": float(abs(sigma_xx - sigma_yy) / max(abs(sigma_xx) + abs(sigma_yy), EPS)),
    }


def case_status_from_metrics(
    *,
    finite_values: bool,
    omega_eV: float,
    sigma_diag_min_real: float,
    relative_offdiag_norm: float,
    ward_max_norm: float,
    frequency_jump_monitor: bool = False,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not finite_values:
        reasons.append("NONFINITE_VALUE")
    if omega_eV <= 0.0:
        reasons.append("NONPOSITIVE_OMEGA")
    if ward_max_norm >= WARD_CLOSED_THRESHOLD:
        reasons.append("WARD_NOT_CLOSED_FOR_CONDUCTIVITY_POINT")
    if sigma_diag_min_real < DIAG_NEGATIVE_TOLERANCE:
        reasons.append("NEGATIVE_DIAGONAL")
    if reasons:
        return "FAIL", reasons
    if relative_offdiag_norm > RELATIVE_OFFDIAG_CLOSED_THRESHOLD:
        reasons.append("OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT")
    if relative_offdiag_norm > RELATIVE_OFFDIAG_MONITOR_THRESHOLD:
        reasons.append("OFFDIAG_ABOVE_MONITOR_THRESHOLD")
    if frequency_jump_monitor:
        reasons.append("FREQUENCY_SMOOTHNESS_MONITOR")
    return ("MONITOR", reasons) if reasons else ("PASS", [])


def _case_failure_row(case: dict[str, Any], reason: list[str], runtime_seconds: float = 0.0) -> dict[str, Any]:
    q = np.asarray(case["q_model"], dtype=float)
    return {
        **{key: value for key, value in case.items() if key != "q_model"},
        "omega_eV": float("nan"),
        "q_model": q,
        "sigma_xx_model": complex(float("nan"), float("nan")),
        "sigma_xy_model": complex(float("nan"), float("nan")),
        "sigma_yx_model": complex(float("nan"), float("nan")),
        "sigma_yy_model": complex(float("nan"), float("nan")),
        "sigma_trace_real": float("nan"),
        "sigma_diag_min_real": float("nan"),
        "sigma_diag_positive": False,
        "offdiag_norm": float("nan"),
        "diag_norm": float("nan"),
        "relative_offdiag_norm": float("nan"),
        "xy_plus_yx_abs": float("nan"),
        "xy_minus_yx_abs": float("nan"),
        "xx_minus_yy_abs": float("nan"),
        "relative_xx_yy_anisotropy": float("nan"),
        "ward_left_norm": float("inf"),
        "ward_right_norm": float("inf"),
        "ward_max_norm": float("inf"),
        "num_quadrature_points": 0,
        "runtime_seconds": float(runtime_seconds),
        "status": "FAIL",
        "status_reasons": reason,
    }


def run_case(case: dict[str, Any], *, eta_eV: float) -> dict[str, Any]:
    start = time.perf_counter()
    q = np.asarray(case["q_model"], dtype=float)
    omega_eV = bosonic_matsubara_energy_eV(case["matsubara_index"], case["temperature_K"])
    config = KuboConfig.from_kelvin(
        omega_eV=omega_eV,
        temperature_K=case["temperature_K"],
        eta_eV=eta_eV,
        output_si=False,
    )
    try:
        cells, refined_count, _flagged = build_adaptive_cells(
            q,
            coarse_grid=case["coarse_grid"],
            refinement_level=case["adaptive_level"],
            fermi_window_eV=case["fermi_window_eV"],
            fermi_level_eV=config.fermi_level_eV,
        )
        points, weights = quadrature_points_for_cells(cells, case["gauss_order"])
        response = integrate_physical_components_on_points(points, weights, q, config)["total"]
        sigma = spatial_response_to_bilayer_sheet_conductivity_model(response, omega_eV)
        metrics = conductivity_metrics(sigma)
        left, right = physical_ward_residuals(response, omega_eV, q)
        ward_left_norm = float(np.linalg.norm(left))
        ward_right_norm = float(np.linalg.norm(right))
        ward_max_norm = max(ward_left_norm, ward_right_norm)
        finite_values = bool(
            np.all(np.isfinite(response))
            and np.all(np.isfinite(sigma))
            and np.isfinite(omega_eV)
            and np.isfinite(ward_max_norm)
        )
        status, reasons = case_status_from_metrics(
            finite_values=finite_values,
            omega_eV=omega_eV,
            sigma_diag_min_real=metrics["sigma_diag_min_real"],
            relative_offdiag_norm=metrics["relative_offdiag_norm"],
            ward_max_norm=ward_max_norm,
        )
        return {
            **{key: value for key, value in case.items() if key != "q_model"},
            "omega_eV": float(omega_eV),
            "q_model": q,
            **metrics,
            "ward_left_norm": ward_left_norm,
            "ward_right_norm": ward_right_norm,
            "ward_max_norm": ward_max_norm,
            "num_quadrature_points": int(len(points)),
            "runtime_seconds": float(time.perf_counter() - start),
            "status": status,
            "status_reasons": reasons,
        }
    except Exception as exc:
        row = _case_failure_row(case, ["CASE_EXCEPTION", type(exc).__name__, str(exc)], time.perf_counter() - start)
        row["omega_eV"] = float(omega_eV) if "omega_eV" in locals() else float("nan")
        return row


def _run_case_job(index: int, case: dict[str, Any], eta_eV: float) -> tuple[int, dict[str, Any]]:
    return index, run_case(case, eta_eV=eta_eV)


def _run_worker_job(
    index: int,
    case: dict[str, Any],
    eta_eV: float,
    worker: Callable[..., dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    return index, worker(case, eta_eV=eta_eV)


def run_cases_parallel(
    cases: list[dict[str, Any]],
    *,
    eta_eV: float,
    workers: int,
    worker: Callable[..., dict[str, Any]] = run_case,
    executor_factory: Callable[..., Any] = ProcessPoolExecutor,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not cases:
        return []
    total = len(cases)
    actual_workers = min(workers, total)
    print(f"Running {total} Stage 5.2 cases with {actual_workers} worker(s)...")
    if actual_workers <= 1:
        rows: list[dict[str, Any]] = []
        for done, case in enumerate(cases, start=1):
            row = worker(case, eta_eV=eta_eV)
            rows.append(row)
            print(
                f"Completed {done}/{total}: {case['q_case']} "
                f"n={case['matsubara_index']} status={row['status']}"
            )
        return rows

    indexed_rows: dict[int, dict[str, Any]] = {}
    with executor_factory(max_workers=actual_workers) as executor:
        future_to_case = {
            executor.submit(_run_worker_job, index, case, eta_eV, worker): (index, case)
            for index, case in enumerate(cases)
        }
        for done, future in enumerate(as_completed(future_to_case), start=1):
            index, case = future_to_case[future]
            try:
                _result_index, row = future.result()
            except Exception as exc:
                row = _case_failure_row(case, ["EXECUTOR_CASE_EXCEPTION", type(exc).__name__, str(exc)])
            indexed_rows[index] = row
            print(
                f"Completed {done}/{total}: {case['q_case']} "
                f"n={case['matsubara_index']} status={row['status']}"
            )
    return [indexed_rows[index] for index in range(total)]


def apply_frequency_smoothness(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["q_case"],
            row["adaptive_level"],
            row["gauss_order"],
            row["fermi_window_eV"],
            row["coarse_grid"],
        )
        grouped.setdefault(key, []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda item: int(item["matsubara_index"]))
        for prev, curr in zip(group_rows, group_rows[1:], strict=False):
            prev_xx = complex(prev["sigma_xx_model"])
            curr_xx = complex(curr["sigma_xx_model"])
            jump = float(abs(curr_xx - prev_xx) / max(abs(prev_xx), abs(curr_xx), EPS))
            curr["frequency_relative_jump_from_previous"] = jump
            if np.isfinite(jump) and jump > FREQUENCY_JUMP_MONITOR_THRESHOLD and curr["status"] == "PASS":
                curr["status"] = "MONITOR"
                curr.setdefault("status_reasons", []).append("FREQUENCY_SMOOTHNESS_MONITOR")
        if group_rows:
            group_rows[0]["frequency_relative_jump_from_previous"] = 0.0


def summary_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_total_cases": 0,
            "num_pass": 0,
            "num_monitor": 0,
            "num_fail": 0,
            "max_ward_norm": None,
            "min_diag_real": None,
            "max_relative_offdiag_norm": None,
            "max_relative_xx_yy_anisotropy": None,
            "worst_offdiag_case": None,
            "worst_ward_case": None,
            "worst_negative_diag_case": None,
        }
    worst_offdiag = max(rows, key=lambda row: float(row["relative_offdiag_norm"]))
    worst_ward = max(rows, key=lambda row: float(row["ward_max_norm"]))
    worst_negative = min(rows, key=lambda row: float(row["sigma_diag_min_real"]))
    return {
        "num_total_cases": len(rows),
        "num_pass": sum(row["status"] == "PASS" for row in rows),
        "num_monitor": sum(row["status"] == "MONITOR" for row in rows),
        "num_fail": sum(row["status"] == "FAIL" for row in rows),
        "max_ward_norm": float(max(float(row["ward_max_norm"]) for row in rows)),
        "min_diag_real": float(min(float(row["sigma_diag_min_real"]) for row in rows)),
        "max_relative_offdiag_norm": float(max(float(row["relative_offdiag_norm"]) for row in rows)),
        "max_relative_xx_yy_anisotropy": float(max(float(row["relative_xx_yy_anisotropy"]) for row in rows)),
        "worst_offdiag_case": _case_id(worst_offdiag),
        "worst_ward_case": _case_id(worst_ward),
        "worst_negative_diag_case": _case_id(worst_negative),
    }


def _case_id(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "q_case": row.get("q_case"),
        "matsubara_index": row.get("matsubara_index"),
        "adaptive_level": row.get("adaptive_level"),
        "gauss_order": row.get("gauss_order"),
        "fermi_window_eV": row.get("fermi_window_eV"),
        "status": row.get("status"),
    }


def diagnostic_status(stats: dict[str, Any]) -> dict[str, str]:
    if int(stats["num_fail"]) > 0:
        status = "CONDUCTIVITY_SANITY_FAILED"
        action = "Do not proceed; diagnose failed conductivity or Ward channel."
    elif int(stats["num_monitor"]) > 0:
        status = "CONDUCTIVITY_SANITY_MONITOR_OFFDIAG"
        action = "Proceed to Stage 5.3 conductivity convergence / symmetry scan before reflection/Casimir."
    else:
        status = "CONDUCTIVITY_SANITY_PASSED"
        action = "Proceed to Stage 5.3 conductivity convergence / symmetry scan before reflection/Casimir."
    return {"conductivity_sanity_status": status, "recommended_next_action": action}


def run_scan(config: dict[str, Any], *, max_cases: int | None = None) -> dict[str, Any]:
    cases = planned_cases(config, max_cases)
    if config["dry_run"]:
        rows: list[dict[str, Any]] = []
    else:
        rows = run_cases_parallel(cases, eta_eV=float(config["eta_eV"]), workers=int(config["workers"]))
        apply_frequency_smoothness(rows)
    stats = summary_statistics(rows)
    return {
        "stage": "Stage 5.2",
        "purpose": "Bilayer sheet conductivity numerical sanity scan",
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
        "summary_statistics": stats,
        "diagnostic_status": diagnostic_status(stats),
    }


def _fmt_float(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6e}"


def _fmt_complex(value: Any) -> str:
    if isinstance(value, dict):
        return f"{float(value['real']):.6e}{float(value['imag']):+.6e}j"
    value = complex(value)
    return f"{value.real:.6e}{value.imag:+.6e}j"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    config = data["config"]
    stats = data["summary_statistics"]
    rows = data["scan_results"]
    boundary = "\n".join(f"- {key}: {value}" for key, value in data["boundary"].items())
    convention = "\n".join(f"- {key}: {value}" for key, value in data["conductivity_convention"].items())
    freq_rows = [
        (
            row["q_case"],
            row["matsubara_index"],
            _fmt_float(row["omega_eV"]),
            _fmt_complex(row["sigma_xx_model"]),
            _fmt_complex(row["sigma_yy_model"]),
            row["status"],
        )
        for row in rows
    ]
    offdiag_rows = [
        (
            row["q_case"],
            row["matsubara_index"],
            _fmt_float(row["relative_offdiag_norm"]),
            _fmt_float(row["relative_xx_yy_anisotropy"]),
            _fmt_float(row.get("frequency_relative_jump_from_previous", 0.0)),
            ", ".join(row.get("status_reasons", [])),
        )
        for row in rows
    ]
    ward_rows = [
        (
            row["q_case"],
            row["matsubara_index"],
            _fmt_float(row["ward_left_norm"]),
            _fmt_float(row["ward_right_norm"]),
            _fmt_float(row["ward_max_norm"]),
            row["num_quadrature_points"],
        )
        for row in rows
    ]
    return "\n\n".join(
        [
            "# Stage 5.2 Bilayer sheet conductivity sanity scan",
            "## Boundary\n\n" + boundary,
            "## Conductivity convention\n\n"
            + convention
            + "\n\nThis is model-level bilayer sheet conductivity. SI scaling is not applied. This is not reflection/Casimir input yet.",
            "## Scan configuration\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("quick", config["quick"]),
                    ("dry_run", config["dry_run"]),
                    ("temperature_K", config["temperature_K"]),
                    ("matsubara_indices", config["matsubara_indices"]),
                    ("q_cases", config["q_cases"]),
                    ("adaptive_levels", config["adaptive_levels"]),
                    ("gauss_orders", config["gauss_orders"]),
                    ("fermi_windows_eV", config["fermi_windows_eV"]),
                    ("coarse_grid", config["coarse_grid"]),
                    ("planned_num_cases", config["planned_num_cases"]),
                ],
            ),
            "## Summary statistics\n\n"
            + _table(("quantity", "value"), [(key, value) for key, value in stats.items()]),
            "## Conductivity sanity by Matsubara frequency\n\n"
            + (_table(("q", "n", "omega_eV", "sigma_xx", "sigma_yy", "status"), freq_rows) if freq_rows else "Dry run: no response integration executed."),
            "## Off-diagonal and anisotropy diagnostics\n\n"
            + (
                _table(("q", "n", "rel offdiag", "rel xx/yy anisotropy", "freq jump", "reasons"), offdiag_rows)
                if offdiag_rows
                else "finite-q angular dependence should not automatically be treated as an error."
            ),
            "## Ward residual diagnostics\n\n"
            + (_table(("q", "n", "left", "right", "max", "points"), ward_rows) if ward_rows else "Dry run: no Ward residuals computed."),
            "## Diagnostic decision\n\n"
            + _table(("quantity", "value"), [(key, value) for key, value in data["diagnostic_status"].items()]),
            "## Recommended next step\n\n"
            + data["diagnostic_status"]["recommended_next_action"]
            + " finite-q angular dependence should not automatically be treated as an error.",
        ]
    ) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matsubara-indices", default="1,2,4,8")
    parser.add_argument("--q-cases", default="q0,qx,qy,q_diag_pos,q_diag_neg")
    parser.add_argument("--levels", default="4,5")
    parser.add_argument("--gauss-orders", default="3,5")
    parser.add_argument("--fermi-windows", default="0.05")
    parser.add_argument("--coarse-grid", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=MD_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    config = build_scan_config(args)
    data = run_scan(config, max_cases=args.max_cases)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
