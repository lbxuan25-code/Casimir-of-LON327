#!/usr/bin/env python3
"""Stage 4.20 user-run targeted Ward refinement scan.

Diagnostic-only.  This script targets the Stage 4.19 worst-case cluster with
optional multiprocessing, resume, dry-run, and max-case controls.  It does not
modify the main response formula, bubble prefactor sign, direct contact,
source/observable split, conductivity, reflection, or Casimir code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lno327.collective.ward import physical_ward_residuals  # noqa: E402

from stage4_19_multi_parameter_ward_robustness_scan import (  # noqa: E402
    CLOSURE_THRESHOLD,
    MONITOR_THRESHOLD,
    _case_status,
    _residual_row,
    _strip_vectors,
    _top_rows,
    to_jsonable,
)

OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "ward_identity"
DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "stage4_20_user_run_targeted_refinement_scan.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "stage4_20_user_run_targeted_refinement_scan.md"

Q_CASES = {
    "qx": np.array([0.02, 0.0], dtype=float),
    "qy": np.array([0.0, 0.02], dtype=float),
    "q_diag_pos": np.array([0.02, 0.013], dtype=float),
    "q_diag_neg": np.array([0.02, -0.013], dtype=float),
}

BOUNDARY = {
    "no_main_response_change": True,
    "no_bubble_sign_change": True,
    "no_direct_contact_change": True,
    "no_source_observable_change": True,
    "no_residual_tuning": True,
    "no_fitted_contact": True,
    "no_E_ET_added": True,
    "no_conductivity_reflection_casimir": True,
    "not_casimir_ready_claim": True,
}


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def preset_config(preset: str) -> dict[str, Any]:
    if preset == "quick":
        return {
            "temperatures_K": [30.0],
            "matsubara_indices": [1],
            "q_cases": {"q_diag_pos": Q_CASES["q_diag_pos"]},
            "q_scales": [1.0, 0.5],
            "adaptive_levels": [1],
            "gauss_orders": [2],
            "fermi_windows_eV": [0.05],
            "coarse_grid": 8,
        }
    if preset == "targeted":
        return {
            "temperatures_K": [30.0],
            "matsubara_indices": [1],
            "q_cases": {"q_diag_pos": Q_CASES["q_diag_pos"], "q_diag_neg": Q_CASES["q_diag_neg"]},
            "q_scales": [1.0, 0.5],
            "adaptive_levels": [3, 4, 5],
            "gauss_orders": [3, 5],
            "fermi_windows_eV": [0.03, 0.05, 0.08, 0.12],
            "coarse_grid": 32,
        }
    if preset == "worst-only":
        return {
            "temperatures_K": [30.0],
            "matsubara_indices": [1],
            "q_cases": {"q_diag_pos": Q_CASES["q_diag_pos"]},
            "q_scales": [1.0],
            "adaptive_levels": [3, 4, 5],
            "gauss_orders": [3, 5],
            "fermi_windows_eV": [0.03, 0.05, 0.08, 0.12],
            "coarse_grid": 32,
        }
    if preset == "confirm":
        return {
            "temperatures_K": [30.0],
            "matsubara_indices": [1],
            "q_cases": {"q_diag_pos": Q_CASES["q_diag_pos"], "q_diag_neg": Q_CASES["q_diag_neg"]},
            "q_scales": [1.0, 0.5],
            "adaptive_levels": [4, 5],
            "gauss_orders": [3, 5],
            "fermi_windows_eV": [0.05, 0.08, 0.12],
            "coarse_grid": 32,
        }
    if preset == "custom":
        return preset_config("quick")
    raise ValueError(f"unknown preset: {preset}")


def apply_overrides(
    config: dict[str, Any],
    *,
    coarse_grid: int | None = None,
    levels: list[int] | None = None,
    gauss_orders: list[int] | None = None,
    fermi_windows: list[float] | None = None,
    temperatures: list[float] | None = None,
    matsubara_indices: list[int] | None = None,
    q_case_names: list[str] | None = None,
    q_scales: list[float] | None = None,
) -> dict[str, Any]:
    updated = dict(config)
    if coarse_grid is not None:
        updated["coarse_grid"] = int(coarse_grid)
    if levels is not None:
        updated["adaptive_levels"] = [int(item) for item in levels]
    if gauss_orders is not None:
        updated["gauss_orders"] = [int(item) for item in gauss_orders]
    if fermi_windows is not None:
        updated["fermi_windows_eV"] = [float(item) for item in fermi_windows]
    if temperatures is not None:
        updated["temperatures_K"] = [float(item) for item in temperatures]
    if matsubara_indices is not None:
        updated["matsubara_indices"] = [int(item) for item in matsubara_indices]
    if q_case_names is not None:
        unknown = sorted(set(q_case_names) - set(Q_CASES))
        if unknown:
            raise ValueError(f"unknown q cases: {unknown}")
        updated["q_cases"] = {name: Q_CASES[name] for name in q_case_names}
    if q_scales is not None:
        updated["q_scales"] = [float(item) for item in q_scales]
    return updated


def build_cases(config: dict[str, Any], *, max_cases: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for temperature_K in config["temperatures_K"]:
        for matsubara_index in config["matsubara_indices"]:
            for q_case, q_base in config["q_cases"].items():
                q_base_array = np.asarray(q_base, dtype=float)
                for q_scale in config["q_scales"]:
                    q = float(q_scale) * q_base_array
                    for adaptive_level in config["adaptive_levels"]:
                        for gauss_order in config["gauss_orders"]:
                            for fermi_window_eV in config["fermi_windows_eV"]:
                                cases.append(
                                    {
                                        "temperature_K": float(temperature_K),
                                        "matsubara_index": int(matsubara_index),
                                        "q_case": str(q_case),
                                        "q_scale": float(q_scale),
                                        "q": q,
                                        "adaptive_level": int(adaptive_level),
                                        "gauss_order": int(gauss_order),
                                        "fermi_window_eV": float(fermi_window_eV),
                                        "coarse_grid": int(config["coarse_grid"]),
                                    }
                                )
    if max_cases is not None:
        return cases[: int(max_cases)]
    return cases


def case_key(case: dict[str, Any] | dict[str, object]) -> str:
    return "|".join(
        (
            f"{float(case['temperature_K']):.12g}",
            str(int(case["matsubara_index"])),
            str(case["q_case"]),
            f"{float(case['q_scale']):.12g}",
            str(int(case["adaptive_level"])),
            str(int(case["gauss_order"])),
            f"{float(case['fermi_window_eV']):.12g}",
            str(int(case["coarse_grid"])),
        )
    )


def estimate_quadrature_points(case: dict[str, Any]) -> int:
    coarse = int(case["coarse_grid"])
    level = int(case["adaptive_level"])
    order = int(case["gauss_order"])
    return int(coarse * coarse * (4**level) * order * order)


def _worker(case: dict[str, Any]) -> dict[str, Any]:
    q = np.asarray(case["q"], dtype=float)
    row = _residual_row(
        temperature_K=float(case["temperature_K"]),
        matsubara_index=int(case["matsubara_index"]),
        q_case=str(case["q_case"]),
        q_scale=float(case["q_scale"]),
        q=q,
        adaptive_level=int(case["adaptive_level"]),
        gauss_order=int(case["gauss_order"]),
        fermi_window_eV=float(case["fermi_window_eV"]),
        coarse_grid=int(case["coarse_grid"]),
    )
    row["coarse_grid"] = int(case["coarse_grid"])
    row["case_key"] = case_key(case)
    return row


def _candidate_existing_paths(
    output_json: Path,
    checkpoint_jsonl: Path,
    *,
    allow_default_checkpoint_fallback: bool,
) -> tuple[list[Path], list[Path]]:
    json_paths = [output_json]
    checkpoint_paths = [checkpoint_jsonl]
    if output_json != DEFAULT_JSON_OUTPUT and not output_json.exists() and DEFAULT_JSON_OUTPUT.exists():
        json_paths.append(DEFAULT_JSON_OUTPUT)
    default_checkpoint = DEFAULT_JSON_OUTPUT.with_suffix(DEFAULT_JSON_OUTPUT.suffix + ".jsonl")
    if (
        allow_default_checkpoint_fallback
        and checkpoint_jsonl != default_checkpoint
        and not checkpoint_jsonl.exists()
        and default_checkpoint.exists()
    ):
        checkpoint_paths.append(default_checkpoint)
    return json_paths, checkpoint_paths


def load_completed(
    output_json: Path,
    checkpoint_jsonl: Path,
    *,
    allow_default_checkpoint_fallback: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    json_paths, checkpoint_paths = _candidate_existing_paths(
        output_json,
        checkpoint_jsonl,
        allow_default_checkpoint_fallback=allow_default_checkpoint_fallback,
    )
    for json_path in json_paths:
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            rows.extend(data.get("scan_results", []))
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "case_key" not in row:
            row["case_key"] = case_key(row)
        deduped[str(row["case_key"])] = row
    return list(deduped.values())


def filter_existing_to_active_grid(
    existing_rows: list[dict[str, Any]],
    active_case_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_by_key: dict[str, dict[str, Any]] = {}
    ignored_keys: list[str] = []
    for row in existing_rows:
        if "case_key" not in row:
            row["case_key"] = case_key(row)
        key = str(row["case_key"])
        if key in active_case_keys:
            active_by_key[key] = row
        else:
            ignored_keys.append(key)
    metadata = {
        "loaded_existing_case_count": int(len(existing_rows)),
        "loaded_existing_active_case_count": int(len(active_by_key)),
        "ignored_existing_case_count": int(len(ignored_keys)),
        "excluded_old_case_examples": ignored_keys[:10],
    }
    return list(active_by_key.values()), metadata


def _best_parameter_set_per_q_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        q_case = str(row["q_case"])
        if q_case not in best or float(row["max_corrected_norm"]) < float(best[q_case]["max_corrected_norm"]):
            best[q_case] = _strip_vectors(row)
    return best


def _summary_statistics(rows: list[dict[str, Any]], total_cases: int) -> dict[str, Any]:
    values = np.array([float(row["max_corrected_norm"]) for row in rows], dtype=float)
    worst = max(rows, key=lambda item: float(item["max_corrected_norm"])) if rows else {}
    return {
        "num_total_cases": int(total_cases),
        "num_completed_cases": int(len(rows)),
        "num_closed": int(sum(1 for row in rows if row["status"] == "CLOSED")),
        "num_acceptable_but_monitor": int(sum(1 for row in rows if row["status"] == "ACCEPTABLE_BUT_MONITOR")),
        "num_not_closed": int(sum(1 for row in rows if row["status"] == "NOT_CLOSED")),
        "max_corrected_norm_global": float(np.max(values)) if len(values) else float("nan"),
        "median_corrected_norm": float(np.median(values)) if len(values) else float("nan"),
        "p95_corrected_norm": float(np.percentile(values, 95)) if len(values) else float("nan"),
        "worst_case_parameters": _strip_vectors(worst) if worst else {},
        "best_parameter_set_per_q_case": _best_parameter_set_per_q_case(rows),
    }


def _dominant_failure_channel(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    target_rows = [row for row in rows if row["status"] != "CLOSED"] or rows
    metrics = {
        "left_density_source": max(float(row["left_density_source_abs"]) for row in target_rows),
        "left_spatial_source": max(float(row["left_spatial_source_norm"]) for row in target_rows),
        "right_density_observable": max(float(row["right_density_observable_abs"]) for row in target_rows),
        "right_spatial_observable": max(float(row["right_spatial_observable_norm"]) for row in target_rows),
        "left_longitudinal": max(float(row["left_longitudinal_abs"]) for row in target_rows),
        "left_transverse": max(float(row["left_transverse_abs"]) for row in target_rows),
        "right_longitudinal": max(float(row["right_longitudinal_abs"]) for row in target_rows),
        "right_transverse": max(float(row["right_transverse_abs"]) for row in target_rows),
    }
    return max(metrics, key=metrics.get)


def _improves_with_refinement(rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[str, float, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["q_case"]), float(row["q_scale"]), int(row["gauss_order"]), float(row["fermi_window_eV"]))
        grouped.setdefault(key, []).append(row)
    checked = False
    for group_rows in grouped.values():
        if len(group_rows) < 2:
            continue
        checked = True
        ordered = sorted(group_rows, key=lambda item: int(item["adaptive_level"]))
        if float(ordered[-1]["max_corrected_norm"]) > float(ordered[0]["max_corrected_norm"]):
            return False
    return checked


def diagnostic_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    closed = sum(1 for row in rows if row["status"] == "CLOSED")
    monitor = sum(1 for row in rows if row["status"] == "ACCEPTABLE_BUT_MONITOR")
    not_closed = sum(1 for row in rows if row["status"] == "NOT_CLOSED")
    if total > 0 and closed == total:
        status = "TARGETED_REFINEMENT_PASSED"
        likely = "TARGETED_REFINEMENT_CLOSED"
        action = "Proceed to the next validation stage; do not claim Casimir readiness."
    elif total > 0 and not_closed == 0:
        status = "TARGETED_REFINEMENT_MOSTLY_PASSED"
        likely = "MINOR_MONITOR_CASES_REMAIN"
        action = "Rerun monitor cases with higher refinement before downstream validation."
    elif _improves_with_refinement(rows):
        status = "NEEDS_HIGHER_REFINEMENT_OR_WINDOW"
        likely = "TARGETED_CLUSTER_STILL_QUADRATURE_LIMITED"
        action = "Increase adaptive level, Gauss order, or Fermi window for remaining NOT_CLOSED cases."
    else:
        status = "POSSIBLE_NON_QUADRATURE_REMAINING_ISSUE"
        likely = "TARGETED_CLUSTER_NOT_MONOTONIC_WITH_REFINEMENT"
        action = "Audit remaining routing/contact assumptions before any downstream use."
    return {
        "targeted_refinement_status": status,
        "closure_threshold": CLOSURE_THRESHOLD,
        "monitor_threshold": MONITOR_THRESHOLD,
        "dominant_failure_channel": _dominant_failure_channel(rows),
        "likely_issue": likely,
        "recommended_next_action": action,
    }


def assemble_output(
    *,
    preset: str,
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    run_mode: str,
    workers: int,
    filtering: dict[str, Any],
) -> dict[str, Any]:
    rows_sorted = sorted(rows, key=lambda row: case_key(row))
    summary = _summary_statistics(rows_sorted, len(cases))
    return {
        "stage": "Stage 4.20",
        "purpose": "User-run targeted worst-case Ward refinement scan with multiprocessing",
        "run_mode": run_mode,
        "config": {
            "preset": preset,
            "workers": int(workers),
            "temperatures_K": [float(item) for item in config["temperatures_K"]],
            "matsubara_indices": [int(item) for item in config["matsubara_indices"]],
            "q_cases": {name: [float(value[0]), float(value[1])] for name, value in config["q_cases"].items()},
            "q_scales": [float(item) for item in config["q_scales"]],
            "adaptive_levels": [int(item) for item in config["adaptive_levels"]],
            "gauss_orders": [int(item) for item in config["gauss_orders"]],
            "fermi_windows_eV": [float(item) for item in config["fermi_windows_eV"]],
            "coarse_grid": int(config["coarse_grid"]),
            "right_residual_convention": "iOmega Pi[mu,0] - qx Pi[mu,x] - qy Pi[mu,y]",
        },
        "scan_results": rows_sorted,
        "summary_statistics": summary,
        "worst_cases": {"top_10_largest_max_corrected_norm": _top_rows(rows_sorted, "max_corrected_norm")},
        "diagnostic_status": diagnostic_status(rows_sorted),
        "filtering": dict(filtering),
        "boundary": dict(BOUNDARY),
    }


def dry_run_output(*, preset: str, config: dict[str, Any], cases: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    planned_cases = []
    for case in cases:
        item = {
            key: value
            for key, value in case.items()
            if key != "q"
        }
        item["q_model"] = [float(case["q"][0]), float(case["q"][1])]
        item["case_key"] = case_key(case)
        item["estimated_quadrature_points_upper_bound"] = estimate_quadrature_points(case)
        planned_cases.append(item)
    return {
        "stage": "Stage 4.20",
        "purpose": "User-run targeted worst-case Ward refinement scan with multiprocessing",
        "run_mode": "dry_run",
        "config": {
            "preset": preset,
            "workers": int(workers),
            "coarse_grid": int(config["coarse_grid"]),
        },
        "num_planned_cases": len(cases),
        "planned_cases": planned_cases,
        "boundary": dict(BOUNDARY),
    }


def run_scan(
    *,
    preset: str = "quick",
    workers: int = 1,
    coarse_grid: int | None = None,
    adaptive_levels: list[int] | None = None,
    gauss_orders: list[int] | None = None,
    fermi_windows: list[float] | None = None,
    temperatures: list[float] | None = None,
    matsubara_indices: list[int] | None = None,
    q_case_names: list[str] | None = None,
    q_scales: list[float] | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
    resume: bool = False,
    fresh: bool = False,
    filter_existing_to_active_grid_only: bool = False,
    checkpoint_jsonl: Path | None = None,
    dry_run: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    if fresh and resume:
        raise ValueError("--fresh and --resume are mutually exclusive")
    if filter_existing_to_active_grid_only and fresh:
        raise ValueError("--filter-existing-to-active-grid cannot be combined with --fresh")
    config = apply_overrides(
        preset_config(preset),
        coarse_grid=coarse_grid,
        levels=adaptive_levels,
        gauss_orders=gauss_orders,
        fermi_windows=fermi_windows,
        temperatures=temperatures,
        matsubara_indices=matsubara_indices,
        q_case_names=q_case_names,
        q_scales=q_scales,
    )
    cases = build_cases(config, max_cases=max_cases)
    if dry_run:
        return dry_run_output(preset=preset, config=config, cases=cases, workers=workers)

    json_path = DEFAULT_JSON_OUTPUT if output_json is None else Path(output_json)
    md_path = DEFAULT_MD_OUTPUT if output_md is None else Path(output_md)
    checkpoint_path = json_path.with_suffix(json_path.suffix + ".jsonl") if checkpoint_jsonl is None else Path(checkpoint_jsonl)
    active_case_keys = {case_key(case) for case in cases}
    should_load_existing = resume or filter_existing_to_active_grid_only
    existing_rows_all = [] if fresh or not should_load_existing else load_completed(
        json_path,
        checkpoint_path,
        allow_default_checkpoint_fallback=checkpoint_jsonl is None,
    )
    existing_rows_active, filtering_metadata = filter_existing_to_active_grid(existing_rows_all, active_case_keys)
    active_completed_keys = {str(row["case_key"]) for row in existing_rows_active}
    pending_cases = [] if filter_existing_to_active_grid_only else [
        case for case in cases if case_key(case) not in active_completed_keys
    ]
    rows = list(existing_rows_active)
    filtering_metadata.update(
        {
            "active_case_count": int(len(cases)),
            "newly_computed_case_count": 0,
            "results_used_for_summary_count": int(len(rows)),
            "fresh_mode": bool(fresh),
            "filter_existing_to_active_grid": bool(filter_existing_to_active_grid_only),
        }
    )

    if (fresh or (not resume and not filter_existing_to_active_grid_only)) and checkpoint_path.exists():
        checkpoint_path.unlink()

    if pending_cases:
        if workers <= 1:
            for case in pending_cases:
                row = _worker(case)
                rows.append(row)
                filtering_metadata["newly_computed_case_count"] += 1
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = {executor.submit(_worker, case): case for case in pending_cases}
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    filtering_metadata["newly_computed_case_count"] += 1
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    with checkpoint_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
    filtering_metadata["results_used_for_summary_count"] = int(len(rows))

    data = assemble_output(
        preset=preset,
        config=config,
        cases=cases,
        rows=rows,
        run_mode="filter_existing_to_active_grid" if filter_existing_to_active_grid_only else ("resume" if resume else "fresh"),
        workers=workers,
        filtering=filtering_metadata,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(data), encoding="utf-8")
    return data


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(data: dict[str, Any]) -> str:
    stats = data["summary_statistics"]
    status = data["diagnostic_status"]
    filtering = data.get("filtering", {})
    worst_rows = [
        (
            row["q_case"],
            _fmt(float(row["q_scale"])),
            row["adaptive_level"],
            row["gauss_order"],
            _fmt(float(row["fermi_window_eV"])),
            _fmt(float(row["max_corrected_norm"])),
            row["status"],
        )
        for row in data["worst_cases"]["top_10_largest_max_corrected_norm"]
    ]
    return "\n\n".join(
        [
            "# Stage 4.20 User-run targeted Ward refinement scan",
            "## Purpose\n\nTargeted user-run refinement scan for the Stage 4.19 worst-case Ward residual cluster.",
            "## Summary\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("num_total_cases", stats["num_total_cases"]),
                    ("num_completed_cases", stats["num_completed_cases"]),
                    ("num_closed", stats["num_closed"]),
                    ("num_acceptable_but_monitor", stats["num_acceptable_but_monitor"]),
                    ("num_not_closed", stats["num_not_closed"]),
                    ("max_corrected_norm_global", _fmt(float(stats["max_corrected_norm_global"]))),
                    ("median_corrected_norm", _fmt(float(stats["median_corrected_norm"]))),
                    ("p95_corrected_norm", _fmt(float(stats["p95_corrected_norm"]))),
                ],
            ),
            "## Worst cases\n\n"
            + _table(("q_case", "q_scale", "level", "order", "window", "max_norm", "status"), worst_rows),
            "## Filtering / resume behavior\n\n"
            + _table(
                ("quantity", "value"),
                [
                    ("active_case_count", filtering.get("active_case_count", 0)),
                    ("loaded_existing_case_count", filtering.get("loaded_existing_case_count", 0)),
                    ("loaded_existing_active_case_count", filtering.get("loaded_existing_active_case_count", 0)),
                    ("ignored_existing_case_count", filtering.get("ignored_existing_case_count", 0)),
                    ("newly_computed_case_count", filtering.get("newly_computed_case_count", 0)),
                    ("results_used_for_summary_count", filtering.get("results_used_for_summary_count", 0)),
                ],
            )
            + "\n\nSummary only includes active CLI grid cases. Old completed cases outside the active grid are ignored, preventing old 0.03 eV scans from polluting no-0.03 summaries.",
            "## Diagnostic decision\n\n"
            + _table(
                ("quantity", "status"),
                [
                    ("targeted_refinement_status", status["targeted_refinement_status"]),
                    ("dominant_failure_channel", status["dominant_failure_channel"]),
                    ("likely_issue", status["likely_issue"]),
                    ("recommended_next_action", status["recommended_next_action"]),
                ],
            ),
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "targeted", "confirm", "worst-only", "custom"), default="quick")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--coarse-grid", type=int)
    parser.add_argument("--levels", type=_int_list)
    parser.add_argument("--gauss-orders", type=_int_list)
    parser.add_argument("--fermi-windows", type=_float_list)
    parser.add_argument("--temperatures", type=_float_list)
    parser.add_argument("--matsubara-indices", type=_int_list)
    parser.add_argument("--q-cases", type=_str_list)
    parser.add_argument("--q-scales", type=_float_list)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD_OUTPUT)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--fresh", action="store_true")
    parser.add_argument("--filter-existing-to-active-grid", action="store_true")
    parser.add_argument("--checkpoint-jsonl", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = run_scan(
        preset=args.preset,
        workers=args.workers,
        coarse_grid=args.coarse_grid,
        adaptive_levels=args.levels,
        gauss_orders=args.gauss_orders,
        fermi_windows=args.fermi_windows,
        temperatures=args.temperatures,
        matsubara_indices=args.matsubara_indices,
        q_case_names=args.q_cases,
        q_scales=args.q_scales,
        output_json=args.output_json,
        output_md=args.output_md,
        resume=args.resume,
        fresh=args.fresh,
        filter_existing_to_active_grid_only=args.filter_existing_to_active_grid,
        checkpoint_jsonl=args.checkpoint_jsonl,
        dry_run=args.dry_run,
        max_cases=args.max_cases,
    )
    if args.dry_run:
        print(json.dumps(to_jsonable(data), indent=2, sort_keys=True))
    else:
        print(f"Wrote {args.output_json}")
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
