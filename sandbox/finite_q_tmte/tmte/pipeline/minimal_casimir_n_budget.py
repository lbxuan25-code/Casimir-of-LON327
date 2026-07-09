"""Offline diagnostic n-budget aggregation for minimal Casimir n scans.

This module merges existing n-scan CSV outputs and optional n-tail-fit JSON
outputs.  It does not rerun BdG/q/phi scans and does not define a production
Matsubara summation policy.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from ..io.writers import write_json
from .minimal_casimir_n_tail_fit import DEFAULT_QUANTITY_COLUMN

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_n_budget_v1"
TERM_CSV_COLUMNS = [
    "matsubara_index",
    "xi_eV",
    "quantity",
    "source_count",
    "sources",
    "max_Rdiff_over_q",
    "max_range_phi_logdet_abs",
    "all_finite_logdet",
    "all_kappa_match",
]
GAP_CSV_COLUMNS = [
    "left_n",
    "right_n",
    "missing_count",
    "left_quantity",
    "right_quantity",
    "loglog_interpolated_missing_sum_diagnostic",
    "trapezoid_interval_integral_diagnostic",
]
TAIL_CSV_COLUMNS = [
    "source_path",
    "tail_start_n_exclusive",
    "best_model_by_rmse_log_space",
    "best_p",
    "best_r2_log_space",
    "best_rmse_log_space",
    "tail_lower_bound_diagnostic",
    "tail_midpoint_estimate_diagnostic",
    "tail_upper_bound_diagnostic",
]


def _bool_from_csv(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _read_n_scan_csv(path: Path, quantity_column: str) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"matsubara_index", "xi_eV", quantity_column}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"missing required CSV columns in {path}: {missing}; available columns: {fieldnames}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            n = int(raw["matsubara_index"])
            xi = float(raw["xi_eV"])
            y = float(raw[quantity_column])
            if n <= 0 or xi <= 0.0 or y < 0.0:
                raise ValueError("n-budget expects positive n/xi and non-negative quantity values")
            rows.append(
                {
                    "matsubara_index": n,
                    "xi_eV": xi,
                    "quantity": y,
                    "sources": [str(path)],
                    "max_Rdiff_over_q": _optional_float(raw, "max_Rdiff_over_q"),
                    "max_range_phi_logdet_abs": _optional_float(raw, "max_range_phi_logdet_abs"),
                    "all_finite_logdet": _bool_from_csv(raw.get("all_finite_logdet")),
                    "all_kappa_match": _bool_from_csv(raw.get("all_kappa_match")),
                }
            )
    return rows


def _merge_rows(rows: Sequence[dict[str, Any]], *, duplicate_rtol: float = 1e-10, duplicate_atol: float = 1e-14) -> list[dict[str, Any]]:
    by_n: dict[int, dict[str, Any]] = {}
    for row in rows:
        n = int(row["matsubara_index"])
        if n not in by_n:
            by_n[n] = dict(row)
            continue
        current = by_n[n]
        if not math.isclose(float(current["quantity"]), float(row["quantity"]), rel_tol=duplicate_rtol, abs_tol=duplicate_atol):
            raise ValueError(f"conflicting duplicate quantity for n={n}: {current['quantity']} vs {row['quantity']}")
        if not math.isclose(float(current["xi_eV"]), float(row["xi_eV"]), rel_tol=duplicate_rtol, abs_tol=duplicate_atol):
            raise ValueError(f"conflicting duplicate xi_eV for n={n}: {current['xi_eV']} vs {row['xi_eV']}")
        current["sources"] = sorted(set([*current["sources"], *row["sources"]]))
        for key in ("max_Rdiff_over_q", "max_range_phi_logdet_abs"):
            values = [v for v in (current.get(key), row.get(key)) if v is not None]
            current[key] = None if not values else max(float(v) for v in values)
        for key in ("all_finite_logdet", "all_kappa_match"):
            values = [v for v in (current.get(key), row.get(key)) if v is not None]
            current[key] = None if not values else all(bool(v) for v in values)
    merged = list(by_n.values())
    merged.sort(key=lambda row: int(row["matsubara_index"]))
    return merged


def _loglog_interpolate_sum(left: dict[str, Any], right: dict[str, Any]) -> float:
    n_left = int(left["matsubara_index"])
    n_right = int(right["matsubara_index"])
    if n_right <= n_left + 1:
        return 0.0
    y_left = float(left["quantity"])
    y_right = float(right["quantity"])
    if y_left <= 0.0 or y_right <= 0.0:
        return 0.0
    log_n_left = math.log(float(n_left))
    log_n_right = math.log(float(n_right))
    log_y_left = math.log(y_left)
    log_y_right = math.log(y_right)
    slope = (log_y_right - log_y_left) / (log_n_right - log_n_left)
    intercept = log_y_left - slope * log_n_left
    total = 0.0
    for n in range(n_left + 1, n_right):
        total += math.exp(intercept + slope * math.log(float(n)))
    return float(total)


def _gap_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for left, right in zip(rows[:-1], rows[1:], strict=True):
        n_left = int(left["matsubara_index"])
        n_right = int(right["matsubara_index"])
        missing = max(0, n_right - n_left - 1)
        if missing <= 0:
            continue
        y_left = float(left["quantity"])
        y_right = float(right["quantity"])
        gaps.append(
            {
                "left_n": n_left,
                "right_n": n_right,
                "missing_count": missing,
                "left_quantity": y_left,
                "right_quantity": y_right,
                "loglog_interpolated_missing_sum_diagnostic": _loglog_interpolate_sum(left, right),
                "trapezoid_interval_integral_diagnostic": float(0.5 * (y_left + y_right) * (n_right - n_left)),
            }
        )
    return gaps


def _read_tail_fit_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", {})
    return {
        "source_path": str(path),
        "tail_start_n_exclusive": summary.get("tail_start_n_exclusive"),
        "best_model_by_rmse_log_space": summary.get("best_model_by_rmse_log_space"),
        "best_p": summary.get("best_p"),
        "best_r2_log_space": summary.get("best_r2_log_space"),
        "best_rmse_log_space": summary.get("best_rmse_log_space"),
        "tail_lower_bound_diagnostic": summary.get("best_tail_lower_bound_diagnostic"),
        "tail_midpoint_estimate_diagnostic": summary.get("best_tail_midpoint_estimate_diagnostic"),
        "tail_upper_bound_diagnostic": summary.get("best_tail_upper_bound_diagnostic"),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def run_minimal_casimir_n_budget(
    *,
    input_csv_paths: Sequence[Path],
    quantity_column: str = DEFAULT_QUANTITY_COLUMN,
    tail_fit_json_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    if not input_csv_paths:
        raise ValueError("at least one n-scan CSV path is required")
    raw_rows: list[dict[str, Any]] = []
    for path in input_csv_paths:
        raw_rows.extend(_read_n_scan_csv(Path(path), quantity_column))
    rows = _merge_rows(raw_rows)
    gaps = _gap_rows(rows)
    tail_entries = [_read_tail_fit_json(Path(path)) for path in tail_fit_json_paths]

    known_sparse_sum = float(sum(float(row["quantity"]) for row in rows))
    gap_interp_sum = float(sum(float(row["loglog_interpolated_missing_sum_diagnostic"]) for row in gaps))
    interp_sum = float(known_sparse_sum + gap_interp_sum)
    tail_mids = [float(row["tail_midpoint_estimate_diagnostic"]) for row in tail_entries if row.get("tail_midpoint_estimate_diagnostic") is not None]
    tail_lowers = [float(row["tail_lower_bound_diagnostic"]) for row in tail_entries if row.get("tail_lower_bound_diagnostic") is not None]
    tail_uppers = [float(row["tail_upper_bound_diagnostic"]) for row in tail_entries if row.get("tail_upper_bound_diagnostic") is not None]
    total_missing_count = int(sum(int(row["missing_count"]) for row in gaps))
    max_gap = 0 if not gaps else int(max(int(row["missing_count"]) for row in gaps))

    term_rows = [
        {
            **row,
            "source_count": len(row["sources"]),
            "sources": ";".join(row["sources"]),
        }
        for row in rows
    ]
    summary = {
        "num_unique_n": len(rows),
        "min_n": int(rows[0]["matsubara_index"]),
        "max_n": int(rows[-1]["matsubara_index"]),
        "known_sparse_sum_diagnostic": known_sparse_sum,
        "num_gaps": len(gaps),
        "total_missing_integer_n_between_known_points": total_missing_count,
        "max_missing_integer_n_in_single_gap": max_gap,
        "has_missing_dense_integer_sum_warning": total_missing_count > 0,
        "loglog_interpolated_missing_sum_between_known_points_diagnostic": gap_interp_sum,
        "loglog_interpolated_sum_through_max_known_n_diagnostic": interp_sum,
        "num_tail_fit_inputs": len(tail_entries),
        "tail_midpoint_min_diagnostic": None if not tail_mids else float(min(tail_mids)),
        "tail_midpoint_max_diagnostic": None if not tail_mids else float(max(tail_mids)),
        "tail_lower_min_diagnostic": None if not tail_lowers else float(min(tail_lowers)),
        "tail_upper_max_diagnostic": None if not tail_uppers else float(max(tail_uppers)),
        "loglog_plus_tail_midpoint_min_diagnostic": None if not tail_mids else float(interp_sum + min(tail_mids)),
        "loglog_plus_tail_midpoint_max_diagnostic": None if not tail_mids else float(interp_sum + max(tail_mids)),
        "all_finite_logdet_known_terms": all(bool(row.get("all_finite_logdet")) for row in rows if row.get("all_finite_logdet") is not None),
        "all_kappa_match_known_terms": all(bool(row.get("all_kappa_match")) for row in rows if row.get("all_kappa_match") is not None),
        "max_Rdiff_over_known_terms": max([float(row["max_Rdiff_over_q"]) for row in rows if row.get("max_Rdiff_over_q") is not None], default=None),
        "max_range_phi_logdet_abs_over_known_terms": max([float(row["max_range_phi_logdet_abs"]) for row in rows if row.get("max_range_phi_logdet_abs") is not None], default=None),
        "valid_for_casimir_input": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "offline_csv_only": True,
            "does_not_rerun_bdg": True,
            "merges_existing_n_scan_csvs": True,
            "tail_fit_optional": True,
            "known_sparse_sum_not_dense_sum": True,
            "loglog_interpolation_diagnostic_only": True,
            "not_a_matsubara_sum_policy": True,
            "not_a_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "input": {
            "input_csv_paths": [str(Path(path)) for path in input_csv_paths],
            "quantity_column": quantity_column,
            "tail_fit_json_paths": [str(Path(path)) for path in tail_fit_json_paths],
            "valid_for_casimir_input": False,
        },
        "summary": summary,
        "term_rows": term_rows,
        "gap_rows": gaps,
        "tail_fit_rows": tail_entries,
        "interpretation_guardrails": {
            "known_sparse_sum_is_not_integer_dense_sum": True,
            "loglog_interpolation_is_diagnostic_only": True,
            "tail_fit_estimate_is_diagnostic_only": True,
            "n0_policy_not_included": True,
            "missing_integer_n_ranges_require_validation": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def run_and_write_minimal_casimir_n_budget(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_n_budget(**kwargs)
    output = Path(output_dir)
    write_json(output / "minimal_casimir_n_budget.json", payload)
    _write_csv(output / "minimal_casimir_n_budget_terms.csv", payload["term_rows"], TERM_CSV_COLUMNS)
    _write_csv(output / "minimal_casimir_n_budget_gaps.csv", payload["gap_rows"], GAP_CSV_COLUMNS)
    _write_csv(output / "minimal_casimir_n_budget_tail_fits.csv", payload["tail_fit_rows"], TAIL_CSV_COLUMNS)
    return payload
