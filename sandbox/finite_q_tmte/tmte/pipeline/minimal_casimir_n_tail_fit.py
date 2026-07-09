"""Offline diagnostic tail fits for minimal Casimir n-scan CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io.writers import write_json

SCHEMA_VERSION = "finite_q_tmte_minimal_casimir_n_tail_fit_v1"
DEFAULT_QUANTITY_COLUMN = "q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic"
SUPPORTED_MODELS = ("power_n", "power_xi")
FIT_CSV_COLUMNS = [
    "model",
    "x_name",
    "quantity_column",
    "num_fit_points",
    "n_min_fit",
    "n_max_fit",
    "A",
    "p",
    "log_A",
    "r2_log_space",
    "rmse_log_space",
    "max_abs_relative_error",
    "tail_start_n_exclusive",
    "tail_lower_bound_diagnostic",
    "tail_midpoint_estimate_diagnostic",
    "tail_upper_bound_diagnostic",
    "tail_convergent",
]
RESIDUAL_CSV_COLUMNS = [
    "model",
    "matsubara_index",
    "xi_eV",
    "x_value",
    "observed",
    "fitted",
    "residual",
    "relative_error",
]


def _read_rows(path: Path, quantity_column: str) -> list[dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"matsubara_index", "xi_eV", quantity_column}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"missing required CSV columns: {missing}; available columns: {fieldnames}")
        rows = []
        for raw in reader:
            n = int(raw["matsubara_index"])
            xi = float(raw["xi_eV"])
            y = float(raw[quantity_column])
            if n <= 0 or xi <= 0.0 or y <= 0.0:
                raise ValueError("tail fit requires positive n, xi_eV, and quantity values")
            rows.append({"matsubara_index": float(n), "xi_eV": xi, "quantity": y})
    if not rows:
        raise ValueError("input n-scan CSV contains no rows")
    rows.sort(key=lambda row: int(row["matsubara_index"]))
    return rows


def _filter_rows(rows: Sequence[dict[str, float]], fit_min_n: int | None, fit_max_n: int | None) -> list[dict[str, float]]:
    out = []
    for row in rows:
        n = int(row["matsubara_index"])
        if fit_min_n is not None and n < int(fit_min_n):
            continue
        if fit_max_n is not None and n > int(fit_max_n):
            continue
        out.append(dict(row))
    if len(out) < 2:
        raise ValueError("tail fit requires at least two rows after filtering")
    return out


def _fit_log_power(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    fitted_log = intercept + slope * log_x
    residual_log = log_y - fitted_log
    ss_res = float(np.sum(residual_log**2))
    ss_tot = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    return {
        "log_A": float(intercept),
        "A": float(np.exp(intercept)),
        "p": float(-slope),
        "r2_log_space": 1.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
        "rmse_log_space": float(np.sqrt(np.mean(residual_log**2))),
    }


def _equivalent_n_amplitude(model: str, A: float, p: float, rows: Sequence[dict[str, float]]) -> float:
    if model == "power_n":
        return float(A)
    if model == "power_xi":
        xi_per_n = float(np.mean([row["xi_eV"] / row["matsubara_index"] for row in rows]))
        return float(A / (xi_per_n**p))
    raise ValueError(f"unsupported model: {model}")


def _tail_bounds(amplitude_n: float, p: float, start_n_exclusive: int) -> dict[str, Any]:
    n0 = int(start_n_exclusive)
    if p <= 1.0:
        return {
            "tail_convergent": False,
            "tail_lower_bound_diagnostic": None,
            "tail_midpoint_estimate_diagnostic": None,
            "tail_upper_bound_diagnostic": None,
        }
    lower = float(amplitude_n * ((n0 + 1) ** (1.0 - p)) / (p - 1.0))
    upper = float(amplitude_n * (n0 ** (1.0 - p)) / (p - 1.0))
    return {
        "tail_convergent": True,
        "tail_lower_bound_diagnostic": lower,
        "tail_midpoint_estimate_diagnostic": float(0.5 * (lower + upper)),
        "tail_upper_bound_diagnostic": upper,
    }


def _fit_model(model: str, rows: Sequence[dict[str, float]], quantity_column: str, tail_start_n_exclusive: int) -> dict[str, Any]:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model!r}; supported models: {SUPPORTED_MODELS}")
    if model == "power_n":
        x_name = "matsubara_index"
        x = np.asarray([row["matsubara_index"] for row in rows], dtype=float)
    else:
        x_name = "xi_eV"
        x = np.asarray([row["xi_eV"] for row in rows], dtype=float)
    y = np.asarray([row["quantity"] for row in rows], dtype=float)
    fit = _fit_log_power(x, y)
    fitted = fit["A"] * np.power(x, -fit["p"])
    residuals = y - fitted
    rel_errors = residuals / y
    tail = _tail_bounds(
        _equivalent_n_amplitude(model, fit["A"], fit["p"], rows),
        fit["p"],
        tail_start_n_exclusive,
    )
    summary = {
        "model": model,
        "x_name": x_name,
        "quantity_column": quantity_column,
        "num_fit_points": len(rows),
        "n_min_fit": int(min(row["matsubara_index"] for row in rows)),
        "n_max_fit": int(max(row["matsubara_index"] for row in rows)),
        "A": fit["A"],
        "p": fit["p"],
        "log_A": fit["log_A"],
        "r2_log_space": fit["r2_log_space"],
        "rmse_log_space": fit["rmse_log_space"],
        "max_abs_relative_error": float(np.max(np.abs(rel_errors))),
        "tail_start_n_exclusive": int(tail_start_n_exclusive),
        **tail,
        "valid_for_casimir_input": False,
    }
    residual_rows = []
    for row, x_value, observed, fit_value, residual, rel in zip(rows, x, y, fitted, residuals, rel_errors, strict=True):
        residual_rows.append(
            {
                "model": model,
                "matsubara_index": int(row["matsubara_index"]),
                "xi_eV": float(row["xi_eV"]),
                "x_value": float(x_value),
                "observed": float(observed),
                "fitted": float(fit_value),
                "residual": float(residual),
                "relative_error": float(rel),
            }
        )
    return {"summary": summary, "residual_rows": residual_rows}


def run_minimal_casimir_n_tail_fit(
    *,
    input_csv_path: Path,
    quantity_column: str = DEFAULT_QUANTITY_COLUMN,
    models: Sequence[str] = SUPPORTED_MODELS,
    fit_min_n: int | None = None,
    fit_max_n: int | None = None,
    tail_start_n_exclusive: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_csv_path)
    rows_all = _read_rows(input_path, quantity_column)
    rows_fit = _filter_rows(rows_all, fit_min_n, fit_max_n)
    tail_start = int(tail_start_n_exclusive) if tail_start_n_exclusive is not None else int(max(row["matsubara_index"] for row in rows_fit))
    if tail_start <= 0:
        raise ValueError("tail_start_n_exclusive must be positive")
    model_results = [_fit_model(model, rows_fit, quantity_column, tail_start) for model in models]
    best = min(model_results, key=lambda result: float(result["summary"]["rmse_log_space"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": True,
            "diagnostic_only_not_a_fix": True,
            "offline_csv_only": True,
            "does_not_rerun_bdg": True,
            "power_law_tail_fit_only": True,
            "tail_estimate_uses_integral_bounds": True,
            "not_a_matsubara_sum_policy": True,
            "not_a_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "input": {
            "input_csv_path": str(input_path),
            "quantity_column": quantity_column,
            "models": list(models),
            "fit_min_n": None if fit_min_n is None else int(fit_min_n),
            "fit_max_n": None if fit_max_n is None else int(fit_max_n),
            "tail_start_n_exclusive": tail_start,
            "num_input_rows": len(rows_all),
            "num_fit_rows": len(rows_fit),
            "valid_for_casimir_input": False,
        },
        "summary": {
            "best_model_by_rmse_log_space": best["summary"]["model"],
            "best_p": best["summary"]["p"],
            "best_rmse_log_space": best["summary"]["rmse_log_space"],
            "best_r2_log_space": best["summary"]["r2_log_space"],
            "best_tail_midpoint_estimate_diagnostic": best["summary"]["tail_midpoint_estimate_diagnostic"],
            "best_tail_lower_bound_diagnostic": best["summary"]["tail_lower_bound_diagnostic"],
            "best_tail_upper_bound_diagnostic": best["summary"]["tail_upper_bound_diagnostic"],
            "tail_start_n_exclusive": tail_start,
            "valid_for_casimir_input": False,
        },
        "fit_summaries": [result["summary"] for result in model_results],
        "residual_rows": [row for result in model_results for row in result["residual_rows"]],
        "interpretation_guardrails": {
            "fits_sparse_diagnostic_points_only": True,
            "tail_bounds_assume_monotone_power_law_beyond_tail_start": True,
            "tail_estimate_is_not_a_validated_policy": True,
            "n0_policy_not_included": True,
            "does_not_modify_main_validation": True,
            "does_not_modify_main_casimir_pipeline": True,
            "not_a_full_casimir_energy": True,
            "not_a_torque_calculation": True,
            "valid_for_casimir_input": False,
        },
        "valid_for_casimir_input": False,
    }


def write_tail_fit_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIT_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIT_CSV_COLUMNS})


def write_tail_fit_residuals_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESIDUAL_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in RESIDUAL_CSV_COLUMNS})


def run_and_write_minimal_casimir_n_tail_fit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_minimal_casimir_n_tail_fit(**kwargs)
    output = Path(output_dir)
    write_json(output / "minimal_casimir_n_tail_fit.json", payload)
    write_tail_fit_summary_csv(output / "minimal_casimir_n_tail_fit_summary.csv", payload["fit_summaries"])
    write_tail_fit_residuals_csv(output / "minimal_casimir_n_tail_fit_residuals.csv", payload["residual_rows"])
    return payload
