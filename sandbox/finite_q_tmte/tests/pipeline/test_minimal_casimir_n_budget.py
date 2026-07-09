from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_n_budget import (
    SCHEMA_VERSION,
    run_and_write_minimal_casimir_n_budget,
    run_minimal_casimir_n_budget,
)
from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_n_tail_fit import DEFAULT_QUANTITY_COLUMN


def _write_n_scan_csv(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "matsubara_index",
                "xi_eV",
                DEFAULT_QUANTITY_COLUMN,
                "max_Rdiff_over_q",
                "max_range_phi_logdet_abs",
                "all_finite_logdet",
                "all_kappa_match",
            ],
        )
        writer.writeheader()
        for n, y in rows:
            writer.writerow(
                {
                    "matsubara_index": n,
                    "xi_eV": 0.01 * n,
                    DEFAULT_QUANTITY_COLUMN: y,
                    "max_Rdiff_over_q": 0.1 / n,
                    "max_range_phi_logdet_abs": 0.01 / n,
                    "all_finite_logdet": True,
                    "all_kappa_match": True,
                }
            )


def _write_tail_fit_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "tail_start_n_exclusive": 5,
                    "best_model_by_rmse_log_space": "power_n",
                    "best_p": 3.0,
                    "best_r2_log_space": 1.0,
                    "best_rmse_log_space": 0.0,
                    "best_tail_lower_bound_diagnostic": 0.01,
                    "best_tail_midpoint_estimate_diagnostic": 0.02,
                    "best_tail_upper_bound_diagnostic": 0.03,
                }
            }
        ),
        encoding="utf-8",
    )


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_n_budget.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_n_budget_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_n_budget_merges_csvs_and_tail_fit(tmp_path):
    csv1 = tmp_path / "scan1.csv"
    csv2 = tmp_path / "scan2.csv"
    tail = tmp_path / "tail.json"
    _write_n_scan_csv(csv1, [(1, 1.0), (2, 0.5), (5, 0.1)])
    _write_n_scan_csv(csv2, [(5, 0.1), (10, 0.01)])
    _write_tail_fit_json(tail)

    payload = run_minimal_casimir_n_budget(input_csv_paths=[csv1, csv2], tail_fit_json_paths=[tail])
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"]["offline_csv_only"] is True
    assert payload["status"]["does_not_rerun_bdg"] is True
    assert payload["summary"]["num_unique_n"] == 4
    assert payload["summary"]["total_missing_integer_n_between_known_points"] == 6
    assert payload["summary"]["has_missing_dense_integer_sum_warning"] is True
    assert payload["summary"]["tail_midpoint_min_diagnostic"] == pytest.approx(0.02)
    assert payload["summary"]["loglog_plus_tail_midpoint_min_diagnostic"] is not None
    assert payload["valid_for_casimir_input"] is False


def test_n_budget_rejects_conflicting_duplicate(tmp_path):
    csv1 = tmp_path / "scan1.csv"
    csv2 = tmp_path / "scan2.csv"
    _write_n_scan_csv(csv1, [(5, 0.1)])
    _write_n_scan_csv(csv2, [(5, 0.2)])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        run_minimal_casimir_n_budget(input_csv_paths=[csv1, csv2])


def test_n_budget_write_outputs(tmp_path):
    csv1 = tmp_path / "scan1.csv"
    _write_n_scan_csv(csv1, [(1, 1.0), (3, 0.2)])
    out = tmp_path / "budget"
    payload = run_and_write_minimal_casimir_n_budget(out, input_csv_paths=[csv1])
    assert payload["summary"]["num_gaps"] == 1
    assert (out / "minimal_casimir_n_budget.json").exists()
    assert (out / "minimal_casimir_n_budget_terms.csv").exists()
    assert (out / "minimal_casimir_n_budget_gaps.csv").exists()
    assert (out / "minimal_casimir_n_budget_tail_fits.csv").exists()


def test_n_budget_cli_parse_args(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--input-csv", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
        "--tail-fit-json", str(tmp_path / "tail.json"),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert len(args.input_csv_paths) == 2
    assert len(args.tail_fit_json_paths) == 1
