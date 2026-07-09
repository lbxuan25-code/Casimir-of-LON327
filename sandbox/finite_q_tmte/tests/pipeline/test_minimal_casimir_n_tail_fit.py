from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_n_tail_fit import (
    DEFAULT_QUANTITY_COLUMN,
    SCHEMA_VERSION,
    run_and_write_minimal_casimir_n_tail_fit,
    run_minimal_casimir_n_tail_fit,
)


def _write_synthetic_n_scan_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in [100, 150, 200, 300, 500]:
        rows.append({
            "matsubara_index": n,
            "xi_eV": 0.01 * n,
            DEFAULT_QUANTITY_COLUMN: 2.0 * n**-3,
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["matsubara_index", "xi_eV", DEFAULT_QUANTITY_COLUMN])
        writer.writeheader()
        writer.writerows(rows)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_n_tail_fit.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_n_tail_fit_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tail_fit_recovers_power_law(tmp_path):
    csv_path = tmp_path / "minimal_casimir_n_scan.csv"
    _write_synthetic_n_scan_csv(csv_path)
    payload = run_minimal_casimir_n_tail_fit(input_csv_path=csv_path)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"]["offline_csv_only"] is True
    assert payload["status"]["does_not_rerun_bdg"] is True
    assert payload["valid_for_casimir_input"] is False
    assert payload["summary"]["best_p"] == pytest.approx(3.0, rel=1e-12)
    assert payload["summary"]["best_r2_log_space"] == pytest.approx(1.0)
    assert payload["summary"]["best_tail_midpoint_estimate_diagnostic"] is not None
    assert len(payload["fit_summaries"]) == 2
    assert len(payload["residual_rows"]) == 10


def test_tail_fit_filter_and_write_outputs(tmp_path):
    csv_path = tmp_path / "minimal_casimir_n_scan.csv"
    _write_synthetic_n_scan_csv(csv_path)
    out_dir = tmp_path / "tail_fit"
    payload = run_and_write_minimal_casimir_n_tail_fit(
        out_dir,
        input_csv_path=csv_path,
        models=("power_n",),
        fit_min_n=150,
        tail_start_n_exclusive=500,
    )
    assert payload["input"]["num_fit_rows"] == 4
    assert (out_dir / "minimal_casimir_n_tail_fit.json").exists()
    assert (out_dir / "minimal_casimir_n_tail_fit_summary.csv").exists()
    assert (out_dir / "minimal_casimir_n_tail_fit_residuals.csv").exists()


def test_tail_fit_rejects_missing_quantity_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("matsubara_index,xi_eV,other\n100,1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required CSV columns"):
        run_minimal_casimir_n_tail_fit(input_csv_path=csv_path)


def test_tail_fit_cli_parse_args(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--input-csv", str(tmp_path / "in.csv"),
        "--models", "power_n",
        "--fit-min-n", "100",
        "--tail-start-n-exclusive", "500",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert args.models == ["power_n"]
    assert args.fit_min_n == 100
    assert args.tail_start_n_exclusive == 500
