from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_health_report import (
    SCHEMA_VERSION,
    run_and_write_minimal_casimir_health_report,
    run_minimal_casimir_health_report,
)


def _load_cli():
    path = Path("sandbox/finite_q_tmte/scripts/debug_minimal_casimir_health_report.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_minimal_casimir_health_report_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_health_report_passes_clean_summary(tmp_path):
    path = tmp_path / "clean.json"
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "all_finite_logdet": True,
                    "all_kappa_match": True,
                    "max_Rdiff_over_q": 0.1,
                    "max_range_phi_logdet_abs": 1e-6,
                },
                "valid_for_casimir_input": False,
            }
        ),
        encoding="utf-8",
    )
    payload = run_minimal_casimir_health_report(input_json_paths=[path])
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"]["health_status"] == "pass"
    assert payload["summary"]["num_pass"] == 1
    assert payload["valid_for_casimir_input"] is False


def test_health_report_flags_reflection_norm_pathology_from_csv(tmp_path):
    path = tmp_path / "phi.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["phi_mod_deg", "Rdiff", "R1_norm", "R2_norm", "range_phi_logdet_abs", "kappa_match"],
        )
        writer.writeheader()
        writer.writerow({"phi_mod_deg": 15, "Rdiff": 21.0, "R1_norm": 4.7, "R2_norm": 25.8, "range_phi_logdet_abs": 0.025, "kappa_match": True})
    payload = run_minimal_casimir_health_report(input_csv_paths=[path])
    assert payload["summary"]["health_status"] == "needs_review"
    assert payload["summary"]["num_needs_review"] == 1
    assert "reflection_norm_pathology" in payload["findings"][0]["classification"]
    assert "valid_for_casimir_input" in payload


def test_health_report_fails_nonfinite_or_kappa_false(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"summary": {"all_finite_logdet": False, "all_kappa_match": False}}),
        encoding="utf-8",
    )
    payload = run_minimal_casimir_health_report(input_json_paths=[path])
    assert payload["summary"]["health_status"] == "fail"
    assert payload["summary"]["num_fail"] == 1
    assert "nonfinite_result" in payload["findings"][0]["classification"]
    assert "kappa_mismatch" in payload["findings"][0]["classification"]


def test_health_report_write_outputs(tmp_path):
    path = tmp_path / "clean.json"
    path.write_text(json.dumps({"summary": {"all_finite_logdet": True, "all_kappa_match": True}}), encoding="utf-8")
    out = tmp_path / "out"
    payload = run_and_write_minimal_casimir_health_report(out, input_json_paths=[path])
    assert payload["summary"]["health_status"] == "pass"
    assert (out / "minimal_casimir_health_report.json").exists()
    assert (out / "minimal_casimir_health_report_findings.csv").exists()
    assert (out / "minimal_casimir_health_report.md").exists()


def test_health_report_requires_input():
    with pytest.raises(ValueError, match="provide at least one"):
        run_minimal_casimir_health_report()


def test_health_report_cli_parse_args(tmp_path):
    module = _load_cli()
    parser = module.build_parser()
    args = parser.parse_args([
        "--input-json", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
        "--input-csv", str(tmp_path / "c.csv"),
        "--r-norm-warning-threshold", "3.0",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert len(args.input_json_paths) == 2
    assert len(args.input_csv_paths) == 1
    assert args.r_norm_warning_threshold == 3.0
