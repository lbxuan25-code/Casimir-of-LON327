import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "archive" / "finite_q_diagnostics" / "diagnose_finite_q_formula_consistency.py"
SUMMARY = ROOT / "outputs" / "archive" / "response" / "finite_q_formula_consistency" / "finite_q_formula_consistency_summary.md"


def test_formula_consistency_quick_runs(tmp_path):
    output_prefix = tmp_path / "finite_q_formula_consistency"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert output_prefix.with_suffix(".csv").exists()
    assert output_prefix.with_suffix(".npz").exists()
    assert SUMMARY.exists()


def test_formula_consistency_fields_are_present(tmp_path):
    output_prefix = tmp_path / "finite_q_formula_consistency"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    required = {
        "vertex_local_limit_relative_error",
        "overlap_unitarity_error",
        "overlap_diagonal_error",
        "overlap_offdiag_norm",
        "error_to_local_sigma",
        "error_to_K_para",
        "error_to_K_total",
        "error_to_K_total_over_omega",
        "best_match_component",
        "gauge_status",
        "final_casimir_input",
        "not_final_Casimir_conclusion",
    }
    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert required.issubset(data.files)
        assert np.all(np.isfinite(data["vertex_local_limit_relative_error"]))
        assert np.all(np.isfinite(data["overlap_unitarity_error"]))
        assert np.all(np.isfinite(data["overlap_diagonal_error"]))
        assert set(data["kind"]) == {"normal", "spm", "dwave"}


def test_formula_consistency_flags_are_correct(tmp_path):
    output_prefix = tmp_path / "finite_q_formula_consistency"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert set(data["gauge_status"]) == {"prototype_not_ward_verified"}
        assert not np.any(data["final_casimir_input"])
        assert np.all(data["not_final_Casimir_conclusion"])
