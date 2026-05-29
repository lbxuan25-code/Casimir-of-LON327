import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_finite_q_local_limit_decomposition.py"
SUMMARY = ROOT / "outputs" / "response" / "finite_q_local_limit" / "finite_q_local_limit_summary.md"


def test_quick_mode_generates_outputs(tmp_path):
    output_prefix = tmp_path / "finite_q_local_limit"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert output_prefix.with_suffix(".csv").exists()
    assert output_prefix.with_suffix(".npz").exists()
    assert SUMMARY.exists()


def test_output_fields_and_kinds(tmp_path):
    output_prefix = tmp_path / "finite_q_local_limit"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    required = {
        "kind",
        "matsubara_index",
        "omega_eV",
        "nk",
        "small_q",
        "q_phi",
        "component",
        "finite_q_xx",
        "local_sigma_xx",
        "local_K_para_xx",
        "local_K_dia_xx",
        "local_K_total_xx",
        "local_K_total_over_omega_xx",
        "normal_kubo_sigma_xx",
        "error_to_local_sigma",
        "error_to_K_para",
        "error_to_K_total",
        "error_to_K_total_over_omega",
        "error_to_normal_kubo_sigma",
        "best_match_component",
        "best_match_relative_error",
        "error_monotonic_in_q",
        "diagnostic_status",
        "gauge_status",
        "final_casimir_input",
        "not_final_Casimir_conclusion",
    }
    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert required.issubset(data.files)
        assert set(data["kind"]) == {"normal", "spm", "dwave"}
        assert data["best_match_component"].size > 0


def test_errors_are_finite_or_reasonable_nan(tmp_path):
    output_prefix = tmp_path / "finite_q_local_limit"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert np.all(np.isfinite(data["error_to_local_sigma"]))
        assert np.all(np.isfinite(data["best_match_relative_error"]))
        normal = data["kind"] == "normal"
        bdg = ~normal
        assert np.all(np.isnan(data["error_to_K_para"][normal]))
        assert np.all(np.isnan(data["error_to_normal_kubo_sigma"][bdg]))


def test_flags_are_correct_and_script_does_not_import_casimir(tmp_path):
    output_prefix = tmp_path / "finite_q_local_limit"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--output-prefix", str(output_prefix)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert set(data["gauge_status"]) == {"prototype_not_ward_verified"}
        assert not np.any(data["final_casimir_input"])
        assert np.all(data["not_final_Casimir_conclusion"])
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert "import lno327.casimir" not in script_text
    assert "from lno327 import casimir" not in script_text
    assert "from lno327.casimir" not in script_text
