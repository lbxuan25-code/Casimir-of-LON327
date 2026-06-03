import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "numerical_stability" / "diagnose_normal_static_gauge_closure.py"


def test_normal_static_gauge_closure_quick_outputs(tmp_path):
    output_prefix = tmp_path / "normal_static_gauge_closure"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--output-prefix",
            str(output_prefix),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "normal-state static gauge closure diagnostic only" in result.stdout
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")
    summary_path = output_prefix.parent / "normal_static_gauge_closure_summary.md"
    figure_dir = output_prefix.parent / "figures"
    assert csv_path.exists()
    assert npz_path.exists()
    assert summary_path.exists()
    assert (figure_dir / "D_fd_xx_vs_nk.png").exists()
    assert (figure_dir / "candidate_K_xx_vs_nk.png").exists()
    assert (figure_dir / "candidate_error_vs_nk.png").exists()
    assert (figure_dir / "intra_inter_dia_decomposition_vs_nk.png").exists()
    assert (figure_dir / "best_candidate_error_vs_omega.png").exists()

    required = {
        "omega_eV",
        "nk",
        "twist_A",
        "K_para_intra_xx",
        "K_para_inter_xx",
        "K_para_total_xx",
        "K_dia_xx",
        "D_fd_xx",
        "D_fd_yy",
        "D_fd_xy",
        "para_plus_dia_xx",
        "minus_para_plus_dia_xx",
        "para_minus_dia_xx",
        "minus_para_minus_dia_xx",
        "best_candidate_name",
        "best_candidate_error",
        "relative_error_to_fd",
        "benchmark_only",
        "local_response",
        "normal_static_gauge_closure_diagnostic",
        "peierls_twist_diagnostic",
        "not_final_response_formula",
        "not_final_optical_conductivity",
        "not_final_Casimir_input",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert required.issubset(reader.fieldnames or [])
        assert list(reader)

    with np.load(npz_path, allow_pickle=True) as data:
        assert required.issubset(data.files)
        for key in [
            "D_fd_xx",
            "D_fd_yy",
            "K_para_intra_xx",
            "K_para_inter_xx",
            "K_dia_xx",
            "best_candidate_error",
            "relative_error_to_fd",
        ]:
            assert np.all(np.isfinite(data[key]))
        assert np.all(data["benchmark_only"])
        assert np.all(data["local_response"])
        assert np.all(data["normal_static_gauge_closure_diagnostic"])
        assert np.all(data["peierls_twist_diagnostic"])
        assert np.all(data["not_final_response_formula"])
        assert np.all(data["not_final_optical_conductivity"])
        assert np.all(data["not_final_Casimir_input"])

    summary = summary_path.read_text(encoding="utf-8")
    assert "Peierls-twist finite-difference" in summary
    assert "does not modify BdG, Casimir, reflection, or finite-q code" in summary
    assert "not a final optical conductivity" in summary
