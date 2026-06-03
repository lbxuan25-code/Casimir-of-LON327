import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "scripts" / "numerical_stability" / "diagnose_bdg_static_gauge_closure.py"


def test_bdg_static_gauge_closure_quick_outputs(tmp_path):
    output_prefix = tmp_path / "bdg_static_gauge_closure"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--scan-kernel-conventions",
            "--output-prefix",
            str(output_prefix),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "BdG static stiffness diagnostic only" in result.stdout
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")
    summary_path = output_prefix.parent / "bdg_static_gauge_closure_summary.md"
    figure_dir = output_prefix.parent / "figures"
    assert csv_path.exists()
    assert npz_path.exists()
    assert summary_path.exists()
    assert (figure_dir / "gauge_residual_vs_delta0.png").exists()
    assert (figure_dir / "rho_s_xx_yy_vs_delta0.png").exists()
    assert (figure_dir / "rho_s_anisotropy_vs_delta0.png").exists()
    assert (figure_dir / "offdiag_ratio_vs_delta0.png").exists()

    convention_csv = ROOT / "validation" / "outputs" / "response" / "bdg_static_gauge_closure" / "data" / "bdg_static_gauge_convention_scan.csv"
    convention_npz = convention_csv.with_suffix(".npz")
    convention_summary = ROOT / "validation" / "outputs" / "response" / "bdg_static_gauge_closure" / "bdg_static_gauge_convention_scan_summary.md"
    convention_figure = ROOT / "validation" / "outputs" / "response" / "bdg_static_gauge_closure" / "figures" / "convention_scan_gauge_residual.png"
    assert convention_csv.exists()
    assert convention_npz.exists()
    assert convention_summary.exists()
    assert convention_figure.exists()

    required = {
        "kind",
        "delta0_eV",
        "omega_eV",
        "K_para_xx",
        "K_dia_xx",
        "K_total_xx",
        "norm_para",
        "norm_dia",
        "norm_total",
        "stiffness_norm",
        "stiffness_reference_mismatch",
        "gauge_residual",
        "rho_s_xx",
        "rho_s_yy",
        "rho_s_anisotropy",
        "offdiag_ratio",
        "benchmark_only",
        "local_response",
        "static_gauge_closure_diagnostic",
        "not_final_optical_conductivity",
        "not_final_Casimir_input",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert required.issubset(reader.fieldnames or [])
        rows = list(reader)
        assert rows

    with np.load(npz_path, allow_pickle=True) as data:
        assert required.issubset(data.files)
        for key in ["norm_para", "norm_dia", "norm_total", "stiffness_norm", "gauge_residual", "rho_s_xx", "rho_s_yy"]:
            assert np.all(np.isfinite(data[key]))
        zero_mask = np.isclose(data["delta0_eV"], 0.0)
        assert np.all(np.isfinite(data["stiffness_reference_mismatch"][zero_mask]))
        assert np.all(data["benchmark_only"])
        assert np.all(data["local_response"])
        assert np.all(data["static_gauge_closure_diagnostic"])
        assert np.all(data["not_final_optical_conductivity"])
        assert np.all(data["not_final_Casimir_input"])

    summary = summary_path.read_text(encoding="utf-8")
    assert "K_total means K_dia - K_para" in summary
    assert "not a final optical conductivity" in summary
    assert "not a final Casimir input" in summary

    convention_required = {
        "convention_name",
        "para_prefactor",
        "dia_prefactor",
        "K_candidate_xx",
        "norm_candidate",
        "candidate_gauge_residual",
        "rho_s_anisotropy_candidate",
        "offdiag_ratio_candidate",
        "convention_scan_diagnostic",
        "not_final_response_formula",
        "not_final_optical_conductivity",
        "not_final_Casimir_input",
    }
    with convention_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert convention_required.issubset(reader.fieldnames or [])
        assert list(reader)
    with np.load(convention_npz, allow_pickle=True) as convention_data:
        assert convention_required.issubset(convention_data.files)
        assert np.all(np.isfinite(convention_data["candidate_gauge_residual"]))
        assert np.all(convention_data["convention_scan_diagnostic"])
        assert np.all(convention_data["not_final_response_formula"])
    convention_text = convention_summary.read_text(encoding="utf-8")
    assert "not a formal response formula change" in convention_text
    assert "best_convention=" in convention_text
