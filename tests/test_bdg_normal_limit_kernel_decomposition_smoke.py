import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "validation"
    / "scripts"
    / "numerical_stability"
    / "diagnose_bdg_normal_limit_kernel_decomposition.py"
)


def test_bdg_normal_limit_kernel_decomposition_quick_outputs(tmp_path):
    output_prefix = tmp_path / "bdg_normal_limit_kernel_decomposition"
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

    assert "BdG normal-limit kernel decomposition diagnostic only" in result.stdout
    csv_path = output_prefix.with_suffix(".csv")
    npz_path = output_prefix.with_suffix(".npz")
    summary_path = output_prefix.parent / "bdg_normal_limit_kernel_decomposition_summary.md"
    figure_dir = output_prefix.parent / "figures"
    assert csv_path.exists()
    assert npz_path.exists()
    assert summary_path.exists()
    assert (figure_dir / "bdg_vs_normal_K_para_xx_vs_omega.png").exists()
    assert (figure_dir / "bdg_vs_normal_K_dia_xx_vs_omega.png").exists()
    assert (figure_dir / "bdg_vs_normal_K_total_xx_vs_omega.png").exists()
    assert (figure_dir / "para_ratio_xx_vs_omega.png").exists()
    assert (figure_dir / "dia_ratio_xx_vs_omega.png").exists()
    assert (figure_dir / "relative_error_vs_omega.png").exists()

    required = {
        "kind",
        "delta0_eV",
        "omega_eV",
        "bdg_K_para_xx",
        "bdg_K_dia_xx",
        "bdg_K_total_xx",
        "normal_K_para_xx",
        "normal_K_dia_xx",
        "normal_K_total_xx",
        "para_ratio_xx",
        "dia_ratio_xx",
        "total_ratio_xx",
        "para_relative_error",
        "dia_relative_error",
        "total_relative_error",
        "benchmark_only",
        "local_response",
        "normal_limit_decomposition_diagnostic",
        "not_final_response_formula",
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
        assert np.allclose(data["delta0_eV"], 0.0)
        for key in [
            "bdg_norm_para",
            "bdg_norm_dia",
            "bdg_norm_total",
            "normal_norm_para",
            "normal_norm_dia",
            "normal_norm_total",
            "para_relative_error",
            "dia_relative_error",
            "total_relative_error",
        ]:
            assert np.all(np.isfinite(data[key]))
        assert np.all(data["benchmark_only"])
        assert np.all(data["local_response"])
        assert np.all(data["normal_limit_decomposition_diagnostic"])
        assert np.all(data["not_final_response_formula"])
        assert np.all(data["not_final_optical_conductivity"])
        assert np.all(data["not_final_Casimir_input"])

    summary = summary_path.read_text(encoding="utf-8")
    assert "not a final response" in summary
    assert "does not modify Casimir" in summary
    assert "contains no finite momentum response" in summary
