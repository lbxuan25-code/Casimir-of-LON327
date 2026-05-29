import subprocess
import sys
from pathlib import Path

import numpy as np

from lno327.finite_q_response import bdg_finite_q_response_imag_axis, finite_q_response_phi_scan

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_finite_q_response_anisotropy.py"


def test_q0_routine_runs_and_local_limit_is_close():
    result = bdg_finite_q_response_imag_axis(
        "dwave",
        matsubara_index=1,
        temperature_K=30.0,
        q_magnitude=0.0,
        q_phi=0.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
    )

    assert np.isfinite(result.local_limit_relative_error)
    assert result.local_limit_relative_error < 1e-8
    assert result.diagnostic_status == "pass_local_limit"


def test_q_phi_scan_runs():
    results = finite_q_response_phi_scan(
        "spm",
        matsubara_index=1,
        temperature_K=30.0,
        q_magnitude=0.05,
        q_phi_list=[0.0, np.pi / 4.0, np.pi / 2.0],
        nk=6,
        delta0=0.04,
        eta=1e-4,
    )

    assert len(results) == 3
    assert all(np.isfinite(result.response_tensor_model).all() for result in results)


def test_normal_spm_dwave_all_run():
    for kind in ["normal", "spm", "dwave"]:
        result = bdg_finite_q_response_imag_axis(
            kind,
            matsubara_index=1,
            temperature_K=30.0,
            q_magnitude=0.02,
            q_phi=0.0,
            nk=6,
            delta0=0.04,
            eta=1e-4,
        )
        assert result.kind == kind
        assert np.isfinite(result.response_tensor_model).all()


def test_unit_interface_fields_and_flags_exist():
    result = bdg_finite_q_response_imag_axis(
        "normal",
        matsubara_index=1,
        temperature_K=30.0,
        q_magnitude=0.02,
        q_phi=0.0,
        nk=6,
        delta0=0.04,
        eta=1e-4,
    )

    assert result.sheet_conductivity_SI.shape == (2, 2)
    assert result.reflection_dimensionless.shape == (2, 2)
    assert result.finite_q_response_diagnostic
    assert not result.final_casimir_input
    assert result.gauge_status == "prototype_not_ward_verified"
    assert result.not_final_Casimir_conclusion


def test_quick_script_outputs_fields(tmp_path):
    output_prefix = tmp_path / "finite_q_anisotropy"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--kinds",
            "normal",
            "spm",
            "dwave",
            "--matsubara-list",
            "1",
            "--q-list",
            "0",
            "0.05",
            "--q-phi-list",
            "0",
            "0.7853981634",
            "--nk",
            "6",
            "--output-prefix",
            str(output_prefix),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    required = {
        "kind",
        "matsubara_index",
        "q_magnitude",
        "q_phi",
        "response_xx",
        "sheet_xx_SI",
        "reflection_xx",
        "finite_q_response_diagnostic",
        "final_casimir_input",
        "not_final_Casimir_conclusion",
        "gauge_status",
        "diagnostic_status",
        "angular_anisotropy_A4_xx",
        "pairing_contrast_dwave_minus_spm",
    }
    with np.load(output_prefix.with_suffix(".npz"), allow_pickle=True) as data:
        assert required.issubset(data.files)
        assert np.all(data["finite_q_response_diagnostic"])
        assert not np.any(data["final_casimir_input"])
        assert set(data["gauge_status"]) == {"prototype_not_ward_verified"}
