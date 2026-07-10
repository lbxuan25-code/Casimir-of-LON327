from __future__ import annotations

import numpy as np

from validation.run_static_projection_scan import _run_task


def test_focused_static_projection_scan_reports_raw_and_projected_contracts():
    row = _run_task(
        {
            "nk": 2,
            "pairing": "spm",
            "qx": 0.03,
            "qy": 0.02,
            "temperature_K": 10.0,
            "delta0_eV": 0.1,
            "eta_eV": 1e-8,
            "ward_tolerance": 1e-6,
            "raw_longitudinal_ceiling": 10.0,
            "longitudinal_tolerance": 1e-7,
            "mixing_tolerance": 10.0,
            "reality_tolerance": 10.0,
            "passivity_tolerance": 10.0,
        }
    )

    assert row["nk"] == 2
    assert row["num_k_points"] == 4
    assert row["ward_passed"] is True
    assert row["ward_condition_ok"] is True
    assert row["projection_eligible"] is True
    assert row["projection_applied"] is True
    assert row["projected_static_validation_passed"] is True
    assert row["projected_relative_longitudinal_gauge_residual"] == 0.0
    assert row["relative_projection_correction_norm"] >= 0.0
    assert row["chi_bar_projection_delta"] == 0.0
    assert row["dbar_t_projection_delta"] == 0.0
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
