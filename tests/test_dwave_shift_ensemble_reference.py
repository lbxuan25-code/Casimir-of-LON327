from __future__ import annotations

import numpy as np

from validation.lib.dwave_shift_ensemble_reference import (
    annotate_drift,
    cross_rule_metrics,
    reference_status,
    run_ensemble_task,
)


def _task(nk: int = 2, rule: str = "gauss2") -> dict:
    return {
        "nk": nk,
        "rule": rule,
        "qx": 0.03,
        "qy": 0.02,
        "temperature_K": 10.0,
        "delta0_eV": 0.1,
        "eta_eV": 1e-8,
        "ward_tolerance": 1e-7,
        "ward_absolute_tolerance": 1e-12,
        "condition_max": 1e12,
        "raw_longitudinal_ceiling": 1e-3,
        "longitudinal_tolerance": 1e-7,
        "mixing_tolerance": 1e-7,
        "reality_tolerance": 1e-9,
        "passivity_tolerance": 1e-10,
        "separation_nm": 20.0,
    }


def test_four_shift_ensemble_worker_smoke():
    row = run_ensemble_task(_task())
    assert row["rule"] == "gauss2"
    assert row["nk"] == 2
    assert row["num_shifts"] == 4
    assert row["num_quadrature_points"] == 16
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
    assert np.isfinite(row["ward_primitive_mixed_ratio_max"])
    assert row["schur_inverse_method"] == "inv"


def test_cross_rule_metrics_are_relative_to_secondary():
    primary = {"chi_bar": 1.01, "dbar_t": 1.98, "raw_longitudinal": 0.11}
    secondary = {"chi_bar": 1.0, "dbar_t": 2.0, "raw_longitudinal": 0.1}
    result = cross_rule_metrics(primary, secondary)
    assert np.isclose(result["relative_chi_cross_rule"], 0.01)
    assert np.isclose(result["relative_dbar_cross_rule"], 0.01)
    assert np.isclose(result["relative_raw_longitudinal_cross_rule"], 0.1)


def test_screening_and_reference_status_are_distinct():
    rows = []
    for nk, chi, dbar in ((100, 1.0, 2.0), (120, 1.002, 2.004), (140, 1.004, 2.008)):
        rows.append(
            {
                "nk": nk,
                "chi_bar": chi,
                "dbar_t": dbar,
                "raw_longitudinal": 0.01,
                "ward_passed": True,
                "schur_inverse_method": "inv",
                "ward_primitive_mixed_ratio_max": 1e-4,
                "ward_effective_mixed_ratio_max": 1e-4,
                "projection_eligible": False,
            }
        )
    annotate_drift(rows)
    secondary = {
        **rows[-1],
        "chi_bar": rows[-1]["chi_bar"] * 1.004,
        "dbar_t": rows[-1]["dbar_t"] * 1.004,
    }
    fits = {
        "chi_bar": {"relative_spread": 5e-3},
        "dbar_t": {"relative_spread": 5e-3},
    }
    status = reference_status(
        rows,
        secondary,
        fits,
        screening_drift=5e-3,
        screening_cross=1e-2,
        drift=1e-3,
        fit_spread=2e-3,
        cross_rule=2e-3,
    )
    assert status["ensemble_screening_promising"]
    assert not status["numerical_reference_converged"]
    assert not status["valid_for_casimir_input"]
