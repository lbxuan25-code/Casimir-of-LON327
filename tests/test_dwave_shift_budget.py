from __future__ import annotations

import numpy as np
import pytest

from validation.lib.dwave_shift_budget import (
    allocation_metrics,
    parse_allocations,
    run_budget_task,
)


def _task(*, shifts: int = 4, nk: int = 4) -> dict[str, object]:
    return {
        "num_shifts": shifts,
        "nk": nk,
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


def test_parse_allocations_sorts_and_validates():
    assert parse_allocations(["16:120", "4:240", "8:170"]) == [
        (4, 240),
        (8, 170),
        (16, 120),
    ]
    with pytest.raises(ValueError):
        parse_allocations(["3:10", "4:10"])
    with pytest.raises(ValueError):
        parse_allocations(["4:10"])


def test_equal_budget_real_four_shift_smoke():
    row = run_budget_task(_task())
    assert row["num_shifts"] == 4
    assert row["base_nk"] == 4
    assert row["num_quadrature_points"] == 64
    assert row["shift_family"] == "nested_halton_bases_2_3_c4_antithetic"
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
    assert np.isfinite(row["ward_primitive_mixed_ratio_max"])
    assert row["schur_inverse_method"] == "inv"


def test_allocation_metrics_distinguish_screening_from_reference():
    rows = [
        {
            "num_shifts": 4,
            "chi_bar": 1.00,
            "dbar_t": 2.00,
            "raw_longitudinal": 2e-3,
            "ward_passed": True,
            "schur_inverse_method": "inv",
            "ward_primitive_mixed_ratio_max": 0.1,
            "ward_effective_mixed_ratio_max": 0.1,
        },
        {
            "num_shifts": 8,
            "chi_bar": 1.004,
            "dbar_t": 2.008,
            "raw_longitudinal": 1.5e-3,
            "ward_passed": True,
            "schur_inverse_method": "inv",
            "ward_primitive_mixed_ratio_max": 0.1,
            "ward_effective_mixed_ratio_max": 0.1,
        },
        {
            "num_shifts": 16,
            "chi_bar": 1.005,
            "dbar_t": 2.010,
            "raw_longitudinal": 9e-4,
            "ward_passed": True,
            "schur_inverse_method": "inv",
            "ward_primitive_mixed_ratio_max": 0.1,
            "ward_effective_mixed_ratio_max": 0.1,
        },
    ]
    metrics = allocation_metrics(rows, agreement_tolerance=0.02)
    assert metrics["equal_budget_agreement"]
    assert metrics["more_shifts_reduce_transition"]
    assert metrics["more_shifts_reduce_raw_longitudinal"]
    assert metrics["allocation_preference"] == "more_shifts"
    assert metrics["production_reference_established"] is False
