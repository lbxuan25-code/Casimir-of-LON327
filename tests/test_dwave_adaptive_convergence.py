from __future__ import annotations

import numpy as np
import pytest

from validation.commands.static.dwave_adaptive_convergence import (
    _add_cross_level_metrics,
    parse_level,
)


def test_parse_adaptive_level_defaults_and_explicit_limits():
    default_limits = parse_level("coarse:1e-5:2e-3:50000")
    assert default_limits.label == "coarse"
    assert default_limits.epsabs == 1e-5
    assert default_limits.epsrel == 2e-3
    assert default_limits.max_point_evaluations == 50_000
    assert default_limits.inner_limit == 160
    assert default_limits.outer_limit == 160

    explicit = parse_level("fine:1e-7:1e-4:300000:240:280")
    assert explicit.inner_limit == 240
    assert explicit.outer_limit == 280


@pytest.mark.parametrize(
    "value",
    (
        "missing-fields",
        "bad label:1e-5:1e-3:1000",
        "x:-1:1e-3:1000",
        "x:1e-5:0:1000",
        "x:1e-5:1e-3:0",
    ),
)
def test_parse_adaptive_level_rejects_invalid_contract(value: str):
    with pytest.raises(Exception):
        parse_level(value)


def test_cross_level_metrics_are_computed_separately_per_order():
    rows = [
        {
            "level_index": 1,
            "order": "yx",
            "chi_bar": 2.02,
            "dbar_t": 3.03,
            "phase_defect_over_q": 2e-5,
            "relative_longitudinal_gauge_residual": 4e-5,
        },
        {
            "level_index": 0,
            "order": "xy",
            "chi_bar": 1.0,
            "dbar_t": 2.0,
            "phase_defect_over_q": 1e-3,
            "relative_longitudinal_gauge_residual": 2e-3,
        },
        {
            "level_index": 1,
            "order": "xy",
            "chi_bar": 1.01,
            "dbar_t": 2.02,
            "phase_defect_over_q": 1e-5,
            "relative_longitudinal_gauge_residual": 2e-5,
        },
        {
            "level_index": 0,
            "order": "yx",
            "chi_bar": 2.0,
            "dbar_t": 3.0,
            "phase_defect_over_q": 2e-3,
            "relative_longitudinal_gauge_residual": 4e-3,
        },
    ]
    _add_cross_level_metrics(rows)
    finest = {(row["order"], row["level_index"]): row for row in rows}

    assert np.isnan(finest[("xy", 0)]["chi_bar_relative_to_previous_level"])
    assert np.isnan(finest[("yx", 0)]["chi_bar_relative_to_previous_level"])
    assert finest[("xy", 1)]["chi_bar_relative_to_previous_level"] > 0.0
    assert finest[("yx", 1)]["chi_bar_relative_to_previous_level"] > 0.0
    assert rows == sorted(rows, key=lambda row: (row["level_index"], row["order"]))
