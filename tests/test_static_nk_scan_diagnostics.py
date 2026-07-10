from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from validation.run_static_nk_scan import (
    _longitudinal_component_diagnostics,
    _run_one,
    _ward_side_diagnostics,
)


def test_longitudinal_component_decomposition_reproduces_aggregate_norm():
    kernel = np.zeros((3, 3), dtype=complex)
    kernel[0, 1] = 3.0 + 4.0j
    kernel[1, 0] = 2.0
    kernel[1, 1] = 1.0j
    kernel[1, 2] = 6.0
    kernel[2, 1] = 8.0

    diagnostics = _longitudinal_component_diagnostics(kernel, 1.0)
    expected = np.linalg.norm(
        [
            diagnostics["relative_k0l"],
            diagnostics["relative_kl0"],
            diagnostics["relative_kll"],
            diagnostics["relative_klt"],
            diagnostics["relative_ktl"],
        ]
    )
    assert np.isclose(diagnostics["longitudinal_components_relative_norm"], expected)
    assert diagnostics["dominant_longitudinal_component"] == "KTL"


def test_ward_side_diagnostics_reports_rhs_projection_cancellation():
    side = SimpleNamespace(
        primitive_rhs=np.asarray([3.0, 0.0, 0.0], dtype=complex),
        collective_projection=np.asarray([2.0, 0.0, 0.0], dtype=complex),
        effective_direct=np.asarray([1.0, 0.0, 0.0], dtype=complex),
        effective_predicted=np.asarray([1.0, 0.0, 0.0], dtype=complex),
        effective_residual=np.zeros(3, dtype=complex),
    )
    diagnostics = _ward_side_diagnostics(side, "ward_left")
    assert diagnostics["ward_left_rhs_norm"] == 3.0
    assert diagnostics["ward_left_collective_projection_norm"] == 2.0
    assert np.isclose(
        diagnostics["ward_left_rhs_projection_cancellation_ratio"],
        1.0 / 3.0,
    )
    assert diagnostics["ward_left_direct_prediction_relative_residual"] == 0.0


def test_static_scan_row_contains_resolved_longitudinal_and_ward_fields():
    row = _run_one(
        {
            "nk": 2,
            "pairing": "spm",
            "qx": 0.03,
            "qy": 0.02,
            "temperature_K": 10.0,
            "delta0_eV": 0.1,
            "eta_eV": 1e-8,
            "ward_tolerance": 1e-7,
        }
    )

    for field in (
        "relative_k0l",
        "relative_kl0",
        "relative_kll",
        "relative_klt",
        "relative_ktl",
        "dominant_longitudinal_component",
        "ward_left_rhs_norm",
        "ward_left_collective_projection_norm",
        "ward_left_rhs_projection_cancellation_ratio",
        "ward_right_rhs_norm",
        "ward_right_collective_projection_norm",
        "ward_right_rhs_projection_cancellation_ratio",
    ):
        assert field in row

    assert np.isclose(
        row["longitudinal_components_relative_norm"],
        row["relative_longitudinal_gauge_residual"],
    )
