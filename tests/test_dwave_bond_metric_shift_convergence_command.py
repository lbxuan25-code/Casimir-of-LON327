from __future__ import annotations

import numpy as np

from validation.__main__ import resolve_command
from validation.commands.static.bond_metric_shift_convergence import _run_one


def test_shift_convergence_small_grid_merges_primitives_before_schur():
    task = {
        "nk": 4,
        "shift_counts": [4],
        "max_quadrature_points": 64,
        "qx": 0.11,
        "qy": 0.07,
        "temperature_K": 10.0,
        "delta0_eV": 0.1,
        "eta_eV": 1e-8,
        "mixed_ward_tolerance": 1.0,
        "mixed_ward_absolute_tolerance": 1.0,
        "primitive_tolerance": 1.0,
        "amplitude_tolerance": 1.0,
        "phase_tolerance": 1.0,
        "effective_direct_tolerance": 1.0,
        "effective_residual_tolerance": 1.0,
        "longitudinal_tolerance": 1.0,
        "condition_max": 1e12,
    }
    rows = _run_one(task)
    assert len(rows) == 1
    row = rows[0]
    assert row["nk"] == 4
    assert row["shift_count"] == 4
    assert row["total_quadrature_points"] == 64
    assert row["primitive_merged_before_schur"] is True
    assert row["ward_rhs_merged_before_validation"] is True
    assert row["phase_hessian_policy"] == "nearest_neighbor_bond_metric"
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
    assert row["diagnostic_only"] is True
    assert row["valid_for_casimir_input"] is False


def test_shift_convergence_public_route():
    assert resolve_command("static", "bond-metric-shift-convergence").endswith(
        "bond_metric_shift_convergence"
    )
