from __future__ import annotations

import numpy as np
import pytest

from lno327.workflows.dwave_periodic_shift_ensemble import (
    DWavePeriodicShiftEnsembleOptions,
    build_dwave_periodic_shift_ensemble,
    merge_shift_components_before_schur,
    nested_c4_antithetic_shifts,
    periodic_shift_mesh,
)
from validation.lib.dwave_shift_batch import ShiftBatchConfig, evaluate_one_shift, merge_prefix


def _config() -> ShiftBatchConfig:
    return ShiftBatchConfig(
        base_nk=2,
        qx=0.03,
        qy=0.02,
        temperature_K=10.0,
        delta0_eV=0.1,
        eta_eV=1e-8,
        ward_tolerance=1e-6,
        ward_absolute_tolerance=1e-12,
        condition_max=1e12,
        raw_longitudinal_ceiling=10.0,
        longitudinal_tolerance=1e-7,
        mixing_tolerance=10.0,
        reality_tolerance=10.0,
        passivity_tolerance=10.0,
        separation_nm=20.0,
    )


def test_nested_shifts_are_unique_and_c4_antithetic_by_orbit():
    shifts = nested_c4_antithetic_shifts(16)
    assert shifts.shape == (16, 2)
    assert np.all(shifts >= 0.0)
    assert np.all(shifts < 1.0)
    assert len({tuple(np.round(value, 14)) for value in shifts}) == 16
    for start in range(0, 16, 4):
        x, y = shifts[start]
        expected = np.asarray(
            [
                [x, y],
                [y, x],
                [(1.0 - x) % 1.0, (1.0 - y) % 1.0],
                [(1.0 - y) % 1.0, (1.0 - x) % 1.0],
            ]
        )
        assert np.allclose(shifts[start : start + 4], expected, rtol=0.0, atol=1e-14)


def test_shift_ensemble_budget_and_complete_mesh_contract():
    shifts, metadata = build_dwave_periodic_shift_ensemble(
        np.asarray([0.03, 0.02]),
        DWavePeriodicShiftEnsembleOptions(
            base_nk=3,
            max_shifts=8,
            max_quadrature_points=100,
        ),
    )
    assert shifts.shape == (8, 2)
    assert metadata["nested_prefixes"] == [4, 8]
    assert metadata["full_periodic_lattice_per_shift"] is True
    points, weights = periodic_shift_mesh(3, shifts[0])
    assert points.shape == (9, 2)
    assert np.allclose(weights, np.full(9, 1.0 / 9.0))
    assert np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-15)
    with pytest.raises(RuntimeError, match="max_quadrature_points"):
        build_dwave_periodic_shift_ensemble(
            np.asarray([0.03, 0.02]),
            DWavePeriodicShiftEnsembleOptions(
                base_nk=4,
                max_shifts=8,
                max_quadrature_points=127,
            ),
        )


def test_identical_cached_shift_primitives_merge_before_one_schur():
    config = _config()
    result = evaluate_one_shift(config, 0, np.asarray([0.5, 1.0 / 3.0]))
    workspace = result["workspace"]
    components, rhs = merge_shift_components_before_schur(
        [result["components"], result["components"]],
        [result["rhs"], result["rhs"]],
        [1.0, 1.0],
        workspace,
        omega_eV=0.0,
    )
    assert np.allclose(
        components.amplitude_phase_schur,
        result["components"].amplitude_phase_schur,
        rtol=1e-11,
        atol=1e-12,
    )
    assert np.allclose(rhs.left, result["rhs"].left, rtol=1e-12, atol=1e-13)
    assert components.metadata["shift_ensemble_merged_before_schur"] is True
    assert components.metadata["num_shift_components"] == 2


def test_cached_four_shift_prefix_reports_full_static_diagnostics():
    config = _config()
    shifts = nested_c4_antithetic_shifts(4)
    results = [evaluate_one_shift(config, index, shift) for index, shift in enumerate(shifts)]
    workspace = results[0]["workspace"]
    row = merge_prefix(results, 4, workspace, config)
    assert np.isfinite(row["chi_bar"])
    assert np.isfinite(row["dbar_t"])
    assert row["schur_inverse_method"] == "inv"
    assert "ward_primitive_mixed_ratio_max" in row
    assert "projection_eligible" in row
    assert "logdet_passed" in row
