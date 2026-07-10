from __future__ import annotations

import pickle

import numpy as np
import pytest

from lno327.workflows.dwave_vector_adaptive_cubature import (
    DWaveCubatureCell,
    DWaveVectorAdaptiveOptions,
    cubature_cell_gauss_rule,
    initial_cubature_cells,
    merge_cell_components_before_schur,
    primitive_component_vector,
    subdivide_cubature_cell,
    validate_vector_adaptive_options,
    vector_error_metrics,
)
from validation.lib.dwave_vector_adaptive import (
    VectorAdaptiveConfig,
    aggregate_cubature_cells,
    choose_refinement_indices,
    evaluate_cubature_cell,
    portable_cubature_cell_result,
    restore_portable_cubature_cell_result,
)


def _config() -> VectorAdaptiveConfig:
    return VectorAdaptiveConfig(
        low_order=1,
        high_order=2,
        qx=0.03,
        qy=0.02,
        temperature_K=10.0,
        delta0_eV=0.1,
        eta_eV=1e-8,
        relative_tolerance=1e-2,
        absolute_tolerance=1e-8,
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


def test_cell_gauss_rule_and_children_preserve_full_bz_measure():
    parent = DWaveCubatureCell(-np.pi, np.pi, -np.pi, np.pi)
    points, weights = cubature_cell_gauss_rule(parent, 3)
    assert points.shape == (9, 2)
    assert np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2e-15)
    children = subdivide_cubature_cell(parent)
    assert len(children) == 4
    assert all(child.level == 1 for child in children)
    assert np.isclose(sum(child.area_fraction for child in children), 1.0)
    child_weight = sum(np.sum(cubature_cell_gauss_rule(child, 2)[1]) for child in children)
    assert np.isclose(child_weight, 1.0, rtol=0.0, atol=2e-15)


def test_option_validation_and_initial_partition():
    options = DWaveVectorAdaptiveOptions(coarse_grid=3, low_order=2, high_order=3)
    validate_vector_adaptive_options(options)
    cells = initial_cubature_cells(3)
    assert len(cells) == 9
    assert np.isclose(sum(cell.area_fraction for cell in cells), 1.0)
    with pytest.raises(ValueError, match="high_order"):
        validate_vector_adaptive_options(
            DWaveVectorAdaptiveOptions(low_order=3, high_order=3)
        )


def test_vector_error_metrics_prioritize_largest_complete_primitive_error():
    low = [np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0])]
    high = [np.asarray([1.001, 2.0]), np.asarray([1.2, 2.0])]
    metrics = vector_error_metrics(
        low,
        high,
        relative_tolerance=1e-2,
        absolute_tolerance=1e-9,
    )
    assert metrics["cell_scores"].shape == (2,)
    assert metrics["cell_scores"][1] > metrics["cell_scores"][0]
    assert metrics["conservative_error_ratio_max"] >= metrics["signed_error_ratio_max"]


def test_single_full_bz_cell_merge_reproduces_high_rule_and_is_pickle_safe():
    config = _config()
    cell = DWaveCubatureCell(-np.pi, np.pi, -np.pi, np.pi)
    result = evaluate_cubature_cell(config, cell)
    workspace = result["workspace"]
    merged, rhs = merge_cell_components_before_schur(
        [result["high_components"]],
        [result["high_rhs"]],
        workspace,
        omega_eV=0.0,
    )
    assert np.allclose(
        merged.amplitude_phase_schur,
        result["high_components"].amplitude_phase_schur,
        rtol=1e-11,
        atol=1e-12,
    )
    assert np.allclose(rhs.left, result["high_rhs"].left, rtol=1e-12, atol=1e-13)
    assert merged.metadata["vector_adaptive_cells_merged_before_schur"] is True
    assert primitive_component_vector(merged, rhs).ndim == 1

    portable = portable_cubature_cell_result(result)
    restored = restore_portable_cubature_cell_result(pickle.loads(pickle.dumps(portable)))
    assert np.allclose(
        restored["high_components"].bare_bubble,
        result["high_components"].bare_bubble,
    )
    assert restored["cell"] == cell


def test_small_cell_set_aggregates_and_budget_selection_is_fail_closed():
    config = _config()
    cells = initial_cubature_cells(2)
    results = [evaluate_cubature_cell(config, cell) for cell in cells]
    template = results[0]["workspace"]
    physical, scores, errors = aggregate_cubature_cells(results, template, config)
    assert np.isfinite(physical["chi_bar"])
    assert np.isfinite(physical["dbar_t"])
    assert scores.shape == (4,)
    assert np.isfinite(errors["conservative_error_ratio_max"])

    selected = choose_refinement_indices(
        results,
        scores,
        refine_fraction=1.0,
        min_refine_cells=1,
        max_level=3,
        max_cells=7,
        remaining_evaluation_points=4 * 5,
        points_per_child=5,
    )
    assert len(selected) == 1
    blocked = choose_refinement_indices(
        results,
        scores,
        refine_fraction=1.0,
        min_refine_cells=1,
        max_level=3,
        max_cells=7,
        remaining_evaluation_points=19,
        points_per_child=5,
    )
    assert blocked == []
