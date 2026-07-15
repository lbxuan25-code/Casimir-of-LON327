from __future__ import annotations

import numpy as np

from lno327.workflows.dwave_vector_adaptive_cubature import vector_error_metrics


def test_global_cell_cancellation_does_not_block_convergence() -> None:
    low = (np.zeros(1, dtype=complex), np.zeros(1, dtype=complex))
    high = (
        np.asarray([1.0 + 0.0j]),
        np.asarray([-1.0 + 0.0j]),
    )
    low_ward = (np.zeros(2, dtype=complex), np.zeros(2, dtype=complex))
    high_ward = (
        np.asarray([1.0 + 0.0j, -2.0 + 0.0j]),
        np.asarray([-1.0 + 0.0j, 2.0 + 0.0j]),
    )

    metrics = vector_error_metrics(
        low,
        high,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-9,
        low_ward_vectors=low_ward,
        high_ward_vectors=high_ward,
        ward_threshold=1e-9,
    )

    assert metrics["error_estimator_contract"] == "global_signed_group_mixed_v3"
    assert metrics["cell_score_contract"] == "primitive_local_group_mixed_only_v1"
    assert metrics["global_group_error_ratio_max"] == 0.0
    assert metrics["ward_global_error_ratio"] == 0.0
    assert metrics["local_absolute_error_ratio_max"] > 1.0
    assert metrics["ward_local_absolute_error_ratio"] > 1.0
    assert np.max(metrics["cell_scores"]) > 1.0
    assert np.max(metrics["ward_cell_scores"]) > 1.0


def test_local_ward_error_is_diagnostic_only_for_cell_ranking() -> None:
    low = (np.zeros(1, dtype=complex), np.zeros(1, dtype=complex))
    high = (np.zeros(1, dtype=complex), np.zeros(1, dtype=complex))
    low_ward = (np.zeros(2, dtype=complex), np.zeros(2, dtype=complex))
    high_ward = (
        np.asarray([1.0 + 0.0j, -2.0 + 0.0j]),
        np.asarray([-1.0 + 0.0j, 2.0 + 0.0j]),
    )

    metrics = vector_error_metrics(
        low,
        high,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-9,
        low_ward_vectors=low_ward,
        high_ward_vectors=high_ward,
        ward_threshold=1e-9,
    )

    np.testing.assert_array_equal(metrics["cell_scores"], np.zeros(2))
    assert np.max(metrics["ward_cell_scores"]) > 1.0
    assert metrics["ward_global_error_ratio"] == 0.0


def test_true_global_low_high_difference_remains_a_hard_failure() -> None:
    low = (np.zeros(1, dtype=complex), np.zeros(1, dtype=complex))
    high = (
        np.asarray([1.0 + 0.0j]),
        np.asarray([1.0 + 0.0j]),
    )

    metrics = vector_error_metrics(
        low,
        high,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-9,
    )

    assert metrics["global_group_error_ratio_max"] > 1.0
    assert metrics["conservative_error_ratio_max"] == metrics[
        "global_group_error_ratio_max"
    ]


def test_packed_physical_blocks_use_shared_mixed_scale() -> None:
    # One-frequency packed primitive width is 18 + 25 = 43. The first nine
    # entries are one direct-response block. A small component change should be
    # judged against the scale of that physical block, not against its own nearly
    # zero component denominator.
    low = np.zeros(43, dtype=complex)
    high = np.zeros(43, dtype=complex)
    low[0] = 1.0
    high[0] = 1.0
    high[1] = 1e-6

    metrics = vector_error_metrics(
        (low,),
        (high,),
        relative_tolerance=1e-3,
        absolute_tolerance=1e-9,
    )

    assert metrics["global_group_error_ratio_max"] < 1.0
    assert len(metrics["group_error_ratios"]) == 8


def test_global_ward_difference_uses_mixed_absolute_relative_scale() -> None:
    low = (np.zeros(1, dtype=complex),)
    high = (np.zeros(1, dtype=complex),)
    low_ward = (np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),)
    high_ward = (np.asarray([1.0 + 1e-6, 0.0 + 0.0j]),)

    metrics = vector_error_metrics(
        low,
        high,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-9,
        low_ward_vectors=low_ward,
        high_ward_vectors=high_ward,
        ward_threshold=1e-9,
    )

    assert metrics["ward_global_error_ratio"] < 1.0
