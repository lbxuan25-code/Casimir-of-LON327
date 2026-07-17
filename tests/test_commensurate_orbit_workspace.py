from __future__ import annotations

import numpy as np
import pytest

from validation.lib.commensurate_orbit_workspace import (
    CompleteOrbitAggregateWorkspace,
    TransverseEvaluationBudgetExceeded,
)


def test_complete_orbit_workspace_caches_periodic_phases_and_enforces_budget() -> None:
    seen_shapes: list[tuple[int, int]] = []

    def evaluator(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        seen_shapes.append(points.shape)
        assert np.sum(weights) == pytest.approx(1.0)
        return np.asarray([np.sum(weights)], dtype=complex)

    workspace = CompleteOrbitAggregateWorkspace(
        evaluator=evaluator,
        nk=8,
        mx=1,
        my=1,
        max_unique_transverse_evaluations=1,
    )

    assert len(workspace.orbit_origins) == 2
    assert workspace.points_per_t == 2 * workspace.nk

    first = workspace.evaluate_phase(0.125)
    cached = workspace.evaluate_phase(1.125)
    np.testing.assert_array_equal(first, cached)

    assert workspace.transverse_evaluations_unique == 1
    assert workspace.cache_hits == 1
    assert workspace.point_evaluations == workspace.points_per_t
    assert seen_shapes == [(workspace.points_per_t, 2)]

    with pytest.raises(TransverseEvaluationBudgetExceeded) as caught:
        workspace.evaluate_phase(0.375)

    assert caught.value.maximum == 1
    assert caught.value.attempted == 2
