from __future__ import annotations

import numpy as np
import pytest

from lno327.response.finite_q import BdGFiniteQResponseComponents
from validation.commands.ward.bond_metric_full_kernel import (
    complementary_subgrid_origins,
)
from validation.lib.dwave_bond_phase_counterterm import (
    apply_nearest_neighbor_dwave_phase_counterterm,
    nearest_neighbor_dwave_bond_metric,
)


def _components() -> BdGFiniteQResponseComponents:
    bare_total = np.asarray(
        [
            [2.0, 0.1, 0.0],
            [0.2, 1.7, 0.3],
            [0.0, 0.4, 1.2],
        ],
        dtype=complex,
    )
    left = np.asarray(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=complex,
    )
    right = np.asarray(
        [
            [0.2, 0.1, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=complex,
    )
    collective_bubble = np.asarray([[1.5, 0.1], [0.2, -2.5]], dtype=complex)
    counterterm = np.asarray([[4.0, 0.0], [0.0, 5.0]], dtype=complex)
    collective_total = collective_bubble + counterterm
    schur = bare_total - left @ np.linalg.inv(collective_total) @ right
    zero3 = np.zeros(3, dtype=complex)
    return BdGFiniteQResponseComponents(
        bare_bubble=bare_total.copy(),
        direct=np.zeros((3, 3), dtype=complex),
        bare_total=bare_total,
        phase_coupling_left=zero3.copy(),
        phase_coupling_right=zero3.copy(),
        phase_phase_bubble=0.0 + 0.0j,
        phase_phase_direct=0.0 + 0.0j,
        phase_phase_total=0.0 + 0.0j,
        minus_schur=bare_total.copy(),
        plus_schur=bare_total.copy(),
        collective_bubble=collective_bubble,
        collective_counterterm=counterterm,
        collective_total=collective_total,
        em_collective_left=left,
        collective_em_right=right,
        amplitude_phase_schur=schur,
        gauge_restored=schur.copy(),
        metadata={
            "collective_mode": "amplitude_phase",
            "selected_gauge_restored": "amplitude_phase_schur",
            "phase_correction_applied": True,
            "model_input_layer": {
                "name": "dwave",
                "phase_vertex": "bond_endpoint_gauge",
            },
            "valid_for_casimir_input": False,
        },
    )


def test_nearest_neighbor_bond_metric_is_exact_and_normalized():
    assert nearest_neighbor_dwave_bond_metric(np.zeros(2)) == pytest.approx(1.0)
    q = np.asarray([0.3, -0.4])
    expected = 0.5 * (np.cos(0.15) ** 2 + np.cos(0.2) ** 2)
    assert nearest_neighbor_dwave_bond_metric(q) == pytest.approx(expected)


def test_bond_metric_changes_only_phase_counterterm_and_rebuilds_schur():
    base = _components()
    q = np.asarray([0.3, 0.4])
    corrected, application = apply_nearest_neighbor_dwave_phase_counterterm(base, q)
    metric = nearest_neighbor_dwave_bond_metric(q)

    assert corrected.collective_counterterm[0, 0] == base.collective_counterterm[0, 0]
    assert corrected.collective_counterterm[0, 1] == base.collective_counterterm[0, 1]
    assert corrected.collective_counterterm[1, 0] == base.collective_counterterm[1, 0]
    assert corrected.collective_counterterm[1, 1] == pytest.approx(
        metric * base.collective_counterterm[1, 1]
    )
    expected_total = base.collective_bubble + corrected.collective_counterterm
    expected_schur = (
        base.bare_total
        - base.em_collective_left
        @ np.linalg.inv(expected_total)
        @ base.collective_em_right
    )
    np.testing.assert_allclose(corrected.collective_total, expected_total)
    np.testing.assert_allclose(corrected.amplitude_phase_schur, expected_schur)
    np.testing.assert_allclose(corrected.gauge_restored, expected_schur)
    assert application.multiplier == pytest.approx(metric)
    assert corrected.metadata["diagnostic_phase_counterterm_changed_only_22"] is True
    assert corrected.metadata["valid_for_casimir_input"] is False
    assert corrected.metadata["projection_applied"] is False


def test_bond_metric_rejects_non_dwave_metadata():
    base = _components()
    base.metadata["model_input_layer"]["name"] = "spm"
    with pytest.raises(ValueError, match="only for the d-wave"):
        apply_nearest_neighbor_dwave_phase_counterterm(base, np.asarray([0.1, 0.2]))


def test_complementary_origins_follow_odd_integer_components():
    assert complementary_subgrid_origins(2, 4, 0.5, 0.5) == ((0.5, 0.5),)
    assert complementary_subgrid_origins(3, 2, 0.5, 0.5) == (
        (0.5, 0.5),
        (0.0, 0.5),
    )
    assert complementary_subgrid_origins(3, 1, 0.5, 0.5) == (
        (0.5, 0.5),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.0, 0.0),
    )
