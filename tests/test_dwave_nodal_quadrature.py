from __future__ import annotations

import numpy as np
import pytest

from lno327.workflows.dwave_nodal_quadrature import (
    DWaveNodalQuadratureOptions,
    build_dwave_nodal_quadrature,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _model_inputs(pairing: str = "dwave"):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing, phase_vertex="bond_endpoint_gauge")
    amplitudes = model.build_pairing_params(0.1)
    return model, ansatz, amplitudes


def test_dwave_nodal_quadrature_is_normalized_and_refines_flagged_cells():
    model, ansatz, amplitudes = _model_inputs()
    options = DWaveNodalQuadratureOptions(
        coarse_grid=2,
        adaptive_level=1,
        gauss_order=2,
        sample_order=2,
        quasiparticle_window_eV=1.0,
        normal_window_eV=1.0,
        gap_window_eV=1.0,
        transition_window_eV=1.0,
        transition_shell_eV=2.0,
        max_quadrature_points=1_000,
    )
    points, weights, metadata = build_dwave_nodal_quadrature(
        model.spec,
        ansatz,
        amplitudes,
        np.array([0.03, 0.02]),
        options,
    )

    assert points.ndim == 2 and points.shape[1] == 2
    assert weights.shape == (len(points),)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2e-12)
    assert np.all(points >= -np.pi)
    assert np.all(points < np.pi)
    assert metadata["integration_strategy"] == "two_band_dwave_nodal_adaptive"
    assert metadata["primitive_merge_before_schur_required"] is True
    assert metadata["parent_child_double_counting"] is False
    assert metadata["num_final_cells"] > metadata["num_base_cells"]
    assert metadata["refinement_history"][0]["num_cells_flagged"] > 0


def test_dwave_nodal_quadrature_rejects_non_dwave_ansatz():
    model, ansatz, amplitudes = _model_inputs("spm")
    with pytest.raises(ValueError, match="ansatz.name == 'dwave'"):
        build_dwave_nodal_quadrature(
            model.spec,
            ansatz,
            amplitudes,
            np.array([0.03, 0.02]),
            DWaveNodalQuadratureOptions(coarse_grid=2, adaptive_level=0, gauss_order=1),
        )
