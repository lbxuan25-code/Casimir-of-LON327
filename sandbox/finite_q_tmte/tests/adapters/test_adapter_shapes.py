from __future__ import annotations

import numpy as np

import pytest

from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from sandbox.finite_q_tmte.tmte.adapters.bubble_adapter import compute_target_bare_blocks
from sandbox.finite_q_tmte.tmte.adapters.model_adapter import build_model_scan_inputs


def test_direct_target_adapter_shapes_nk1():
    inputs = build_model_scan_inputs(model_name="symmetry_bdg_2band", pairing_name="dwave", xi=0.01, nk=1)
    blocks = compute_target_bare_blocks(
        spec=inputs.spec,
        ansatz=inputs.ansatz,
        q_model=np.asarray([0.02, 0.0]),
        xi=0.01,
        k_points=inputs.k_points,
        weights=inputs.weights,
        config=inputs.config,
        pairing_params=inputs.pairing_params,
        options=FiniteQEngineOptions(include_phase_correction=False, collective_mode="amplitude_phase"),
    )
    assert blocks.k_ss.shape == (3, 3)
    assert blocks.k_seta.shape == (3, 2)
    assert blocks.k_etas.shape == (2, 3)
    assert blocks.k_etaeta.shape == (2, 2)
    assert blocks.metadata["valid_for_casimir_input"] is False


def test_xi_equal_omega_is_accepted():
    inputs = build_model_scan_inputs(model_name="symmetry_bdg_2band", pairing_name="dwave", xi=0.01, omega_eV=0.01, nk=1)
    assert float(inputs.config.omega_eV) == 0.01


def test_xi_not_equal_omega_raises():
    with pytest.raises(ValueError, match="xi == omega_eV"):
        build_model_scan_inputs(model_name="symmetry_bdg_2band", pairing_name="dwave", xi=0.01, omega_eV=0.02, nk=1)
