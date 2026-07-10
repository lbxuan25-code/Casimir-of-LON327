from __future__ import annotations

import numpy as np
import pytest

from lno327.response.effective_kernel import (
    AMPLITUDE_PHASE_ORDER,
    PRIMITIVE_EM_BASIS,
    PRIMITIVE_EM_ORDER,
    EffectiveEMKernel,
    effective_em_kernel_from_components,
)
from lno327.response.finite_q import BdGFiniteQResponseComponents


def _components(*, selected: str = "amplitude_phase_schur") -> BdGFiniteQResponseComponents:
    bare_bubble = np.diag([1.0, 2.0, 3.0]).astype(complex)
    direct = np.diag([0.0, 0.2, 0.3]).astype(complex)
    bare_total = bare_bubble + direct
    k_seta = np.array(
        [
            [0.2, 0.1j],
            [0.3, 0.2j],
            [0.1, -0.15j],
        ],
        dtype=complex,
    )
    k_etas = np.array(
        [
            [0.2, 0.3, 0.1],
            [-0.1j, -0.2j, 0.15j],
        ],
        dtype=complex,
    )
    k_etaeta = np.array([[2.0, 0.05j], [-0.05j, 1.5]], dtype=complex)
    k_eff = bare_total - k_seta @ np.linalg.solve(k_etaeta, k_etas)
    gauge_restored = k_eff.copy() if selected == "amplitude_phase_schur" else bare_total.copy()
    return BdGFiniteQResponseComponents(
        bare_bubble=bare_bubble,
        direct=direct,
        bare_total=bare_total,
        phase_coupling_left=np.zeros(3, dtype=complex),
        phase_coupling_right=np.zeros(3, dtype=complex),
        phase_phase_bubble=0.0 + 0.0j,
        phase_phase_direct=0.0 + 0.0j,
        phase_phase_total=1.0 + 0.0j,
        minus_schur=bare_total.copy(),
        plus_schur=bare_total.copy(),
        collective_bubble=k_etaeta.copy(),
        collective_counterterm=np.zeros((2, 2), dtype=complex),
        collective_total=k_etaeta,
        em_collective_left=k_seta,
        collective_em_right=k_etas,
        amplitude_phase_schur=k_eff,
        gauge_restored=gauge_restored,
        metadata={
            "collective_mode": "amplitude_phase",
            "selected_gauge_restored": selected,
            "phase_correction_applied": selected == "amplitude_phase_schur",
            "collective_total_condition_number": float(np.linalg.cond(k_etaeta)),
            "collective_inverse_method": "solve",
        },
    )


def test_effective_kernel_extracts_existing_primitive_blocks_without_recomputation():
    components = _components()
    kernel = effective_em_kernel_from_components(
        components,
        q_model=np.array([0.03, -0.02]),
        xi_eV=0.01,
    )

    np.testing.assert_array_equal(kernel.k_ss, components.bare_total)
    np.testing.assert_array_equal(kernel.k_seta, components.em_collective_left)
    np.testing.assert_array_equal(kernel.k_etas, components.collective_em_right)
    np.testing.assert_array_equal(kernel.k_etaeta, components.collective_total)
    np.testing.assert_array_equal(kernel.k_eff, components.amplitude_phase_schur)
    np.testing.assert_array_equal(kernel.matrix, components.amplitude_phase_schur)
    np.testing.assert_array_equal(kernel.spatial_xy, components.amplitude_phase_schur[1:, 1:])
    assert kernel.primitive_order == PRIMITIVE_EM_ORDER
    assert kernel.collective_order == AMPLITUDE_PHASE_ORDER
    assert kernel.metadata["basis"] == PRIMITIVE_EM_BASIS
    assert kernel.metadata["casimir_stage"] == "microscopic_response_only"


def test_effective_kernel_owns_read_only_copies():
    components = _components()
    original = components.amplitude_phase_schur.copy()
    kernel = effective_em_kernel_from_components(
        components,
        q_model=np.array([0.01, 0.0]),
        xi_eV=0.02,
    )

    components.amplitude_phase_schur[0, 0] += 10.0
    np.testing.assert_array_equal(kernel.k_eff, original)
    assert not kernel.k_eff.flags.writeable
    assert not kernel.q_model.flags.writeable
    with pytest.raises(ValueError):
        kernel.k_eff[0, 0] = 0.0


def test_effective_kernel_rejects_non_amplitude_phase_selection():
    with pytest.raises(ValueError, match="amplitude/phase Schur response"):
        effective_em_kernel_from_components(
            _components(selected="bare_total"),
            q_model=np.array([0.01, 0.0]),
            xi_eV=0.02,
        )


def test_effective_kernel_validates_shapes_and_frequency():
    with pytest.raises(ValueError, match="q_model"):
        EffectiveEMKernel(
            k_ss=np.eye(3),
            k_seta=np.zeros((3, 2)),
            k_etas=np.zeros((2, 3)),
            k_etaeta=np.eye(2),
            k_eff=np.eye(3),
            q_model=np.array([0.1]),
            xi_eV=0.01,
            schur_condition_number=1.0,
            schur_inverse_method="solve",
            metadata={},
        )

    with pytest.raises(ValueError, match="non-negative"):
        EffectiveEMKernel(
            k_ss=np.eye(3),
            k_seta=np.zeros((3, 2)),
            k_etas=np.zeros((2, 3)),
            k_etaeta=np.eye(2),
            k_eff=np.eye(3),
            q_model=np.array([0.1, 0.0]),
            xi_eV=-0.01,
            schur_condition_number=1.0,
            schur_inverse_method="solve",
            metadata={},
        )
