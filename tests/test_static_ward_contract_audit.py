from __future__ import annotations

import numpy as np
import pytest

from lno327.electrodynamics.basis import xy_to_lt_rotation
from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import PrimitiveWardRHS, primitive_ward_vectors_xy
from validation.lib.static_ward_contract_audit import audit_static_ward_contract


def _rhs(q: np.ndarray, delta0: float, equal: np.ndarray, delta_v: np.ndarray, contact: np.ndarray) -> PrimitiveWardRHS:
    total = equal - delta_v + contact
    return PrimitiveWardRHS(
        left=total,
        right=total.copy(),
        q_model=q,
        xi_eV=0.0,
        delta0_eV=delta0,
        metadata={
            "equal_forward": equal,
            "delta_v_mid": delta_v,
            "qM_mid": contact,
        },
    )


def _kernel(
    q: np.ndarray,
    k_ss: np.ndarray,
    k_seta: np.ndarray,
    k_etas: np.ndarray,
    k_etaeta: np.ndarray,
) -> EffectiveEMKernel:
    inverse = np.linalg.inv(k_etaeta)
    k_eff = k_ss - k_seta @ inverse @ k_etas
    return EffectiveEMKernel(
        k_ss=k_ss,
        k_seta=k_seta,
        k_etas=k_etas,
        k_etaeta=k_etaeta,
        k_eff=k_eff,
        q_model=q,
        xi_eV=0.0,
        schur_condition_number=float(np.linalg.cond(k_etaeta)),
        schur_inverse_method="inv",
        metadata={"test": True},
    )


def test_static_ward_contract_audit_reconstructs_both_sides_and_lt_mapping():
    q = np.asarray([0.3, 0.4])
    delta0 = 0.2
    k_etaeta = np.asarray([[2.0 + 0.1j, 0.2], [0.1, 1.4 - 0.05j]])
    k_seta = np.asarray(
        [
            [0.2 + 0.1j, -0.1],
            [0.3, 0.05 - 0.02j],
            [-0.2j, 0.4],
        ]
    )
    k_etas = np.asarray(
        [
            [0.1, -0.3j, 0.2],
            [0.05 + 0.02j, 0.25, -0.1],
        ]
    )
    k_ss = np.asarray(
        [
            [1.1, 0.2j, -0.1],
            [0.15, -0.7, 0.08j],
            [-0.05j, 0.12, -0.4],
        ]
    )
    equal = np.asarray([0.01j, 0.03, -0.02])
    delta_v = np.asarray([0.0, 0.004, -0.003])
    contact = np.asarray([0.0, -0.011, 0.007])

    kernel = _kernel(q, k_ss, k_seta, k_etas, k_etaeta)
    audit = audit_static_ward_contract(
        kernel, _rhs(q, delta0, equal, delta_v, contact)
    )

    for side in ("left", "right"):
        payload = audit[side]
        assert np.allclose(
            payload["effective_direct"],
            payload["effective_predicted"] + payload["primitive_residual"],
            rtol=1e-13,
            atol=1e-13,
        )
        assert np.allclose(
            payload["effective_residual"],
            payload["primitive_residual"],
            rtol=1e-13,
            atol=1e-13,
        )
        assert np.allclose(
            np.sum(payload["collective_projection_by_channel"], axis=0),
            payload["collective_projection"],
            rtol=1e-13,
            atol=1e-13,
        )
        assert payload["norms"]["reconstruction_error"] < 1e-12
        assert payload["norms"]["residual_identity_error"] < 1e-12

    transform = np.eye(3)
    transform[1:3, 1:3] = xy_to_lt_rotation(q[0], q[1])
    kernel_lt = transform @ kernel.k_eff @ transform.T
    left_lt = audit["left"]["effective_direct_lt_over_q"]
    right_lt = audit["right"]["effective_direct_lt_over_q"]
    assert np.allclose(left_lt, kernel_lt[1, :], rtol=1e-13, atol=1e-13)
    assert np.allclose(right_lt, kernel_lt[:, 1], rtol=1e-13, atol=1e-13)
    assert audit["lt_contraction_mapping_error_norm"] < 1e-12
    assert audit["rhs_metadata_error_norm"] < 1e-14
    assert audit["status"]["projection_applied"] is False
    assert audit["status"]["valid_for_casimir_input"] is False


def test_static_ward_contract_audit_identifies_exact_combined_gauge_zero_mode():
    q = np.asarray([0.3, 0.4])
    delta0 = 0.2
    u_left, u_right, w_left, w_right = primitive_ward_vectors_xy(0.0, q, delta0)
    assert np.allclose(u_left, u_right)
    assert np.allclose(w_left, w_right)
    u = u_left
    w = w_left
    u2 = float(np.vdot(u, u).real)

    transform = np.eye(3)
    transform[1:3, 1:3] = xy_to_lt_rotation(q[0], q[1])
    k_eff_lt = np.asarray(
        [
            [1.2, 0.0, 0.15],
            [0.0, 0.0, 0.0],
            [-0.08, 0.0, -0.7],
        ],
        dtype=complex,
    )
    k_eff = transform.T @ k_eff_lt @ transform

    k_etaeta = np.asarray([[1.8, 0.2], [0.1, 1.3]], dtype=complex)
    right_target = -(k_etaeta @ w)
    k_etas = np.stack([right_target[a] * np.conjugate(u) / u2 for a in range(2)])
    left_target = -(w @ k_etaeta)
    k_seta = np.stack(
        [np.conjugate(u) * left_target[a] / u2 for a in range(2)], axis=1
    )
    k_ss = k_eff + k_seta @ np.linalg.inv(k_etaeta) @ k_etas

    zero = np.zeros(3, dtype=complex)
    kernel = _kernel(q, k_ss, k_seta, k_etas, k_etaeta)
    audit = audit_static_ward_contract(kernel, _rhs(q, delta0, zero, zero, zero))

    assert np.linalg.norm(audit["left"]["collective_defect"]) < 1e-12
    assert np.linalg.norm(audit["right"]["collective_defect"]) < 1e-12
    assert np.linalg.norm(audit["left"]["primitive_residual"]) < 1e-12
    assert np.linalg.norm(audit["right"]["primitive_residual"]) < 1e-12
    assert audit["max_effective_direct_over_q"] < 1e-12
    assert audit["relative_longitudinal_gauge_residual"] < 1e-12
    assert np.allclose(audit["kernel_lt"], k_eff_lt, rtol=1e-12, atol=1e-12)


def test_static_ward_contract_audit_requires_separate_rhs_pieces():
    q = np.asarray([0.3, 0.4])
    k_etaeta = np.eye(2, dtype=complex)
    k_seta = np.zeros((3, 2), dtype=complex)
    k_etas = np.zeros((2, 3), dtype=complex)
    k_ss = np.eye(3, dtype=complex)
    kernel = _kernel(q, k_ss, k_seta, k_etas, k_etaeta)
    rhs = PrimitiveWardRHS(
        left=np.zeros(3),
        right=np.zeros(3),
        q_model=q,
        xi_eV=0.0,
        delta0_eV=0.1,
        metadata={},
    )
    with pytest.raises(ValueError, match="equal_forward"):
        audit_static_ward_contract(kernel, rhs)


def test_dwave_static_ward_contract_audit_runner_imports():
    import validation.run_dwave_static_ward_contract_audit as runner

    assert callable(runner.main)
