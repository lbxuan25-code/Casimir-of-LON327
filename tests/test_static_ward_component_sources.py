from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import PrimitiveWardRHS
from validation.lib.static_ward_component_sources import (
    audit_static_ward_contract_with_components,
)


def test_component_source_decomposition_reconstructs_all_reported_terms():
    q = np.asarray([0.3, 0.4])
    delta0 = 0.2
    bubble = np.asarray(
        [
            [0.7, 0.1j, -0.04],
            [0.03, -0.5, 0.02j],
            [-0.01j, 0.05, -0.2],
        ],
        dtype=complex,
    )
    direct = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, -0.15, 0.01],
            [0.0, 0.02, -0.12],
        ],
        dtype=complex,
    )
    k_seta = np.asarray(
        [
            [0.1, -0.05j],
            [0.2, 0.03],
            [-0.04j, 0.12],
        ],
        dtype=complex,
    )
    k_etas = np.asarray(
        [
            [0.08, -0.1j, 0.05],
            [0.02j, 0.16, -0.03],
        ],
        dtype=complex,
    )
    collective_bubble = np.asarray([[0.9, 0.08], [0.03, 0.55]], dtype=complex)
    collective_counterterm = np.asarray([[-0.2, 0.0], [0.0, -0.2]], dtype=complex)
    k_etaeta = collective_bubble + collective_counterterm
    k_ss = bubble + direct
    k_eff = k_ss - k_seta @ np.linalg.inv(k_etaeta) @ k_etas

    kernel = EffectiveEMKernel(
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
    equal = np.asarray([0.01j, 0.02, -0.01])
    delta_v = np.asarray([0.0, 0.003, -0.002])
    contact = np.asarray([0.0, -0.006, 0.004])
    total_rhs = equal - delta_v + contact
    rhs = PrimitiveWardRHS(
        left=total_rhs,
        right=total_rhs.copy(),
        q_model=q,
        xi_eV=0.0,
        delta0_eV=delta0,
        metadata={
            "equal_forward": equal,
            "delta_v_mid": delta_v,
            "qM_mid": contact,
        },
    )
    components = SimpleNamespace(
        bare_bubble=bubble,
        direct=direct,
        collective_bubble=collective_bubble,
        collective_counterterm=collective_counterterm,
    )

    audit = audit_static_ward_contract_with_components(kernel, rhs, components)

    for side in ("left", "right"):
        detailed = audit["component_sources"][side]
        base = audit[side]
        assert np.allclose(
            detailed["primitive_split_sum"],
            base["primitive_residual"],
            rtol=1e-13,
            atol=1e-13,
        )
        assert np.allclose(
            detailed["collective_defect_sum"],
            base["collective_defect"],
            rtol=1e-13,
            atol=1e-13,
        )
        assert np.allclose(
            detailed["collective_projection_sum"],
            base["collective_projection"],
            rtol=1e-13,
            atol=1e-13,
        )
        assert np.allclose(
            detailed["effective_predicted_sources"]["sum"],
            base["effective_predicted"],
            rtol=1e-13,
            atol=1e-13,
        )
        assert detailed["primitive_split_error_norm"] < 1e-12
        assert detailed["collective_split_error_norm"] < 1e-12
        assert detailed["projection_split_error_norm"] < 1e-12
        assert detailed["effective_source_error_norm"] < 1e-12

    assert audit["component_source_consistency_max"] < 1e-12
    assert audit["schema"] == "static_ward_contract_component_source_audit_v1"
