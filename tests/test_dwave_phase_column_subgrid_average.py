from __future__ import annotations

import numpy as np

from validation.lib.dwave_phase_column_subgrid_average import (
    average_dwave_phase_column_payloads,
)
from validation.lib.dwave_phase_hessian_analysis import (
    analyze_dwave_phase_hessian_payload,
)


def _parts(em: complex, bubble: complex, counterterm: complex):
    return {
        "em_collective_contraction": [0.0, em],
        "phase_rotation_bubble": [0.0, bubble],
        "phase_rotation_counterterm": [0.0, counterterm],
    }


def _payload(
    qx: float,
    qy: float,
    *,
    required: float,
    counterterm: complex,
    shift_x: float,
):
    em = 0.0 + 0.0j
    bubble = -required * counterterm
    parts = _parts(em, bubble, counterterm)
    return {
        "schema": "dwave_static_commensurate_phase_column_audit_v1",
        "audit": {
            "q_model": [qx, qy],
            "q_norm": float(np.hypot(qx, qy)),
            "delta0_eV": 0.1,
            "w_left": [0.0, -0.2j],
            "w_right": [0.0, -0.2j],
            "component_sources": {
                "left": {"collective_defect_parts": parts},
                "right": {"collective_defect_parts": parts},
            },
        },
        "primitive_metadata": {},
        "metadata": {"grid_shift": [shift_x, 0.5]},
    }


def test_subgrid_average_forms_multiplier_after_component_average():
    qx, qy = 0.01, 0.0
    bond = float(0.5 * (np.cos(0.5 * qx) ** 2 + 1.0))
    counterterm_a = -0.5j
    counterterm_b = -0.7j
    required_a = bond - 2.0e-5
    required_b = float(
        (bond * (counterterm_a + counterterm_b) - required_a * counterterm_a)
        / counterterm_b
    )

    first = _payload(
        qx,
        qy,
        required=required_a,
        counterterm=counterterm_a,
        shift_x=0.0,
    )
    second = _payload(
        qx,
        qy,
        required=required_b,
        counterterm=counterterm_b,
        shift_x=0.5,
    )

    averaged = average_dwave_phase_column_payloads(
        (first, second), labels=("integer", "half")
    )
    analysis = analyze_dwave_phase_hessian_payload(averaged)

    assert np.isclose(analysis.left.required_counterterm_multiplier, bond, atol=1e-14)
    assert np.isclose(analysis.right.required_counterterm_multiplier, bond, atol=1e-14)
    assert abs(analysis.left.bond_metric_phase_defect) < 1e-14
    assert averaged["metadata"]["average_formed_before_required_multiplier"]
    assert averaged["status"]["subgrid_averaged"]


def test_subgrid_average_rejects_different_q_vectors():
    first = _payload(0.01, 0.0, required=0.99, counterterm=-0.5j, shift_x=0.0)
    second = _payload(0.02, 0.0, required=0.99, counterterm=-0.5j, shift_x=0.5)

    try:
        average_dwave_phase_column_payloads((first, second))
    except ValueError as exc:
        assert "same q vector" in str(exc)
    else:
        raise AssertionError("different q vectors must be rejected")
