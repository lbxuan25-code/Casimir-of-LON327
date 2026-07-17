from __future__ import annotations

import numpy as np
import pytest

from validation.lib.dwave_phase_hessian_scaling import (
    analyze_dwave_phase_hessian_family,
)


def _z(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {"real": scalar.real, "imag": scalar.imag}


def _payload(qx: float, qy: float, required_multiplier: float) -> dict[str, object]:
    counterterm = -0.5j
    em = 0.05j
    bubble = 0.5j * required_multiplier - em
    parts = {
        "em_collective_contraction": [_z(0.0), _z(em)],
        "phase_rotation_bubble": [_z(0.0), _z(bubble)],
        "phase_rotation_counterterm": [_z(0.0), _z(counterterm)],
    }
    return {
        "schema": "dwave_static_commensurate_periodic_ward_audit_v1",
        "audit": {
            "q_model": [qx, qy],
            "delta0_eV": 0.1,
            "w_left": [_z(0.0), _z(-0.2j)],
            "component_sources": {
                "left": {"collective_defect_parts": parts},
                "right": {"collective_defect_parts": parts},
            },
        },
        "primitive_metadata": {},
    }


def _bond_metric(qx: float, qy: float) -> float:
    return float(0.5 * (np.cos(0.5 * qx) ** 2 + np.cos(0.5 * qy) ** 2))


def test_scaling_classifies_q4_remainder_after_bond_metric():
    payloads = []
    for scale in (0.5, 1.0, 2.0):
        qx, qy = 0.03 * scale, 0.02 * scale
        q_norm = float(np.hypot(qx, qy))
        required = _bond_metric(qx, qy) - 0.2 * q_norm**4
        payloads.append(_payload(qx, qy, required))

    analysis = analyze_dwave_phase_hessian_family(payloads)

    assert 1.9 < analysis.required_shift_exponent < 2.2
    assert 1.9 < analysis.bond_shift_exponent < 2.1
    assert 3.9 < analysis.bond_error_exponent < 4.1
    assert analysis.classification == "bond_metric_matches_leading_q2_geometry"
    assert analysis.diagnostic_only
    assert not analysis.valid_for_casimir_input


def test_scaling_classifies_leading_q2_bond_metric_error():
    payloads = []
    for scale in (0.5, 1.0, 2.0):
        qx, qy = 0.03 * scale, 0.02 * scale
        q_norm = float(np.hypot(qx, qy))
        required = _bond_metric(qx, qy) - 0.01 * q_norm**2
        payloads.append(_payload(qx, qy, required))

    analysis = analyze_dwave_phase_hessian_family(payloads)

    assert 1.9 < analysis.bond_error_exponent < 2.1
    assert analysis.classification == "bond_metric_misses_leading_q2_curvature"


def test_scaling_rejects_mixed_q_directions():
    first = _payload(0.03, 0.02, _bond_metric(0.03, 0.02))
    second = _payload(0.02, 0.03, _bond_metric(0.02, 0.03))

    with pytest.raises(ValueError, match="same direction"):
        analyze_dwave_phase_hessian_family((first, second))
