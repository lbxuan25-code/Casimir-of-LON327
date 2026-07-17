from __future__ import annotations

import numpy as np

from validation.lib.dwave_phase_hessian_scaling import (
    analyze_dwave_phase_hessian_family,
)


def _z(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {"real": scalar.real, "imag": scalar.imag}


def _payload(qx: float, qy: float, required_shift: float) -> dict[str, object]:
    required = 1.0 - required_shift
    counterterm = -0.5j
    em = 0.05j
    bubble = 0.5j * required - em
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


def test_unstable_required_pairwise_power_is_not_accepted_as_clean_q2():
    q_values = (
        (0.0300152164356, 0.0200101442904, 1.769411473420e-4),
        (0.0400202885808, 0.0266801923872, 3.217221925937e-4),
        (0.0600304328711, 0.0400202885808, 5.197648405220e-4),
    )
    analysis = analyze_dwave_phase_hessian_family(
        [_payload(qx, qy, shift) for qx, qy, shift in q_values]
    )

    assert analysis.required_shift_exponent < 1.6
    assert analysis.required_shift_pairwise_exponents[-1] < 1.3
    assert analysis.classification == "not_in_clean_small_q_regime"
    assert np.isclose(
        analysis.smallest_two_required_q2_coefficient,
        0.131991,
        rtol=2e-4,
    )
    assert np.isclose(
        analysis.smallest_two_bond_error_q2_coefficient,
        0.006991,
        rtol=3e-4,
    )
