from __future__ import annotations

import numpy as np

from validation.lib.dwave_phase_hessian_analysis import (
    analyze_dwave_phase_hessian_payload,
    complex_jsonable,
)


def _z(value: complex) -> dict[str, float]:
    scalar = complex(value)
    return {"real": scalar.real, "imag": scalar.imag}


def _collective_parts() -> dict[str, list[dict[str, float]]]:
    return {
        "em_collective_contraction": [_z(0.0), _z(0.0999j)],
        "phase_rotation_bubble": [_z(0.0), _z(0.4j)],
        "phase_rotation_counterterm": [_z(0.0), _z(-0.5j)],
    }


def test_phase_hessian_postprocessor_infers_required_and_direct_multipliers():
    required = 0.9998
    delta0 = 0.1
    counterterm_curvature = 2.5
    payload = {
        "schema": "dwave_static_commensurate_periodic_ward_audit_v1",
        "audit": {
            "q_model": [0.03, 0.02],
            "delta0_eV": delta0,
            "w_left": [_z(0.0), _z(-0.2j)],
            "component_sources": {
                "left": {"collective_defect_parts": _collective_parts()},
                "right": {"collective_defect_parts": _collective_parts()},
            },
        },
        "primitive_metadata": {
            "phase_phase_direct_plus": _z(
                delta0 * delta0 * counterterm_curvature * required
            )
        },
    }

    analysis = analyze_dwave_phase_hessian_payload(payload)

    assert analysis.q_norm == np.hypot(0.03, 0.02)
    assert analysis.counterterm_curvature == 2.5
    assert analysis.phase_direct_counterterm_multiplier == required
    assert analysis.left.required_counterterm_multiplier == required
    assert analysis.right.required_counterterm_multiplier == required
    assert abs(analysis.left.phase_direct_phase_defect) < 1e-15
    assert abs(analysis.right.phase_direct_phase_defect) < 1e-15
    assert analysis.diagnostic_only
    assert not analysis.valid_for_casimir_input


def test_complex_jsonable_encodes_nested_complex_values():
    converted = complex_jsonable({"value": 1.5 - 2.0j, "items": (3.0j,)})
    assert converted == {
        "value": {"real": 1.5, "imag": -2.0},
        "items": [{"real": 0.0, "imag": 3.0}],
    }
