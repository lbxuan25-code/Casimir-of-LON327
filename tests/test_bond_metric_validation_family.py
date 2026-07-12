from __future__ import annotations

import pytest

from validation.commands.ward.bond_metric_family import (
    c4_comparisons,
    parse_integer_point,
    strict_gate_from_point_payload,
)


def _payload(*, phase: float = 1e-12, effective: float = 2e-12):
    row = {
        "corrected_primitive_residual_over_q": 1e-13,
        "corrected_amplitude_defect_over_q": 1e-13,
        "corrected_phase_defect_over_q": phase,
        "corrected_effective_direct_over_q": effective,
        "corrected_raw_longitudinal": 3e-12,
        "corrected_collective_condition": 10.0,
        "corrected_collective_inverse_method": "inv",
    }
    return {
        "row": row,
        "corrected": {
            "ward_audit": {
                "left": {"q_normalized_norms": {"effective_residual": 4e-13}},
                "right": {"q_normalized_norms": {"effective_residual": 5e-13}},
            }
        },
    }


def test_parse_integer_point():
    assert parse_integer_point("3,2") == (3, 2)
    assert parse_integer_point(" -2, 4 ") == (-2, 4)
    with pytest.raises(ValueError, match="form mx,my"):
        parse_integer_point("3")
    with pytest.raises(ValueError, match="q=0"):
        parse_integer_point("0,0")


def test_strict_gate_from_full_kernel_payload():
    gate = strict_gate_from_point_payload(
        _payload(),
        "corrected",
        primitive_tolerance=1e-9,
        amplitude_tolerance=1e-9,
        phase_tolerance=1e-9,
        effective_direct_tolerance=1e-9,
        effective_residual_tolerance=1e-9,
        longitudinal_tolerance=1e-9,
        condition_max=1e12,
    )
    assert gate["effective_residual_over_q"] == pytest.approx(5e-13)
    assert gate["passed"] is True

    failed = strict_gate_from_point_payload(
        _payload(phase=2e-3),
        "corrected",
        primitive_tolerance=1e-9,
        amplitude_tolerance=1e-9,
        phase_tolerance=1e-9,
        effective_direct_tolerance=1e-9,
        effective_residual_tolerance=1e-9,
        longitudinal_tolerance=1e-9,
        condition_max=1e12,
    )
    assert failed["passed"] is False


def test_c4_comparisons_use_corrected_observables():
    rows = [
        {
            "mx": 4,
            "my": 2,
            "q_norm": 0.1,
            "corrected_chi_bar": 0.45,
            "corrected_dbar_t": 0.59,
            "corrected_phase_defect_over_q": 1e-13,
            "corrected_effective_direct_over_q": 2e-13,
        },
        {
            "mx": 2,
            "my": 4,
            "q_norm": 0.1,
            "corrected_chi_bar": 0.45 * (1.0 + 1e-10),
            "corrected_dbar_t": 0.59 * (1.0 - 1e-10),
            "corrected_phase_defect_over_q": 2e-13,
            "corrected_effective_direct_over_q": 3e-13,
        },
    ]
    comparisons = c4_comparisons(rows, 1e-8)
    assert len(comparisons) == 1
    assert comparisons[0]["passed"] is True
