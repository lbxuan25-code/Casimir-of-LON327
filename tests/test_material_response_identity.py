"""Stable identity and policy-completeness guards for material responses."""
from __future__ import annotations

from lno327.casimir.material_response import MaterialResponsePolicy


def test_material_policy_fingerprint_covers_positive_and_static_gates() -> None:
    baseline = MaterialResponsePolicy()
    changed_positive = MaterialResponsePolicy(positive_symmetry_tolerance=2e-9)
    changed_static = MaterialResponsePolicy(static_mixing_tolerance=2e-6)

    payload = baseline.as_dict()
    assert "positive_reality_tolerance" in payload
    assert "positive_symmetry_tolerance" in payload
    assert "positive_passivity_tolerance" in payload
    assert "static_reality_tolerance" in payload
    assert "static_mixing_tolerance" in payload
    assert "static_passivity_tolerance" in payload
    assert baseline.fingerprint != changed_positive.fingerprint
    assert baseline.fingerprint != changed_static.fingerprint
    assert not {
        "q_lab",
        "theta_rad",
        "plate_angles_rad",
        "separation_nm",
        "outer_order",
    }.intersection(payload)
