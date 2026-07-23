"""Stable identity and policy-completeness guards for material responses."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir.material_response import (
    MaterialResponsePolicy,
    MaterialResponseSample,
)
from lno327.casimir.material_response_certification import (
    MaterialResponseConvergencePolicy,
    compare_material_responses,
)
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
)


def _sample(
    *,
    material_state: str,
    policy_fingerprint: str,
    cache_fingerprint: str,
    grid_fingerprint: str,
    value: float = 1.0,
) -> MaterialResponseSample:
    q = np.array([0.01, 0.02])
    validation = StaticSheetValidation(
        finite=True,
        ward_passed=True,
        relative_imaginary_norm=0.0,
        relative_longitudinal_gauge_residual=0.0,
        relative_density_transverse_mixing=0.0,
        chi_bar=value,
        dbar_t=2.0 * value,
        reality_tolerance=1e-8,
        longitudinal_tolerance=1e-6,
        mixing_tolerance=1e-6,
        passivity_tolerance=1e-10,
    )
    response = StaticSheetResponse(
        kernel_lt=np.eye(3, dtype=complex),
        chi_bar=value,
        dbar_t=2.0 * value,
        q_model=q,
        energy_scale_eV=1.0,
        degeneracy=1.0,
        basis=STATIC_LOCAL_BASIS,
        validation=validation,
        metadata={"source": "identity-test"},
    )
    side = SimpleNamespace(effective_mixed_ratio=0.0)
    return MaterialResponseSample(
        frequency_index=0,
        frequency_sector="zero_matsubara",
        q_crystal=q,
        xi_eV=0.0,
        material_cache_fingerprint=cache_fingerprint,
        kernel=SimpleNamespace(q_model=q, xi_eV=0.0),
        operator_ward=SimpleNamespace(passed=True),
        effective_ward=SimpleNamespace(
            passed=True,
            left=side,
            right=side,
            schur_condition_number=1.0,
        ),
        strict_static_ward=SimpleNamespace(passed=True),
        response=response,
        sheet_validation=validation,
        metadata={
            "material_state_fingerprint": material_state,
            "response_policy_fingerprint": policy_fingerprint,
            "primitive_contract_version": "primitive-test-v1",
            "post_integral_phase_hessian_policy": "q_independent",
            "basis": STATIC_LOCAL_BASIS,
            "grid_fingerprint": grid_fingerprint,
            "grid": {"N": 64, "shift": [0.5, 0.5]},
            "canonical_reduction_block_size": 4096,
        },
    )


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
    assert baseline.fingerprint == MaterialResponsePolicy().fingerprint
    assert baseline.fingerprint != changed_positive.fingerprint
    assert baseline.fingerprint != changed_static.fingerprint
    assert not {
        "q_lab",
        "theta_rad",
        "plate_angles_rad",
        "separation_nm",
        "outer_order",
    }.intersection(payload)


def test_physical_identity_excludes_grid_and_cache_provenance() -> None:
    first = _sample(
        material_state="same-material",
        policy_fingerprint="same-policy",
        cache_fingerprint="cache-N64-shift-a",
        grid_fingerprint="grid-N64-shift-a",
    )
    second = _sample(
        material_state="same-material",
        policy_fingerprint="same-policy",
        cache_fingerprint="cache-N96-shift-b",
        grid_fingerprint="grid-N96-shift-b",
        value=1.001,
    )

    assert first.identity_payload == second.identity_payload
    assert first.identity_fingerprint == second.identity_fingerprint
    assert first.provenance_payload != second.provenance_payload
    comparison = compare_material_responses(
        first,
        second,
        policy=MaterialResponseConvergencePolicy(relative_tolerance=1e-2),
    )
    assert comparison["passed"] is True


def test_certification_rejects_material_or_policy_identity_mismatch() -> None:
    baseline = _sample(
        material_state="material-a",
        policy_fingerprint="policy-a",
        cache_fingerprint="cache-a",
        grid_fingerprint="grid-a",
    )
    changed_material = _sample(
        material_state="material-b",
        policy_fingerprint="policy-a",
        cache_fingerprint="cache-b",
        grid_fingerprint="grid-b",
    )
    changed_policy = _sample(
        material_state="material-a",
        policy_fingerprint="policy-b",
        cache_fingerprint="cache-c",
        grid_fingerprint="grid-c",
    )
    convergence = MaterialResponseConvergencePolicy()

    with pytest.raises(ValueError, match="material/policy identity"):
        compare_material_responses(baseline, changed_material, policy=convergence)
    with pytest.raises(ValueError, match="material/policy identity"):
        compare_material_responses(baseline, changed_policy, policy=convergence)
