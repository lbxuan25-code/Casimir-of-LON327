"""Pairwise completeness guard for the response-space oscillatory envelope."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lno327.casimir.material_response import MaterialResponseSample
from lno327.casimir.material_response_certification import (
    MaterialResponseConvergencePolicy,
    MaterialResponseLevelRecord,
    assess_material_response_envelope,
    assess_material_response_level,
)
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
)


def _sample(value: float, *, fingerprint: str) -> MaterialResponseSample:
    q = np.array([0.01, 0.02])
    validation = StaticSheetValidation(
        finite=True,
        ward_passed=True,
        relative_imaginary_norm=0.0,
        relative_longitudinal_gauge_residual=0.0,
        relative_density_transverse_mixing=0.0,
        chi_bar=value,
        dbar_t=value,
        reality_tolerance=1e-9,
        longitudinal_tolerance=1e-7,
        mixing_tolerance=1e-7,
        passivity_tolerance=1e-10,
    )
    response = StaticSheetResponse(
        kernel_lt=np.eye(3, dtype=complex),
        chi_bar=value,
        dbar_t=value,
        q_model=q,
        energy_scale_eV=1.0,
        degeneracy=1.0,
        basis=STATIC_LOCAL_BASIS,
        validation=validation,
        metadata={},
    )
    side = SimpleNamespace(effective_mixed_ratio=0.0)
    return MaterialResponseSample(
        frequency_index=0,
        frequency_sector="zero_matsubara",
        q_crystal=q,
        xi_eV=0.0,
        material_cache_fingerprint=fingerprint,
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
        metadata={},
    )


def test_envelope_compares_every_N_shift_pair() -> None:
    policy = MaterialResponseConvergencePolicy(
        relative_tolerance=7.5e-3,
        absolute_tolerance=0.0,
    )
    values = ((1.000, 1.002), (1.004, 1.006), (1.008, 1.010))
    history = []
    previous = None
    for n_grid, pair in zip((64, 96, 128), values, strict=True):
        samples = {
            "shift_a": _sample(pair[0], fingerprint=f"a-{n_grid}"),
            "shift_b": _sample(pair[1], fingerprint=f"b-{n_grid}"),
        }
        assessment = assess_material_response_level(
            current_by_shift=samples,
            previous_by_shift=previous,
            policy=policy,
        )
        history.append(
            MaterialResponseLevelRecord(
                n_grid=n_grid,
                samples_by_shift=samples,
                assessment=assessment,
            )
        )
        previous = samples

    envelope = assess_material_response_envelope(history, policy=policy, levels=3)
    assert envelope["pairwise_complete"] is True
    assert envelope["comparison_count"] == 15
    assert envelope["passed"] is False
    assert envelope["joint_response_envelope"]["N64:shift_a|N128:shift_b"][
        "passed"
    ] is False
