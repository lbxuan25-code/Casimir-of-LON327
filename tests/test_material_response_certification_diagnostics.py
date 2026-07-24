from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from lno327.casimir.material_response import MaterialResponseSample
from lno327.casimir.material_response_certification import (
    MaterialResponseConvergencePolicy,
    MaterialResponseLevelRecord,
    assess_material_response_level,
)
from lno327.casimir.material_response_certification_diagnostics import (
    summarize_material_frequency_result,
)
from lno327.casimir.material_response_engine import MaterialFrequencyResult
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
)


def _side() -> SimpleNamespace:
    return SimpleNamespace(
        primitive_absolute_residual=0.0,
        effective_absolute_residual=0.0,
        primitive_reference_scale=1.0,
        effective_reference_scale=1.0,
        primitive_relative_residual=0.0,
        effective_relative_residual=0.0,
        primitive_mixed_threshold=1e-7,
        effective_mixed_threshold=1e-7,
        primitive_mixed_ratio=0.0,
        effective_mixed_ratio=0.0,
        primitive_mixed_passed=True,
        effective_mixed_passed=True,
        primitive_denominator_collapsed=False,
        effective_denominator_collapsed=False,
    )


def _sample(value: float, *, fingerprint: str) -> MaterialResponseSample:
    q = np.array([0.02, 0.0])
    validation = StaticSheetValidation(
        finite=True,
        ward_passed=True,
        relative_imaginary_norm=0.0,
        relative_longitudinal_gauge_residual=0.0,
        relative_density_transverse_mixing=0.0,
        chi_bar=value,
        dbar_t=value,
        reality_tolerance=1e-8,
        longitudinal_tolerance=1e-6,
        mixing_tolerance=1e-6,
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
    side = _side()
    effective = SimpleNamespace(
        passed=True,
        condition_ok=True,
        primitive_closed=True,
        effective_closed=True,
        denominator_collapse_detected=False,
        schur_condition_number=1.0,
        schur_inverse_method="inv",
        residual_tolerance=1e-7,
        absolute_residual_tolerance=1e-12,
        condition_max=1e12,
        left=side,
        right=side,
    )
    strict = SimpleNamespace(
        passed=True,
        generic_ward_passed=True,
        condition_ok=True,
        schur_condition_number=1.0,
        schur_inverse_method="inv",
        primitive_residual_over_q=0.0,
        amplitude_defect_over_q=0.0,
        phase_defect_over_q=0.0,
        effective_direct_over_q=0.0,
        effective_residual_over_q=0.0,
        relative_longitudinal_gauge_residual=0.0,
        primitive_tolerance=1e-6,
        amplitude_tolerance=1e-6,
        phase_tolerance=1e-6,
        effective_direct_tolerance=1e-6,
        effective_residual_tolerance=1e-6,
        longitudinal_tolerance=1e-6,
        longitudinal_warning=False,
        condition_max=1e12,
    )
    return MaterialResponseSample(
        frequency_index=0,
        frequency_sector="zero_matsubara",
        q_crystal=q,
        xi_eV=0.0,
        material_cache_fingerprint=fingerprint,
        kernel=SimpleNamespace(q_model=q, xi_eV=0.0),
        operator_ward=SimpleNamespace(passed=True),
        effective_ward=effective,
        strict_static_ward=strict,
        response=response,
        sheet_validation=validation,
        metadata={
            "grid": {
                "N": 128,
                "shift_hex": [(0.5).hex(), (0.5).hex()],
            },
            "canonical_reduction_block_size": 4096,
        },
    )


def test_unresolved_summary_preserves_all_N_shift_failure_evidence() -> None:
    policy = MaterialResponseConvergencePolicy(
        relative_tolerance=1e-3,
        absolute_tolerance=1e-6,
    )
    history = []
    previous = None
    for n_grid, values in zip(
        (128, 192, 256),
        ((1.0, 1.1), (1.2, 1.3), (1.4, 1.5)),
        strict=True,
    ):
        samples = {
            "shift_a": _sample(values[0], fingerprint=f"a-{n_grid}"),
            "shift_b": _sample(values[1], fingerprint=f"b-{n_grid}"),
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

    frequency = MaterialFrequencyResult(
        matsubara_index=0,
        xi_eV=0.0,
        history=tuple(history),
        certification=None,
    )
    payload = summarize_material_frequency_result(
        frequency,
        policy=policy,
        required_consecutive_passes=2,
        envelope_levels=3,
    )

    assert payload["status"] == "unresolved"
    assert payload["evaluated_N"] == [128, 192, 256]
    assert payload["hard_physical_failure_levels"] == []
    assert payload["cross_shift_failure_levels"] == [128, 192, 256]
    assert payload["adjacent_N_failure_levels"] == [192, 256]
    assert payload["terminal_consecutive_accepted_transitions"] == 0
    assert set(payload["blockers"]) == {
        "cross_shift_response",
        "adjacent_N_response",
        "required_consecutive_transitions",
        "complete_pairwise_envelope",
    }
    assert len(payload["levels"]) == 3
    assert payload["levels"][0]["samples_by_shift"]["shift_a"][
        "hard_physical_passed"
    ] is True
    assert payload["oscillatory_envelope"]["pairwise_complete"] is True
    json.dumps(payload, allow_nan=False)
