"""Compact diagnostics for unresolved material-response certification histories.

The response engine deliberately returns the full in-memory N/shift history for an
unresolved frequency.  This module converts that history into a deterministic,
JSON-safe audit payload without persisting uncertified response samples or changing
any scientific policy.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from lno327.casimir.material_response import MaterialResponseSample
from lno327.casimir.material_response_certification import (
    MaterialResponseConvergencePolicy,
    assess_material_response_envelope,
)
from lno327.casimir.material_response_engine import MaterialFrequencyResult
from lno327.electrodynamics.static_sheet import StaticSheetResponse

MATERIAL_RESPONSE_UNRESOLVED_DIAGNOSTIC_SCHEMA = (
    "material-response-unresolved-diagnostic-v1"
)


def _ward_side_payload(side: object) -> dict[str, Any]:
    return {
        "primitive_absolute_residual": float(side.primitive_absolute_residual),
        "effective_absolute_residual": float(side.effective_absolute_residual),
        "primitive_reference_scale": float(side.primitive_reference_scale),
        "effective_reference_scale": float(side.effective_reference_scale),
        "primitive_relative_residual": float(side.primitive_relative_residual),
        "effective_relative_residual": float(side.effective_relative_residual),
        "primitive_mixed_threshold": float(side.primitive_mixed_threshold),
        "effective_mixed_threshold": float(side.effective_mixed_threshold),
        "primitive_mixed_ratio": float(side.primitive_mixed_ratio),
        "effective_mixed_ratio": float(side.effective_mixed_ratio),
        "primitive_mixed_passed": bool(side.primitive_mixed_passed),
        "effective_mixed_passed": bool(side.effective_mixed_passed),
        "primitive_denominator_collapsed": bool(
            side.primitive_denominator_collapsed
        ),
        "effective_denominator_collapsed": bool(
            side.effective_denominator_collapsed
        ),
    }


def _effective_ward_payload(ward: object) -> dict[str, Any]:
    return {
        "passed": bool(ward.passed),
        "condition_ok": bool(ward.condition_ok),
        "primitive_closed": bool(ward.primitive_closed),
        "effective_closed": bool(ward.effective_closed),
        "denominator_collapse_detected": bool(
            ward.denominator_collapse_detected
        ),
        "schur_condition_number": float(ward.schur_condition_number),
        "schur_inverse_method": str(ward.schur_inverse_method),
        "relative_residual_tolerance": float(ward.residual_tolerance),
        "absolute_residual_tolerance": float(
            ward.absolute_residual_tolerance
        ),
        "condition_max": float(ward.condition_max),
        "left": _ward_side_payload(ward.left),
        "right": _ward_side_payload(ward.right),
    }


def _strict_static_payload(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "passed": bool(value.passed),
        "generic_ward_passed": bool(value.generic_ward_passed),
        "condition_ok": bool(value.condition_ok),
        "schur_condition_number": float(value.schur_condition_number),
        "schur_inverse_method": str(value.schur_inverse_method),
        "primitive_residual_over_q": float(value.primitive_residual_over_q),
        "amplitude_defect_over_q": float(value.amplitude_defect_over_q),
        "phase_defect_over_q": float(value.phase_defect_over_q),
        "effective_direct_over_q": float(value.effective_direct_over_q),
        "effective_residual_over_q": float(value.effective_residual_over_q),
        "relative_longitudinal_gauge_residual": float(
            value.relative_longitudinal_gauge_residual
        ),
        "primitive_tolerance": float(value.primitive_tolerance),
        "amplitude_tolerance": float(value.amplitude_tolerance),
        "phase_tolerance": float(value.phase_tolerance),
        "effective_direct_tolerance": float(value.effective_direct_tolerance),
        "effective_residual_tolerance": float(
            value.effective_residual_tolerance
        ),
        "longitudinal_tolerance": float(value.longitudinal_tolerance),
        "longitudinal_warning": bool(value.longitudinal_warning),
        "condition_max": float(value.condition_max),
    }


def _sheet_validation_payload(sample: MaterialResponseSample) -> dict[str, Any]:
    value = sample.sheet_validation
    if sample.frequency_sector == "zero_matsubara":
        return {
            "frequency_sector": sample.frequency_sector,
            "passed": bool(value.passed),
            "finite": bool(value.finite),
            "ward_passed": bool(value.ward_passed),
            "relative_imaginary_norm": float(value.relative_imaginary_norm),
            "relative_longitudinal_gauge_residual": float(
                value.relative_longitudinal_gauge_residual
            ),
            "relative_density_transverse_mixing": float(
                value.relative_density_transverse_mixing
            ),
            "chi_bar": float(value.chi_bar),
            "dbar_t": float(value.dbar_t),
            "reality_tolerance": float(value.reality_tolerance),
            "longitudinal_tolerance": float(value.longitudinal_tolerance),
            "mixing_tolerance": float(value.mixing_tolerance),
            "passivity_tolerance": float(value.passivity_tolerance),
            "longitudinal_warning": bool(value.longitudinal_warning),
            "longitudinal_is_hard_gate": False,
        }
    return {
        "frequency_sector": sample.frequency_sector,
        "passed": bool(value.passed),
        "finite": bool(value.finite),
        "relative_imaginary_norm": float(value.relative_imaginary_norm),
        "relative_symmetry_residual": float(
            value.relative_symmetry_residual
        ),
        "minimum_symmetric_eigenvalue": float(
            value.minimum_symmetric_eigenvalue
        ),
        "reality_tolerance": float(value.reality_tolerance),
        "symmetry_tolerance": float(value.symmetry_tolerance),
        "passivity_tolerance": float(value.passivity_tolerance),
    }


def _response_payload(sample: MaterialResponseSample) -> dict[str, Any]:
    response = sample.response
    if isinstance(response, StaticSheetResponse):
        return {
            "frequency_sector": sample.frequency_sector,
            "chi_bar": float(response.chi_bar),
            "dbar_t": float(response.dbar_t),
            "primary_norm": float(np.linalg.norm(sample.primary_matrix)),
        }
    matrix = np.asarray(response.matrix_tilde, dtype=complex)
    return {
        "frequency_sector": sample.frequency_sector,
        "matrix_tilde_real": matrix.real.tolist(),
        "matrix_tilde_imag": matrix.imag.tolist(),
        "spectral_norm": float(np.linalg.norm(matrix, ord=2)),
    }


def material_response_sample_diagnostic(
    sample: MaterialResponseSample,
) -> dict[str, Any]:
    if not isinstance(sample, MaterialResponseSample):
        raise TypeError("sample must be a MaterialResponseSample")
    operator = sample.operator_ward
    operator_payload = (
        operator.as_dict()
        if callable(getattr(operator, "as_dict", None))
        else {"passed": bool(operator.passed)}
    )
    return {
        "frequency_sector": sample.frequency_sector,
        "xi_eV_hex": float(sample.xi_eV).hex(),
        "q_crystal_hex": [float(value).hex() for value in sample.q_crystal],
        "hard_physical_passed": bool(sample.hard_physical_passed),
        "operator_ward": operator_payload,
        "effective_ward": _effective_ward_payload(sample.effective_ward),
        "strict_static_ward": _strict_static_payload(sample.strict_static_ward),
        "sheet_validation": _sheet_validation_payload(sample),
        "response": _response_payload(sample),
        "provenance": sample.provenance_payload,
    }


def _terminal_consecutive_passes(history: tuple[object, ...]) -> int:
    count = 0
    for row in reversed(history):
        if not bool(row.assessment.accepted_transition):
            break
        count += 1
    return count


def summarize_material_frequency_result(
    frequency: MaterialFrequencyResult,
    *,
    policy: MaterialResponseConvergencePolicy,
    required_consecutive_passes: int,
    envelope_levels: int,
) -> dict[str, Any]:
    """Return complete compact evidence for one established or unresolved frequency."""

    if not isinstance(frequency, MaterialFrequencyResult):
        raise TypeError("frequency must be a MaterialFrequencyResult")
    if not isinstance(policy, MaterialResponseConvergencePolicy):
        raise TypeError("policy must be a MaterialResponseConvergencePolicy")
    required = int(required_consecutive_passes)
    if required <= 0:
        raise ValueError("required_consecutive_passes must be positive")
    levels = int(envelope_levels)
    if levels < 3:
        raise ValueError("envelope_levels must be at least three")

    history = tuple(frequency.history)
    envelope = assess_material_response_envelope(
        history,
        policy=policy,
        levels=levels,
    )
    level_payloads = []
    for row in history:
        level_payloads.append(
            {
                "N": int(row.n_grid),
                "assessment": row.assessment.as_dict(),
                "samples_by_shift": {
                    label: material_response_sample_diagnostic(sample)
                    for label, sample in row.samples_by_shift.items()
                },
            }
        )

    hard_failure_levels = [
        int(row.n_grid)
        for row in history
        if not row.assessment.hard_physical_closure_across_shifts
    ]
    cross_shift_failure_levels = [
        int(row.n_grid)
        for row in history
        if not row.assessment.cross_shift_all_passed
    ]
    adjacent_failure_levels = [
        int(row.n_grid)
        for row in history[1:]
        if not row.assessment.adjacent_N_all_shifts_passed
    ]
    terminal = _terminal_consecutive_passes(history)
    blockers: list[str] = []
    if hard_failure_levels:
        blockers.append("hard_physical_closure")
    if cross_shift_failure_levels:
        blockers.append("cross_shift_response")
    if adjacent_failure_levels:
        blockers.append("adjacent_N_response")
    if terminal < required:
        blockers.append("required_consecutive_transitions")
    if not bool(envelope["passed"]):
        blockers.append("complete_pairwise_envelope")

    certification = frequency.certification
    return {
        "schema": MATERIAL_RESPONSE_UNRESOLVED_DIAGNOSTIC_SCHEMA,
        "matsubara_index": int(frequency.matsubara_index),
        "xi_eV_hex": float(frequency.xi_eV).hex(),
        "status": "established" if certification is not None else "unresolved",
        "evaluated_N": [int(row.n_grid) for row in history],
        "required_consecutive_passes": required,
        "terminal_consecutive_accepted_transitions": terminal,
        "envelope_levels": levels,
        "blockers": blockers if certification is None else [],
        "hard_physical_failure_levels": hard_failure_levels,
        "cross_shift_failure_levels": cross_shift_failure_levels,
        "adjacent_N_failure_levels": adjacent_failure_levels,
        "oscillatory_envelope": envelope,
        "levels": level_payloads,
        "certification": (
            None
            if certification is None
            else {
                "working_N": int(certification.working_N),
                "audit_N": int(certification.audit_N),
                "primary_shift": str(certification.primary_shift),
                "establishment_mode": str(certification.establishment_mode),
            }
        ),
        "diagnostic_only": True,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }


__all__ = [
    "MATERIAL_RESPONSE_UNRESOLVED_DIAGNOSTIC_SCHEMA",
    "material_response_sample_diagnostic",
    "summarize_material_frequency_result",
]
