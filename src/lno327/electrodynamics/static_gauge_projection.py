"""Validated longitudinal gauge projection for exact zero-Matsubara kernels.

The exact ``xi=0``, nonzero-q effective kernel has a pure-gauge longitudinal
vector-potential direction after the collective Schur correction.  Finite
Brillouin-zone quadrature can leave a small longitudinal row/column because two
independently integrated O(1) terms must cancel.  This module provides an
explicit, fail-closed projection onto the physical ``(A0,T)`` subspace.

Projection is never inferred silently.  The caller must select
``project_after_validated_ward`` and the raw response must first pass the mixed
Ward, Schur-condition, reality, density-transverse mixing, passivity, and raw
longitudinal-ceiling checks.  Positive Matsubara frequencies are rejected by
the underlying static-sheet contract.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
    static_matsubara_kernel_to_sheet_response,
)
from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import EffectiveWardValidation

RAW_FAIL_CLOSED = "raw_fail_closed"
PROJECT_AFTER_VALIDATED_WARD = "project_after_validated_ward"
StaticLongitudinalPolicy = Literal[
    "raw_fail_closed",
    "project_after_validated_ward",
]

DEFAULT_PROJECTION_RAW_LONGITUDINAL_CEILING = 1e-5


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _readonly_complex_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=complex, copy=True)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def _readonly_real_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def _dimensionless_static_kernel(kernel_lt: np.ndarray, energy_scale_eV: float) -> np.ndarray:
    matrix = np.asarray(kernel_lt, dtype=complex).copy()
    matrix[0, 0] *= float(energy_scale_eV)
    matrix[1:3, 1:3] /= float(energy_scale_eV)
    return matrix


def _longitudinal_relative_norm(scaled: np.ndarray, scale: float) -> float:
    entries = np.asarray(
        [
            scaled[0, 1],
            scaled[1, 0],
            scaled[1, 1],
            scaled[1, 2],
            scaled[2, 1],
        ],
        dtype=complex,
    )
    return float(np.linalg.norm(entries) / max(float(scale), 1e-30))


def static_longitudinal_gauge_projector_lt() -> np.ndarray:
    """Return the exact-static projector retaining only ``A0`` and ``A_T``."""

    return _readonly_real_matrix(np.diag([1.0, 0.0, 1.0]), "projector_lt")


def _response_with_metadata(
    response: StaticSheetResponse,
    updates: dict[str, object],
) -> StaticSheetResponse:
    metadata = dict(response.metadata)
    metadata.update(updates)
    return StaticSheetResponse(
        kernel_lt=response.kernel_lt,
        chi_bar=response.chi_bar,
        dbar_t=response.dbar_t,
        q_model=response.q_model,
        energy_scale_eV=response.energy_scale_eV,
        degeneracy=response.degeneracy,
        basis=response.basis,
        validation=response.validation,
        metadata=metadata,
    )


def _projection_prerequisite_error(
    response: StaticSheetResponse,
    ward_validation: EffectiveWardValidation,
    raw_ceiling: float,
) -> ValueError:
    validation = response.validation
    return ValueError(
        "static longitudinal gauge projection prerequisites failed: "
        f"ward_passed={ward_validation.passed}, "
        f"condition_ok={ward_validation.condition_ok}, "
        f"inverse_method={ward_validation.schur_inverse_method}, "
        f"raw_validation_passed={validation.passed}, "
        "raw_relative_longitudinal_gauge_residual="
        f"{validation.relative_longitudinal_gauge_residual:.3e}, "
        f"raw_longitudinal_ceiling={raw_ceiling:.3e}, "
        f"relative_imaginary_norm={validation.relative_imaginary_norm:.3e}, "
        "relative_density_transverse_mixing="
        f"{validation.relative_density_transverse_mixing:.3e}, "
        f"chi_bar={validation.chi_bar:.3e}, dbar_t={validation.dbar_t:.3e}"
    )


def static_matsubara_kernel_to_sheet_response_with_policy(
    kernel: EffectiveEMKernel,
    ward_validation: EffectiveWardValidation,
    *,
    longitudinal_policy: StaticLongitudinalPolicy = RAW_FAIL_CLOSED,
    projection_raw_longitudinal_ceiling: float = (
        DEFAULT_PROJECTION_RAW_LONGITUDINAL_CEILING
    ),
    energy_scale_eV: float = 1.0,
    degeneracy: float = 1.0,
    frequency_tolerance_eV: float = 1e-14,
    reality_tolerance: float = 1e-9,
    longitudinal_tolerance: float = 1e-7,
    mixing_tolerance: float = 1e-7,
    passivity_tolerance: float = 1e-10,
) -> StaticSheetResponse:
    """Convert an exact-static kernel under an explicit longitudinal policy.

    ``raw_fail_closed`` preserves the original static-sheet contract.

    ``project_after_validated_ward`` first validates the unprojected response
    against ``projection_raw_longitudinal_ceiling`` and all other physical
    gates.  It then applies ``P K P`` with ``P=diag(1,0,1)`` in local
    ``(A0,L,T)``.  The returned response contains the projected kernel, while
    its metadata retains the raw kernel and all projection diagnostics.
    """

    policy = str(longitudinal_policy)
    if policy not in {RAW_FAIL_CLOSED, PROJECT_AFTER_VALIDATED_WARD}:
        raise ValueError(
            "longitudinal_policy must be 'raw_fail_closed' or "
            "'project_after_validated_ward'"
        )

    target_longitudinal = _finite_nonnegative(
        longitudinal_tolerance, "longitudinal_tolerance"
    )
    raw_ceiling = _finite_nonnegative(
        projection_raw_longitudinal_ceiling,
        "projection_raw_longitudinal_ceiling",
    )
    if raw_ceiling < target_longitudinal:
        raise ValueError(
            "projection_raw_longitudinal_ceiling must be greater than or equal "
            "to longitudinal_tolerance"
        )

    if policy == RAW_FAIL_CLOSED:
        raw = static_matsubara_kernel_to_sheet_response(
            kernel,
            ward_validation,
            energy_scale_eV=energy_scale_eV,
            degeneracy=degeneracy,
            frequency_tolerance_eV=frequency_tolerance_eV,
            reality_tolerance=reality_tolerance,
            longitudinal_tolerance=target_longitudinal,
            mixing_tolerance=mixing_tolerance,
            passivity_tolerance=passivity_tolerance,
        )
        return _response_with_metadata(
            raw,
            {
                "static_longitudinal_policy": RAW_FAIL_CLOSED,
                "gauge_projection_applied": False,
                "raw_relative_longitudinal_gauge_residual": (
                    raw.validation.relative_longitudinal_gauge_residual
                ),
                "projection_raw_longitudinal_ceiling": raw_ceiling,
            },
        )

    raw = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward_validation,
        energy_scale_eV=energy_scale_eV,
        degeneracy=degeneracy,
        frequency_tolerance_eV=frequency_tolerance_eV,
        reality_tolerance=reality_tolerance,
        longitudinal_tolerance=raw_ceiling,
        mixing_tolerance=mixing_tolerance,
        passivity_tolerance=passivity_tolerance,
    )

    if not ward_validation.passed or not ward_validation.condition_ok:
        raise _projection_prerequisite_error(raw, ward_validation, raw_ceiling)
    if ward_validation.schur_inverse_method != "inv":
        raise _projection_prerequisite_error(raw, ward_validation, raw_ceiling)
    if not raw.validation.passed:
        raise _projection_prerequisite_error(raw, ward_validation, raw_ceiling)

    projector = static_longitudinal_gauge_projector_lt()
    raw_kernel_lt = np.asarray(raw.kernel_lt, dtype=complex)
    projected_kernel_lt = projector @ raw_kernel_lt @ projector

    physical_indices = np.ix_([0, 2], [0, 2])
    if not np.array_equal(
        projected_kernel_lt[physical_indices],
        raw_kernel_lt[physical_indices],
    ):
        raise RuntimeError("static gauge projection changed the physical A0/T block")

    scaled_raw = _dimensionless_static_kernel(raw_kernel_lt, raw.energy_scale_eV)
    scaled_projected = _dimensionless_static_kernel(
        projected_kernel_lt,
        raw.energy_scale_eV,
    )
    raw_scale = max(float(np.linalg.norm(scaled_raw.real)), 1.0)
    projected_scale = max(float(np.linalg.norm(scaled_projected.real)), 1.0)
    projected_longitudinal = _longitudinal_relative_norm(
        scaled_projected,
        projected_scale,
    )
    projection_correction = float(
        np.linalg.norm(scaled_projected - scaled_raw) / raw_scale
    )
    projected_imaginary = float(
        np.linalg.norm(scaled_projected.imag) / projected_scale
    )
    projected_density_transverse = float(
        np.linalg.norm(
            np.asarray(
                [scaled_projected[0, 2], scaled_projected[2, 0]],
                dtype=complex,
            )
        )
        / projected_scale
    )

    # Reality and density-transverse gates deliberately retain the worse of raw
    # and projected diagnostics so the gauge projection cannot hide another
    # failure confined to the longitudinal row/column.
    validation = StaticSheetValidation(
        finite=bool(
            raw.validation.finite
            and np.isfinite(projected_kernel_lt.real).all()
            and np.isfinite(projected_kernel_lt.imag).all()
        ),
        ward_passed=bool(ward_validation.passed),
        relative_imaginary_norm=max(
            raw.validation.relative_imaginary_norm,
            projected_imaginary,
        ),
        relative_longitudinal_gauge_residual=projected_longitudinal,
        relative_density_transverse_mixing=max(
            raw.validation.relative_density_transverse_mixing,
            projected_density_transverse,
        ),
        chi_bar=raw.chi_bar,
        dbar_t=raw.dbar_t,
        reality_tolerance=float(reality_tolerance),
        longitudinal_tolerance=target_longitudinal,
        mixing_tolerance=float(mixing_tolerance),
        passivity_tolerance=float(passivity_tolerance),
    )
    if not validation.passed:
        raise RuntimeError(
            "projected zero-Matsubara sheet response failed physical validation"
        )

    readonly_raw = _readonly_complex_matrix(raw_kernel_lt, "raw_kernel_lt")
    readonly_projected = _readonly_complex_matrix(
        projected_kernel_lt,
        "projected_kernel_lt",
    )
    metadata = dict(raw.metadata)
    metadata.update(
        {
            "static_longitudinal_policy": PROJECT_AFTER_VALIDATED_WARD,
            "gauge_projection_applied": True,
            "gauge_projection_formula": "K_projected = P_static K_raw P_static",
            "gauge_projection_basis": STATIC_LOCAL_BASIS,
            "gauge_projection_scope": "exact xi=0 and nonzero q only",
            "gauge_projection_matrix_lt": projector,
            "raw_kernel_lt": readonly_raw,
            "projected_kernel_lt": readonly_projected,
            "raw_relative_longitudinal_gauge_residual": (
                raw.validation.relative_longitudinal_gauge_residual
            ),
            "projected_relative_longitudinal_gauge_residual": projected_longitudinal,
            "relative_projection_correction_norm": projection_correction,
            "projection_raw_longitudinal_ceiling": raw_ceiling,
            "target_longitudinal_tolerance": target_longitudinal,
            "projection_prerequisites": {
                "mixed_ward_passed": bool(ward_validation.passed),
                "schur_condition_ok": bool(ward_validation.condition_ok),
                "schur_inverse_method": ward_validation.schur_inverse_method,
                "raw_static_validation_passed_under_ceiling": bool(
                    raw.validation.passed
                ),
                "raw_reality_passed": bool(
                    raw.validation.relative_imaginary_norm
                    <= raw.validation.reality_tolerance
                ),
                "raw_density_transverse_mixing_passed": bool(
                    raw.validation.relative_density_transverse_mixing
                    <= raw.validation.mixing_tolerance
                ),
                "raw_passivity_passed": bool(
                    raw.validation.chi_bar >= -raw.validation.passivity_tolerance
                    and raw.validation.dbar_t >= -raw.validation.passivity_tolerance
                ),
            },
            "physical_A0_T_block_preserved_exactly": True,
            "chi_bar_unchanged_by_projection": True,
            "dbar_t_unchanged_by_projection": True,
            "positive_matsubara_projection_forbidden": True,
        }
    )
    return StaticSheetResponse(
        kernel_lt=projected_kernel_lt,
        chi_bar=raw.chi_bar,
        dbar_t=raw.dbar_t,
        q_model=raw.q_model,
        energy_scale_eV=raw.energy_scale_eV,
        degeneracy=raw.degeneracy,
        basis=raw.basis,
        validation=validation,
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_PROJECTION_RAW_LONGITUDINAL_CEILING",
    "PROJECT_AFTER_VALIDATED_WARD",
    "RAW_FAIL_CLOSED",
    "StaticLongitudinalPolicy",
    "static_longitudinal_gauge_projector_lt",
    "static_matsubara_kernel_to_sheet_response_with_policy",
]
