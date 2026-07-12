"""Bond-metric and strict-Ward postprocessing for adaptive d-wave integrals.

The adaptive integrand must integrate the q=0 Goldstone counterterm together with
all other primitive response blocks.  The finite-q nearest-neighbour bond metric
is therefore applied only after the complete Brillouin-zone primitive vector has
been assembled.  This module then rebuilds the amplitude/phase Schur complement,
validates the same adaptive Ward RHS, and extracts the raw static sheet response.

No projection is performed and every result remains fail-closed for Casimir use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327.electrodynamics.static_sheet import (
    StaticSheetResponse,
    static_matsubara_kernel_to_sheet_response,
)
from lno327.response.effective_kernel import (
    EffectiveEMKernel,
    effective_em_kernel_from_components,
)
from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.phase_hessian import (
    PhaseHessianApplication,
    apply_phase_hessian_policy_to_components,
)
from lno327.response.static_ward_gate import (
    StrictStaticWardClosure,
    validate_strict_static_ward_closure,
)
from lno327.response.ward_validation import (
    EffectiveWardValidation,
    PrimitiveWardRHS,
    validate_effective_ward_xy,
)


@dataclass(frozen=True)
class AdaptiveStaticValidationConfig:
    """Numerical gates applied after one complete adaptive primitive integral."""

    mixed_ward_tolerance: float = 1e-7
    mixed_ward_absolute_tolerance: float = 1e-12
    primitive_tolerance: float = 1e-6
    amplitude_tolerance: float = 1e-6
    phase_tolerance: float = 1e-6
    effective_direct_tolerance: float = 1e-6
    effective_residual_tolerance: float = 1e-6
    longitudinal_tolerance: float = 1e-6
    condition_max: float = 1e12
    reality_tolerance: float = 1e-8
    mixing_tolerance: float = 1e-6
    passivity_tolerance: float = 1e-10
    energy_scale_eV: float = 1.0
    degeneracy: float = 1.0

    def __post_init__(self) -> None:
        nonnegative = (
            "mixed_ward_tolerance",
            "mixed_ward_absolute_tolerance",
            "primitive_tolerance",
            "amplitude_tolerance",
            "phase_tolerance",
            "effective_direct_tolerance",
            "effective_residual_tolerance",
            "longitudinal_tolerance",
            "reality_tolerance",
            "mixing_tolerance",
            "passivity_tolerance",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("condition_max", "energy_scale_eV", "degeneracy"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AdaptiveBondMetricStaticResult:
    """Corrected response and validation records for one adaptive order."""

    components: BdGFiniteQResponseComponents
    kernel: EffectiveEMKernel
    ward: EffectiveWardValidation
    strict: StrictStaticWardClosure
    sheet: StaticSheetResponse
    application: PhaseHessianApplication

    def to_row_fields(self) -> dict[str, Any]:
        strict_fields = {
            key: value
            for key, value in self.strict.to_dict().items()
            if key not in {"metadata", "passed"}
        }
        return {
            "phase_hessian_policy": self.application.policy,
            "phase_hessian_multiplier": self.application.multiplier,
            "phase_hessian_base_counterterm_22_real": float(
                self.application.base_counterterm[1, 1].real
            ),
            "phase_hessian_base_counterterm_22_imag": float(
                self.application.base_counterterm[1, 1].imag
            ),
            "phase_hessian_applied_counterterm_22_real": float(
                self.application.applied_counterterm[1, 1].real
            ),
            "phase_hessian_applied_counterterm_22_imag": float(
                self.application.applied_counterterm[1, 1].imag
            ),
            "phase_hessian_changed_only_22": bool(
                self.components.metadata.get("phase_hessian_changed_only_22", False)
            ),
            "ward_passed": bool(self.ward.passed),
            "ward_primitive_mixed_ratio_max": max(
                self.ward.left.primitive_mixed_ratio,
                self.ward.right.primitive_mixed_ratio,
            ),
            "ward_effective_mixed_ratio_max": max(
                self.ward.left.effective_mixed_ratio,
                self.ward.right.effective_mixed_ratio,
            ),
            "strict_gate_passed": bool(self.strict.passed),
            **strict_fields,
            "sheet_validation_passed": bool(self.sheet.validation.passed),
            "relative_imaginary_norm": float(
                self.sheet.validation.relative_imaginary_norm
            ),
            "relative_density_transverse_mixing": float(
                self.sheet.validation.relative_density_transverse_mixing
            ),
            "chi_bar": float(self.sheet.chi_bar),
            "dbar_t": float(self.sheet.dbar_t),
            "projection_applied": False,
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        }


def postprocess_adaptive_bond_metric_static(
    components: BdGFiniteQResponseComponents,
    rhs: PrimitiveWardRHS,
    *,
    ansatz: object,
    q_model: np.ndarray,
    config: AdaptiveStaticValidationConfig,
) -> AdaptiveBondMetricStaticResult:
    """Apply the diagnosed bond metric after integration and validate fail-closed."""

    q = np.asarray(q_model, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all() or float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_model must be a finite nonzero vector with shape (2,)")
    if not np.allclose(rhs.q_model, q, rtol=0.0, atol=1e-14):
        raise ValueError("adaptive components and Ward RHS must share q_model")
    if components.metadata.get("phase_hessian_policy") == "nearest_neighbor_bond_metric":
        raise ValueError("adaptive primitive components already have the bond metric applied")

    corrected, application = apply_phase_hessian_policy_to_components(
        components,
        ansatz,
        q,
        "nearest_neighbor_bond_metric",
        condition_threshold=config.condition_max,
    )
    kernel = effective_em_kernel_from_components(
        corrected,
        q_model=q,
        xi_eV=0.0,
    )
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=config.mixed_ward_tolerance,
        absolute_residual_tolerance=config.mixed_ward_absolute_tolerance,
        condition_max=config.condition_max,
    )
    strict = validate_strict_static_ward_closure(
        kernel,
        ward,
        energy_scale_eV=config.energy_scale_eV,
        primitive_tolerance=config.primitive_tolerance,
        amplitude_tolerance=config.amplitude_tolerance,
        phase_tolerance=config.phase_tolerance,
        effective_direct_tolerance=config.effective_direct_tolerance,
        effective_residual_tolerance=config.effective_residual_tolerance,
        longitudinal_tolerance=config.longitudinal_tolerance,
        condition_max=config.condition_max,
    )
    sheet = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        energy_scale_eV=config.energy_scale_eV,
        degeneracy=config.degeneracy,
        reality_tolerance=config.reality_tolerance,
        longitudinal_tolerance=config.longitudinal_tolerance,
        mixing_tolerance=config.mixing_tolerance,
        passivity_tolerance=config.passivity_tolerance,
    )
    return AdaptiveBondMetricStaticResult(
        components=corrected,
        kernel=kernel,
        ward=ward,
        strict=strict,
        sheet=sheet,
        application=application,
    )


__all__ = [
    "AdaptiveBondMetricStaticResult",
    "AdaptiveStaticValidationConfig",
    "postprocess_adaptive_bond_metric_static",
]
