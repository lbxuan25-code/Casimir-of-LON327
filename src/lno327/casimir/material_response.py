"""Geometry-independent finite-temperature material-response boundary.

This module owns the conversion from an integrated arbitrary-q microscopic result
into a typed sheet response in the crystal frame. It deliberately contains no
plate rotation, reflection, propagation, two-plate logdet, or outer-integration
logic. The separation is structural: later caches and interpolants may depend on
material identity without accidentally depending on a Casimir geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np

from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetResponseValidation,
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.electrodynamics.static_sheet import (
    StaticSheetResponse,
    StaticSheetValidation,
    static_matsubara_kernel_to_sheet_response,
)
from lno327.response.effective_kernel import (
    EffectiveEMKernel,
    effective_em_kernel_from_components,
)
from lno327.response.primitive_kernel import OperatorWardReport
from lno327.response.static_ward_gate import (
    StrictStaticWardClosure,
    validate_strict_static_ward_closure,
)
from lno327.response.ward_validation import (
    EffectiveWardValidation,
    validate_effective_ward_xy,
)
from lno327.workflows.arbitrary_q_matsubara import ArbitraryQPeriodicBZResult

MATERIAL_RESPONSE_SAMPLE_SCHEMA = "material-response-sample-v1"
FrequencySector: TypeAlias = Literal["zero_matsubara", "positive_matsubara"]
SheetResponse: TypeAlias = StaticSheetResponse | PositiveMatsubaraSheetResponse
SheetValidation: TypeAlias = StaticSheetValidation | SheetResponseValidation


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _finite_positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _readonly_q(value: np.ndarray) -> np.ndarray:
    q = np.array(value, dtype=float, copy=True)
    if q.shape != (2,):
        raise ValueError(f"q_crystal must have shape (2,), got {q.shape}")
    if not np.isfinite(q).all() or float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_crystal must be finite and nonzero")
    q.setflags(write=False)
    return q


@dataclass(frozen=True)
class MaterialResponsePolicy:
    """Physical validation policy for one geometry-independent response sample."""

    degeneracy: float = 1.0
    ward_tolerance: float = 1e-7
    ward_absolute_tolerance: float = 1e-12
    condition_max: float = 1e12
    static_energy_scale_eV: float = 1.0
    static_primitive_tolerance: float = 1e-6
    static_amplitude_tolerance: float = 1e-6
    static_phase_tolerance: float = 1e-6
    static_effective_direct_tolerance: float = 1e-6
    static_effective_residual_tolerance: float = 1e-6
    static_reality_tolerance: float = 1e-8
    static_longitudinal_tolerance: float = 1e-6
    static_mixing_tolerance: float = 1e-6
    static_passivity_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "degeneracy",
            _finite_positive(self.degeneracy, "degeneracy"),
        )
        object.__setattr__(
            self,
            "condition_max",
            _finite_positive(self.condition_max, "condition_max"),
        )
        object.__setattr__(
            self,
            "static_energy_scale_eV",
            _finite_positive(self.static_energy_scale_eV, "static_energy_scale_eV"),
        )
        for name in (
            "ward_tolerance",
            "ward_absolute_tolerance",
            "static_primitive_tolerance",
            "static_amplitude_tolerance",
            "static_phase_tolerance",
            "static_effective_direct_tolerance",
            "static_effective_residual_tolerance",
            "static_reality_tolerance",
            "static_longitudinal_tolerance",
            "static_mixing_tolerance",
            "static_passivity_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "schema": "material-response-policy-v1",
            "degeneracy": self.degeneracy,
            "ward_tolerance": self.ward_tolerance,
            "ward_absolute_tolerance": self.ward_absolute_tolerance,
            "condition_max": self.condition_max,
            "static_energy_scale_eV": self.static_energy_scale_eV,
            "static_primitive_tolerance": self.static_primitive_tolerance,
            "static_amplitude_tolerance": self.static_amplitude_tolerance,
            "static_phase_tolerance": self.static_phase_tolerance,
            "static_effective_direct_tolerance": (
                self.static_effective_direct_tolerance
            ),
            "static_effective_residual_tolerance": (
                self.static_effective_residual_tolerance
            ),
            "static_reality_tolerance": self.static_reality_tolerance,
            "static_longitudinal_tolerance": self.static_longitudinal_tolerance,
            "static_mixing_tolerance": self.static_mixing_tolerance,
            "static_passivity_tolerance": self.static_passivity_tolerance,
        }


@dataclass(frozen=True)
class MaterialResponseSample:
    """One N/shift material response before response-level certification."""

    frequency_index: int
    frequency_sector: FrequencySector
    q_crystal: np.ndarray
    xi_eV: float
    material_cache_fingerprint: str
    kernel: EffectiveEMKernel
    operator_ward: OperatorWardReport
    effective_ward: EffectiveWardValidation
    strict_static_ward: StrictStaticWardClosure | None
    response: SheetResponse
    sheet_validation: SheetValidation
    metadata: Mapping[str, Any]
    schema: str = MATERIAL_RESPONSE_SAMPLE_SCHEMA

    def __post_init__(self) -> None:
        index = int(self.frequency_index)
        if index < 0:
            raise ValueError("frequency_index must be non-negative")
        object.__setattr__(self, "frequency_index", index)

        if self.schema != MATERIAL_RESPONSE_SAMPLE_SCHEMA:
            raise ValueError(f"schema must be {MATERIAL_RESPONSE_SAMPLE_SCHEMA!r}")
        if self.frequency_sector not in (
            "zero_matsubara",
            "positive_matsubara",
        ):
            raise ValueError("unsupported frequency_sector")

        q = _readonly_q(self.q_crystal)
        object.__setattr__(self, "q_crystal", q)
        xi = _finite_nonnegative(self.xi_eV, "xi_eV")
        object.__setattr__(self, "xi_eV", xi)

        fingerprint = str(self.material_cache_fingerprint)
        if not fingerprint:
            raise ValueError("material_cache_fingerprint must be non-empty")
        object.__setattr__(self, "material_cache_fingerprint", fingerprint)

        if not np.array_equal(np.asarray(self.kernel.q_model, dtype=float), q):
            raise ValueError("kernel q_model does not match q_crystal")
        if float(self.kernel.xi_eV) != xi:
            raise ValueError("kernel xi_eV does not match sample xi_eV")
        if not np.array_equal(np.asarray(self.response.q_model, dtype=float), q):
            raise ValueError("sheet response q_model does not match q_crystal")

        if self.frequency_sector == "zero_matsubara":
            if xi != 0.0:
                raise ValueError("zero_matsubara sample requires xi_eV == 0")
            if not isinstance(self.response, StaticSheetResponse):
                raise TypeError("zero_matsubara sample requires StaticSheetResponse")
            if not isinstance(self.sheet_validation, StaticSheetValidation):
                raise TypeError("zero_matsubara sample requires StaticSheetValidation")
            if self.strict_static_ward is None:
                raise ValueError(
                    "zero_matsubara sample requires strict_static_ward telemetry"
                )
        else:
            if xi <= 0.0:
                raise ValueError("positive_matsubara sample requires xi_eV > 0")
            if not isinstance(self.response, PositiveMatsubaraSheetResponse):
                raise TypeError(
                    "positive_matsubara sample requires "
                    "PositiveMatsubaraSheetResponse"
                )
            if not isinstance(self.sheet_validation, SheetResponseValidation):
                raise TypeError(
                    "positive_matsubara sample requires SheetResponseValidation"
                )
            if self.strict_static_ward is not None:
                raise ValueError(
                    "positive_matsubara sample cannot carry strict_static_ward"
                )
            if float(self.response.xi_eV) != xi:
                raise ValueError(
                    "positive sheet response xi_eV does not match sample xi_eV"
                )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def primary_matrix(self) -> np.ndarray:
        if isinstance(self.response, StaticSheetResponse):
            matrix = np.diag(
                [float(self.response.chi_bar), float(self.response.dbar_t)]
            ).astype(complex)
        else:
            matrix = np.array(self.response.matrix_tilde, dtype=complex, copy=True)
        matrix.setflags(write=False)
        return matrix

    @property
    def hard_physical_passed(self) -> bool:
        return bool(
            self.operator_ward.passed
            and self.effective_ward.passed
            and self.sheet_validation.passed
        )

    def diagnostics(self) -> dict[str, Any]:
        ward_ratio = max(
            self.effective_ward.left.effective_mixed_ratio,
            self.effective_ward.right.effective_mixed_ratio,
        )
        state: dict[str, Any] = {
            "material_response_schema": self.schema,
            "frequency_index": self.frequency_index,
            "frequency_sector": self.frequency_sector,
            "xi_eV": self.xi_eV,
            "q_crystal": self.q_crystal.tolist(),
            "material_cache_fingerprint": self.material_cache_fingerprint,
            "operator_ward_passed": bool(self.operator_ward.passed),
            "ward_passed": bool(self.effective_ward.passed),
            "ward_effective_mixed_ratio_max": float(ward_ratio),
            "schur_condition_number": float(
                self.effective_ward.schur_condition_number
            ),
            "sheet_validation_passed": bool(self.sheet_validation.passed),
            "material_hard_physical_passed": self.hard_physical_passed,
            "primary_norm": float(np.linalg.norm(self.primary_matrix)),
        }
        if isinstance(self.response, StaticSheetResponse):
            validation = self.response.validation
            state.update(
                {
                    "strict_static_ward_passed": bool(
                        self.strict_static_ward.passed
                    ),
                    "strict_static_hard_gate": False,
                    "chi_bar": float(self.response.chi_bar),
                    "dbar_t": float(self.response.dbar_t),
                    "static_longitudinal_residual": float(
                        validation.relative_longitudinal_gauge_residual
                    ),
                    "static_longitudinal_tolerance": float(
                        validation.longitudinal_tolerance
                    ),
                    "static_longitudinal_warning": bool(
                        validation.longitudinal_warning
                    ),
                    "relative_imaginary_norm": float(
                        validation.relative_imaginary_norm
                    ),
                    "relative_density_transverse_mixing": float(
                        validation.relative_density_transverse_mixing
                    ),
                }
            )
        else:
            state.update(
                {
                    "strict_static_ward_passed": False,
                    "strict_static_hard_gate": False,
                    "chi_bar": float("nan"),
                    "dbar_t": float("nan"),
                    "static_longitudinal_residual": float("nan"),
                    "static_longitudinal_tolerance": float("nan"),
                    "static_longitudinal_warning": False,
                    "relative_imaginary_norm": float(
                        self.sheet_validation.relative_imaginary_norm
                    ),
                    "relative_density_transverse_mixing": float("nan"),
                }
            )
        return state


def build_material_response_sample(
    result: ArbitraryQPeriodicBZResult,
    *,
    frequency_index: int,
    policy: MaterialResponsePolicy,
) -> MaterialResponseSample:
    """Convert one integrated microscopic frequency into a crystal-frame response."""

    index = int(frequency_index)
    if index < 0 or index >= len(result.components):
        raise IndexError("frequency_index is outside the integrated result")

    xi_eV = float(result.xi_eV_values[index])
    q_crystal = np.asarray(result.q_model, dtype=float)
    kernel = effective_em_kernel_from_components(
        result.components[index],
        q_model=q_crystal,
        xi_eV=xi_eV,
    )
    ward = validate_effective_ward_xy(
        kernel,
        result.rhs[index],
        residual_tolerance=policy.ward_tolerance,
        absolute_residual_tolerance=policy.ward_absolute_tolerance,
        condition_max=policy.condition_max,
    )

    if xi_eV == 0.0:
        sector: FrequencySector = "zero_matsubara"
        strict = validate_strict_static_ward_closure(
            kernel,
            ward,
            energy_scale_eV=policy.static_energy_scale_eV,
            primitive_tolerance=policy.static_primitive_tolerance,
            amplitude_tolerance=policy.static_amplitude_tolerance,
            phase_tolerance=policy.static_phase_tolerance,
            effective_direct_tolerance=(
                policy.static_effective_direct_tolerance
            ),
            effective_residual_tolerance=(
                policy.static_effective_residual_tolerance
            ),
            longitudinal_tolerance=policy.static_longitudinal_tolerance,
            condition_max=policy.condition_max,
        )
        response: SheetResponse = static_matsubara_kernel_to_sheet_response(
            kernel,
            ward,
            energy_scale_eV=policy.static_energy_scale_eV,
            degeneracy=policy.degeneracy,
            reality_tolerance=policy.static_reality_tolerance,
            longitudinal_tolerance=policy.static_longitudinal_tolerance,
            mixing_tolerance=policy.static_mixing_tolerance,
            passivity_tolerance=policy.static_passivity_tolerance,
        )
        validation: SheetValidation = response.validation
    else:
        sector = "positive_matsubara"
        strict = None
        response = positive_matsubara_kernel_to_sheet_response(
            kernel,
            degeneracy=policy.degeneracy,
        )
        validation = validate_positive_matsubara_sheet_response(response)

    source_metadata = dict(result.metadata)
    metadata = {
        "source": "ArbitraryQPeriodicBZResult",
        "casimir_stage": "geometry_independent_material_response",
        "frequency_sector": sector,
        "basis": getattr(response, "basis", "unknown"),
        "post_integral_phase_hessian_policy": source_metadata.get(
            "post_integral_phase_hessian_policy"
        ),
        "canonical_reduction_block_size": source_metadata.get(
            "canonical_reduction_block_size"
        ),
        "primitive_contract_version": source_metadata.get(
            "primitive_contract_version"
        ),
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    return MaterialResponseSample(
        frequency_index=index,
        frequency_sector=sector,
        q_crystal=q_crystal,
        xi_eV=xi_eV,
        material_cache_fingerprint=result.material_cache_fingerprint,
        kernel=kernel,
        operator_ward=result.operator_ward,
        effective_ward=ward,
        strict_static_ward=strict,
        response=response,
        sheet_validation=validation,
        metadata=metadata,
    )


__all__ = [
    "MATERIAL_RESPONSE_SAMPLE_SCHEMA",
    "FrequencySector",
    "MaterialResponsePolicy",
    "MaterialResponseSample",
    "SheetResponse",
    "SheetValidation",
    "build_material_response_sample",
]
