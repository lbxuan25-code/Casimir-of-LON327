"""Exact-static Ward closure telemetry for finite-q BdG kernels.

The generic RHS-aware Ward validator is the hard gauge-closure gate. This module
adds q-normalized microscopic diagnostics and a local-LT longitudinal view. The
longitudinal residual is always recorded but is not itself a hard gate; no q,
pairing, or direction receives a special-case exemption.

The diagnostic is observational: it never projects or modifies the kernel and
never promotes a response to Casimir-ready status.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.electrodynamics.basis import xy_to_lt_rotation
from lno327.response.effective_kernel import EffectiveEMKernel
from lno327.response.ward_validation import EffectiveWardValidation

DEFAULT_STATIC_PRIMITIVE_OVER_Q_TOLERANCE = 1e-9
DEFAULT_STATIC_AMPLITUDE_OVER_Q_TOLERANCE = 1e-9
DEFAULT_STATIC_PHASE_OVER_Q_TOLERANCE = 1e-9
DEFAULT_STATIC_EFFECTIVE_DIRECT_OVER_Q_TOLERANCE = 1e-9
DEFAULT_STATIC_EFFECTIVE_RESIDUAL_OVER_Q_TOLERANCE = 1e-9
DEFAULT_STATIC_LONGITUDINAL_TOLERANCE = 1e-9
DEFAULT_STATIC_CONDITION_MAX = 1e12


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


@dataclass(frozen=True)
class StrictStaticWardClosure:
    """Q-normalized static Ward diagnostics plus nonblocking longitudinal telemetry."""

    primitive_residual_over_q: float
    amplitude_defect_over_q: float
    phase_defect_over_q: float
    effective_direct_over_q: float
    effective_residual_over_q: float
    relative_longitudinal_gauge_residual: float
    schur_condition_number: float
    schur_inverse_method: str

    primitive_tolerance: float
    amplitude_tolerance: float
    phase_tolerance: float
    effective_direct_tolerance: float
    effective_residual_tolerance: float
    longitudinal_tolerance: float
    condition_max: float

    generic_ward_passed: bool
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "primitive_residual_over_q",
            "amplitude_defect_over_q",
            "phase_defect_over_q",
            "effective_direct_over_q",
            "effective_residual_over_q",
            "relative_longitudinal_gauge_residual",
            "schur_condition_number",
            "primitive_tolerance",
            "amplitude_tolerance",
            "phase_tolerance",
            "effective_direct_tolerance",
            "effective_residual_tolerance",
            "longitudinal_tolerance",
            "condition_max",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        object.__setattr__(self, "schur_inverse_method", str(self.schur_inverse_method))
        object.__setattr__(self, "generic_ward_passed", bool(self.generic_ward_passed))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def condition_ok(self) -> bool:
        return bool(
            self.schur_inverse_method == "inv"
            and self.schur_condition_number <= self.condition_max
        )

    @property
    def longitudinal_within_tolerance(self) -> bool:
        return bool(
            self.relative_longitudinal_gauge_residual <= self.longitudinal_tolerance
        )

    @property
    def longitudinal_warning(self) -> bool:
        return not self.longitudinal_within_tolerance

    @property
    def passed(self) -> bool:
        """Return hard microscopic closure without gating on longitudinal telemetry."""

        return bool(
            self.generic_ward_passed
            and self.primitive_residual_over_q <= self.primitive_tolerance
            and self.amplitude_defect_over_q <= self.amplitude_tolerance
            and self.phase_defect_over_q <= self.phase_tolerance
            and self.effective_direct_over_q <= self.effective_direct_tolerance
            and self.effective_residual_over_q <= self.effective_residual_tolerance
            and self.condition_ok
        )

    def require_passed(self) -> None:
        if not self.passed:
            raise ValueError(
                "exact-static Ward closure failed hard validation: "
                f"generic_ward_passed={self.generic_ward_passed}, "
                f"primitive/q={self.primitive_residual_over_q:.3e}, "
                f"amplitude/q={self.amplitude_defect_over_q:.3e}, "
                f"phase/q={self.phase_defect_over_q:.3e}, "
                f"effective_direct/q={self.effective_direct_over_q:.3e}, "
                f"effective_residual/q={self.effective_residual_over_q:.3e}, "
                f"condition={self.schur_condition_number:.3e}, "
                f"inverse_method={self.schur_inverse_method}; "
                "longitudinal is diagnostic-only: "
                f"residual={self.relative_longitudinal_gauge_residual:.3e}, "
                f"tolerance={self.longitudinal_tolerance:.3e}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_residual_over_q": self.primitive_residual_over_q,
            "amplitude_defect_over_q": self.amplitude_defect_over_q,
            "phase_defect_over_q": self.phase_defect_over_q,
            "effective_direct_over_q": self.effective_direct_over_q,
            "effective_residual_over_q": self.effective_residual_over_q,
            "relative_longitudinal_gauge_residual": (
                self.relative_longitudinal_gauge_residual
            ),
            "schur_condition_number": self.schur_condition_number,
            "schur_inverse_method": self.schur_inverse_method,
            "primitive_tolerance": self.primitive_tolerance,
            "amplitude_tolerance": self.amplitude_tolerance,
            "phase_tolerance": self.phase_tolerance,
            "effective_direct_tolerance": self.effective_direct_tolerance,
            "effective_residual_tolerance": self.effective_residual_tolerance,
            "longitudinal_tolerance": self.longitudinal_tolerance,
            "longitudinal_within_tolerance": self.longitudinal_within_tolerance,
            "longitudinal_warning": self.longitudinal_warning,
            "longitudinal_is_hard_gate": False,
            "condition_max": self.condition_max,
            "condition_ok": self.condition_ok,
            "generic_ward_passed": self.generic_ward_passed,
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }


def _relative_longitudinal_residual(
    kernel: EffectiveEMKernel,
    energy_scale_eV: float,
) -> float:
    energy = float(energy_scale_eV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_scale_eV must be finite and positive")
    q = np.asarray(kernel.q_model, dtype=float)
    rotation = np.eye(3, dtype=float)
    rotation[1:3, 1:3] = xy_to_lt_rotation(float(q[0]), float(q[1]))
    kernel_lt = rotation @ np.asarray(kernel.k_eff, dtype=complex) @ rotation.T
    scaled = np.array(kernel_lt, dtype=complex, copy=True)
    scaled[0, 0] *= energy
    scaled[1:3, 1:3] /= energy
    scale = max(float(np.linalg.norm(scaled.real)), 1.0)
    entries = np.asarray(
        [scaled[0, 1], scaled[1, 0], scaled[1, 1], scaled[1, 2], scaled[2, 1]],
        dtype=complex,
    )
    return _norm(entries) / scale


def validate_strict_static_ward_closure(
    kernel: EffectiveEMKernel,
    ward: EffectiveWardValidation,
    *,
    energy_scale_eV: float = 1.0,
    primitive_tolerance: float = DEFAULT_STATIC_PRIMITIVE_OVER_Q_TOLERANCE,
    amplitude_tolerance: float = DEFAULT_STATIC_AMPLITUDE_OVER_Q_TOLERANCE,
    phase_tolerance: float = DEFAULT_STATIC_PHASE_OVER_Q_TOLERANCE,
    effective_direct_tolerance: float = DEFAULT_STATIC_EFFECTIVE_DIRECT_OVER_Q_TOLERANCE,
    effective_residual_tolerance: float = DEFAULT_STATIC_EFFECTIVE_RESIDUAL_OVER_Q_TOLERANCE,
    longitudinal_tolerance: float = DEFAULT_STATIC_LONGITUDINAL_TOLERANCE,
    condition_max: float = DEFAULT_STATIC_CONDITION_MAX,
) -> StrictStaticWardClosure:
    """Evaluate uniform exact-static closure and longitudinal telemetry.

    The generic crystal-xy Ward pass and q-normalized microscopic defects remain
    hard checks. The local-LT longitudinal residual is reported against its
    configured tolerance but never changes ``passed``.
    """

    if float(kernel.xi_eV) != 0.0 or float(ward.xi_eV) != 0.0:
        raise ValueError("strict static Ward closure requires xi_eV == 0 exactly")
    if not np.allclose(kernel.q_model, ward.q_model, rtol=0.0, atol=1e-14):
        raise ValueError("kernel and Ward validation q_model do not match")
    q_norm = float(np.linalg.norm(np.asarray(kernel.q_model, dtype=float)))
    if q_norm == 0.0:
        raise ValueError("strict static Ward closure requires nonzero q_model")

    tolerances = {
        "primitive_tolerance": primitive_tolerance,
        "amplitude_tolerance": amplitude_tolerance,
        "phase_tolerance": phase_tolerance,
        "effective_direct_tolerance": effective_direct_tolerance,
        "effective_residual_tolerance": effective_residual_tolerance,
        "longitudinal_tolerance": longitudinal_tolerance,
        "condition_max": condition_max,
    }
    checked = {
        name: _finite_nonnegative(value, name) for name, value in tolerances.items()
    }
    if checked["condition_max"] == 0.0:
        raise ValueError("condition_max must be positive")

    primitive = max(
        _norm(ward.left.primitive_residual),
        _norm(ward.right.primitive_residual),
    ) / q_norm
    amplitude = max(
        abs(complex(ward.left.collective_residual[0])),
        abs(complex(ward.right.collective_residual[0])),
    ) / q_norm
    phase = max(
        abs(complex(ward.left.collective_residual[1])),
        abs(complex(ward.right.collective_residual[1])),
    ) / q_norm
    effective_direct = max(
        _norm(ward.left.effective_direct),
        _norm(ward.right.effective_direct),
    ) / q_norm
    effective_residual = max(
        _norm(ward.left.effective_residual),
        _norm(ward.right.effective_residual),
    ) / q_norm

    condition = (
        float(kernel.schur_condition_number)
        if kernel.schur_condition_number is not None
        else float(np.linalg.cond(np.asarray(kernel.k_etaeta, dtype=complex)))
    )
    return StrictStaticWardClosure(
        primitive_residual_over_q=primitive,
        amplitude_defect_over_q=amplitude,
        phase_defect_over_q=phase,
        effective_direct_over_q=effective_direct,
        effective_residual_over_q=effective_residual,
        relative_longitudinal_gauge_residual=_relative_longitudinal_residual(
            kernel, energy_scale_eV
        ),
        schur_condition_number=condition,
        schur_inverse_method=str(kernel.schur_inverse_method),
        primitive_tolerance=checked["primitive_tolerance"],
        amplitude_tolerance=checked["amplitude_tolerance"],
        phase_tolerance=checked["phase_tolerance"],
        effective_direct_tolerance=checked["effective_direct_tolerance"],
        effective_residual_tolerance=checked["effective_residual_tolerance"],
        longitudinal_tolerance=checked["longitudinal_tolerance"],
        condition_max=checked["condition_max"],
        generic_ward_passed=bool(ward.passed),
        metadata={
            "criterion": "static_q_normalized_ward_plus_longitudinal_telemetry_v2",
            "basis": "primitive_crystal_A0_xy_with_diagnostic_LT_view",
            "projection_applied": False,
            "generic_mixed_ward_is_hard_gate": True,
            "longitudinal_is_hard_gate": False,
            "uniform_policy_all_q_pairings_directions": True,
            "valid_for_casimir_input": False,
        },
    )


__all__ = [
    "DEFAULT_STATIC_AMPLITUDE_OVER_Q_TOLERANCE",
    "DEFAULT_STATIC_CONDITION_MAX",
    "DEFAULT_STATIC_EFFECTIVE_DIRECT_OVER_Q_TOLERANCE",
    "DEFAULT_STATIC_EFFECTIVE_RESIDUAL_OVER_Q_TOLERANCE",
    "DEFAULT_STATIC_LONGITUDINAL_TOLERANCE",
    "DEFAULT_STATIC_PHASE_OVER_Q_TOLERANCE",
    "DEFAULT_STATIC_PRIMITIVE_OVER_Q_TOLERANCE",
    "StrictStaticWardClosure",
    "validate_strict_static_ward_closure",
]
