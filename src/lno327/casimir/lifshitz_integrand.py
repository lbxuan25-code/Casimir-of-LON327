"""Main-pipeline Lifshitz trace-log single-point helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.electrodynamics.reflection import LAB_LT_TANGENTIAL_E_BASIS, SheetReflection

from .readiness import round_trip_factor, trace_log_integrand, trace_log_matrix


def _readonly_complex_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=complex, copy=True)
    if matrix.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {matrix.shape}")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


def _readonly_real_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class LifshitzPoint:
    """Signed real trace-log result for two passive positive-frequency sheets."""

    logdet: float
    trace_log_matrix: np.ndarray
    product_eigenvalues: np.ndarray
    round_trip_eigenvalues: np.ndarray
    propagation_factor: float
    kappa_m_inv: float
    separation_m: float
    basis: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = float(self.logdet)
        if not np.isfinite(value):
            raise ValueError("logdet must be finite")
        object.__setattr__(self, "logdet", value)
        object.__setattr__(
            self,
            "trace_log_matrix",
            _readonly_complex_matrix(self.trace_log_matrix, "trace_log_matrix"),
        )
        object.__setattr__(
            self,
            "product_eigenvalues",
            _readonly_real_vector(self.product_eigenvalues, "product_eigenvalues"),
        )
        object.__setattr__(
            self,
            "round_trip_eigenvalues",
            _readonly_real_vector(self.round_trip_eigenvalues, "round_trip_eigenvalues"),
        )
        propagation = float(self.propagation_factor)
        if not np.isfinite(propagation) or propagation < 0.0 or propagation > 1.0:
            raise ValueError("propagation_factor must lie in [0, 1]")
        object.__setattr__(self, "propagation_factor", propagation)
        for name in ("kappa_m_inv", "separation_m"):
            scalar = float(getattr(self, name))
            if not np.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, scalar)
        if self.basis != LAB_LT_TANGENTIAL_E_BASIS:
            raise ValueError(f"basis must be {LAB_LT_TANGENTIAL_E_BASIS!r}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _compatible_reflections(
    reflection_1: SheetReflection,
    reflection_2: SheetReflection,
    tolerance: float,
) -> None:
    if reflection_1.basis != LAB_LT_TANGENTIAL_E_BASIS or reflection_2.basis != LAB_LT_TANGENTIAL_E_BASIS:
        raise ValueError("both reflections must use the common lab LT tangential-E basis")
    if not reflection_1.sheet_validation.passed or not reflection_2.sheet_validation.passed:
        raise ValueError("passive signed logdet requires both sheet validations to pass")
    if not np.allclose(
        reflection_1.q_lab_si_m_inv,
        reflection_2.q_lab_si_m_inv,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("reflection wavevectors do not match")
    if not np.isclose(reflection_1.xi_si_s_inv, reflection_2.xi_si_s_inv, rtol=tolerance, atol=tolerance):
        raise ValueError("reflection frequencies do not match")
    if not np.isclose(reflection_1.kappa_m_inv, reflection_2.kappa_m_inv, rtol=tolerance, atol=tolerance):
        raise ValueError("reflection vacuum kappa values do not match")


def passive_sheet_logdet(
    reflection_1: SheetReflection,
    reflection_2: SheetReflection,
    *,
    separation_m: float,
    compatibility_tolerance: float = 1e-11,
    eigenvalue_imag_tolerance: float = 1e-9,
    eigenvalue_lower_tolerance: float = 1e-10,
) -> LifshitzPoint:
    """Return the signed real LT trace-log for reciprocal passive sheets.

    For the admitted passive-sheet domain the two eigenvalues of ``R1 @ R2``
    must be real and non-negative.  The propagated round-trip eigenvalues must
    lie in ``[0, 1)``.  The implementation therefore evaluates
    ``sum(log1p(-lambda_round_trip))`` and treats a complex determinant or a
    branch-cut crossing as a hard error instead of discarding an imaginary part.
    """

    compatibility = float(compatibility_tolerance)
    imag_tolerance = float(eigenvalue_imag_tolerance)
    lower_tolerance = float(eigenvalue_lower_tolerance)
    if compatibility < 0.0 or imag_tolerance < 0.0 or lower_tolerance < 0.0:
        raise ValueError("logdet tolerances must be non-negative")
    _compatible_reflections(reflection_1, reflection_2, compatibility)

    separation = float(separation_m)
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("separation_m must be finite and positive")
    kappa = 0.5 * (reflection_1.kappa_m_inv + reflection_2.kappa_m_inv)
    propagation = round_trip_factor(kappa, separation)
    product = reflection_1.matrix_lt @ reflection_2.matrix_lt
    raw_eigenvalues = np.linalg.eigvals(product)
    eigenvalue_scale = max(float(np.max(np.abs(raw_eigenvalues))), 1.0)
    maximum_imaginary = float(np.max(np.abs(raw_eigenvalues.imag)))
    if maximum_imaginary > imag_tolerance * eigenvalue_scale:
        raise ValueError(
            "round-trip product eigenvalues are not real within tolerance: "
            f"max_imag={maximum_imaginary:.3e}"
        )

    product_eigenvalues = raw_eigenvalues.real
    if np.any(product_eigenvalues < -lower_tolerance):
        raise ValueError("round-trip product has a negative eigenvalue outside tolerance")
    product_eigenvalues = np.maximum(product_eigenvalues, 0.0)
    round_trip_eigenvalues = propagation * product_eigenvalues
    if np.any(round_trip_eigenvalues >= 1.0):
        raise ValueError("round-trip eigenvalue reached or crossed the logarithm branch point at 1")

    value = float(np.sum(np.log1p(-round_trip_eigenvalues)))
    matrix = np.eye(2, dtype=complex) - propagation * product
    determinant = complex(np.linalg.det(matrix))
    determinant_scale = max(abs(determinant.real), 1.0)
    if abs(determinant.imag) > imag_tolerance * determinant_scale:
        raise ValueError("trace-log determinant has a significant imaginary part")
    if determinant.real <= 0.0:
        raise ValueError("trace-log determinant must be positive for passive sheets")
    determinant_log = float(np.log(determinant.real))
    if not np.isclose(value, determinant_log, rtol=1e-9, atol=1e-12):
        raise ValueError("eigenvalue and determinant logdet evaluations disagree")
    if value > max(1e-12, lower_tolerance):
        raise ValueError("passive-sheet logdet must be non-positive")

    return LifshitzPoint(
        logdet=value,
        trace_log_matrix=matrix,
        product_eigenvalues=product_eigenvalues,
        round_trip_eigenvalues=round_trip_eigenvalues,
        propagation_factor=propagation,
        kappa_m_inv=kappa,
        separation_m=separation,
        basis=LAB_LT_TANGENTIAL_E_BASIS,
        metadata={
            "formula": "sum_alpha log1p(-exp(-2 kappa d) lambda_alpha(R1 R2))",
            "signed_real_logdet": True,
            "uses_tangential_E_LT_basis": True,
            "silent_real_part_discard_forbidden": True,
            "absolute_determinant_forbidden": True,
            "maximum_product_eigenvalue_imaginary_abs": maximum_imaginary,
            "determinant_real": determinant.real,
            "determinant_imaginary": determinant.imag,
        },
    )


def trace_log_point(
    R1_te_tm: np.ndarray,
    R2_te_tm: np.ndarray,
    kappa_m_inv: float,
    separation_m: float,
) -> dict[str, Any]:
    """Return a legacy complex trace-log point for diagnostic comparison."""

    return {
        "round_trip_factor": round_trip_factor(kappa_m_inv, separation_m),
        "trace_log_matrix": trace_log_matrix(R1_te_tm, R2_te_tm, kappa_m_inv, separation_m),
        "logdet_integrand": trace_log_integrand(R1_te_tm, R2_te_tm, kappa_m_inv, separation_m),
        "single_point_trace_log_for_main_pipeline": True,
        "does_not_by_itself_perform_full_integral": True,
        "used_by_main_pipeline_after_external_grid_sum": True,
        "legacy_complex_diagnostic": True,
    }


def lifshitz_integrand_metadata() -> dict[str, Any]:
    """Return trace-log metadata for the main Casimir pipeline."""

    return {
        "single_point_trace_log_for_main_pipeline": True,
        "does_not_by_itself_perform_full_integral": True,
        "used_by_main_pipeline_after_external_grid_sum": True,
        "formula": "log det[I - exp(-2*kappa*d) R1 @ R2]",
        "round_trip_factor_formula": "exp(-2*kappa*d)",
        "production_positive_frequency_function": "passive_sheet_logdet",
        "legacy_complex_function": "trace_log_point",
    }
