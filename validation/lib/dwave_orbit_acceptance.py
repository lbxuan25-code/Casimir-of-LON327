"""Shared physical postprocessing for positive-Matsubara orbit acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.conventions import (
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import (
    positive_matsubara_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy


@dataclass(frozen=True)
class OrbitAcceptancePhysicsConfig:
    degeneracy: float = 1.0
    separation_nm: float = 20.0
    ward_tolerance: float = 1e-7
    ward_absolute_tolerance: float = 1e-12
    condition_max: float = 1e12


def matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=complex)
    result = {f"{prefix}_frobenius_norm": float(np.linalg.norm(value))}
    for label, row, col in (
        ("xx", 0, 0),
        ("xy", 0, 1),
        ("yx", 1, 0),
        ("yy", 1, 1),
    ):
        scalar = complex(value[row, col])
        result[f"{prefix}_{label}_real"] = float(scalar.real)
        result[f"{prefix}_{label}_imag"] = float(scalar.imag)
    return result


def mixed_matrix_gate(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float, bool]:
    a = np.asarray(left, dtype=complex)
    b = np.asarray(right, dtype=complex)
    delta = float(np.linalg.norm(b - a))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    tolerance = float(atol) + float(rtol) * scale
    ratio = delta / max(tolerance, np.finfo(float).tiny)
    relative = delta / max(scale, np.finfo(float).tiny)
    return delta, relative, ratio, bool(np.isfinite(ratio) and ratio <= 1.0)


def mixed_scalar_gate(
    left: float,
    right: float,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float, bool]:
    delta = abs(float(right) - float(left))
    scale = max(abs(float(left)), abs(float(right)))
    tolerance = float(atol) + float(rtol) * scale
    ratio = delta / max(tolerance, np.finfo(float).tiny)
    relative = delta / max(scale, np.finfo(float).tiny)
    return delta, relative, ratio, bool(np.isfinite(ratio) and ratio <= 1.0)


def evaluate_positive_matsubara_pipeline(
    *,
    components: object,
    rhs: object,
    q_model: np.ndarray,
    xi_eV: float,
    config: OrbitAcceptancePhysicsConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "physical_passed": False,
        "ward_passed": False,
        "sheet_validation_passed": False,
        "reflection_constructed": False,
        "logdet_passed": False,
        "error": "",
        "sigma": np.full((2, 2), np.nan + 1j * np.nan, dtype=complex),
        "reflection": np.full((2, 2), np.nan + 1j * np.nan, dtype=complex),
        "logdet": float("nan"),
        "ward_effective_mixed_ratio_max": float("nan"),
        "schur_condition_number": float("nan"),
    }
    try:
        kernel = effective_em_kernel_from_components(
            components,
            q_model=np.asarray(q_model, dtype=float),
            xi_eV=float(xi_eV),
        )
        ward = validate_effective_ward_xy(
            kernel,
            rhs,
            residual_tolerance=float(config.ward_tolerance),
            absolute_residual_tolerance=float(config.ward_absolute_tolerance),
            condition_max=float(config.condition_max),
        )
        sheet = positive_matsubara_kernel_to_sheet_response(
            kernel,
            degeneracy=float(config.degeneracy),
        )
        sheet_validation = validate_positive_matsubara_sheet_response(sheet)
        reflection = positive_matsubara_sheet_response_to_reflection(
            sheet,
            q_lab_model=np.asarray(q_model, dtype=float),
            theta_rad=0.0,
            lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m,
            require_physical=True,
        )
        point = passive_sheet_logdet(
            reflection,
            reflection,
            separation_m=float(config.separation_nm) * 1e-9,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        result["error"] = str(exc)
        return result

    result.update(
        {
            "ward_passed": bool(ward.passed),
            "sheet_validation_passed": bool(sheet_validation.passed),
            "reflection_constructed": True,
            "logdet_passed": True,
            "sigma": np.asarray(sheet.matrix_tilde, dtype=complex),
            "reflection": np.asarray(reflection.matrix_lt, dtype=complex),
            "logdet": float(point.logdet),
            "ward_effective_mixed_ratio_max": max(
                ward.left.effective_mixed_ratio,
                ward.right.effective_mixed_ratio,
            ),
            "schur_condition_number": float(ward.schur_condition_number),
        }
    )
    result["physical_passed"] = bool(
        result["ward_passed"]
        and result["sheet_validation_passed"]
        and result["reflection_constructed"]
        and result["logdet_passed"]
    )
    return result


__all__ = [
    "OrbitAcceptancePhysicsConfig",
    "evaluate_positive_matsubara_pipeline",
    "matrix_fields",
    "mixed_matrix_gate",
    "mixed_scalar_gate",
]
