"""Classify the small-q scaling of d-wave phase-Hessian candidates.

Each input is an existing full or reduced commensurate periodic phase-column JSON
payload.  The analysis keeps the momentum direction fixed and compares the
counterterm shift required by the collective phase-column identity with the
nearest-neighbour scalar bond metric.

A classification is issued only when the required shift itself shows a stable
quadratic small-q law.  This fail-closed rule prevents a formally acceptable global
fit from hiding an unstable pairwise exponent at the largest q point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from validation.lib.dwave_phase_hessian_analysis import (
    DWavePhaseHessianAnalysis,
    analyze_dwave_phase_hessian_payload,
)


@dataclass(frozen=True)
class PhaseHessianScalingPoint:
    label: str
    q_model: tuple[float, float]
    q_norm: float
    direction: tuple[float, float]
    required_multiplier: complex
    left_right_required_mismatch: float
    required_shift_abs: float
    required_shift_over_q2: float
    bond_metric_multiplier: float
    bond_shift_abs: float
    bond_shift_over_q2: float
    bond_multiplier_error_abs: float
    bond_multiplier_error_over_q2: float
    current_phase_defect_over_q: float
    bond_phase_defect_over_q: float
    phase_direct_phase_defect_over_q: float | None


@dataclass(frozen=True)
class DWavePhaseHessianScalingAnalysis:
    points: tuple[PhaseHessianScalingPoint, ...]
    direction: tuple[float, float]
    delta0_eV: float
    required_shift_exponent: float | None
    bond_shift_exponent: float | None
    bond_error_exponent: float | None
    current_defect_over_q_exponent: float | None
    bond_defect_over_q_exponent: float | None
    required_shift_pairwise_exponents: tuple[float, ...]
    bond_error_pairwise_exponents: tuple[float, ...]
    smallest_two_required_q2_coefficient: float | None
    smallest_two_required_q4_coefficient: float | None
    smallest_two_bond_error_q2_coefficient: float | None
    smallest_two_bond_error_q4_coefficient: float | None
    classification: str
    interpretation: str
    diagnostic_only: bool = True
    projection_applied: bool = False
    production_reference_established: bool = False
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fit_power(q_values: Sequence[float], values: Sequence[float]) -> float | None:
    q = np.asarray(q_values, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(q) & np.isfinite(y) & (q > 0.0) & (y > 1e-30)
    if int(np.count_nonzero(mask)) < 2:
        return None
    return float(np.polyfit(np.log(q[mask]), np.log(y[mask]), 1)[0])


def _pairwise_powers(q_values: Sequence[float], values: Sequence[float]) -> tuple[float, ...]:
    q = np.asarray(q_values, dtype=float)
    y = np.asarray(values, dtype=float)
    output: list[float] = []
    for index in range(len(q) - 1):
        q0, q1 = float(q[index]), float(q[index + 1])
        y0, y1 = float(y[index]), float(y[index + 1])
        if q0 <= 0.0 or q1 <= 0.0 or y0 <= 1e-30 or y1 <= 1e-30 or q0 == q1:
            output.append(float("nan"))
        else:
            output.append(float(np.log(y1 / y0) / np.log(q1 / q0)))
    return tuple(output)


def _smallest_two_even_fit(
    q_values: Sequence[float], values: Sequence[float]
) -> tuple[float | None, float | None]:
    if len(q_values) < 2:
        return None, None
    q = np.asarray(q_values[:2], dtype=float)
    y = np.asarray(values[:2], dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(y).all() or np.any(q <= 0.0):
        return None, None
    matrix = np.column_stack((q**2, q**4))
    if abs(float(np.linalg.det(matrix))) <= 1e-30:
        return None, None
    coefficients = np.linalg.solve(matrix, y)
    return float(coefficients[0]), float(coefficients[1])


def _point_from_analysis(
    analysis: DWavePhaseHessianAnalysis,
    *,
    label: str,
) -> PhaseHessianScalingPoint:
    q = np.asarray(analysis.q_model, dtype=float)
    q_norm = float(analysis.q_norm)
    q2 = q_norm * q_norm
    direction = q / q_norm
    left_required = complex(analysis.left.required_counterterm_multiplier)
    right_required = complex(analysis.right.required_counterterm_multiplier)
    required = 0.5 * (left_required + right_required)
    current_over_q = max(
        abs(complex(analysis.left.current_phase_defect)) / q_norm,
        abs(complex(analysis.right.current_phase_defect)) / q_norm,
    )
    bond_over_q = max(
        abs(complex(analysis.left.bond_metric_phase_defect)) / q_norm,
        abs(complex(analysis.right.bond_metric_phase_defect)) / q_norm,
    )
    direct_values = (
        analysis.left.phase_direct_phase_defect,
        analysis.right.phase_direct_phase_defect,
    )
    direct_over_q: float | None
    if any(value is None for value in direct_values):
        direct_over_q = None
    else:
        direct_over_q = max(abs(complex(value)) / q_norm for value in direct_values)
    bond = float(analysis.bond_metric_multiplier)
    required_shift = float(abs(1.0 - required))
    bond_shift = float(abs(1.0 - bond))
    bond_error = float(abs(complex(bond) - required))
    return PhaseHessianScalingPoint(
        label=str(label),
        q_model=(float(q[0]), float(q[1])),
        q_norm=q_norm,
        direction=(float(direction[0]), float(direction[1])),
        required_multiplier=required,
        left_right_required_mismatch=float(abs(left_required - right_required)),
        required_shift_abs=required_shift,
        required_shift_over_q2=required_shift / q2,
        bond_metric_multiplier=bond,
        bond_shift_abs=bond_shift,
        bond_shift_over_q2=bond_shift / q2,
        bond_multiplier_error_abs=bond_error,
        bond_multiplier_error_over_q2=bond_error / q2,
        current_phase_defect_over_q=float(current_over_q),
        bond_phase_defect_over_q=float(bond_over_q),
        phase_direct_phase_defect_over_q=direct_over_q,
    )


def _finite_in_range(values: Sequence[float], low: float, high: float) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(
        array.size > 0
        and np.isfinite(array).all()
        and np.all(array >= low)
        and np.all(array <= high)
    )


def _classify(
    required_exponent: float | None,
    bond_exponent: float | None,
    error_exponent: float | None,
    required_pairwise: Sequence[float],
    error_pairwise: Sequence[float],
) -> tuple[str, str]:
    if required_exponent is None or bond_exponent is None or error_exponent is None:
        return (
            "insufficient_scaling_data",
            "At least two nonzero points are required for each fitted quantity.",
        )

    required_clean = (
        1.8 <= required_exponent <= 2.2
        and _finite_in_range(required_pairwise, 1.6, 2.4)
    )
    bond_clean = 1.8 <= bond_exponent <= 2.2
    if not required_clean or not bond_clean:
        return (
            "not_in_clean_small_q_regime",
            "The required counterterm shift does not show a stable quadratic law "
            "across both the global and adjacent-point fits.  Do not infer a finite-q "
            "Hessian from this family; add a smaller-q commensurate point or improve "
            "grid convergence.",
        )

    if error_exponent >= 3.5 and _finite_in_range(error_pairwise, 3.0, 5.0):
        return (
            "bond_metric_matches_leading_q2_geometry",
            "The scalar bond metric captures the leading q^2 counterterm geometry.  "
            "The remaining defect is higher order and should be resolved with an "
            "exact bond-resolved Hessian rather than a fitted scalar multiplier.",
        )
    if 1.5 <= error_exponent <= 2.5 and _finite_in_range(error_pairwise, 1.3, 2.7):
        return (
            "bond_metric_misses_leading_q2_curvature",
            "The residual mismatch is itself quadratic in q.  A scalar finite-q "
            "counterterm is not adequate; the next audit must retain independent "
            "x/y bond collective channels.",
        )
    return (
        "scaling_inconclusive",
        "The required shift is quadratic, but the residual exponent is neither "
        "stably q^2 nor stably q^4.  Add a smaller-q commensurate point before "
        "changing the collective kernel.",
    )


def analyze_dwave_phase_hessian_family(
    payloads: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    direction_tolerance: float = 1e-12,
    delta0_tolerance: float = 1e-12,
) -> DWavePhaseHessianScalingAnalysis:
    """Analyze two or more commensurate audits along one fixed q direction."""

    if len(payloads) < 2:
        raise ValueError("phase-Hessian scaling analysis requires at least two payloads")
    if labels is None:
        labels = tuple(f"point_{index}" for index in range(len(payloads)))
    if len(labels) != len(payloads):
        raise ValueError("labels and payloads must have the same length")

    analyses = [analyze_dwave_phase_hessian_payload(payload) for payload in payloads]
    paired = sorted(
        zip(analyses, labels, strict=True),
        key=lambda item: item[0].q_norm,
    )
    analyses = [item[0] for item in paired]
    points = [
        _point_from_analysis(analysis, label=str(label))
        for analysis, label in paired
    ]

    reference_direction = np.asarray(points[0].direction, dtype=float)
    reference_delta0 = float(analyses[0].delta0_eV)
    for point, analysis in zip(points, analyses, strict=True):
        if not np.allclose(
            np.asarray(point.direction, dtype=float),
            reference_direction,
            rtol=0.0,
            atol=float(direction_tolerance),
        ):
            raise ValueError("all commensurate q vectors must have the same direction")
        if not np.isclose(
            float(analysis.delta0_eV),
            reference_delta0,
            rtol=0.0,
            atol=float(delta0_tolerance),
        ):
            raise ValueError("all payloads must use the same delta0_eV")

    q_values = [point.q_norm for point in points]
    required_values = [point.required_shift_abs for point in points]
    bond_values = [point.bond_shift_abs for point in points]
    error_values = [point.bond_multiplier_error_abs for point in points]
    current_values = [point.current_phase_defect_over_q for point in points]
    bond_defect_values = [point.bond_phase_defect_over_q for point in points]

    required_exponent = _fit_power(q_values, required_values)
    bond_exponent = _fit_power(q_values, bond_values)
    error_exponent = _fit_power(q_values, error_values)
    current_exponent = _fit_power(q_values, current_values)
    bond_defect_exponent = _fit_power(q_values, bond_defect_values)
    required_pairwise = _pairwise_powers(q_values, required_values)
    error_pairwise = _pairwise_powers(q_values, error_values)
    required_q2, required_q4 = _smallest_two_even_fit(q_values, required_values)
    error_q2, error_q4 = _smallest_two_even_fit(q_values, error_values)
    classification, interpretation = _classify(
        required_exponent,
        bond_exponent,
        error_exponent,
        required_pairwise,
        error_pairwise,
    )

    return DWavePhaseHessianScalingAnalysis(
        points=tuple(points),
        direction=(float(reference_direction[0]), float(reference_direction[1])),
        delta0_eV=reference_delta0,
        required_shift_exponent=required_exponent,
        bond_shift_exponent=bond_exponent,
        bond_error_exponent=error_exponent,
        current_defect_over_q_exponent=current_exponent,
        bond_defect_over_q_exponent=bond_defect_exponent,
        required_shift_pairwise_exponents=required_pairwise,
        bond_error_pairwise_exponents=error_pairwise,
        smallest_two_required_q2_coefficient=required_q2,
        smallest_two_required_q4_coefficient=required_q4,
        smallest_two_bond_error_q2_coefficient=error_q2,
        smallest_two_bond_error_q4_coefficient=error_q4,
        classification=classification,
        interpretation=interpretation,
    )


__all__ = [
    "DWavePhaseHessianScalingAnalysis",
    "PhaseHessianScalingPoint",
    "analyze_dwave_phase_hessian_family",
]
