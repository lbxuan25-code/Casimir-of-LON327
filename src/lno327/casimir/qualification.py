"""Pairing-blind 0-degree qualification runner and analytic outer-tail certificate.

The ordinary production runner remains unchanged.  This module is used only by the
frozen ``0deg_qualification_v5`` workflow.  It first executes the existing numerical
geometric tail controller.  If that controller reaches the end of the common cutoff
ladder with every finite-domain and microscopic hard gate passed but cannot resolve a
geometric shell ratio, the runner applies a pairing-independent passive-vacuum bound.

Mathematical contract
---------------------
For a validated passive positive-Matsubara sheet, in the vacuum-admittance power
metric the tangential-electric reflection operator is similar to

    -(2 I + A)^(-1) A,

where A is positive semidefinite.  Its singular values therefore lie in [0, 1).
The zero-Matsubara reflection is diagonal with passive channel parameters and obeys
the same contraction bound.  Similarity leaves the determinant invariant.  The
vacuum round trip contributes exp(-u), so for the two-polarization determinant

    |log det(I - R1 R2 exp(-u))| <= -2 log(1 - exp(-u)).

The resulting radial tail is integrated explicitly.  This path is accepted only
when the existing microscopic hard-physical contract and the complete finite-domain
controller have both passed.  It never bypasses Ward, sheet, passivity, reflection,
radial, angular, or offset gates.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.constants import KB

from .adaptive_matsubara_tail import (
    AdaptiveMatsubaraCasimirConfig,
    AdaptiveMatsubaraCasimirResult,
    run_adaptive_matsubara_casimir,
)
from .adaptive_outer_tail import (
    AdaptiveOuterTailCasimirConfig,
    AdaptiveOuterTailCasimirResult,
    run_adaptive_outer_tail_casimir,
)
from .certified_point_provider import FrequencyExtendableCertifiedOuterQProvider
from .outer_quadrature import matsubara_prime_weights
from .production import _quarantine_invalid_telemetry
from .strict_transverse_runner import run_strict_transverse_certifier

PASSIVE_REFLECTION_BOUND_VERSION = "passive-vacuum-power-metric-v1"


def passive_vacuum_tail_series(u0: float) -> tuple[float, int]:
    start = float(u0)
    if not math.isfinite(start) or start <= 0.0:
        raise ValueError("u0 must be finite and positive")
    total = 0.0
    for m in range(1, 100_000):
        term = math.exp(-m * start) * (
            start / (m * m) + 1.0 / (m * m * m)
        )
        total += term
        if term <= max(1e-300, 1e-15 * max(total, 1e-300)):
            return total, m
    raise RuntimeError("passive-vacuum tail series did not converge")


def passive_vacuum_channel_bounds_J_m2(
    *,
    u0: float,
    separation_nm: float,
    temperature_K: float,
    matsubara_indices: Sequence[int],
) -> dict[str, Any]:
    series, terms = passive_vacuum_tail_series(float(u0))
    separation_m = float(separation_nm) * 1e-9
    temperature = float(temperature_K)
    if not math.isfinite(separation_m) or separation_m <= 0.0:
        raise ValueError("separation_nm must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    indices = tuple(int(value) for value in matsubara_indices)
    weights = matsubara_prime_weights(indices)
    bounds = [
        float(KB * temperature * weight * series / (4.0 * math.pi * separation_m**2))
        for weight in weights
    ]
    return {
        "schema": "passive-vacuum-tail-certificate-v1",
        "proof_version": PASSIVE_REFLECTION_BOUND_VERSION,
        "u0": float(u0),
        "series_value": series,
        "series_terms_used": terms,
        "matsubara_indices": list(indices),
        "channel_bounds_J_m2": bounds,
        "matsubara_prime_weights": [float(value) for value in weights],
        "determinant_similarity_invariant": True,
        "two_polarization_coefficient": 2,
        "round_trip_propagation_bound": "exp(-u)",
        "pairing_independent": True,
    }


def _finite_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with shape ({count},)")
    return array


def _geometric_certificate_tag(
    result: AdaptiveOuterTailCasimirResult,
) -> AdaptiveOuterTailCasimirResult:
    tagged: dict[str, Any] = {}
    for pairing, raw in result.pairing_results.items():
        row = dict(raw)
        row.setdefault("outer_tail_certificate_path", "geometric_numerical")
        row.setdefault("outer_tail_certificate_pairing_independent", True)
        tagged[str(pairing)] = row
    return replace(result, pairing_results=tagged)


def _analytic_upgrade(
    config: AdaptiveOuterTailCasimirConfig,
    result: AdaptiveOuterTailCasimirResult,
) -> AdaptiveOuterTailCasimirResult:
    if result.status == "adaptive_finite_partial" and result.cutoff_converged:
        return _geometric_certificate_tag(result)
    if not result.all_microscopic_nodes_certified:
        return result
    if not result.all_finite_domain_runs_converged:
        return result
    if not result.cutoff_records or result.selected_u_max is None:
        return result

    latest = dict(result.cutoff_records[-1])
    point = config.joint_config.radial_config.point_config
    indices = tuple(int(value) for value in point.matsubara_indices)
    certificate = passive_vacuum_channel_bounds_J_m2(
        u0=float(result.selected_u_max),
        separation_nm=float(point.separation_nm),
        temperature_K=float(point.temperature_K),
        matsubara_indices=indices,
    )
    analytic = _finite_vector(
        certificate["channel_bounds_J_m2"],
        name="analytic tail bound",
        count=len(indices),
    )
    finite_by_pairing = latest.get("finite_domain_error_bounds_J_m2")
    latest_pairings = latest.get("pairing_results")
    if not isinstance(finite_by_pairing, Mapping) or not isinstance(
        latest_pairings, Mapping
    ):
        return result

    upgraded: dict[str, Any] = {}
    all_passed = True
    for pairing in point.pairings:
        base_raw = latest_pairings.get(pairing)
        finite_raw = finite_by_pairing.get(pairing)
        if not isinstance(base_raw, Mapping):
            return result
        base = dict(base_raw)
        values = _finite_vector(
            base.get("contributions_J_m2"),
            name=f"{pairing} contributions",
            count=len(indices),
        )
        finite = _finite_vector(
            finite_raw,
            name=f"{pairing} finite-domain errors",
            count=len(indices),
        )
        if np.any(finite < 0.0) or np.any(analytic < 0.0):
            return result
        total_tolerance = np.maximum(
            float(config.total_outer_atol_J_m2),
            float(config.total_outer_rtol) * np.abs(values),
        )
        finite_tolerance = (
            float(config.finite_domain_budget_fraction) * total_tolerance
        )
        tail_tolerance = float(config.tail_budget_fraction) * total_tolerance
        total_error = finite + analytic
        finite_pass = finite <= finite_tolerance
        tail_pass = analytic <= tail_tolerance
        total_pass = total_error <= total_tolerance
        channel_pass = bool(
            np.all(finite_pass) and np.all(tail_pass) and np.all(total_pass)
        )
        all_passed = all_passed and channel_pass
        base.update(
            {
                "status": "integrated_with_outer_tail_bound",
                "finite_domain_error_bounds_J_m2": finite.tolist(),
                "estimated_outer_tail_bounds_J_m2": analytic.tolist(),
                "estimated_total_outer_errors_J_m2": total_error.tolist(),
                "total_outer_tolerances_J_m2": total_tolerance.tolist(),
                "finite_domain_budget_tolerances_J_m2": finite_tolerance.tolist(),
                "tail_budget_tolerances_J_m2": tail_tolerance.tolist(),
                "finite_domain_channel_passed": finite_pass.tolist(),
                "outer_tail_channel_passed": tail_pass.tolist(),
                "total_outer_channel_passed": total_pass.tolist(),
                "outer_tail_certificate_path": "analytic_passive_vacuum",
                "outer_tail_certificate_pairing_independent": True,
                "passive_vacuum_tail_certificate": dict(certificate),
                "power_metric_contraction_premise": {
                    "status": "proved_by_validated_sheet_contract",
                    "proof_version": PASSIVE_REFLECTION_BOUND_VERSION,
                    "requires_all_microscopic_hard_physical_gates": True,
                    "all_microscopic_hard_physical_gates_passed": True,
                    "finite_domain_controller_passed": True,
                    "non_normal_control": "singular-value bound in vacuum-admittance metric",
                },
            }
        )
        upgraded[str(pairing)] = base

    if not all_passed:
        return result
    return replace(
        result,
        status="adaptive_finite_partial",
        cutoff_converged=True,
        outer_tail_estimated_flag=True,
        all_finite_domain_runs_converged=True,
        all_microscopic_nodes_certified=True,
        pairing_results=upgraded,
        termination_reason="analytic_passive_vacuum_tail_bound_met",
    )


def run_qualification_outer_tail_casimir(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    provider: Any | None = None,
) -> AdaptiveOuterTailCasimirResult:
    """Run the common geometric controller, then the common analytic fallback."""

    result = run_adaptive_outer_tail_casimir(config, provider=provider)
    return _analytic_upgrade(config, result)


def run_qualification_casimir(
    config: AdaptiveMatsubaraCasimirConfig,
) -> AdaptiveMatsubaraCasimirResult:
    """Run the frozen v5 qualification path with one pairing-blind tail structure."""

    if not isinstance(config, AdaptiveMatsubaraCasimirConfig):
        raise TypeError("config must be an AdaptiveMatsubaraCasimirConfig")
    _quarantine_invalid_telemetry(config)
    first_cutoff = int(config.matsubara_cutoff_values[0])
    base = config.outer_tail_config.joint_config.radial_config.point_config
    first_point = replace(base, matsubara_indices=tuple(range(first_cutoff + 1)))
    provider = FrequencyExtendableCertifiedOuterQProvider(
        first_point,
        cache_path=config.point_cache_path,
        runner=run_strict_transverse_certifier,
        certifier_q_batch_size=config.certifier_q_batch_size,
    )
    return run_adaptive_matsubara_casimir(
        config,
        provider=provider,
        outer_tail_runner=run_qualification_outer_tail_casimir,
    )


__all__ = [
    "PASSIVE_REFLECTION_BOUND_VERSION",
    "passive_vacuum_channel_bounds_J_m2",
    "passive_vacuum_tail_series",
    "run_qualification_casimir",
    "run_qualification_outer_tail_casimir",
]
