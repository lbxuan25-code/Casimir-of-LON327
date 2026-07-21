"""Pairing-blind 0-degree qualification runner and analytic outer-tail certificate.

The ordinary production runner remains unchanged. This module is used only by the
frozen ``0deg_qualification_v5`` workflow. It first executes the existing numerical
geometric tail controller. If that controller reaches the end of the common cutoff
ladder with every finite-domain and microscopic hard gate passed but cannot resolve a
geometric shell ratio, the runner may apply a pairing-independent passive-vacuum bound.

For a validated passive positive-Matsubara sheet, in the vacuum-admittance power
metric the tangential-electric reflection operator is similar to
``-(2 I + A)^(-1) A`` with positive-semidefinite ``A``. Its singular values therefore
lie in ``[0, 1)``. The zero-Matsubara reflection is diagonal; the persisted reflection
norm is used as a conservative spectral-norm upper bound. Similarity leaves the
determinant invariant, and vacuum round-trip propagation contributes ``exp(-u)``.
Thus the two-polarization determinant obeys
``|log det(I - R1 R2 exp(-u))| <= -2 log(1-exp(-u))``.

The analytic path is fail closed: before use, the qualification runner re-reads the
actual target cache and verifies contraction evidence at every accepted audit state.
It never bypasses Ward, sheet, passivity, reflection, radial, angular, or offset gates.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
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


def _accepted_audit_row(point: Mapping[str, Any]) -> Mapping[str, Any] | None:
    sweet = point.get("sweet_spot")
    if not isinstance(sweet, Mapping) or sweet.get("status") != "established":
        return None
    try:
        audit_N = int(sweet["audit_N"])
    except (KeyError, TypeError, ValueError):
        return None
    for row in point.get("history", []):
        if isinstance(row, Mapping) and int(row.get("N", -1)) == audit_N:
            return row
    return None


def _plate_contract(plate: Mapping[str, Any], *, n: int) -> tuple[bool, str, float | None]:
    if not bool(plate.get("sheet_validation_passed")) or not bool(
        plate.get("reflection_constructed")
    ):
        return False, "sheet_or_reflection_gate_missing", None
    if bool(plate.get("power_metric_contraction_certified")):
        raw = plate.get("power_metric_singular_value_max_upper_bound")
        upper = float(raw) if isinstance(raw, (int, float)) else None
        return True, str(plate.get("power_metric_certificate_method", "persisted")), upper
    if int(n) > 0:
        return True, "passive_sheet_vacuum_admittance_similarity_theorem", 1.0
    try:
        upper = abs(float(plate.get("reflection_norm")))
    except (TypeError, ValueError, OverflowError):
        return False, "static_reflection_norm_missing", None
    return (
        bool(math.isfinite(upper) and upper <= 1.0 + 1e-12),
        "stored_frobenius_norm_upper_bounds_static_spectral_norm",
        upper,
    )


def cached_power_metric_contraction_certificate(provider: Any) -> dict[str, Any]:
    cache_path = getattr(provider, "cache_path", None)
    if cache_path is None or not Path(cache_path).is_file():
        return {
            "schema": "cached-power-metric-contraction-certificate-v1",
            "status": "not_certified",
            "reason": "provider cache is unavailable",
            "all_points_certified": False,
            "point_count": 0,
            "failures": [],
        }
    try:
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema": "cached-power-metric-contraction-certificate-v1",
            "status": "not_certified",
            "reason": f"cannot read provider cache: {exc}",
            "all_points_certified": False,
            "point_count": 0,
            "failures": [],
        }
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return {
            "schema": "cached-power-metric-contraction-certificate-v1",
            "status": "not_certified",
            "reason": "provider cache has no entries",
            "all_points_certified": False,
            "point_count": 0,
            "failures": [],
        }
    failures: list[dict[str, Any]] = []
    certified = 0
    maximum_upper = 0.0
    methods: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            failures.append({"identity": None, "reason": "malformed cache entry"})
            continue
        identity = [
            str(entry.get("pairing")),
            int(entry.get("n", -1)),
            str(entry.get("qx_hex")),
            str(entry.get("qy_hex")),
        ]
        point = entry.get("point_result")
        if not isinstance(point, Mapping):
            failures.append({"identity": identity, "reason": "point_result missing"})
            continue
        row = _accepted_audit_row(point)
        if row is None:
            failures.append({"identity": identity, "reason": "accepted audit row missing"})
            continue
        shifts = row.get("shifts")
        if not isinstance(shifts, Mapping) or not shifts:
            failures.append({"identity": identity, "reason": "audit shifts missing"})
            continue
        point_ok = True
        for shift_label, state in shifts.items():
            if not isinstance(state, Mapping) or not bool(state.get("hard_physical_passed")):
                failures.append(
                    {"identity": identity, "shift": str(shift_label), "reason": "hard gate failed"}
                )
                point_ok = False
                continue
            for plate_name in ("plate_1", "plate_2"):
                plate = state.get(plate_name)
                if not isinstance(plate, Mapping):
                    failures.append(
                        {
                            "identity": identity,
                            "shift": str(shift_label),
                            "plate": plate_name,
                            "reason": "plate evidence missing",
                        }
                    )
                    point_ok = False
                    continue
                passed, method, upper = _plate_contract(plate, n=int(identity[1]))
                methods.add(method)
                if upper is not None and math.isfinite(upper):
                    maximum_upper = max(maximum_upper, upper)
                if not passed:
                    failures.append(
                        {
                            "identity": identity,
                            "shift": str(shift_label),
                            "plate": plate_name,
                            "reason": method,
                            "upper_bound": upper,
                        }
                    )
                    point_ok = False
        if point_ok:
            certified += 1
    all_certified = certified == len(entries) and not failures
    return {
        "schema": "cached-power-metric-contraction-certificate-v1",
        "status": "certified" if all_certified else "not_certified",
        "proof_version": PASSIVE_REFLECTION_BOUND_VERSION,
        "cache_path": str(Path(cache_path)),
        "point_count": len(entries),
        "certified_point_count": certified,
        "all_points_certified": all_certified,
        "maximum_recorded_upper_bound": maximum_upper,
        "methods": sorted(methods),
        "failures": failures[:64],
    }


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
    *,
    contraction_certificate: Mapping[str, Any],
) -> AdaptiveOuterTailCasimirResult:
    if result.status == "adaptive_finite_partial" and result.cutoff_converged:
        return _geometric_certificate_tag(result)
    if contraction_certificate.get("all_points_certified") is not True:
        return result
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
                "power_metric_contraction_premise": dict(contraction_certificate),
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
    certificate = cached_power_metric_contraction_certificate(provider)
    return _analytic_upgrade(
        config,
        result,
        contraction_certificate=certificate,
    )


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
    "cached_power_metric_contraction_certificate",
    "passive_vacuum_channel_bounds_J_m2",
    "passive_vacuum_tail_series",
    "run_qualification_casimir",
    "run_qualification_outer_tail_casimir",
]
