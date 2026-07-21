"""Production outer-Q tail certificates with explicit analytic and numerical paths.

The numerical shell-ratio controller remains available as diagnostic evidence. Formal
production acceptance additionally attempts a pairing-independent passive-vacuum bound
at every finite-domain cutoff. The zero-Matsubara contraction premise is evaluated
with the exact static spectral norm, never the Frobenius norm.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.constants import C0, E2_OVER_HBAR, EV_TO_J, HBAR, KB, SIGMA0
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE

from .adaptive_joint_q import run_adaptive_joint_casimir
from .adaptive_outer_tail import (
    AdaptiveOuterTailCasimirConfig,
    AdaptiveOuterTailCasimirResult,
    _channel_arrays,
    _cutoff_record,
    _final_pairing_results,
    _joint_run_usable,
    _provider_statistics,
    _scaled_joint_config,
    _shell_record,
    _tail_window_metrics,
)
from .certified_point_provider import CertifiedOuterQProvider, CertifiedPointCacheError
from .fixed_chain import FixedCasimirConfig, FixedCasimirExecutionError
from .outer_quadrature import matsubara_prime_weights

OUTER_TAIL_CERTIFICATE_CONTRACT = "outer-tail-certificate-contract-v2"
PASSIVE_REFLECTION_BOUND_VERSION = "passive-vacuum-power-metric-v2-exact-static-spectral"


def passive_vacuum_tail_series(u0: float) -> tuple[float, int]:
    """Return the exact scalar series for the passive-vacuum outer-Q bound."""

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
    """Bound each fixed Matsubara channel beyond the radial cutoff ``u0``."""

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
        "schema": "passive-vacuum-outer-tail-certificate-v2",
        "certificate_contract": OUTER_TAIL_CERTIFICATE_CONTRACT,
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


def _static_exact_spectral_norm(
    plate: Mapping[str, Any],
    *,
    point_config: FixedCasimirConfig,
) -> tuple[float, dict[str, float]]:
    """Reconstruct the exact spectral norm of the diagonal static reflection."""

    persisted = plate.get("reflection_spectral_norm")
    if isinstance(persisted, (int, float)):
        spectral = abs(float(persisted))
        if math.isfinite(spectral):
            return spectral, {"persisted_spectral_norm": spectral}

    q = np.asarray(plate.get("q_crystal"), dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("static q_crystal is missing or malformed")
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 0.0:
        raise ValueError("static q_crystal must be nonzero")
    chi_bar = float(plate["chi_bar"])
    dbar_t = float(plate["dbar_t"])
    if not math.isfinite(chi_bar) or not math.isfinite(dbar_t):
        raise ValueError("static chi_bar/dbar_t must be finite")

    lattice = float(LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m)
    beta = (
        float(point_config.static_energy_scale_eV)
        * EV_TO_J
        * lattice
        / (HBAR * C0)
    )
    gamma = E2_OVER_HBAR / SIGMA0
    lambda_l = float(point_config.degeneracy) * (gamma / beta) * chi_bar / q_norm
    lambda_t = float(point_config.degeneracy) * (gamma * beta) * dbar_t / q_norm
    if 2.0 + lambda_l <= 0.0 or 2.0 + lambda_t <= 0.0:
        raise ValueError("static reflection denominator reached a nonphysical pole")
    r_l = -lambda_l / (2.0 + lambda_l)
    r_t = -lambda_t / (2.0 + lambda_t)
    spectral = max(abs(r_l), abs(r_t))
    if not math.isfinite(spectral):
        raise ValueError("static spectral norm is not finite")
    return spectral, {
        "lambda_l": lambda_l,
        "lambda_t": lambda_t,
        "r_l": r_l,
        "r_t": r_t,
        "reconstructed_frobenius_norm": math.hypot(abs(r_l), abs(r_t)),
    }


def _active_points(
    provider: Any,
    *,
    point_config: FixedCasimirConfig,
) -> list[Mapping[str, Any]]:
    """Return only entries in the active pairing/frequency scope."""

    pairings = set(point_config.pairings)
    indices = set(int(value) for value in point_config.matsubara_indices)
    raw_entries = getattr(provider, "_entries", None)
    points: list[Mapping[str, Any]] = []
    if isinstance(raw_entries, Mapping):
        for raw in raw_entries.values():
            if not isinstance(raw, Mapping):
                continue
            point = raw.get("point_result") if "point_result" in raw else raw
            if not isinstance(point, Mapping):
                continue
            if str(point.get("pairing")) in pairings and int(point.get("n", -1)) in indices:
                points.append(point)
        return points
    raise ValueError("active provider does not expose certified point entries")


def active_power_metric_contraction_certificate(
    provider: Any,
    *,
    point_config: FixedCasimirConfig,
) -> dict[str, Any]:
    """Certify reflection contraction on the active run scope."""

    try:
        points = _active_points(provider, point_config=point_config)
    except (TypeError, ValueError) as exc:
        return {
            "schema": "active-power-metric-contraction-certificate-v2",
            "status": "not_certified",
            "reason": str(exc),
            "all_points_certified": False,
            "point_count": 0,
            "certified_point_count": 0,
            "failures": [],
        }
    if not points:
        return {
            "schema": "active-power-metric-contraction-certificate-v2",
            "status": "not_certified",
            "reason": "active provider has no entries in the requested scope",
            "all_points_certified": False,
            "point_count": 0,
            "certified_point_count": 0,
            "failures": [],
        }

    failures: list[dict[str, Any]] = []
    certified = 0
    maximum_upper = 0.0
    methods: set[str] = set()
    for point in points:
        identity = [
            str(point.get("pairing")),
            int(point.get("n", -1)),
            str(point.get("q_label", "")),
        ]
        n = int(point.get("n", -1))
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
                if not bool(plate.get("sheet_validation_passed")) or not bool(
                    plate.get("reflection_constructed")
                ):
                    failures.append(
                        {
                            "identity": identity,
                            "shift": str(shift_label),
                            "plate": plate_name,
                            "reason": "sheet_or_reflection_gate_missing",
                        }
                    )
                    point_ok = False
                    continue
                if n > 0:
                    upper = 1.0
                    method = "passive_sheet_vacuum_admittance_similarity_theorem"
                    details: Mapping[str, Any] = {}
                else:
                    method = "exact_static_diagonal_spectral_norm"
                    try:
                        upper, details = _static_exact_spectral_norm(
                            plate,
                            point_config=point_config,
                        )
                    except (KeyError, TypeError, ValueError, OverflowError) as exc:
                        failures.append(
                            {
                                "identity": identity,
                                "shift": str(shift_label),
                                "plate": plate_name,
                                "reason": f"static_spectral_reconstruction_failed: {exc}",
                            }
                        )
                        point_ok = False
                        continue
                methods.add(method)
                maximum_upper = max(maximum_upper, float(upper))
                if not math.isfinite(float(upper)) or float(upper) > 1.0 + 1e-12:
                    failures.append(
                        {
                            "identity": identity,
                            "shift": str(shift_label),
                            "plate": plate_name,
                            "reason": method,
                            "upper_bound": float(upper),
                            "details": dict(details),
                        }
                    )
                    point_ok = False
        if point_ok:
            certified += 1

    all_certified = certified == len(points) and not failures
    return {
        "schema": "active-power-metric-contraction-certificate-v2",
        "status": "certified" if all_certified else "not_certified",
        "proof_version": PASSIVE_REFLECTION_BOUND_VERSION,
        "scope": {
            "source": "active_provider_entries",
            "pairings": list(point_config.pairings),
            "matsubara_indices": list(point_config.matsubara_indices),
            "historical_unrequested_entries_excluded": True,
        },
        "point_count": len(points),
        "certified_point_count": certified,
        "all_points_certified": all_certified,
        "maximum_recorded_upper_bound": maximum_upper,
        "methods": sorted(methods),
        "failures": failures[:64],
    }


def _finite_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with shape ({count},)")
    return array


def _analytic_metrics(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    u_max: float,
    values: Mapping[str, np.ndarray],
    finite_errors: Mapping[str, np.ndarray],
    indices: Sequence[int],
    contraction_certificate: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    point = config.joint_config.radial_config.point_config
    certificate = passive_vacuum_channel_bounds_J_m2(
        u0=float(u_max),
        separation_nm=float(point.separation_nm),
        temperature_K=float(point.temperature_K),
        matsubara_indices=indices,
    )
    analytic = _finite_vector(
        certificate["channel_bounds_J_m2"],
        name="analytic outer tail bound",
        count=len(indices),
    )
    premise_passed = contraction_certificate.get("all_points_certified") is True
    output: dict[str, Any] = {
        "certificate_path": "analytic_passive_vacuum",
        "certificate_contract": OUTER_TAIL_CERTIFICATE_CONTRACT,
        "premise_passed": premise_passed,
        "premise": dict(contraction_certificate),
        "passive_vacuum_tail_certificate": dict(certificate),
        "pairings": {},
    }
    all_passed = bool(premise_passed)
    for pairing in values:
        finite = np.asarray(finite_errors[pairing], dtype=float)
        total_tolerance = np.maximum(
            config.total_outer_atol_J_m2,
            config.total_outer_rtol * np.abs(values[pairing]),
        )
        finite_tolerance = config.finite_domain_budget_fraction * total_tolerance
        tail_tolerance = config.tail_budget_fraction * total_tolerance
        total_error = finite + analytic
        finite_pass = finite <= finite_tolerance
        tail_pass = analytic <= tail_tolerance
        total_pass = total_error <= total_tolerance
        pairing_passed = bool(
            premise_passed
            and np.all(finite_pass)
            and np.all(tail_pass)
            and np.all(total_pass)
        )
        all_passed = all_passed and pairing_passed
        output["pairings"][pairing] = {
            "finite_domain_error_bounds_J_m2": finite.tolist(),
            "estimated_tail_bounds_J_m2": analytic.tolist(),
            "estimated_total_outer_errors_J_m2": total_error.tolist(),
            "total_outer_tolerances_J_m2": total_tolerance.tolist(),
            "finite_domain_budget_tolerances_J_m2": finite_tolerance.tolist(),
            "tail_budget_tolerances_J_m2": tail_tolerance.tolist(),
            "finite_domain_channel_passed": finite_pass.tolist(),
            "tail_channel_passed": tail_pass.tolist(),
            "total_outer_channel_passed": total_pass.tolist(),
            "matsubara_indices": list(indices),
            "certificate_passed": pairing_passed,
        }
    output["certificate_passed"] = bool(all_passed)
    if not premise_passed:
        output["rejection_reason"] = "reflection_contraction_premise_not_certified"
    elif not all_passed:
        output["rejection_reason"] = "analytic_outer_tail_or_total_budget_not_met"
    else:
        output["rejection_reason"] = None
    return output, bool(all_passed)


def _analytic_pairing_results(result: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pairing in result.config.radial_config.point_config.pairings:
        base = dict(result.pairing_results[pairing])
        tail = dict(metrics["pairings"][pairing])
        base.update(
            {
                "status": "integrated_with_outer_tail_bound",
                "finite_domain_error_bounds_J_m2": tail["finite_domain_error_bounds_J_m2"],
                "estimated_outer_tail_bounds_J_m2": tail["estimated_tail_bounds_J_m2"],
                "estimated_total_outer_errors_J_m2": tail["estimated_total_outer_errors_J_m2"],
                "total_outer_tolerances_J_m2": tail["total_outer_tolerances_J_m2"],
                "finite_domain_budget_tolerances_J_m2": tail["finite_domain_budget_tolerances_J_m2"],
                "tail_budget_tolerances_J_m2": tail["tail_budget_tolerances_J_m2"],
                "finite_domain_channel_passed": tail["finite_domain_channel_passed"],
                "outer_tail_channel_passed": tail["tail_channel_passed"],
                "total_outer_channel_passed": tail["total_outer_channel_passed"],
                "outer_tail_certificate_path": "analytic_passive_vacuum",
                "outer_tail_certificate_pairing_independent": True,
                "outer_tail_certificate_contract": OUTER_TAIL_CERTIFICATE_CONTRACT,
                "passive_vacuum_tail_certificate": dict(metrics["passive_vacuum_tail_certificate"]),
                "power_metric_contraction_premise": dict(metrics["premise"]),
            }
        )
        output[str(pairing)] = base
    return output


def _geometric_pairing_results(result: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    output = _final_pairing_results(result, metrics)
    for row in output.values():
        row["outer_tail_certificate_path"] = "geometric_numerical_shell_envelope"
        row["outer_tail_certificate_pairing_independent"] = True
        row["outer_tail_certificate_contract"] = OUTER_TAIL_CERTIFICATE_CONTRACT
    return output


def _unresolved_result(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    cutoff_records: Sequence[Mapping[str, Any]],
    shell_records: Sequence[Mapping[str, Any]],
    pairing_results: Mapping[str, Any],
    selected_u_max: float | None,
    all_finite: bool,
    all_certified: bool,
    reason: str,
    provider: Any,
) -> AdaptiveOuterTailCasimirResult:
    return AdaptiveOuterTailCasimirResult(
        status="unresolved",
        config=config,
        cutoff_converged=False,
        outer_tail_estimated_flag=False,
        all_finite_domain_runs_converged=all_finite,
        all_microscopic_nodes_certified=all_certified,
        selected_u_max=selected_u_max,
        pairing_results=dict(pairing_results),
        cutoff_records=tuple(cutoff_records),
        shell_records=tuple(shell_records),
        termination_reason=reason,
        provider_statistics=_provider_statistics(provider),
    )


def run_certified_outer_tail_casimir(
    config: AdaptiveOuterTailCasimirConfig,
    *,
    provider: Any | None = None,
    joint_runner: Any = run_adaptive_joint_casimir,
) -> AdaptiveOuterTailCasimirResult:
    """Try the analytic certificate at every cutoff and retain geometric diagnostics."""

    if not isinstance(config, AdaptiveOuterTailCasimirConfig):
        raise TypeError("config must be an AdaptiveOuterTailCasimirConfig")
    active_provider = provider
    cutoff_records: list[Mapping[str, Any]] = []
    shell_records: list[Mapping[str, Any]] = []
    previous_values: dict[str, np.ndarray] | None = None
    previous_errors: dict[str, np.ndarray] | None = None
    previous_u = 0.0
    last_pairing_results: dict[str, Any] = {}
    all_finite = True
    all_certified = True
    pairings = tuple(config.joint_config.radial_config.point_config.pairings)
    indices = tuple(config.joint_config.radial_config.point_config.matsubara_indices)
    last_geometric: Mapping[str, Any] | None = None
    last_analytic: Mapping[str, Any] | None = None

    try:
        if active_provider is None:
            radial = config.joint_config.radial_config
            active_provider = CertifiedOuterQProvider(
                radial.point_config,
                cache_path=radial.point_cache_path,
            )
        for cutoff_index, u_max in enumerate(config.cutoff_u_values):
            run_config = _scaled_joint_config(config, cutoff_index=cutoff_index)
            result = joint_runner(run_config, provider=active_provider)
            usable, usable_reason = _joint_run_usable(result)
            all_finite = all_finite and usable
            all_certified = all_certified and bool(result.all_microscopic_nodes_certified)
            values: dict[str, np.ndarray] = {}
            finite_errors: dict[str, np.ndarray] = {}
            if usable:
                for pairing in pairings:
                    values[pairing], finite_errors[pairing] = _channel_arrays(
                        result, pairing, len(indices)
                    )
            base_record = dict(_cutoff_record(result, u_max=u_max, finite_errors=finite_errors))
            last_pairing_results = dict(result.pairing_results)
            if not usable:
                base_record["outer_tail_certificates"] = {
                    "geometric": None,
                    "analytic": {
                        "certificate_passed": False,
                        "rejection_reason": "finite_domain_run_unresolved",
                    },
                }
                cutoff_records.append(MappingProxyType(base_record))
                return _unresolved_result(
                    config,
                    cutoff_records=cutoff_records,
                    shell_records=shell_records,
                    pairing_results=last_pairing_results,
                    selected_u_max=u_max,
                    all_finite=False,
                    all_certified=all_certified,
                    reason=f"finite_domain_run_unresolved: {usable_reason}",
                    provider=active_provider,
                )
            if int(getattr(active_provider, "unique_q_count", 0)) > config.max_total_microscopic_q_nodes:
                base_record["outer_tail_certificates"] = {
                    "geometric": None,
                    "analytic": {
                        "certificate_passed": False,
                        "rejection_reason": "microscopic_q_node_budget_exhausted",
                    },
                }
                cutoff_records.append(MappingProxyType(base_record))
                return _unresolved_result(
                    config,
                    cutoff_records=cutoff_records,
                    shell_records=shell_records,
                    pairing_results=last_pairing_results,
                    selected_u_max=u_max,
                    all_finite=all_finite,
                    all_certified=all_certified,
                    reason="outer_tail_microscopic_q_node_budget_exhausted",
                    provider=active_provider,
                )

            zeros = {pairing: np.zeros(len(indices), dtype=float) for pairing in pairings}
            shell_records.append(
                _shell_record(
                    left_u=previous_u,
                    right_u=u_max,
                    previous_values=zeros if previous_values is None else previous_values,
                    current_values=values,
                    previous_errors=zeros if previous_errors is None else previous_errors,
                    current_errors=finite_errors,
                    indices=indices,
                )
            )
            geometric, decay_passed, geometric_tail_passed, geometric_total_passed = _tail_window_metrics(
                config,
                shell_records=shell_records,
                current_values=values,
                current_finite_errors=finite_errors,
                indices=indices,
            )
            geometric_passed = bool(
                geometric is not None
                and decay_passed
                and geometric_tail_passed
                and geometric_total_passed
            )
            if geometric is not None:
                geometric = dict(geometric)
                geometric["certificate_path"] = "geometric_numerical_shell_envelope"
                geometric["certificate_passed"] = geometric_passed
                geometric["rejection_reason"] = (
                    None if geometric_passed else "geometric_decay_or_budget_not_met"
                )
            last_geometric = geometric

            point_config = run_config.radial_config.point_config
            contraction = active_power_metric_contraction_certificate(
                active_provider,
                point_config=point_config,
            )
            analytic, analytic_passed = _analytic_metrics(
                config,
                u_max=u_max,
                values=values,
                finite_errors=finite_errors,
                indices=indices,
                contraction_certificate=contraction,
            )
            last_analytic = analytic
            base_record["outer_tail_certificates"] = {
                "geometric": None if geometric is None else dict(geometric),
                "analytic": dict(analytic),
            }
            cutoff_records.append(MappingProxyType(base_record))

            if analytic_passed:
                return AdaptiveOuterTailCasimirResult(
                    status="adaptive_finite_partial",
                    config=config,
                    cutoff_converged=True,
                    outer_tail_estimated_flag=True,
                    all_finite_domain_runs_converged=True,
                    all_microscopic_nodes_certified=True,
                    selected_u_max=u_max,
                    pairing_results=_analytic_pairing_results(result, analytic),
                    cutoff_records=tuple(cutoff_records),
                    shell_records=tuple(shell_records),
                    termination_reason="analytic_passive_vacuum_tail_bound_met",
                    provider_statistics=_provider_statistics(active_provider),
                )
            if geometric_passed and geometric is not None:
                return AdaptiveOuterTailCasimirResult(
                    status="adaptive_finite_partial",
                    config=config,
                    cutoff_converged=True,
                    outer_tail_estimated_flag=True,
                    all_finite_domain_runs_converged=True,
                    all_microscopic_nodes_certified=True,
                    selected_u_max=u_max,
                    pairing_results=_geometric_pairing_results(result, geometric),
                    cutoff_records=tuple(cutoff_records),
                    shell_records=tuple(shell_records),
                    termination_reason="geometric_outer_tail_bound_met",
                    provider_statistics=_provider_statistics(active_provider),
                )
            previous_values = values
            previous_errors = finite_errors
            previous_u = u_max

        geometric_reason = (
            "not_attempted"
            if last_geometric is None
            else str(last_geometric.get("rejection_reason", "unresolved"))
        )
        analytic_reason = (
            "not_attempted"
            if last_analytic is None
            else str(last_analytic.get("rejection_reason", "unresolved"))
        )
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            shell_records=shell_records,
            pairing_results=last_pairing_results,
            selected_u_max=config.cutoff_u_values[-1],
            all_finite=all_finite,
            all_certified=all_certified,
            reason=(
                "outer_tail_certificates_unresolved: "
                f"analytic={analytic_reason}; geometric={geometric_reason}"
            ),
            provider=active_provider,
        )
    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            shell_records=shell_records,
            pairing_results=last_pairing_results,
            selected_u_max=(None if not cutoff_records else float(cutoff_records[-1]["u_max"])),
            all_finite=False,
            all_certified=False,
            reason=f"point_provider_failure: {exc}",
            provider=active_provider,
        )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        return _unresolved_result(
            config,
            cutoff_records=cutoff_records,
            shell_records=shell_records,
            pairing_results=last_pairing_results,
            selected_u_max=(None if not cutoff_records else float(cutoff_records[-1]["u_max"])),
            all_finite=False,
            all_certified=False,
            reason=f"outer_tail_certificate_contract_failure: {exc}",
            provider=active_provider,
        )


__all__ = [
    "OUTER_TAIL_CERTIFICATE_CONTRACT",
    "PASSIVE_REFLECTION_BOUND_VERSION",
    "active_power_metric_contraction_certificate",
    "passive_vacuum_channel_bounds_J_m2",
    "passive_vacuum_tail_series",
    "run_certified_outer_tail_casimir",
]
