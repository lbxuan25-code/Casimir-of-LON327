from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

from lno327.constants import KB
from lno327.casimir.outer_quadrature import matsubara_prime_weights

from ._diagnostic_io import mapping, sequence


def audit_evidence_gaps(
    *,
    run_reports: Sequence[Mapping[str, Any]],
    policy_parity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the legacy fail-closed ledger from source artifacts alone."""

    tolerance_available = all(
        bool(mapping(report).get("tolerance_replay")) for report in run_reports
    )
    tail_runs = [
        mapping(report).get("outer_tail_replay")
        for report in run_reports
        if mapping(report).get("outer_tail_replay") is not None
    ]
    unresolved_counts = [
        int(mapping(mapping(report).get("point_cache")).get("unresolved_count", 0))
        for report in run_reports
    ]
    missing = []
    if not tolerance_available:
        missing.append("tolerance_replay_for_every_run")
    missing.extend(
        [
            "quadrature_trace_with_signed_and_absolute_point_weights",
            "quadrature_weighted_microscopic_error_bound",
            "independent_high_N_holdout_validation",
            "proved_pairing_independent_analytic_outer_tail_bound",
            "nonduplicating_end_to_end_error_ledger",
        ]
    )
    if mapping(policy_parity).get("pairing_blind_scientific_policy") is not True:
        missing.append("pairing_blind_numerical_policy")
    return {
        "schema": "convergence-audit-evidence-ledger-v1",
        "status": "incomplete" if missing else "complete",
        "current_artifact_evidence": {
            "run_count": len(run_reports),
            "unresolved_point_counts": unresolved_counts,
            "tolerance_replay_available_for_all_runs": tolerance_available,
            "outer_tail_replay_run_count": len(tail_runs),
            "policy_parity_evaluated": mapping(policy_parity).get("status")
            == "analyzed",
        },
        "missing_evidence": missing,
        "production_policy_change_authorized": not missing,
        "fairness_rule": (
            "No candidate tolerance or tail path is recommended for production until the "
            "same evidence contract is satisfied for every pairing under comparison."
        ),
    }


def candidate_policy_screen(run_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Screen candidates using stored histories without claiming final acceptance."""

    by_rtol: dict[float, dict[str, Any]] = {}
    for report in run_reports:
        run_dir = str(mapping(report).get("run_dir"))
        replay = mapping(mapping(report).get("tolerance_replay"))
        for raw in sequence(replay.get("candidate_policies")):
            row = mapping(raw)
            rtol = float(row.get("logdet_rtol"))
            state = by_rtol.setdefault(
                rtol,
                {
                    "logdet_rtol": rtol,
                    "runs": [],
                    "all_stored_points_established": True,
                    "hard_physical_failure_count": 0,
                },
            )
            unresolved = int(row.get("unresolved_count", 0))
            hard = int(row.get("hard_physical_failure_count", 0))
            state["runs"].append(
                {
                    "run_dir": run_dir,
                    "established_count": int(row.get("established_count", 0)),
                    "unresolved_count": unresolved,
                    "hard_physical_failure_count": hard,
                    "point_level_N2_work_proxy_saved_fraction": row.get(
                        "point_level_N2_work_proxy_saved_fraction"
                    ),
                }
            )
            state["all_stored_points_established"] = bool(
                state["all_stored_points_established"] and unresolved == 0
            )
            state["hard_physical_failure_count"] += hard
    candidates = []
    for rtol in sorted(by_rtol):
        state = by_rtol[rtol]
        replay_eligible = bool(
            state["all_stored_points_established"]
            and state["hard_physical_failure_count"] == 0
        )
        state["replay_screen_passed"] = replay_eligible
        state["production_ready"] = False
        state["remaining_requirements"] = [
            "quadrature_weighted_error_bound",
            "independent_high_N_holdout",
            "unified_tail_bound",
        ]
        candidates.append(state)
    return {
        "schema": "candidate-policy-screen-v1",
        "candidates": candidates,
        "interpretation": (
            "Passing this screen means only that stored histories would establish under "
            "the candidate policy. It is not a production authorization."
        ),
    }


def tail_resolution_audit(outer_tail_replay: Mapping[str, Any]) -> dict[str, Any]:
    """Separate central shell signal from the finite-domain error floor."""

    runs = []
    for raw_run in sequence(mapping(outer_tail_replay).get("outer_tail_runs")):
        run = mapping(raw_run)
        result = mapping(run.get("result"))
        config = mapping(result.get("config"))
        tail_start = float(config.get("tail_start_u", 0.0))
        window_size = int(config.get("tail_window_shells", 0))
        shells = [
            mapping(record)
            for record in sequence(result.get("shell_records"))
            if float(mapping(record).get("left_u", -1.0)) >= tail_start
        ]
        window = shells[-window_size:] if window_size > 0 else []
        pairing_rows: dict[str, Any] = {}
        pairing_names = set()
        for shell in window:
            pairing_names.update(mapping(shell.get("pairings")))
        for pairing in sorted(pairing_names):
            central = []
            quadrature = []
            envelope = []
            indices: list[int] = []
            for shell in window:
                channel = mapping(mapping(shell.get("pairings")).get(pairing))
                if not indices:
                    indices = [
                        int(value)
                        for value in sequence(channel.get("matsubara_indices"))
                    ]
                central.append(
                    [
                        abs(float(value))
                        for value in sequence(
                            channel.get("shell_contributions_J_m2")
                        )
                    ]
                )
                quadrature.append(
                    [
                        float(value)
                        for value in sequence(
                            channel.get("shell_quadrature_error_bounds_J_m2")
                        )
                    ]
                )
                envelope.append(
                    [
                        float(value)
                        for value in sequence(
                            channel.get("shell_envelope_amplitudes_J_m2")
                        )
                    ]
                )
            channel_records = []
            count = len(indices)
            for position in range(count):
                signal_values = [row[position] for row in central if len(row) == count]
                resolution_values = [
                    row[position] for row in quadrature if len(row) == count
                ]
                envelope_values = [
                    row[position] for row in envelope if len(row) == count
                ]
                if not signal_values or len(signal_values) != len(window):
                    channel_records.append(
                        {"n": indices[position], "classification": "shape_mismatch"}
                    )
                    continue
                latest_signal = signal_values[-1]
                latest_resolution = resolution_values[-1]
                signal_to_resolution = (
                    float("inf")
                    if latest_resolution == 0.0 and latest_signal > 0.0
                    else 0.0
                    if latest_resolution == 0.0
                    else latest_signal / latest_resolution
                )
                if latest_signal <= latest_resolution:
                    classification = "below_finite_domain_resolution"
                else:
                    central_ratios = [
                        signal_values[index] / signal_values[index - 1]
                        if signal_values[index - 1] > 0.0
                        else float("inf")
                        for index in range(1, len(signal_values))
                    ]
                    ratio_max = float(config.get("tail_ratio_max", 0.8))
                    classification = (
                        "resolved_contracting"
                        if central_ratios and max(central_ratios) <= ratio_max
                        else "resolved_noncontracting"
                    )
                channel_records.append(
                    {
                        "n": indices[position],
                        "central_shell_signals_J_m2": signal_values,
                        "finite_domain_resolution_J_m2": resolution_values,
                        "shell_envelope_amplitudes_J_m2": envelope_values,
                        "latest_signal_to_resolution_ratio": signal_to_resolution,
                        "classification": classification,
                        "production_tail_bound_established": False,
                    }
                )
            pairing_rows[pairing] = {
                "matsubara_indices": indices,
                "channels": channel_records,
            }
        runs.append(
            {
                "matsubara_cutoff": run.get("matsubara_cutoff"),
                "tail_start_u": tail_start,
                "tail_window_shells": window_size,
                "window_available": len(window) == window_size and window_size >= 2,
                "pairings": pairing_rows,
            }
        )
    return {
        "schema": "outer-tail-resolution-audit-v1",
        "status": "analyzed" if runs else "not_available",
        "outer_tail_runs": runs,
        "acceptance_effect": "diagnostic_only",
        "analytic_bound_status": "not_established",
        "warning": (
            "Below-resolution shell signals explain why the geometric envelope can stall, "
            "but they do not by themselves certify the omitted infinite tail."
        ),
    }


def weighted_microscopic_impact(
    trace: Sequence[Mapping[str, Any]],
    point_evidence: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    by_channel: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {
            "signed_delta_J_m2": 0.0,
            "absolute_error_bound_J_m2": 0.0,
        }
    )
    rows: list[dict[str, Any]] = []
    missing: list[list[Any]] = []
    for raw in trace:
        row = mapping(raw)
        identity_list = sequence(row.get("identity"))
        identity = (
            str(identity_list[0]),
            int(identity_list[1]),
            str(identity_list[2]),
            str(identity_list[3]),
        )
        evidence = point_evidence.get(identity)
        if evidence is None:
            missing.append(list(identity))
            continue
        signed_weight = float(row.get("signed_weight_J_m2_per_logdet"))
        absolute_weight = float(row.get("absolute_weight_J_m2_per_logdet"))
        delta = float(evidence.get("empirical_delta"))
        uncertainty = float(evidence.get("local_absolute_uncertainty"))
        signed_delta = signed_weight * delta
        bound = absolute_weight * uncertainty
        by_channel[(identity[0], identity[1])]["signed_delta_J_m2"] += signed_delta
        by_channel[(identity[0], identity[1])][
            "absolute_error_bound_J_m2"
        ] += bound
        rows.append(
            {
                "identity": list(identity),
                "signed_weight_J_m2_per_logdet": signed_weight,
                "empirical_delta_logdet": delta,
                "local_absolute_uncertainty_logdet": uncertainty,
                "signed_energy_delta_J_m2": signed_delta,
                "absolute_energy_error_bound_J_m2": bound,
                "point_level_N2_work_proxy_saved": evidence.get(
                    "point_level_N2_work_proxy_saved", 0
                ),
            }
        )
    rows.sort(
        key=lambda item: float(item["absolute_energy_error_bound_J_m2"]),
        reverse=True,
    )
    channel_rows = [
        {"pairing": pairing, "n": n, **values}
        for (pairing, n), values in sorted(by_channel.items())
    ]
    return {
        "schema": "quadrature-weighted-microscopic-impact-v1",
        "trace_entry_count": len(trace),
        "matched_entry_count": len(rows),
        "missing_identity_count": len(missing),
        "missing_identities": missing,
        "channels": channel_rows,
        "total_signed_delta_J_m2": math.fsum(
            float(item["signed_energy_delta_J_m2"]) for item in rows
        ),
        "total_absolute_error_bound_J_m2": math.fsum(
            float(item["absolute_energy_error_bound_J_m2"]) for item in rows
        ),
        "top_point_contributors": rows[:64],
    }


def holdout_plan(
    point_evidence: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    impact: Mapping[str, Any],
    *,
    max_points: int = 32,
) -> dict[str, Any]:
    ranked = [mapping(row) for row in sequence(impact.get("top_point_contributors"))]
    selected: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in ranked:
        identity_list = sequence(row.get("identity"))
        identity = (
            str(identity_list[0]),
            int(identity_list[1]),
            str(identity_list[2]),
            str(identity_list[3]),
        )
        evidence = mapping(point_evidence.get(identity))
        reference_N = int(evidence.get("reference_highest_valid_N", 0))
        step = max(2, 2 * round(max(reference_N // 10, 64) / 2))
        selected[identity] = {
            "identity": list(identity),
            "reason": "largest_quadrature_weighted_uncertainty",
            "reference_highest_valid_N": reference_N,
            "suggested_holdout_N": [reference_N + step, reference_N + 2 * step],
            "predicted_local_uncertainty": evidence.get(
                "local_absolute_uncertainty"
            ),
            "weighted_error_contribution_J_m2": row.get(
                "absolute_energy_error_bound_J_m2"
            ),
        }
        if len(selected) >= max_points:
            break
    return {
        "schema": "independent-high-N-holdout-plan-v1",
        "status": "planned_not_executed",
        "selection_count": len(selected),
        "points": list(selected.values()),
        "independence_requirement": (
            "Holdout N values must be computed after the candidate policy is frozen and "
            "must not be used to retune that candidate."
        ),
    }


def _tail_series(u0: float) -> tuple[float, int]:
    total = 0.0
    for m in range(1, 100000):
        term = math.exp(-m * u0) * (
            u0 / (m * m) + 1.0 / (m * m * m)
        )
        total += term
        if term <= max(1e-300, 1e-15 * max(total, 1e-300)):
            return total, m
    raise RuntimeError("analytic tail series did not converge")


def conditional_analytic_outer_tail_bound(
    *,
    u0: float,
    separation_nm: float,
    temperature_K: float,
    matsubara_indices: Sequence[int],
) -> dict[str, Any]:
    series, terms = _tail_series(float(u0))
    d = float(separation_nm) * 1e-9
    indices = tuple(int(value) for value in matsubara_indices)
    prime = matsubara_prime_weights(indices)
    bounds = [
        float(KB * float(temperature_K) * weight * series / (4.0 * math.pi * d * d))
        for weight in prime
    ]
    return {
        "schema": "conditional-passive-vacuum-tail-bound-v1",
        "status": "conditional_bound_derived",
        "u0": float(u0),
        "series_value": series,
        "series_terms_used": terms,
        "matsubara_indices": list(indices),
        "channel_bounds_J_m2": bounds,
        "assumptions": [
            "the round-trip reflection operator is a contraction in the determinant-preserving power metric",
            "the two-polarization logdet magnitude is bounded by -2 log(1-exp(-u))",
            "the current u=2Qd normalization and full angular measure are used",
        ],
        "production_usable": False,
        "missing_proof": (
            "The repository does not yet persist the power-metric singular-value "
            "certificate needed to verify the contraction premise for every point."
        ),
    }


def end_to_end_error_ledger(
    *,
    matsubara_payload: Mapping[str, Any],
    outer_payload: Mapping[str, Any],
    microscopic_impact: Mapping[str, Any],
    analytic_tail: Mapping[str, Any],
) -> dict[str, Any]:
    cutoff_records = sequence(outer_payload.get("cutoff_records"))
    latest = mapping(cutoff_records[-1]) if cutoff_records else {}
    finite_by_pairing = mapping(latest.get("finite_domain_error_bounds_J_m2"))
    micro_by_channel = {
        (str(row.get("pairing")), int(row.get("n"))): float(
            row.get("absolute_error_bound_J_m2", 0.0)
        )
        for row in sequence(microscopic_impact.get("channels"))
    }
    pairings: dict[str, Any] = {}
    for pairing, values_raw in finite_by_pairing.items():
        values = [float(value) for value in sequence(values_raw)]
        indices = [
            int(value)
            for value in sequence(
                mapping(mapping(latest.get("pairing_results")).get(pairing)).get(
                    "matsubara_indices"
                )
            )
        ]
        micro_values = [micro_by_channel.get((str(pairing), n)) for n in indices]
        analytic_values = [
            float(value)
            for value in sequence(analytic_tail.get("channel_bounds_J_m2"))
        ]
        if len(analytic_values) != len(indices):
            analytic_values = []
        pairings[str(pairing)] = {
            "matsubara_indices": indices,
            "microscopic_error_bounds_J_m2": micro_values,
            "finite_domain_error_bounds_J_m2": values,
            "conditional_analytic_outer_tail_bounds_J_m2": (
                analytic_values if analytic_values else None
            ),
            "certified_outer_tail_bounds_J_m2": None,
            "nonduplicating_sum_rule": (
                "microscopic + combined finite-domain + certified outer tail; "
                "radial/angular/offset subcomponents are diagnostics and are not added again"
            ),
            "status": "incomplete_until_analytic_tail_premise_is_proved",
        }
    matsubara_pairings = mapping(matsubara_payload.get("pairing_results"))
    return {
        "schema": "nonduplicating-end-to-end-error-ledger-v1",
        "pairings": pairings,
        "matsubara_tail_evidence": {
            str(pairing): {
                "estimated_matsubara_tail_bound_J_m2": mapping(payload).get(
                    "estimated_matsubara_tail_bound_J_m2"
                ),
                "estimated_total_error_J_m2": mapping(payload).get(
                    "estimated_total_error_J_m2"
                ),
                "total_free_energy_tolerance_J_m2": mapping(payload).get(
                    "total_free_energy_tolerance_J_m2"
                ),
            }
            for pairing, payload in matsubara_pairings.items()
        },
        "status": "incomplete",
        "double_counting_prevented": True,
    }


def budget_fraction_sensitivity(outer_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-score stored radial/angular errors under common budget splits."""

    fractions = (0.75, 0.80, 0.85)
    records = []
    for raw_cutoff in sequence(outer_payload.get("cutoff_records")):
        cutoff = mapping(raw_cutoff)
        pairings = mapping(cutoff.get("pairing_results"))
        for fraction in fractions:
            all_passed = True
            channels = []
            for pairing, raw_payload in pairings.items():
                payload = mapping(raw_payload)
                radial = [
                    float(v)
                    for v in sequence(
                        payload.get("combined_comparison_radial_errors_J_m2")
                    )
                ]
                angular = [
                    float(v)
                    for v in sequence(payload.get("estimated_angular_errors_J_m2"))
                ]
                total_tol = [
                    float(v)
                    for v in sequence(payload.get("outer_tolerances_J_m2"))
                ]
                count = min(len(radial), len(angular), len(total_tol))
                radial_pass = [
                    radial[i] <= fraction * total_tol[i] for i in range(count)
                ]
                angular_pass = [
                    angular[i] <= (1.0 - fraction) * total_tol[i]
                    for i in range(count)
                ]
                joint_pass = [
                    radial[i] + angular[i] <= total_tol[i] for i in range(count)
                ]
                passed = bool(
                    count
                    and all(radial_pass)
                    and all(angular_pass)
                    and all(joint_pass)
                )
                all_passed = all_passed and passed
                channels.append(
                    {
                        "pairing": str(pairing),
                        "matsubara_indices": payload.get("matsubara_indices"),
                        "radial_passed": radial_pass,
                        "angular_passed": angular_pass,
                        "joint_passed": joint_pass,
                        "all_passed": passed,
                    }
                )
            records.append(
                {
                    "u_max": cutoff.get("u_max"),
                    "radial_budget_fraction": fraction,
                    "angular_budget_fraction": 1.0 - fraction,
                    "all_channels_passed": all_passed,
                    "channels": channels,
                }
            )
    return {
        "schema": "radial-angular-budget-counterfactual-v1",
        "status": "screened" if records else "not_available",
        "records": records,
        "exact_replay_still_required": True,
    }


def audit_completion_ledger(
    *,
    run_reports: Sequence[Mapping[str, Any]],
    closure_replays: Sequence[Mapping[str, Any]],
    policy_parity: Mapping[str, Any],
) -> dict[str, Any]:
    equivalence = all(
        mapping(mapping(report).get("production_replay_equivalence")).get(
            "equivalent"
        )
        is True
        for report in run_reports
    )
    trace_complete = bool(closure_replays) and all(
        all(
            mapping(mapping(outer).get("microscopic_impact")).get(
                "missing_identity_count"
            )
            == 0
            for outer in sequence(mapping(replay).get("outer_runs"))
        )
        for replay in closure_replays
    )
    holdout_planned = bool(closure_replays) and all(
        all(
            mapping(mapping(outer).get("holdout_plan")).get("status")
            == "planned_not_executed"
            for outer in sequence(mapping(replay).get("outer_runs"))
        )
        for replay in closure_replays
    )
    conditional_tail_available = bool(closure_replays) and all(
        all(
            mapping(
                mapping(outer).get("conditional_analytic_outer_tail_bound")
            ).get("status")
            == "conditional_bound_derived"
            for outer in sequence(mapping(replay).get("outer_runs"))
        )
        for replay in closure_replays
    )
    error_ledgers_present = bool(closure_replays) and all(
        mapping(replay).get("error_ledger") is not None for replay in closure_replays
    )
    missing = []
    if not equivalence:
        missing.append("production_replay_pointwise_equivalence")
    if not trace_complete:
        missing.append("complete_quadrature_weight_trace")
    if not holdout_planned:
        missing.append("independent_high_N_holdout_plan")
    missing.extend(
        [
            "independent_high_N_holdout_execution",
            "power_metric_round_trip_contraction_proof",
            "production_wall_time_benchmark_under_frozen_candidate",
        ]
    )
    if not conditional_tail_available:
        missing.append("conditional_analytic_outer_tail_formula")
    if not error_ledgers_present:
        missing.append("nonduplicating_error_ledger_structure")
    if mapping(policy_parity).get("pairing_blind_scientific_policy") is not True:
        missing.append("source_pairing_blind_numerical_policy")
    framework_blockers = {
        "production_replay_pointwise_equivalence",
        "complete_quadrature_weight_trace",
        "independent_high_N_holdout_plan",
        "conditional_analytic_outer_tail_formula",
        "nonduplicating_error_ledger_structure",
    }
    return {
        "schema": "convergence-audit-completion-ledger-v1",
        "framework_components": {
            "production_replay_equivalence": equivalence,
            "quadrature_weighted_impact": trace_complete,
            "holdout_plan": holdout_planned,
            "conditional_analytic_tail_bound": conditional_tail_available,
            "nonduplicating_error_ledger": error_ledgers_present,
            "cache_only_replay_wall_time_recorded": bool(closure_replays),
        },
        "framework_implementation_complete": not any(
            item in framework_blockers for item in missing
        ),
        "production_parameter_change_authorized": False,
        "missing_execution_or_proof_evidence": missing,
    }


__all__ = [
    "audit_completion_ledger",
    "audit_evidence_gaps",
    "budget_fraction_sensitivity",
    "candidate_policy_screen",
    "conditional_analytic_outer_tail_bound",
    "end_to_end_error_ledger",
    "holdout_plan",
    "tail_resolution_audit",
    "weighted_microscopic_impact",
]
