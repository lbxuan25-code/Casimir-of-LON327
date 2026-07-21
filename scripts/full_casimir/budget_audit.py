from __future__ import annotations

from typing import Any, Mapping, Sequence

from ._diagnostic_io import mapping, sequence


def audit_evidence_gaps(
    *,
    run_reports: Sequence[Mapping[str, Any]],
    policy_parity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fail-closed ledger of what current artifacts can and cannot prove."""

    tolerance_available = all(bool(mapping(report).get("tolerance_replay")) for report in run_reports)
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
            "policy_parity_evaluated": mapping(policy_parity).get("status") == "analyzed",
        },
        "quadrature_impact": {
            "status": "not_computable_from_current_artifacts",
            "reason": (
                "Certified-point caches contain point histories but do not retain the "
                "signed and absolute quadrature weights needed to propagate local logdet "
                "uncertainty into a total free-energy bound."
            ),
            "required_artifact": "read-only quadrature trace keyed by exact point identity",
        },
        "holdout_validation": {
            "status": "not_performed",
            "reason": (
                "A looser global policy must be checked against selected higher-N values "
                "that were not used to choose the candidate tolerance."
            ),
        },
        "analytic_outer_tail_bound": {
            "status": "not_established",
            "reason": (
                "Noise-floor classification is diagnostic evidence only. Production use "
                "requires a proved reflection/passivity and vacuum-propagation bound with "
                "the exact normalization and Matsubara prefactors used by this code."
            ),
        },
        "missing_evidence": missing,
        "production_policy_change_authorized": not missing,
        "fairness_rule": (
            "No candidate tolerance or tail path is recommended for production until the "
            "same evidence contract is satisfied for every pairing under comparison."
        ),
    }


def candidate_policy_screen(run_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Screen candidates using only replay evidence, without claiming final acceptance."""

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
    """Separate central shell signal from the finite-domain error floor.

    This is deliberately diagnostic-only: a signal below numerical resolution does not
    by itself bound the omitted infinite tail.
    """

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
                    indices = [int(value) for value in sequence(channel.get("matsubara_indices"))]
                central.append(
                    [abs(float(value)) for value in sequence(channel.get("shell_contributions_J_m2"))]
                )
                quadrature.append(
                    [float(value) for value in sequence(channel.get("shell_quadrature_error_bounds_J_m2"))]
                )
                envelope.append(
                    [float(value) for value in sequence(channel.get("shell_envelope_amplitudes_J_m2"))]
                )
            channel_records = []
            count = len(indices)
            for position in range(count):
                signal_values = [row[position] for row in central if len(row) == count]
                resolution_values = [row[position] for row in quadrature if len(row) == count]
                envelope_values = [row[position] for row in envelope if len(row) == count]
                if not signal_values or len(signal_values) != len(window):
                    channel_records.append(
                        {
                            "n": indices[position],
                            "classification": "shape_mismatch",
                        }
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


__all__ = [
    "audit_evidence_gaps",
    "candidate_policy_screen",
    "tail_resolution_audit",
]
