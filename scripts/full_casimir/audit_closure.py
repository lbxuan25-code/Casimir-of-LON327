from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.constants import KB
from lno327.casimir.adaptive_joint_q import run_adaptive_joint_casimir
from lno327.casimir.adaptive_matsubara_tail import run_adaptive_matsubara_casimir
from lno327.casimir.adaptive_outer_q import (
    AdaptiveRadialPanel,
    build_adaptive_outer_q_panel_grid,
    run_adaptive_radial_casimir,
)
from lno327.casimir.adaptive_outer_tail import run_adaptive_outer_tail_casimir
from lno327.casimir.certified_point_provider import (
    FrequencyExtendableCertifiedOuterQProvider,
    certified_point_policy_fingerprint,
    certified_point_policy_payload,
)
from lno327.casimir.outer_quadrature import matsubara_prime_weights
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE

from ._diagnostic_io import mapping, sequence, sha256
from .outer_tail_diagnostics import _config_from_payload
from .point_diagnostics import replay_point_policy

_SCHEMA = "convergence-audit-closure-v1"


def _entry_identity(entry: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(entry.get("pairing")),
        int(entry.get("n")),
        str(entry.get("qx_hex")),
        str(entry.get("qy_hex")),
    )


def _primary_value_at_N(point: Mapping[str, Any], N: int) -> float | None:
    for raw in sequence(point.get("history")):
        row = mapping(raw)
        if int(row.get("N", -1)) != int(N):
            continue
        shifts = mapping(row.get("shifts"))
        if not shifts:
            return None
        state = mapping(next(iter(shifts.values())))
        value = state.get("two_plate_logdet")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number) or not bool(state.get("hard_physical_passed")):
            return None
        return number
    return None


def _highest_valid_primary(point: Mapping[str, Any]) -> tuple[int | None, float | None]:
    selected_N: int | None = None
    selected_value: float | None = None
    for raw in sequence(point.get("history")):
        row = mapping(raw)
        N = int(row.get("N", -1))
        value = _primary_value_at_N(point, N)
        if value is not None and (selected_N is None or N > selected_N):
            selected_N = N
            selected_value = value
    return selected_N, selected_value


def _max_absolute_uncertainty(replay: Mapping[str, Any]) -> float:
    candidates: list[float] = []
    cross = mapping(replay.get("latest_cross_shift"))
    if isinstance(cross.get("absolute"), (int, float)):
        value = float(cross["absolute"])
        if math.isfinite(value):
            candidates.append(abs(value))
    for record in mapping(replay.get("latest_adjacent_N_by_shift")).values():
        row = mapping(record)
        if isinstance(row.get("absolute"), (int, float)):
            value = float(row["absolute"])
            if math.isfinite(value):
                candidates.append(abs(value))
    envelope = mapping(replay.get("latest_oscillatory_envelope"))
    joint = mapping(envelope.get("joint_logdet_envelope"))
    if isinstance(joint.get("absolute"), (int, float)):
        value = float(joint["absolute"])
        if math.isfinite(value):
            candidates.append(abs(value))
    return max(candidates, default=math.inf)


def production_equivalence_audit(
    cache: Mapping[str, Any],
    *,
    source_logdet_rtol: float,
    source_logdet_atol: float,
    required_consecutive_passes: int,
) -> dict[str, Any]:
    """Require the audit replay to reproduce every persisted production decision."""

    mismatches: list[dict[str, Any]] = []
    checked = 0
    for raw_entry in sequence(cache.get("entries")):
        entry = mapping(raw_entry)
        point = mapping(entry.get("point_result"))
        observed = mapping(point.get("sweet_spot"))
        replay = replay_point_policy(
            point,
            logdet_rtol=float(source_logdet_rtol),
            logdet_atol=float(source_logdet_atol),
            required_consecutive_passes=int(required_consecutive_passes),
        )
        checked += 1
        fields = ("status", "working_N", "audit_N", "establishment_mode")
        differences = {
            name: {"production": observed.get(name), "replay": replay.get(name)}
            for name in fields
            if observed.get(name) != replay.get(name)
        }
        if differences:
            mismatches.append(
                {"identity": list(_entry_identity(entry)), "differences": differences}
            )
    return {
        "schema": "production-replay-equivalence-audit-v1",
        "checked_point_count": checked,
        "mismatch_count": len(mismatches),
        "equivalent": not mismatches,
        "mismatches": mismatches,
    }


def _candidate_config(
    base_config: Any,
    *,
    logdet_rtol: float,
    cache_path: Path,
    radial_fraction: float | None = None,
    N_candidates_override: Sequence[int] | None = None,
):
    point = base_config.outer_tail_config.joint_config.radial_config.point_config
    point = replace(
        point,
        logdet_rtol=float(logdet_rtol),
        N_candidates=(
            point.N_candidates
            if N_candidates_override is None
            else tuple(int(value) for value in N_candidates_override)
        ),
    )
    radial = replace(
        base_config.outer_tail_config.joint_config.radial_config,
        point_config=point,
        point_cache_path=Path(cache_path),
    )
    joint = replace(base_config.outer_tail_config.joint_config, radial_config=radial)
    if radial_fraction is not None:
        fraction = float(radial_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("radial_fraction must lie strictly between zero and one")
        joint = replace(
            joint,
            radial_budget_fraction=fraction,
            angular_budget_fraction=1.0 - fraction,
        )
    outer = replace(base_config.outer_tail_config, joint_config=joint)
    return replace(base_config, outer_tail_config=outer, point_cache_path=Path(cache_path))


def _project_cache(
    cache: Mapping[str, Any],
    *,
    candidate_config: Any,
) -> tuple[dict[str, Any], dict[tuple[str, int, str, str], dict[str, Any]]]:
    point_config = candidate_config.outer_tail_config.joint_config.radial_config.point_config
    projected = deepcopy(dict(cache))
    projected_entries: list[dict[str, Any]] = []
    point_evidence: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    unresolved: list[tuple[str, int, str, str]] = []

    for raw_entry in sequence(cache.get("entries")):
        entry = deepcopy(dict(mapping(raw_entry)))
        identity = _entry_identity(entry)
        point = deepcopy(dict(mapping(entry.get("point_result"))))
        replay = replay_point_policy(
            point,
            logdet_rtol=float(point_config.logdet_rtol),
            logdet_atol=float(point_config.logdet_atol),
            required_consecutive_passes=int(point_config.required_consecutive_passes),
        )
        if replay.get("status") != "established":
            unresolved.append(identity)
            projected_entries.append(entry)
            continue
        audit_N = int(replay["audit_N"])
        candidate_value = _primary_value_at_N(point, audit_N)
        reference_N, reference_value = _highest_valid_primary(point)
        if candidate_value is None or reference_value is None:
            unresolved.append(identity)
            projected_entries.append(entry)
            continue
        history = []
        for raw_row in sequence(point.get("history")):
            row = deepcopy(dict(mapping(raw_row)))
            if int(row.get("N", -1)) == audit_N:
                row["two_plate_logdet_cross_shift"] = deepcopy(
                    replay.get("latest_cross_shift")
                )
                row["adjacent_N_by_shift"] = deepcopy(
                    replay.get("latest_adjacent_N_by_shift")
                )
                row["adjacent_N_all_shifts_passed"] = all(
                    bool(mapping(value).get("passed"))
                    for value in mapping(replay.get("latest_adjacent_N_by_shift")).values()
                )
                row["oscillatory_envelope"] = deepcopy(
                    replay.get("latest_oscillatory_envelope")
                )
            history.append(row)
        point["history"] = history
        point["sweet_spot"] = {
            "status": "established",
            "working_N": int(replay["working_N"]),
            "audit_N": audit_N,
            "establishment_mode": str(replay["establishment_mode"]),
        }
        point["audit_projection"] = {
            "schema": "candidate-point-projection-v1",
            "logdet_rtol": float(point_config.logdet_rtol),
            "source_history_unchanged_except_selected_gate_metadata": True,
        }
        entry["point_result"] = point
        projected_entries.append(entry)
        point_evidence[identity] = {
            "identity": list(identity),
            "candidate_working_N": int(replay["working_N"]),
            "candidate_audit_N": audit_N,
            "candidate_value": candidate_value,
            "reference_highest_valid_N": reference_N,
            "reference_value": reference_value,
            "empirical_delta": candidate_value - reference_value,
            "local_absolute_uncertainty": _max_absolute_uncertainty(replay),
            "point_level_N2_work_proxy_saved": int(
                replay.get("point_level_N2_work_proxy_saved", 0)
            ),
        }

    if unresolved:
        raise ValueError(
            "candidate cache projection left unresolved points: "
            + ", ".join("|".join(map(str, item)) for item in unresolved[:8])
        )
    projected["entries"] = projected_entries
    projected["point_policy"] = certified_point_policy_payload(
        point_config, frequency_extendable=True
    )
    projected["policy_fingerprint"] = certified_point_policy_fingerprint(
        point_config, frequency_extendable=True
    )
    projected["frequency_extendable"] = True
    projected["active_matsubara_indices"] = list(point_config.matsubara_indices)
    return projected, point_evidence


def _forbid_new_certification(*args: Any, **kwargs: Any):
    del args, kwargs
    raise RuntimeError(
        "candidate audit replay encountered a cache miss; new microscopic work is forbidden"
    )


def _select_primary_radial_result(
    radial_runs: Sequence[Mapping[str, Any]],
    outer_payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    cutoff_records = sequence(outer_payload.get("cutoff_records"))
    if not cutoff_records:
        return None
    latest = mapping(cutoff_records[-1])
    u_max = float(latest.get("u_max"))
    angular_order = latest.get("selected_angular_order")
    radial_cap = latest.get("selected_radial_round_cap")
    primary_offset = float(
        mapping(mapping(outer_payload.get("config")).get("joint_config")).get(
            "primary_offset_fraction", 0.5
        )
    )
    matches: list[Mapping[str, Any]] = []
    for raw in radial_runs:
        row = mapping(raw)
        config = mapping(row.get("config"))
        edges = sequence(config.get("initial_panel_edges"))
        if not edges or not math.isclose(
            float(edges[-1]), u_max, rel_tol=0.0, abs_tol=1e-12
        ):
            continue
        if angular_order is not None and int(config.get("angular_order")) != int(
            angular_order
        ):
            continue
        if radial_cap is not None and int(config.get("max_refinement_rounds")) != int(
            radial_cap
        ):
            continue
        if not math.isclose(
            float(config.get("angular_offset_fraction")),
            primary_offset,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            continue
        matches.append(row)
    return matches[-1] if matches else None


def _quadrature_trace_from_radial_result(
    radial_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    config = mapping(radial_payload.get("config"))
    point = mapping(config.get("point_config"))
    temperature = float(point.get("temperature_K"))
    separation = float(point.get("separation_nm")) * 1e-9
    angular_order = int(config.get("angular_order"))
    radial_order = int(config.get("radial_order"))
    offset = float(config.get("angular_offset_fraction"))
    pairings = [str(value) for value in sequence(point.get("pairings"))]
    indices = [int(value) for value in sequence(point.get("matsubara_indices"))]
    prime = matsubara_prime_weights(indices)
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    aggregated: dict[tuple[str, int, str, str], float] = defaultdict(float)

    for raw_panel in sequence(radial_payload.get("panel_records")):
        panel_row = mapping(raw_panel)
        parent = AdaptiveRadialPanel(
            float(panel_row.get("left_u")),
            float(panel_row.get("right_u")),
            int(panel_row.get("depth", 0)),
        )
        for child in parent.split():
            grid = build_adaptive_outer_q_panel_grid(
                child,
                separation_m=separation,
                lattice_a_x_m=material.lattice_a_x_m,
                lattice_a_y_m=material.lattice_a_y_m,
                radial_order=radial_order,
                angular_order=angular_order,
                angular_offset_fraction=offset,
            )
            for q, measure in zip(
                grid.q_model, grid.measure_weights_m_inv2, strict=True
            ):
                qx_hex = float(q[0]).hex()
                qy_hex = float(q[1]).hex()
                for pairing in pairings:
                    for position, n in enumerate(indices):
                        coefficient = float(
                            KB * temperature * prime[position] * measure
                        )
                        aggregated[(pairing, n, qx_hex, qy_hex)] += coefficient

    return [
        {
            "identity": [pairing, n, qx_hex, qy_hex],
            "signed_weight_J_m2_per_logdet": coefficient,
            "absolute_weight_J_m2_per_logdet": abs(coefficient),
        }
        for (pairing, n, qx_hex, qy_hex), coefficient in sorted(
            aggregated.items()
        )
    ]


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
    ranked = [
        mapping(row) for row in sequence(impact.get("top_point_contributors"))
    ]
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
            "suggested_holdout_N": [
                reference_N + step,
                reference_N + 2 * step,
            ],
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
        float(
            KB
            * float(temperature_K)
            * weight
            * series
            / (4.0 * math.pi * d * d)
        )
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
            "certificate needed to verify the contraction premise for every microscopic point."
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
    finite_by_pairing = mapping(
        latest.get("finite_domain_error_bounds_J_m2")
    )
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
                mapping(
                    mapping(latest.get("pairing_results")).get(pairing)
                ).get("matsubara_indices")
            )
        ]
        micro = [
            micro_by_channel.get((str(pairing), n), math.inf) for n in indices
        ]
        analytic = [
            float(value)
            for value in sequence(analytic_tail.get("channel_bounds_J_m2"))
        ]
        if len(analytic) != len(indices):
            analytic = [math.inf] * len(indices)
        pairings[str(pairing)] = {
            "matsubara_indices": indices,
            "microscopic_error_bounds_J_m2": micro,
            "finite_domain_error_bounds_J_m2": values,
            "conditional_analytic_outer_tail_bounds_J_m2": analytic,
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


def candidate_cache_replay_audit(
    *,
    run_dir: Path,
    config_payload: Mapping[str, Any],
    cache_payload: Mapping[str, Any],
    point_cache_path: Path,
    candidate_logdet_rtol: float,
    radial_fraction: float | None = None,
    N_candidates_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    source_hash_before = sha256(point_cache_path)
    with TemporaryDirectory(prefix="casimir-candidate-audit-") as temporary:
        temporary_cache = Path(temporary) / "certified_points.json"
        shutil.copy2(point_cache_path, temporary_cache)
        base_config = _config_from_payload(
            config_payload, point_cache_path=temporary_cache
        )
        candidate_config = _candidate_config(
            base_config,
            logdet_rtol=float(candidate_logdet_rtol),
            cache_path=temporary_cache,
            radial_fraction=radial_fraction,
            N_candidates_override=N_candidates_override,
        )
        projected, point_evidence = _project_cache(
            cache_payload, candidate_config=candidate_config
        )
        temporary_cache.write_text(
            json.dumps(projected, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        first_cutoff = int(candidate_config.matsubara_cutoff_values[0])
        base_point = (
            candidate_config.outer_tail_config.joint_config.radial_config.point_config
        )
        first_point = replace(
            base_point, matsubara_indices=tuple(range(first_cutoff + 1))
        )
        provider = FrequencyExtendableCertifiedOuterQProvider(
            first_point,
            cache_path=temporary_cache,
            runner=_forbid_new_certification,
            certifier_q_batch_size=candidate_config.certifier_q_batch_size,
        )
        captured: list[dict[str, Any]] = []

        def outer_runner(outer_config: Any, *, provider: Any = None):
            radial_runs: list[dict[str, Any]] = []

            def radial_runner(radial_config: Any, *, provider: Any = None):
                started = perf_counter()
                result = run_adaptive_radial_casimir(
                    radial_config, provider=provider
                )
                radial_runs.append(
                    {
                        "wall_seconds": perf_counter() - started,
                        "config": result.config.as_dict(),
                        "result": result.as_dict(),
                    }
                )
                return result

            def joint_runner(joint_config: Any, *, provider: Any = None):
                return run_adaptive_joint_casimir(
                    joint_config,
                    provider=provider,
                    radial_runner=radial_runner,
                )

            started = perf_counter()
            result = run_adaptive_outer_tail_casimir(
                outer_config,
                provider=provider,
                joint_runner=joint_runner,
            )
            captured.append(
                {
                    "wall_seconds": perf_counter() - started,
                    "result": result.as_dict(),
                    "radial_runs": radial_runs,
                }
            )
            return result

        started = perf_counter()
        matsubara_result = run_adaptive_matsubara_casimir(
            candidate_config,
            provider=provider,
            outer_tail_runner=outer_runner,
        )
        replay_wall = perf_counter() - started

        outer_reports = []
        for captured_run in captured:
            outer_payload = mapping(captured_run.get("result"))
            selected_radial = _select_primary_radial_result(
                sequence(captured_run.get("radial_runs")), outer_payload
            )
            trace = []
            impact: dict[str, Any] = {
                "status": "not_available",
                "reason": (
                    "selected primary radial estimator could not be identified"
                ),
            }
            if selected_radial is not None:
                radial_result = mapping(selected_radial.get("result"))
                trace = _quadrature_trace_from_radial_result(radial_result)
                impact = weighted_microscopic_impact(trace, point_evidence)
            radial_config = mapping(
                mapping(
                    mapping(outer_payload.get("config")).get("joint_config")
                ).get("radial_config")
            )
            microscopic = mapping(radial_config.get("point_config"))
            u0 = float(outer_payload.get("selected_u_max") or 0.0)
            analytic = conditional_analytic_outer_tail_bound(
                u0=u0,
                separation_nm=float(microscopic.get("separation_nm")),
                temperature_K=float(microscopic.get("temperature_K")),
                matsubara_indices=[
                    int(value)
                    for value in sequence(microscopic.get("matsubara_indices"))
                ],
            )
            outer_reports.append(
                {
                    "outer_result": outer_payload,
                    "outer_wall_seconds": captured_run.get("wall_seconds"),
                    "radial_run_count": len(
                        sequence(captured_run.get("radial_runs"))
                    ),
                    "selected_primary_radial_result_found": (
                        selected_radial is not None
                    ),
                    "quadrature_trace": {
                        "schema": "exact-final-estimator-quadrature-trace-v1",
                        "entry_count": len(trace),
                        "entries": trace,
                    },
                    "microscopic_impact": impact,
                    "holdout_plan": holdout_plan(point_evidence, impact),
                    "conditional_analytic_outer_tail_bound": analytic,
                }
            )
        ledger = None
        if outer_reports:
            last_outer = outer_reports[-1]
            ledger = end_to_end_error_ledger(
                matsubara_payload=matsubara_result.as_dict(),
                outer_payload=mapping(last_outer.get("outer_result")),
                microscopic_impact=mapping(
                    last_outer.get("microscopic_impact")
                ),
                analytic_tail=mapping(
                    last_outer.get("conditional_analytic_outer_tail_bound")
                ),
            )

    source_hash_after = sha256(point_cache_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source cache changed during candidate audit replay")
    return {
        "schema": _SCHEMA,
        "run_dir": str(run_dir),
        "candidate_logdet_rtol": float(candidate_logdet_rtol),
        "radial_budget_fraction_override": radial_fraction,
        "N_candidates_override": (
            None
            if N_candidates_override is None
            else [int(value) for value in N_candidates_override]
        ),
        "source_cache_sha256": source_hash_before,
        "source_cache_unchanged": True,
        "new_microscopic_work_forbidden": True,
        "replay_wall_seconds": replay_wall,
        "matsubara_result": matsubara_result.as_dict(),
        "outer_runs": outer_reports,
        "error_ledger": ledger,
        "point_evidence_count": len(point_evidence),
    }


def budget_fraction_sensitivity(
    outer_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-score stored radial/angular errors under common budget splits.

    This is a counterfactual screen only: changing the split can alter the adaptive path,
    so a selected split still requires one exact cache-only controller replay.
    """

    fractions = (0.75, 0.80, 0.85)
    cutoff_records = sequence(outer_payload.get("cutoff_records"))
    records = []
    for raw_cutoff in cutoff_records:
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
                        payload.get(
                            "combined_comparison_radial_errors_J_m2"
                        )
                    )
                ]
                angular = [
                    float(v)
                    for v in sequence(
                        payload.get("estimated_angular_errors_J_m2")
                    )
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
                    radial[i] + angular[i] <= total_tol[i]
                    for i in range(count)
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
        mapping(
            mapping(report).get("production_replay_equivalence")
        ).get("equivalent")
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
        mapping(replay).get("error_ledger") is not None
        for replay in closure_replays
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
    "budget_fraction_sensitivity",
    "candidate_cache_replay_audit",
    "conditional_analytic_outer_tail_bound",
    "end_to_end_error_ledger",
    "holdout_plan",
    "production_equivalence_audit",
    "weighted_microscopic_impact",
]
