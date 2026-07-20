from __future__ import annotations

from scripts.full_casimir.budget_audit import (
    audit_evidence_gaps,
    candidate_policy_screen,
    tail_resolution_audit,
)
from scripts.full_casimir.point_diagnostics import (
    replay_point_policy,
    tolerance_replay_audit,
)
from scripts.full_casimir.policy_audit import compare_policy_snapshots


def _point_result() -> dict:
    labels = ("s0", "s1", "s2")
    levels = [
        (128, (-1.0, -1.0000001, -0.9999999)),
        (192, (-1.0018, -1.0018001, -1.0017999)),
        (256, (-1.0036, -1.0036001, -1.0035999)),
    ]
    return {
        "sweet_spot": {"status": "not_established"},
        "history": [
            {
                "N": N,
                "shifts": {
                    label: {
                        "two_plate_logdet": value,
                        "hard_physical_passed": True,
                    }
                    for label, value in zip(labels, values, strict=True)
                },
            }
            for N, values in levels
        ],
    }


def _cache() -> dict:
    return {
        "schema": "cache",
        "point_policy": {
            "logdet_rtol": 0.0015,
            "logdet_atol": 0.0,
            "required_consecutive_passes": 2,
        },
        "entries": [
            {
                "pairing": "dwave",
                "n": 1,
                "qx_hex": float(0.1).hex(),
                "qy_hex": float(0.2).hex(),
                "point_result": _point_result(),
            }
        ],
    }


def test_policy_replay_keeps_hard_gates_and_replays_full_history() -> None:
    strict = replay_point_policy(
        _point_result(),
        logdet_rtol=0.0015,
        logdet_atol=0.0,
        required_consecutive_passes=2,
    )
    moderate = replay_point_policy(
        _point_result(),
        logdet_rtol=0.002,
        logdet_atol=0.0,
        required_consecutive_passes=2,
    )

    assert strict["status"] == "not_established"
    assert moderate["status"] == "established"
    assert moderate["establishment_mode"] == "strict_consecutive_adjacent"
    assert moderate["audit_N"] == 256
    assert moderate["policy"]["hard_physical_gates_unchanged"] is True


def test_tolerance_replay_reports_proxy_as_non_wall_time_evidence() -> None:
    result = tolerance_replay_audit(
        _cache(), candidate_logdet_rtols=(0.0015, 0.002)
    )

    strict, moderate = result["candidate_policies"]
    assert strict["unresolved_count"] == 1
    assert moderate["unresolved_count"] == 0
    assert result["scientific_limitations"]["production_policy_change_authorized"] is False
    assert "not reported as wall time" in result["work_proxy_note"]


def _config(radial_fraction: float) -> dict:
    return {
        "matsubara_cutoff_values": [1, 3],
        "total_free_energy_rtol": 0.005,
        "total_free_energy_atol_J_m2": 1e-12,
        "finite_matsubara_budget_fraction": 0.7,
        "matsubara_tail_budget_fraction": 0.3,
        "tail_start_n": 8,
        "tail_window_terms": 4,
        "tail_ratio_max": 0.8,
        "max_total_microscopic_point_entries": 1000,
        "certifier_q_batch_size": 32,
        "outer_tail_config": {
            "cutoff_u_values": [6.0, 10.0, 14.0],
            "total_outer_rtol": 0.005,
            "total_outer_atol_J_m2": 1e-12,
            "finite_domain_budget_fraction": 0.7,
            "tail_budget_fraction": 0.3,
            "tail_start_u": 6.0,
            "tail_window_shells": 2,
            "tail_ratio_max": 0.8,
            "joint_config": {
                "radial_budget_fraction": radial_fraction,
                "angular_budget_fraction": 1.0 - radial_fraction,
                "radial_config": {
                    "point_config": {
                        "pairings": ["spm"],
                        "N_candidates": [128, 192],
                        "required_consecutive_passes": 2,
                        "logdet_rtol": 0.0015,
                        "logdet_atol": 1e-6,
                        "workers": 4,
                    }
                },
            },
        },
    }


def test_policy_parity_reports_pairing_dependent_budget_as_scientific_difference() -> None:
    result = compare_policy_snapshots(
        (("spm", _config(0.85)), ("dwave", _config(0.75)))
    )

    assert result["pairing_blind_scientific_policy"] is False
    differences = result["comparisons"][0]["differences"]
    assert any(
        row["path"] == "$.joint_controller.radial_budget_fraction"
        for row in differences
    )


def test_tail_resolution_separates_central_signal_from_quadrature_floor() -> None:
    replay = {
        "outer_tail_runs": [
            {
                "matsubara_cutoff": 1,
                "result": {
                    "config": {
                        "tail_start_u": 24.0,
                        "tail_window_shells": 3,
                        "tail_ratio_max": 0.8,
                    },
                    "shell_records": [
                        {
                            "left_u": left,
                            "right_u": left + 6.0,
                            "pairings": {
                                "spm": {
                                    "matsubara_indices": [0],
                                    "shell_contributions_J_m2": [signal],
                                    "shell_quadrature_error_bounds_J_m2": [1e-11],
                                    "shell_envelope_amplitudes_J_m2": [abs(signal) + 1e-11],
                                }
                            },
                        }
                        for left, signal in (
                            (24.0, 1e-12),
                            (30.0, 1e-13),
                            (36.0, 1e-14),
                        )
                    ],
                },
            }
        ]
    }

    result = tail_resolution_audit(replay)
    channel = result["outer_tail_runs"][0]["pairings"]["spm"]["channels"][0]
    assert channel["classification"] == "below_finite_domain_resolution"
    assert channel["production_tail_bound_established"] is False
    assert result["acceptance_effect"] == "diagnostic_only"


def test_evidence_ledger_refuses_production_change_without_weight_and_holdout() -> None:
    run_report = {
        "run_dir": "case",
        "point_cache": {"unresolved_count": 0},
        "tolerance_replay": {
            "candidate_policies": [
                {
                    "logdet_rtol": 0.002,
                    "established_count": 1,
                    "unresolved_count": 0,
                    "hard_physical_failure_count": 0,
                }
            ]
        },
    }
    parity = {
        "status": "analyzed",
        "pairing_blind_scientific_policy": True,
    }

    screen = candidate_policy_screen([run_report])
    ledger = audit_evidence_gaps(run_reports=[run_report], policy_parity=parity)

    assert screen["candidates"][0]["replay_screen_passed"] is True
    assert screen["candidates"][0]["production_ready"] is False
    assert ledger["production_policy_change_authorized"] is False
    assert "quadrature_weighted_microscopic_error_bound" in ledger["missing_evidence"]
