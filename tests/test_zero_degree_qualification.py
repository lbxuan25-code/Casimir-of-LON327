from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lno327.casimir.adaptive_outer_tail import AdaptiveOuterTailCasimirResult
from lno327.casimir.production import build_full_casimir_config
from lno327.casimir.qualification import (
    _analytic_upgrade,
    cached_power_metric_contraction_certificate,
    passive_vacuum_channel_bounds_J_m2,
)
from scripts.full_casimir.policy_audit import compare_policy_snapshots
from scripts.full_casimir.qualification import (
    LOGDET_ATOL,
    LOGDET_RTOL,
    N_CANDIDATES,
    RADIAL_BUDGET_FRACTION,
    REQUIRED_CONSECUTIVE_PASSES,
    _reassess_complete_history,
)


def _state(value: float, *, reflection_norm: float = 0.5) -> dict:
    plate = {
        "sheet_validation_passed": True,
        "reflection_constructed": True,
        "reflection_norm": reflection_norm,
    }
    return {
        "two_plate_logdet": value,
        "hard_physical_passed": True,
        "plate_1": dict(plate),
        "plate_2": dict(plate),
    }


def _point_history() -> dict:
    return {
        "sweet_spot": {"status": "not_established"},
        "history": [
            {
                "N": 128,
                "shifts": {
                    "a": _state(-1.0000),
                    "b": _state(-1.0005),
                },
            },
            {
                "N": 192,
                "shifts": {
                    "a": _state(-1.0010),
                    "b": _state(-1.0014),
                },
            },
            {
                "N": 256,
                "shifts": {
                    "a": _state(-1.0015),
                    "b": _state(-1.0018),
                },
            },
            {
                "N": 384,
                "shifts": {
                    "a": _state(-1.0017),
                    "b": _state(-1.0019),
                },
            },
        ],
    }


def test_full_config_is_pairing_blind_under_frozen_budget() -> None:
    kwargs = {
        "N_candidates": N_CANDIDATES,
        "logdet_rtol": LOGDET_RTOL,
        "logdet_atol": LOGDET_ATOL,
        "required_consecutive_passes": REQUIRED_CONSECUTIVE_PASSES,
        "radial_budget_fraction": RADIAL_BUDGET_FRACTION,
    }
    spm = build_full_casimir_config(pairings=("spm",), **kwargs)
    dwave = build_full_casimir_config(pairings=("dwave",), **kwargs)

    assert spm.outer_tail_config.joint_config.radial_budget_fraction == pytest.approx(0.8)
    assert dwave.outer_tail_config.joint_config.radial_budget_fraction == pytest.approx(0.8)
    parity = compare_policy_snapshots(
        (("spm", spm.as_dict()), ("dwave", dwave.as_dict()))
    )
    assert parity["pairing_blind_scientific_policy"] is True


def test_history_projection_preserves_all_levels_and_reassesses_candidate() -> None:
    source = _point_history()
    projected = _reassess_complete_history(
        source,
        rtol=LOGDET_RTOL,
        atol=LOGDET_ATOL,
        required_consecutive_passes=REQUIRED_CONSECUTIVE_PASSES,
    )

    assert len(projected["history"]) == len(source["history"])
    assert projected["sweet_spot"]["status"] == "established"
    assert projected["sweet_spot"]["audit_N"] in {256, 384}
    assert source["sweet_spot"]["status"] == "not_established"


def test_passive_vacuum_bound_has_zero_mode_half_weight_and_decays() -> None:
    at_24 = passive_vacuum_channel_bounds_J_m2(
        u0=24.0,
        separation_nm=20.0,
        temperature_K=10.0,
        matsubara_indices=(0, 1),
    )
    at_60 = passive_vacuum_channel_bounds_J_m2(
        u0=60.0,
        separation_nm=20.0,
        temperature_K=10.0,
        matsubara_indices=(0, 1),
    )

    assert at_24["matsubara_prime_weights"] == [0.5, 1.0]
    assert at_24["channel_bounds_J_m2"][0] == pytest.approx(
        0.5 * at_24["channel_bounds_J_m2"][1]
    )
    assert at_60["channel_bounds_J_m2"][1] < at_24["channel_bounds_J_m2"][1]


def test_cached_contraction_certificate_checks_accepted_audit_state(tmp_path: Path) -> None:
    point = _point_history()
    point["sweet_spot"] = {
        "status": "established",
        "working_N": 256,
        "audit_N": 384,
    }
    cache = tmp_path / "points.json"
    cache.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "pairing": "spm",
                        "n": 0,
                        "qx_hex": float(0.1).hex(),
                        "qy_hex": float(0.2).hex(),
                        "point_result": point,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    certificate = cached_power_metric_contraction_certificate(
        SimpleNamespace(cache_path=cache)
    )
    assert certificate["all_points_certified"] is True

    point["history"][-1]["shifts"]["a"]["plate_1"]["reflection_norm"] = 1.1
    cache.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "pairing": "spm",
                        "n": 0,
                        "qx_hex": float(0.1).hex(),
                        "qy_hex": float(0.2).hex(),
                        "point_result": point,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rejected = cached_power_metric_contraction_certificate(
        SimpleNamespace(cache_path=cache)
    )
    assert rejected["all_points_certified"] is False


def test_analytic_upgrade_requires_contraction_and_budget() -> None:
    full = build_full_casimir_config(
        pairings=("spm",),
        plate_angles_deg=(0.0, 0.0),
        cutoff_u_values=(24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0),
        total_free_energy_rtol=5e-3,
        total_free_energy_atol_J_m2=1e-12,
        radial_budget_fraction=0.8,
    )
    outer = full.outer_tail_config
    latest_pairing = {
        "matsubara_indices": [0, 1],
        "contributions_J_m2": [-1e-6, -1e-6],
    }
    unresolved = AdaptiveOuterTailCasimirResult(
        status="unresolved",
        config=outer,
        cutoff_converged=False,
        outer_tail_estimated_flag=False,
        all_finite_domain_runs_converged=True,
        all_microscopic_nodes_certified=True,
        selected_u_max=60.0,
        pairing_results={"spm": latest_pairing},
        cutoff_records=(
            {
                "u_max": 60.0,
                "pairing_results": {"spm": latest_pairing},
                "finite_domain_error_bounds_J_m2": {"spm": [1e-15, 1e-15]},
            },
        ),
        shell_records=(),
        termination_reason="outer_tail_decay_ratio_not_established",
        provider_statistics={},
    )
    rejected = _analytic_upgrade(
        outer,
        unresolved,
        contraction_certificate={"all_points_certified": False},
    )
    assert rejected.status == "unresolved"

    upgraded = _analytic_upgrade(
        outer,
        unresolved,
        contraction_certificate={
            "all_points_certified": True,
            "status": "certified",
        },
    )
    assert upgraded.cutoff_converged is True
    assert upgraded.termination_reason == "analytic_passive_vacuum_tail_bound_met"
    assert (
        upgraded.pairing_results["spm"]["outer_tail_certificate_path"]
        == "analytic_passive_vacuum"
    )
