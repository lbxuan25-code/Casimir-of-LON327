from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np

from lno327.casimir.adaptive_outer_tail import AdaptiveOuterTailCasimirResult
from lno327.casimir.certified_matsubara import run_certified_matsubara_casimir
from lno327.casimir.production import build_full_casimir_config
from lno327.casimir.transverse_policy import (
    CONDITIONAL_AUDIT_SHIFT,
    FORMAL_TRANSVERSE_SHIFTS,
    LADDER_EARLY_STOP_CONTRACT,
)
from scripts.full_casimir import scan
from scripts.full_casimir.shift_audit import replay_point_two_shift


def _history_row(N: int, values: tuple[float, float, float]) -> dict:
    shifts = (
        (0.5, 0.5),
        (0.25, 0.75),
        (0.75, 0.25),
    )
    return {
        "N": N,
        "shifts": {
            f"shift_{index}:{shift}": {
                "two_plate_logdet": value,
                "hard_physical_passed": True,
            }
            for index, (shift, value) in enumerate(zip(shifts, values, strict=True))
        },
    }


def test_production_defaults_use_two_formal_shifts_only() -> None:
    config = build_full_casimir_config(pairings=("spm",))
    point = config.outer_tail_config.joint_config.radial_config.point_config
    assert point.shifts == FORMAL_TRANSVERSE_SHIFTS
    assert CONDITIONAL_AUDIT_SHIFT not in point.shifts


def test_plan_identity_contains_shift_and_early_stop_contracts() -> None:
    args = scan._parser().parse_args(
        ["plan", "--pairings", "spm", "--distances-nm", "20", "--angles-deg", "0"]
    )
    plan = scan.build_scan_plan(
        args,
        code_identity={"git_commit": "a" * 40, "tracked_worktree_clean": True},
    )
    budget = plan["scientific_policy"]["error_budget"]
    transverse = budget["transverse_certification"]
    assert transverse["formal_shifts"] == [list(value) for value in FORMAL_TRANSVERSE_SHIFTS]
    assert transverse["conditional_audit_shift"] == list(CONDITIONAL_AUDIT_SHIFT)
    assert transverse["single_shift_production_acceptance_forbidden"] is True
    early = budget["adaptive_ladder_early_stop"]
    assert early["contract"] == LADDER_EARLY_STOP_CONTRACT
    assert early["higher_levels_after_certificate_forbidden"] is True


def test_two_shift_replay_matches_stable_historical_three_shift_point() -> None:
    point = {
        "pairing": "dwave",
        "q_label": "q",
        "n": 1,
        "sweet_spot": {"status": "established"},
        "history": [
            _history_row(128, (1.0, 1.0002, 0.9999)),
            _history_row(192, (1.0001, 1.0002, 1.0000)),
            _history_row(256, (1.0001, 1.0002, 1.0000)),
        ],
    }
    report = replay_point_two_shift(
        point,
        rtol=2e-3,
        atol=1e-6,
        required_consecutive_passes=2,
    )
    assert report["two_shift_established"] is True
    assert report["decision_matches"] is True
    assert report["evaluated_N"] == [128, 192, 256]


def test_conditional_audit_shift_cannot_rescue_or_hide_policy_mismatch() -> None:
    point = {
        "pairing": "dwave",
        "q_label": "q",
        "n": 1,
        "sweet_spot": {"status": "not_established"},
        "history": [
            _history_row(128, (1.0, 1.0001, 2.0)),
            _history_row(192, (1.0, 1.0001, 2.0)),
            _history_row(256, (1.0, 1.0001, 2.0)),
        ],
    }
    report = replay_point_two_shift(
        point,
        rtol=2e-3,
        atol=1e-6,
        required_consecutive_passes=2,
    )
    assert report["two_shift_established"] is True
    assert report["decision_matches"] is False


class _Provider:
    def __init__(self) -> None:
        self.configs: list[tuple[int, ...]] = []
        self.cached_point_count = 0
        self.unique_q_count = 0
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0
        self.requested_point_evaluations = 0
        self.new_point_evaluations = 0
        self.cache_hit_point_evaluations = 0
        self._entries = {}

    def reconfigure(self, config) -> None:
        self.configs.append(tuple(config.matsubara_indices))
        self.cached_point_count = len(config.matsubara_indices)


class _ZeroOuterRunner:
    def __init__(self) -> None:
        self.cutoffs: list[int] = []

    def __call__(self, config, *, provider=None):
        point = config.joint_config.radial_config.point_config
        indices = tuple(point.matsubara_indices)
        self.cutoffs.append(indices[-1])
        values = np.zeros(len(indices), dtype=float)
        errors = np.zeros(len(indices), dtype=float)
        payload = {
            pairing: {
                "status": "integrated_with_outer_tail_bound",
                "partial_free_energy_J_m2": 0.0,
                "contributions_J_m2": values.tolist(),
                "estimated_total_outer_errors_J_m2": errors.tolist(),
                "matsubara_indices": list(indices),
                "outer_tail_certificate_path": "analytic_passive_vacuum",
            }
            for pairing in point.pairings
        }
        return AdaptiveOuterTailCasimirResult(
            status="adaptive_finite_partial",
            config=config,
            cutoff_converged=True,
            outer_tail_estimated_flag=True,
            all_finite_domain_runs_converged=True,
            all_microscopic_nodes_certified=True,
            selected_u_max=config.cutoff_u_values[0],
            pairing_results=MappingProxyType(payload),
            cutoff_records=(),
            shell_records=(),
            termination_reason="analytic_passive_vacuum_tail_bound_met",
            provider_statistics=MappingProxyType({"cached_point_count": len(indices)}),
        )


def test_matsubara_ladder_stops_before_unused_higher_block() -> None:
    config = build_full_casimir_config(
        pairings=("spm",),
        N_candidates=(128, 192, 256),
        matsubara_cutoff_values=(1, 3, 7, 15, 31, 63),
        matsubara_tail_start_n=4,
        matsubara_tail_window_terms=3,
        total_free_energy_rtol=0.0,
        total_free_energy_atol_J_m2=1e-6,
    )
    provider = _Provider()
    runner = _ZeroOuterRunner()
    result = run_certified_matsubara_casimir(
        config,
        provider=provider,
        outer_tail_runner=runner,
    )
    assert result.matsubara_converged
    assert result.selected_matsubara_cutoff == 31
    assert runner.cutoffs == [1, 3, 7, 15, 31]
    assert 63 not in runner.cutoffs


def test_transverse_N_ladder_removes_resolved_point_before_higher_levels(monkeypatch, tmp_path) -> None:
    from lno327.casimir import fixed_transverse_point_certification as cert

    args = SimpleNamespace(
        q_points=({"label": "q", "q_lab": np.asarray([0.1, 0.2])},),
        pairings=("spm",),
        matsubara_indices=(0,),
        N_candidates=(128, 192, 256, 384),
        shifts=FORMAL_TRANSVERSE_SHIFTS,
        required_consecutive_passes=2,
        memory_safety_factor=1.5,
        fallback_context_bytes_per_point=16384.0,
        parallel_mode="q",
        workers=2,
        memory_budget_gb=1.0,
        max_context_workers=1,
        logdet_rtol=2e-3,
        logdet_atol=1e-6,
        output=tmp_path / "result.json",
    )
    monkeypatch.setattr(cert, "_parse_args", lambda _argv: args)
    built_levels: list[int] = []

    def build_jobs(*, n_grid, args, active, q_by_label):
        built_levels.append(int(n_grid))
        return [SimpleNamespace(n_grid=int(n_grid))] if any(active.values()) else []

    monkeypatch.setattr(cert._engine, "_build_context_jobs", build_jobs)
    monkeypatch.setattr(cert._engine, "_max_q_tasks_per_context", lambda jobs: 1)
    monkeypatch.setattr(cert._engine, "_total_flat_tasks", lambda jobs: 1)
    monkeypatch.setattr(cert, "estimate_context_bytes", lambda **kwargs: 1)
    plan = SimpleNamespace(
        as_dict=lambda: {
            "strategy": "serial",
            "total_worker_budget": 1,
            "context_workers": 1,
            "q_workers": 1,
            "flat_workers": 1,
            "total_flat_tasks": 1,
            "wave_count": 1,
            "reason": "test",
        }
    )
    monkeypatch.setattr(cert, "choose_cpu_parallel_plan", lambda **kwargs: plan)

    def execute_level(*, jobs, plan):
        N = jobs[0].n_grid
        records = {}
        for index, shift in enumerate(FORMAL_TRANSVERSE_SHIFTS):
            records[("spm", index)] = {
                "shift": shift,
                "point_count": 1,
                "material_cache_array_bytes": 1,
                "points": {
                    "q": {
                        "0": {
                            "two_plate_logdet": 1.0 + index * 1e-4,
                            "hard_physical_passed": True,
                        }
                    }
                },
            }
        return records, []

    monkeypatch.setattr(cert._engine, "_execute_level", execute_level)
    monkeypatch.setattr(cert._engine, "_atomic_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cert,
        "_build_payload",
        lambda **kwargs: {
            "schema": "transverse-point-sweet-spot-v4",
            "all_requested_sweet_spots_established": not any(
                row["sweet_spot"]["status"] != "established"
                for row in kwargs["result_records"].values()
            ),
            "convergence_policy": {},
            "point_results": list(kwargs["result_records"].values()),
        },
    )
    cert.main([])
    assert built_levels == [128, 192, 256]
    assert 384 not in built_levels
