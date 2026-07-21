from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np

from lno327.casimir.adaptive_joint_q import AdaptiveJointCasimirResult
from lno327.casimir.certified_tail import (
    passive_vacuum_channel_bounds_J_m2,
    run_certified_outer_tail_casimir,
)
from lno327.casimir.production import build_full_casimir_config


class _ActivePositiveProvider:
    def __init__(self) -> None:
        self.cached_point_count = 1
        self.unique_q_count = 1
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0
        self.requested_point_evaluations = 0
        self.new_point_evaluations = 0
        self.cache_hit_point_evaluations = 0
        plate = {
            "sheet_validation_passed": True,
            "reflection_constructed": True,
        }
        point = {
            "pairing": "spm",
            "n": 1,
            "q_label": "q-positive",
            "sweet_spot": {"status": "established", "audit_N": 256},
            "history": [
                {
                    "N": 256,
                    "shifts": {
                        "primary": {
                            "hard_physical_passed": True,
                            "plate_1": dict(plate),
                            "plate_2": dict(plate),
                        }
                    },
                }
            ],
        }
        self._entries = {"spm|1|x|y": point}


class _FiniteJointRunner:
    def __init__(self) -> None:
        self.cutoffs: list[float] = []

    def __call__(self, config, *, provider=None):
        u_max = float(config.radial_config.initial_panel_edges[-1])
        self.cutoffs.append(u_max)
        values = np.asarray([-1.0e-8], dtype=float)
        payload = {
            "spm": {
                "status": "integrated",
                "partial_free_energy_J_m2": float(np.sum(values)),
                "contributions_J_m2": values.tolist(),
                "estimated_joint_errors_J_m2": [1.0e-12],
                "estimated_offset_errors_J_m2": [1.0e-12],
                "matsubara_indices": [1],
            }
        }
        return AdaptiveJointCasimirResult(
            status="adaptive_finite_partial",
            config=config,
            joint_converged=True,
            radial_budget_passed=True,
            angular_budget_passed=True,
            offset_audit_passed=True,
            all_microscopic_nodes_certified=True,
            selected_angular_order=config.angular_orders[-1],
            selected_radial_round_cap=config.initial_radial_round_cap,
            pairing_results=MappingProxyType(payload),
            direction_records=(),
            radial_run_records=(),
            offset_audit_record=None,
            termination_reason="joint_radial_angular_budget_and_offset_tolerances_met",
            provider_statistics=MappingProxyType(
                {
                    "cached_point_count": provider.cached_point_count,
                    "unique_q_count": provider.unique_q_count,
                }
            ),
        )


def _positive_frequency_outer_config():
    full = build_full_casimir_config(
        pairings=("spm",),
        total_free_energy_rtol=0.0,
        total_free_energy_atol_J_m2=1.0e-6,
    )
    outer = full.outer_tail_config
    radial = outer.joint_config.radial_config
    point = replace(radial.point_config, matsubara_indices=(1,))
    radial = replace(radial, point_config=point)
    joint = replace(outer.joint_config, radial_config=radial)
    return replace(outer, joint_config=joint)


def test_passive_vacuum_bound_decreases_with_outer_cutoff() -> None:
    common = {
        "separation_nm": 20.0,
        "temperature_K": 10.0,
        "matsubara_indices": (1,),
    }
    at_6 = passive_vacuum_channel_bounds_J_m2(u0=6.0, **common)
    at_18 = passive_vacuum_channel_bounds_J_m2(u0=18.0, **common)
    assert at_18["channel_bounds_J_m2"][0] < at_6["channel_bounds_J_m2"][0]


def test_common_outer_controller_attempts_analytic_certificate_at_first_cutoff() -> None:
    runner = _FiniteJointRunner()
    result = run_certified_outer_tail_casimir(
        _positive_frequency_outer_config(),
        provider=_ActivePositiveProvider(),
        joint_runner=runner,
    )
    assert result.cutoff_converged
    assert result.outer_tail_estimated
    assert result.selected_u_max == 6.0
    assert runner.cutoffs == [6.0]
    assert result.termination_reason == "analytic_passive_vacuum_tail_bound_met"
    channel = result.pairing_results["spm"]
    assert channel["outer_tail_certificate_path"] == "analytic_passive_vacuum"
    certificates = result.cutoff_records[0]["outer_tail_certificates"]
    assert certificates["analytic"]["certificate_passed"] is True
    assert certificates["analytic"]["premise"]["all_points_certified"] is True
