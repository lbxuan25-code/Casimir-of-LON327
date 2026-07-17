from __future__ import annotations

import math

import numpy as np
import pytest

from lno327.casimir.adaptive_joint_q import (
    AdaptiveJointCasimirConfig,
    run_adaptive_joint_casimir,
)
from lno327.casimir.adaptive_outer_q import (
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialCasimirResult,
)
from lno327.casimir.certified_point_provider import CertifiedPointBatch
from lno327.casimir.fixed_chain import FixedCasimirConfig
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


class _Provider:
    def __init__(self):
        self.cached_point_count = 0
        self.unique_q_count = 0
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0


def _result(
    config: AdaptiveRadialCasimirConfig,
    contributions,
    errors,
    *,
    status="adaptive_finite_partial",
    reason="radial_tolerance_met",
    certified=True,
):
    values = np.asarray(contributions, dtype=float)
    radial = np.asarray(errors, dtype=float)
    pairing_results = {
        pairing: {
            "status": "integrated" if status == "adaptive_finite_partial" else "radial_unresolved",
            "partial_free_energy_J_m2": float(np.sum(values)),
            "contributions_J_m2": values.tolist(),
            "outer_q_integrals_m_inv2": values.tolist(),
            "estimated_radial_errors_J_m2": radial.tolist(),
            "radial_tolerances_J_m2": [1.0] * len(values),
            "radial_channel_passed": [status == "adaptive_finite_partial"] * len(values),
            "matsubara_indices": list(config.point_config.matsubara_indices),
            "prime_weights": [1.0] * len(values),
        }
        for pairing in config.point_config.pairings
    }
    return AdaptiveRadialCasimirResult(
        status=status,
        config=config,
        radial_converged=status == "adaptive_finite_partial",
        all_microscopic_nodes_certified=certified,
        pairing_results=pairing_results,
        panel_records=(),
        refinement_rounds=config.max_refinement_rounds,
        unique_microscopic_q_node_count=0,
        unresolved_points=() if certified else ({"reason": reason},),
        termination_reason=reason,
        provider_statistics={},
    )


class _ScriptedRunner:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def __call__(self, config, *, provider=None):
        self.calls.append(
            (
                config.angular_order,
                config.angular_offset_fraction,
                config.max_refinement_rounds,
            )
        )
        contributions, errors, status, reason, certified = self.function(config)
        return _result(
            config,
            contributions,
            errors,
            status=status,
            reason=reason,
            certified=certified,
        )


def _base_config(*, matsubara_indices=(0,), max_rounds=3):
    point = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=matsubara_indices,
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
    )
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 2.0),
        radial_order=1,
        angular_order=4,
        radial_rtol=1e-2,
        radial_atol_J_m2=1e-12,
        max_refinement_rounds=max_rounds,
        max_panel_depth=4,
    )
    return AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(4, 8, 16),
        outer_rtol=0.0,
        outer_atol_J_m2=1.0,
        radial_budget_fraction=0.5,
        angular_budget_fraction=0.5,
        offset_rtol=0.0,
        offset_atol_J_m2=1.0,
        initial_radial_round_cap=0,
        radial_round_step=1,
    )


def test_budget_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        AdaptiveJointCasimirConfig(
            radial_budget_fraction=0.7,
            angular_budget_fraction=0.4,
        )


def test_radial_dominated_error_refines_radial_before_accepting():
    config = _base_config()

    def script(radial_config):
        cap = radial_config.max_refinement_rounds
        error = 0.40 if cap == 0 else 0.10
        status = "unresolved" if cap == 0 else "adaptive_finite_partial"
        reason = "maximum_refinement_rounds_reached" if cap == 0 else "radial_tolerance_met"
        value = 1.0 if radial_config.angular_order == 4 else 1.1
        if radial_config.angular_offset_fraction == 0.0:
            value += 0.05
        return [value], [error], status, reason, True

    runner = _ScriptedRunner(script)
    result = run_adaptive_joint_casimir(
        config,
        provider=_Provider(),
        radial_runner=runner,
    )

    assert result.status == "adaptive_finite_partial"
    assert result.selected_radial_round_cap == 1
    assert result.direction_records[0]["selected_direction"] == "radial"
    assert result.direction_records[-1]["selected_direction"] == "accept"
    assert result.pairing_results["spm"]["joint_channel_passed"] == [True]


def test_angular_dominated_error_advances_angular_order():
    config = _base_config()

    def script(radial_config):
        values = {4: 1.0, 8: 2.0, 16: 2.1}
        value = values[radial_config.angular_order]
        if radial_config.angular_offset_fraction == 0.0:
            value += 0.05
        return [value], [0.05], "adaptive_finite_partial", "radial_tolerance_met", True

    runner = _ScriptedRunner(script)
    result = run_adaptive_joint_casimir(
        config,
        provider=_Provider(),
        radial_runner=runner,
    )

    assert result.status == "adaptive_finite_partial"
    assert result.selected_angular_order == 16
    assert result.direction_records[0]["selected_direction"] == "angular"
    assert result.direction_records[-1]["selected_direction"] == "accept"


def test_channelwise_budget_prevents_matsubara_cancellation():
    config = _base_config(matsubara_indices=(0, 1))
    config = AdaptiveJointCasimirConfig(
        radial_config=config.radial_config,
        angular_orders=(4, 8),
        outer_rtol=0.0,
        outer_atol_J_m2=1.0,
        radial_budget_fraction=0.5,
        angular_budget_fraction=0.5,
        offset_rtol=0.0,
        offset_atol_J_m2=1.0,
    )

    def script(radial_config):
        values = [1.0, -1.0] if radial_config.angular_order == 4 else [2.0, -2.0]
        return values, [0.01, 0.01], "adaptive_finite_partial", "radial_tolerance_met", True

    result = run_adaptive_joint_casimir(
        config,
        provider=_Provider(),
        radial_runner=_ScriptedRunner(script),
    )

    assert result.status == "unresolved"
    assert result.termination_reason == "joint_angular_order_ladder_exhausted"
    record = result.direction_records[0]
    assert sum(record["pairings"]["spm"]["previous_contributions_J_m2"]) == 0.0
    assert sum(record["pairings"]["spm"]["current_contributions_J_m2"]) == 0.0
    assert record["angular_budget_passed"] is False


def test_offset_failure_selects_angular_and_fails_at_maximum_order():
    config = AdaptiveJointCasimirConfig(
        radial_config=_base_config().radial_config,
        angular_orders=(4, 8),
        outer_rtol=0.0,
        outer_atol_J_m2=1.0,
        radial_budget_fraction=0.5,
        angular_budget_fraction=0.5,
        offset_rtol=0.0,
        offset_atol_J_m2=0.1,
    )

    def script(radial_config):
        value = 1.0 if radial_config.angular_order == 4 else 1.1
        if radial_config.angular_offset_fraction == 0.0:
            value = 3.0
        return [value], [0.01], "adaptive_finite_partial", "radial_tolerance_met", True

    result = run_adaptive_joint_casimir(
        config,
        provider=_Provider(),
        radial_runner=_ScriptedRunner(script),
    )

    assert result.status == "unresolved"
    assert result.termination_reason == "joint_offset_audit_failed_at_maximum_angular_order"
    assert result.direction_records[-1]["selected_direction"] == "angular"
    assert result.offset_audit_record["offset_audit_passed"] is False


def test_unresolved_microscopic_inner_run_fails_closed():
    config = _base_config()

    def script(radial_config):
        return [1.0], [1.0], "unresolved", "microscopic_point_unresolved", False

    result = run_adaptive_joint_casimir(
        config,
        provider=_Provider(),
        radial_runner=_ScriptedRunner(script),
    )

    assert result.status == "unresolved"
    assert result.termination_reason.startswith("radial_run_unresolved")
    assert result.all_microscopic_nodes_certified is False


class _AnalyticProvider:
    def __init__(self, config):
        self.config = config
        self._points = {}
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0

    @property
    def unique_q_count(self):
        return len(self._points)

    @property
    def cached_point_count(self):
        return len(self._points) * len(self.config.pairings) * len(
            self.config.matsubara_indices
        )

    def evaluate(self, q_model):
        array = np.asarray(q_model, dtype=float)
        keys = [(float(q[0]).hex(), float(q[1]).hex()) for q in array]
        unique = dict(zip(keys, array, strict=False))
        new = [key for key in unique if key not in self._points]
        self.requested_q_evaluations += len(unique)
        self.new_q_evaluations += len(new)
        self.cache_hit_q_evaluations += len(unique) - len(new)
        self.certification_batches += int(bool(new))
        self._points.update(unique)
        return CertifiedPointBatch(
            point_results=(),
            unresolved_points=(),
            requested_q_count=len(unique),
            new_q_count=len(new),
            cache_hit_q_count=len(unique) - len(new),
            certification_batches=int(bool(new)),
        )

    def primary_logdet(self, pairing, n, q):
        material = LNO327_THIN_FILM_SLAO_IN_PLANE
        qx = float(q[0]) / material.lattice_a_x_m
        qy = float(q[1]) / material.lattice_a_y_m
        u = 2.0 * self.config.separation_m * math.hypot(qx, qy)
        phi = math.atan2(qy, qx)
        return math.exp(-u) * (1.0 + 0.2 * math.cos(4.0 * phi))


def test_joint_controller_composes_with_real_radial_controller():
    point = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=(0,),
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
        separation_nm=1e9,
    )
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 2.0),
        radial_order=1,
        angular_order=4,
        radial_rtol=1e-2,
        radial_atol_J_m2=1e-30,
        max_refinement_rounds=5,
        refine_panels_per_round=2,
    )
    config = AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(4, 8, 16),
        outer_rtol=5e-2,
        outer_atol_J_m2=1e-30,
        offset_rtol=5e-2,
        offset_atol_J_m2=1e-30,
        max_joint_iterations=16,
    )
    provider = _AnalyticProvider(point)

    result = run_adaptive_joint_casimir(config, provider=provider)

    assert result.status == "adaptive_finite_partial"
    assert result.joint_converged
    assert result.selected_angular_order >= 8
    assert result.offset_audit_passed
    assert provider.unique_q_count > 0
    assert any(
        row["selected_direction"] in {"radial", "angular"}
        for row in result.direction_records[:-1]
    )
