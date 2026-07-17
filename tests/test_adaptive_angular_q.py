from __future__ import annotations

import math

import numpy as np
import pytest

from lno327.casimir.adaptive_angular_q import (
    AdaptiveAngularCasimirConfig,
    run_adaptive_angular_casimir,
)
from lno327.casimir.adaptive_outer_q import (
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialCasimirResult,
)
from lno327.casimir.certified_point_provider import CertifiedPointBatch
from lno327.casimir.fixed_chain import FixedCasimirConfig
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


class _AnalyticAngularProvider:
    def __init__(self, config: FixedCasimirConfig, function):
        self.config = config
        self.function = function
        self._points: dict[tuple[str, str], np.ndarray] = {}
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0

    @property
    def unique_q_count(self) -> int:
        return len(self._points)

    @property
    def cached_point_count(self) -> int:
        return len(self._points) * len(self.config.pairings) * len(
            self.config.matsubara_indices
        )

    def evaluate(self, q_model: np.ndarray) -> CertifiedPointBatch:
        array = np.asarray(q_model, dtype=float)
        unique = {
            (float(q[0]).hex(), float(q[1]).hex()): q
            for q in array
        }
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

    def primary_logdet(self, pairing: str, n: int, q) -> float:
        material = LNO327_THIN_FILM_SLAO_IN_PLANE
        qx = float(q[0]) / material.lattice_a_x_m
        qy = float(q[1]) / material.lattice_a_y_m
        u = 2.0 * self.config.separation_m * math.hypot(qx, qy)
        phi = math.atan2(qy, qx)
        return float(self.function(pairing, n, u, phi))


class _StatsProvider:
    cached_point_count = 0
    unique_q_count = 0
    certification_batches = 0
    requested_q_evaluations = 0
    new_q_evaluations = 0
    cache_hit_q_evaluations = 0


def _point_config() -> FixedCasimirConfig:
    return FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=(0, 1),
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
        separation_nm=1e9,
    )


def _radial_template() -> AdaptiveRadialCasimirConfig:
    return AdaptiveRadialCasimirConfig(
        point_config=_point_config(),
        initial_panel_edges=(0.0, 2.0),
        radial_order=2,
        angular_order=2,
        angular_offset_fraction=0.5,
        radial_rtol=1e-12,
        radial_atol_J_m2=1e-20,
        max_refinement_rounds=2,
        max_panel_depth=2,
        refine_panels_per_round=1,
        max_microscopic_q_nodes=10_000,
    )


def _fake_radial_result(
    config: AdaptiveRadialCasimirConfig,
    contributions: tuple[float, float],
    *,
    status: str = "adaptive_finite_partial",
) -> AdaptiveRadialCasimirResult:
    converged = status == "adaptive_finite_partial"
    channel = {
        "status": "integrated" if converged else "radial_unresolved",
        "partial_free_energy_J_m2": float(sum(contributions)),
        "contributions_J_m2": list(contributions),
        "outer_q_integrals_m_inv2": [0.0, 0.0],
        "estimated_radial_errors_J_m2": [0.0, 0.0],
        "radial_tolerances_J_m2": [1.0, 1.0],
        "radial_channel_passed": [converged, converged],
        "matsubara_indices": [0, 1],
        "prime_weights": [0.5, 1.0],
    }
    return AdaptiveRadialCasimirResult(
        status=status,
        config=config,
        radial_converged=converged,
        all_microscopic_nodes_certified=converged,
        pairing_results={"spm": channel},
        panel_records=(),
        refinement_rounds=0,
        unique_microscopic_q_node_count=0,
        unresolved_points=() if converged else ({"reason": "synthetic"},),
        termination_reason=(
            "radial_tolerance_met" if converged else "microscopic_point_unresolved"
        ),
        provider_statistics={},
    )


def test_config_requires_strict_angular_doubling_and_distinct_offsets() -> None:
    with pytest.raises(ValueError, match="doubling"):
        AdaptiveAngularCasimirConfig(
            radial_config=_radial_template(),
            angular_orders=(4, 8, 12),
        )
    with pytest.raises(ValueError, match="offsets must differ"):
        AdaptiveAngularCasimirConfig(
            radial_config=_radial_template(),
            primary_offset_fraction=0.5,
            audit_offset_fraction=0.5,
        )


def test_actual_radial_controller_resolves_fourfold_angular_aliasing() -> None:
    radial = _radial_template()
    provider = _AnalyticAngularProvider(
        radial.point_config,
        lambda pairing, n, u, phi: 1.0 + 0.2 * math.cos(4.0 * phi),
    )
    config = AdaptiveAngularCasimirConfig(
        radial_config=radial,
        angular_orders=(4, 8, 16),
        primary_offset_fraction=0.5,
        audit_offset_fraction=0.0,
        angular_rtol=1e-12,
        angular_atol_J_m2=1e-30,
        offset_rtol=1e-12,
        offset_atol_J_m2=1e-30,
        required_consecutive_passes=1,
    )

    result = run_adaptive_angular_casimir(config, provider=provider)

    assert result.status == "adaptive_finite_partial"
    assert result.angular_converged
    assert result.offset_audit_passed
    assert result.selected_angular_order == 16
    assert len(result.angular_order_records) == 3
    assert result.termination_reason == "angular_order_and_offset_tolerances_met"
    assert result.pairing_results["spm"]["angular_channel_passed"] == [True, True]
    assert result.pairing_results["spm"]["offset_channel_passed"] == [True, True]
    assert result.production_casimir_allowed is False
    assert result.outer_tail_estimated is False
    assert result.matsubara_tail_estimated is False


def test_zero_offset_doubling_reuses_exact_angular_nodes() -> None:
    radial = _radial_template()
    provider = _AnalyticAngularProvider(
        radial.point_config,
        lambda pairing, n, u, phi: 1.0,
    )
    config = AdaptiveAngularCasimirConfig(
        radial_config=radial,
        angular_orders=(2, 4),
        primary_offset_fraction=0.0,
        audit_offset_fraction=0.5,
        angular_rtol=1e-12,
        angular_atol_J_m2=1e-20,
        offset_rtol=1e-12,
        offset_atol_J_m2=1e-20,
    )

    result = run_adaptive_angular_casimir(config, provider=provider)

    assert result.status == "adaptive_finite_partial"
    assert result.selected_angular_order == 4
    assert result.provider_statistics["cache_hit_q_evaluations"] > 0
    assert result.unique_microscopic_q_node_count == provider.unique_q_count


def test_channelwise_gate_rejects_cancellation_in_total_energy() -> None:
    radial = _radial_template()
    values = {
        (2, 0.5): (1.0, -1.0),
        (4, 0.5): (2.0, -2.0),
    }

    def runner(config, *, provider=None):
        return _fake_radial_result(
            config,
            values[(config.angular_order, config.angular_offset_fraction)],
        )

    config = AdaptiveAngularCasimirConfig(
        radial_config=radial,
        angular_orders=(2, 4),
        primary_offset_fraction=0.5,
        audit_offset_fraction=0.0,
        angular_rtol=0.1,
        angular_atol_J_m2=0.0,
        offset_rtol=0.1,
        offset_atol_J_m2=0.0,
    )
    result = run_adaptive_angular_casimir(
        config,
        provider=_StatsProvider(),
        radial_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.termination_reason == "angular_order_ladder_exhausted"
    assert result.angular_converged is False


def test_offset_audit_is_independent_fail_closed_gate() -> None:
    radial = _radial_template()
    values = {
        (2, 0.5): (1.0, 2.0),
        (4, 0.5): (1.001, 2.001),
        (4, 0.0): (1.5, 2.5),
    }

    def runner(config, *, provider=None):
        return _fake_radial_result(
            config,
            values[(config.angular_order, config.angular_offset_fraction)],
        )

    config = AdaptiveAngularCasimirConfig(
        radial_config=radial,
        angular_orders=(2, 4),
        primary_offset_fraction=0.5,
        audit_offset_fraction=0.0,
        angular_rtol=0.01,
        angular_atol_J_m2=0.0,
        offset_rtol=0.01,
        offset_atol_J_m2=0.0,
    )
    result = run_adaptive_angular_casimir(
        config,
        provider=_StatsProvider(),
        radial_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.angular_converged is True
    assert result.offset_audit_passed is False
    assert result.selected_angular_order == 4
    assert result.termination_reason == "angular_offset_audit_failed"
    assert result.pairing_results["spm"]["offset_channel_passed"] == [False, False]


def test_unresolved_radial_inner_run_stops_angular_ladder() -> None:
    radial = _radial_template()

    def runner(config, *, provider=None):
        return _fake_radial_result(
            config,
            (0.0, 0.0),
            status="unresolved",
        )

    result = run_adaptive_angular_casimir(
        AdaptiveAngularCasimirConfig(
            radial_config=radial,
            angular_orders=(2, 4),
        ),
        provider=_StatsProvider(),
        radial_runner=runner,
    )

    assert result.status == "unresolved"
    assert result.termination_reason == "radial_run_unresolved"
    assert result.all_radial_runs_converged is False
    assert result.all_microscopic_nodes_certified is False
