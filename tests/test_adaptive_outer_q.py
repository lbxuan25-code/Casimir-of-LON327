from __future__ import annotations

import math

import numpy as np

from lno327.casimir.adaptive_outer_q import (
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialPanel,
    build_adaptive_outer_q_panel_grid,
    run_adaptive_radial_casimir,
)
from lno327.casimir.certified_point_provider import CertifiedPointBatch
from lno327.casimir.fixed_chain import FixedCasimirConfig
from lno327.constants import KB
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


class _AnalyticProvider:
    def __init__(self, config, function, *, unresolved=False):
        self.config = config
        self.function = function
        self.unresolved = unresolved
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
        self.certification_batches += bool(new)
        self._points.update(unique)
        unresolved = (
            ({"reason": "synthetic_unresolved"},) if self.unresolved else ()
        )
        return CertifiedPointBatch(
            point_results=(),
            unresolved_points=unresolved,
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
        return float(self.function(pairing, n, u))


def test_adaptive_panel_grid_preserves_exact_annulus_measure() -> None:
    panel = AdaptiveRadialPanel(2.0, 5.0)
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    grid = build_adaptive_outer_q_panel_grid(
        panel,
        separation_m=20e-9,
        lattice_a_x_m=material.lattice_a_x_m,
        lattice_a_y_m=material.lattice_a_y_m,
        radial_order=4,
        angular_order=8,
        angular_offset_fraction=0.5,
    )
    exact = (5.0**2 - 2.0**2) / (16.0 * np.pi * (20e-9) ** 2)
    np.testing.assert_allclose(
        np.sum(grid.measure_weights_m_inv2),
        exact,
        rtol=2e-15,
    )
    assert grid.node_count == 32
    assert not np.any(np.all(grid.q_model == 0.0, axis=1))


def test_constant_logdet_converges_without_refinement_and_applies_prime_weight() -> None:
    point_config = FixedCasimirConfig(
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
        separation_nm=1e9,
    )
    config = AdaptiveRadialCasimirConfig(
        point_config=point_config,
        initial_panel_edges=(0.0, 1.0, 2.0),
        radial_order=2,
        angular_order=4,
        radial_rtol=1e-12,
        radial_atol_J_m2=1e-20,
    )
    provider = _AnalyticProvider(
        point_config,
        lambda pairing, n, u: 2.0 if n == 0 else 3.0,
    )
    result = run_adaptive_radial_casimir(config, provider=provider)

    assert result.status == "adaptive_finite_partial"
    assert result.radial_converged
    assert result.refinement_rounds == 0
    measure = 2.0**2 / (16.0 * np.pi * 1.0**2)
    expected = KB * point_config.temperature_K * measure * np.asarray([1.0, 3.0])
    actual = result.pairing_results["spm"]
    np.testing.assert_allclose(actual["contributions_J_m2"], expected, rtol=1e-14)
    np.testing.assert_allclose(
        actual["partial_free_energy_J_m2"],
        float(np.sum(expected)),
        rtol=1e-14,
    )
    assert result.production_casimir_allowed is False
    assert result.outer_tail_estimated is False
    assert result.matsubara_tail_estimated is False


def test_nonpolynomial_radial_integrand_triggers_refinement_then_converges() -> None:
    point_config = FixedCasimirConfig(
        matsubara_indices=(0,),
        u_max_values=(1.0, 4.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
        separation_nm=1e9,
    )
    config = AdaptiveRadialCasimirConfig(
        point_config=point_config,
        initial_panel_edges=(0.0, 4.0),
        radial_order=1,
        angular_order=4,
        radial_rtol=2e-3,
        radial_atol_J_m2=1e-30,
        max_refinement_rounds=8,
        refine_panels_per_round=2,
    )
    provider = _AnalyticProvider(
        point_config,
        lambda pairing, n, u: math.exp(-u),
    )
    result = run_adaptive_radial_casimir(config, provider=provider)

    assert result.status == "adaptive_finite_partial"
    assert result.radial_converged
    assert result.refinement_rounds > 0
    assert len(result.panel_records) > 1
    channel = result.pairing_results["spm"]
    assert channel["estimated_radial_errors_J_m2"][0] <= channel[
        "radial_tolerances_J_m2"
    ][0]


def test_unresolved_microscopic_point_fails_closed() -> None:
    point_config = FixedCasimirConfig(
        matsubara_indices=(0,),
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
    )
    config = AdaptiveRadialCasimirConfig(
        point_config=point_config,
        initial_panel_edges=(0.0, 2.0),
        radial_order=1,
        angular_order=2,
    )
    provider = _AnalyticProvider(
        point_config,
        lambda pairing, n, u: 1.0,
        unresolved=True,
    )
    result = run_adaptive_radial_casimir(config, provider=provider)

    assert result.status == "unresolved"
    assert not result.radial_converged
    assert result.termination_reason == "microscopic_point_unresolved"
    assert result.unresolved_points
    assert result.production_casimir_allowed is False
