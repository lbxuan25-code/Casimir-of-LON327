"""Fail-closed adaptive radial outer-Q integration on a fixed finite domain.

This module adapts only the radial panel partition on ``u in [0, u_max]``.  The
full periodic angular rule, finite Matsubara index set, microscopic certification
policy, and physical gates remain fixed.  Neither the omitted outer-Q tail nor the
Matsubara tail is estimated here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from lno327.constants import KB
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE

from .certified_point_provider import (
    CertifiedOuterQProvider,
    CertifiedPointBatch,
    CertifiedPointCacheError,
)
from .fixed_chain import FixedCasimirConfig, FixedCasimirExecutionError
from .outer_quadrature import matsubara_prime_weights

_TWO_PI = 2.0 * np.pi


class _PointProvider(Protocol):
    def evaluate(self, q_model: np.ndarray) -> CertifiedPointBatch: ...

    def primary_logdet(self, pairing: str, n: int, q: Sequence[float]) -> float: ...


@dataclass(frozen=True, order=True)
class AdaptiveRadialPanel:
    """One radial estimator interval."""

    left_u: float
    right_u: float
    depth: int = 0

    def __post_init__(self) -> None:
        left = float(self.left_u)
        right = float(self.right_u)
        depth = int(self.depth)
        if not np.isfinite(left) or not np.isfinite(right):
            raise ValueError("panel bounds must be finite")
        if left < 0.0 or right <= left:
            raise ValueError("panel bounds must satisfy 0 <= left < right")
        if depth < 0:
            raise ValueError("panel depth must be non-negative")
        object.__setattr__(self, "left_u", left)
        object.__setattr__(self, "right_u", right)
        object.__setattr__(self, "depth", depth)

    @property
    def midpoint_u(self) -> float:
        return 0.5 * (self.left_u + self.right_u)

    def split(self) -> tuple["AdaptiveRadialPanel", "AdaptiveRadialPanel"]:
        middle = self.midpoint_u
        return (
            AdaptiveRadialPanel(self.left_u, middle, self.depth + 1),
            AdaptiveRadialPanel(middle, self.right_u, self.depth + 1),
        )


@dataclass(frozen=True)
class AdaptiveOuterQPanelGrid:
    """Tensor-product quadrature restricted to one radial annulus."""

    panel: AdaptiveRadialPanel
    u: np.ndarray
    phi_rad: np.ndarray
    q_model: np.ndarray
    measure_weights_m_inv2: np.ndarray
    radial_order: int
    angular_order: int
    angular_offset_fraction: float
    separation_m: float

    def __post_init__(self) -> None:
        for name in ("u", "phi_rad", "measure_weights_m_inv2"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite one-dimensional array")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        q = np.array(self.q_model, dtype=float, copy=True)
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError("q_model must have shape (N, 2)")
        q.setflags(write=False)
        object.__setattr__(self, "q_model", q)
        count = int(self.radial_order) * int(self.angular_order)
        if not all(
            len(getattr(self, name)) == count
            for name in ("u", "phi_rad", "q_model", "measure_weights_m_inv2")
        ):
            raise ValueError("panel grid arrays have inconsistent node counts")
        if not np.all(self.measure_weights_m_inv2 > 0.0):
            raise ValueError("panel measure weights must be positive")

    @property
    def node_count(self) -> int:
        return len(self.u)


@dataclass(frozen=True)
class AdaptiveRadialCasimirConfig:
    """Configuration for finite-domain radial adaptivity."""

    point_config: FixedCasimirConfig = field(default_factory=FixedCasimirConfig)
    initial_panel_edges: tuple[float, ...] = (0.0, 6.0, 10.0, 14.0, 18.0, 24.0)
    radial_order: int = 8
    angular_order: int = 8
    angular_offset_fraction: float = 0.5
    radial_rtol: float = 5e-2
    radial_atol_J_m2: float = 1e-10
    max_refinement_rounds: int = 8
    max_panel_depth: int = 8
    refine_panels_per_round: int = 4
    max_microscopic_q_nodes: int = 20_000
    point_cache_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.point_config, FixedCasimirConfig):
            raise TypeError("point_config must be a FixedCasimirConfig")
        edges = tuple(float(value) for value in self.initial_panel_edges)
        if len(edges) < 2 or not np.isfinite(edges).all():
            raise ValueError("initial_panel_edges must contain finite values")
        if edges[0] != 0.0:
            raise ValueError("initial_panel_edges must start at zero")
        if any(right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)):
            raise ValueError("initial_panel_edges must be strictly increasing")
        object.__setattr__(self, "initial_panel_edges", edges)
        for name in ("radial_order", "angular_order"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        offset = float(self.angular_offset_fraction)
        if not np.isfinite(offset) or not 0.0 <= offset < 1.0:
            raise ValueError("angular_offset_fraction must lie in [0, 1)")
        object.__setattr__(self, "angular_offset_fraction", offset)
        for name in ("radial_rtol", "radial_atol_J_m2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.radial_rtol == 0.0 and self.radial_atol_J_m2 == 0.0:
            raise ValueError("at least one radial tolerance must be positive")
        for name in ("max_refinement_rounds", "max_panel_depth"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("refine_panels_per_round", "max_microscopic_q_nodes"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.point_cache_path is not None:
            object.__setattr__(self, "point_cache_path", Path(self.point_cache_path))

    @property
    def u_max(self) -> float:
        return self.initial_panel_edges[-1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_config": self.point_config.as_dict(),
            "initial_panel_edges": list(self.initial_panel_edges),
            "u_max": self.u_max,
            "radial_order": self.radial_order,
            "angular_order": self.angular_order,
            "angular_offset_fraction": self.angular_offset_fraction,
            "radial_rtol": self.radial_rtol,
            "radial_atol_J_m2": self.radial_atol_J_m2,
            "max_refinement_rounds": self.max_refinement_rounds,
            "max_panel_depth": self.max_panel_depth,
            "refine_panels_per_round": self.refine_panels_per_round,
            "max_microscopic_q_nodes": self.max_microscopic_q_nodes,
            "point_cache_path": (
                None if self.point_cache_path is None else str(self.point_cache_path)
            ),
        }


@dataclass(frozen=True)
class _LeafEstimate:
    panel: AdaptiveRadialPanel
    coarse_integrals: Mapping[str, np.ndarray]
    fine_integrals: Mapping[str, np.ndarray]
    error_integrals: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class AdaptiveRadialCasimirResult:
    """Fail-closed finite Matsubara result from radial adaptivity."""

    status: Literal["adaptive_finite_partial", "unresolved"]
    config: AdaptiveRadialCasimirConfig
    radial_converged: bool
    all_microscopic_nodes_certified: bool
    pairing_results: Mapping[str, Any]
    panel_records: tuple[Mapping[str, Any], ...]
    refinement_rounds: int
    unique_microscopic_q_node_count: int
    unresolved_points: tuple[Mapping[str, Any], ...]
    termination_reason: str
    provider_statistics: Mapping[str, Any]

    @property
    def production_casimir_allowed(self) -> bool:
        return False

    @property
    def partial_sum_only(self) -> bool:
        return True

    @property
    def outer_tail_estimated(self) -> bool:
        return False

    @property
    def matsubara_tail_estimated(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "adaptive-radial-casimir-result-v1",
            "status": self.status,
            "production_casimir_allowed": False,
            "partial_sum_only": True,
            "outer_cutoff_fixed": True,
            "outer_tail_estimated": False,
            "matsubara_tail_estimated": False,
            "radial_converged": self.radial_converged,
            "all_microscopic_nodes_certified": self.all_microscopic_nodes_certified,
            "config": self.config.as_dict(),
            "pairing_results": dict(self.pairing_results),
            "panel_records": [dict(record) for record in self.panel_records],
            "refinement_rounds": self.refinement_rounds,
            "unique_microscopic_q_node_count": self.unique_microscopic_q_node_count,
            "unresolved_points": [dict(value) for value in self.unresolved_points],
            "termination_reason": self.termination_reason,
            "provider_statistics": dict(self.provider_statistics),
        }


def build_adaptive_outer_q_panel_grid(
    panel: AdaptiveRadialPanel,
    *,
    separation_m: float,
    lattice_a_x_m: float,
    lattice_a_y_m: float,
    radial_order: int,
    angular_order: int,
    angular_offset_fraction: float,
) -> AdaptiveOuterQPanelGrid:
    """Build the established polar measure on one finite radial panel."""

    d = float(separation_m)
    ax = float(lattice_a_x_m)
    ay = float(lattice_a_y_m)
    nr = int(radial_order)
    nphi = int(angular_order)
    offset = float(angular_offset_fraction)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("separation_m must be finite and positive")
    if not np.isfinite(ax) or ax <= 0.0 or not np.isfinite(ay) or ay <= 0.0:
        raise ValueError("lattice constants must be finite and positive")
    if nr <= 0 or nphi <= 0:
        raise ValueError("radial_order and angular_order must be positive")
    if not np.isfinite(offset) or not 0.0 <= offset < 1.0:
        raise ValueError("angular_offset_fraction must lie in [0, 1)")

    roots, root_weights = np.polynomial.legendre.leggauss(nr)
    half_width = 0.5 * (panel.right_u - panel.left_u)
    midpoint = panel.midpoint_u
    radial_u = midpoint + half_width * roots
    radial_weights = half_width * root_weights
    angular_phi = _TWO_PI * (np.arange(nphi, dtype=float) + offset) / nphi
    angular_weight = _TWO_PI / nphi
    u_mesh, phi_mesh = np.meshgrid(radial_u, angular_phi, indexing="ij")
    radial_weight_mesh, _ = np.meshgrid(
        radial_weights,
        angular_phi,
        indexing="ij",
    )
    q_radius = u_mesh / (2.0 * d)
    qx = q_radius * np.cos(phi_mesh)
    qy = q_radius * np.sin(phi_mesh)
    q_model = np.column_stack([(ax * qx).ravel(), (ay * qy).ravel()])
    weights = (
        u_mesh
        * radial_weight_mesh
        * angular_weight
        / (16.0 * np.pi**2 * d**2)
    ).ravel()
    exact_measure = (
        panel.right_u**2 - panel.left_u**2
    ) / (16.0 * np.pi * d**2)
    error = abs(float(np.sum(weights)) - exact_measure)
    tolerance = 128.0 * np.finfo(float).eps * max(exact_measure, 1.0)
    if error > tolerance:
        raise RuntimeError("adaptive panel weights fail the exact annulus measure")
    return AdaptiveOuterQPanelGrid(
        panel=panel,
        u=u_mesh.ravel(),
        phi_rad=phi_mesh.ravel(),
        q_model=q_model,
        measure_weights_m_inv2=weights,
        radial_order=nr,
        angular_order=nphi,
        angular_offset_fraction=offset,
        separation_m=d,
    )


def _grid_integrals(
    provider: _PointProvider,
    grid: AdaptiveOuterQPanelGrid,
    config: AdaptiveRadialCasimirConfig,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for pairing in config.point_config.pairings:
        by_n = []
        for n in config.point_config.matsubara_indices:
            values = np.asarray(
                [provider.primary_logdet(pairing, n, q) for q in grid.q_model],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise ValueError("certified logdet values must be finite")
            by_n.append(float(np.dot(values, grid.measure_weights_m_inv2)))
        output[pairing] = np.asarray(by_n, dtype=float)
    return output


def _candidate_grids(
    panels: Sequence[AdaptiveRadialPanel],
    config: AdaptiveRadialCasimirConfig,
) -> tuple[
    tuple[
        AdaptiveRadialPanel,
        AdaptiveOuterQPanelGrid,
        AdaptiveOuterQPanelGrid,
        AdaptiveOuterQPanelGrid,
    ],
    ...,
]:
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    common = {
        "separation_m": config.point_config.separation_m,
        "lattice_a_x_m": material.lattice_a_x_m,
        "lattice_a_y_m": material.lattice_a_y_m,
        "radial_order": config.radial_order,
        "angular_order": config.angular_order,
        "angular_offset_fraction": config.angular_offset_fraction,
    }
    records = []
    for panel in panels:
        left, right = panel.split()
        records.append(
            (
                panel,
                build_adaptive_outer_q_panel_grid(panel, **common),
                build_adaptive_outer_q_panel_grid(left, **common),
                build_adaptive_outer_q_panel_grid(right, **common),
            )
        )
    return tuple(records)


def build_initial_adaptive_outer_q_model(
    config: AdaptiveRadialCasimirConfig,
) -> np.ndarray:
    """Build the exact initial parent/child q nodes for one radial run."""

    panels = tuple(
        AdaptiveRadialPanel(left, right, 0)
        for left, right in zip(
            config.initial_panel_edges[:-1],
            config.initial_panel_edges[1:],
            strict=True,
        )
    )
    records = _candidate_grids(panels, config)
    return np.concatenate(
        [
            grid.q_model
            for _, parent, left, right in records
            for grid in (parent, left, right)
        ],
        axis=0,
    )


def _q_hex_keys(q_model: np.ndarray) -> set[tuple[str, str]]:
    return {
        (float(q[0]).hex(), float(q[1]).hex())
        for q in np.asarray(q_model, dtype=float)
    }


def _evaluate_panels(
    panels: Sequence[AdaptiveRadialPanel],
    *,
    provider: _PointProvider,
    config: AdaptiveRadialCasimirConfig,
    requested_q_keys: set[tuple[str, str]],
) -> tuple[tuple[_LeafEstimate, ...], tuple[Mapping[str, Any], ...], str | None]:
    records = _candidate_grids(panels, config)
    q_arrays = [
        grid.q_model
        for _, parent, left, right in records
        for grid in (parent, left, right)
    ]
    combined = np.concatenate(q_arrays, axis=0)
    candidate_keys = _q_hex_keys(combined)
    if len(requested_q_keys | candidate_keys) > config.max_microscopic_q_nodes:
        return (), (), "microscopic_q_node_budget_exhausted"
    requested_q_keys.update(candidate_keys)
    batch = provider.evaluate(combined)
    if not batch.all_established:
        return (), batch.unresolved_points, "microscopic_point_unresolved"

    estimates: list[_LeafEstimate] = []
    for panel, parent_grid, left_grid, right_grid in records:
        coarse = _grid_integrals(provider, parent_grid, config)
        left = _grid_integrals(provider, left_grid, config)
        right = _grid_integrals(provider, right_grid, config)
        fine = {pairing: left[pairing] + right[pairing] for pairing in coarse}
        error = {
            pairing: np.abs(fine[pairing] - coarse[pairing])
            for pairing in coarse
        }
        estimates.append(
            _LeafEstimate(
                panel=panel,
                coarse_integrals=MappingProxyType(coarse),
                fine_integrals=MappingProxyType(fine),
                error_integrals=MappingProxyType(error),
            )
        )
    return tuple(estimates), (), None


def _summarize(
    leaves: Sequence[_LeafEstimate],
    config: AdaptiveRadialCasimirConfig,
) -> tuple[dict[str, Any], bool, dict[str, np.ndarray]]:
    prime = matsubara_prime_weights(config.point_config.matsubara_indices)
    factor = KB * config.point_config.temperature_K * prime
    results: dict[str, Any] = {}
    tolerances: dict[str, np.ndarray] = {}
    all_passed = True
    for pairing in config.point_config.pairings:
        count = len(config.point_config.matsubara_indices)
        outer = np.asarray(
            [
                math.fsum(
                    float(leaf.fine_integrals[pairing][index]) for leaf in leaves
                )
                for index in range(count)
            ],
            dtype=float,
        )
        error_outer = np.asarray(
            [
                math.fsum(
                    float(leaf.error_integrals[pairing][index]) for leaf in leaves
                )
                for index in range(count)
            ],
            dtype=float,
        )
        contributions = factor * outer
        errors = factor * error_outer
        channel_tolerance = np.maximum(
            config.radial_atol_J_m2,
            config.radial_rtol * np.abs(contributions),
        )
        passed = errors <= channel_tolerance
        all_passed = all_passed and bool(np.all(passed))
        tolerances[pairing] = channel_tolerance
        results[pairing] = {
            "status": "integrated" if bool(np.all(passed)) else "radial_unresolved",
            "partial_free_energy_J_m2": float(math.fsum(contributions.tolist())),
            "contributions_J_m2": contributions.tolist(),
            "outer_q_integrals_m_inv2": outer.tolist(),
            "estimated_radial_errors_J_m2": errors.tolist(),
            "radial_tolerances_J_m2": channel_tolerance.tolist(),
            "radial_channel_passed": passed.tolist(),
            "matsubara_indices": list(config.point_config.matsubara_indices),
            "prime_weights": prime.tolist(),
        }
    return results, all_passed, tolerances


def _leaf_score(
    leaf: _LeafEstimate,
    config: AdaptiveRadialCasimirConfig,
    tolerances: Mapping[str, np.ndarray],
) -> float:
    prime = matsubara_prime_weights(config.point_config.matsubara_indices)
    factor = KB * config.point_config.temperature_K * prime
    scores = []
    for pairing in config.point_config.pairings:
        errors = factor * leaf.error_integrals[pairing]
        denominator = tolerances[pairing]
        ratio = np.divide(
            errors,
            denominator,
            out=np.full_like(errors, np.inf),
            where=denominator > 0.0,
        )
        scores.append(float(np.max(ratio)))
    return max(scores, default=0.0)


def _panel_records(
    leaves: Sequence[_LeafEstimate],
    config: AdaptiveRadialCasimirConfig,
    tolerances: Mapping[str, np.ndarray],
) -> tuple[Mapping[str, Any], ...]:
    records = []
    prime = matsubara_prime_weights(config.point_config.matsubara_indices)
    factor = KB * config.point_config.temperature_K * prime
    for leaf in sorted(leaves, key=lambda value: value.panel):
        pairing_payload = {}
        for pairing in config.point_config.pairings:
            pairing_payload[pairing] = {
                "coarse_outer_q_integrals_m_inv2": leaf.coarse_integrals[
                    pairing
                ].tolist(),
                "fine_outer_q_integrals_m_inv2": leaf.fine_integrals[
                    pairing
                ].tolist(),
                "estimated_errors_J_m2": (
                    factor * leaf.error_integrals[pairing]
                ).tolist(),
            }
        records.append(
            {
                "left_u": leaf.panel.left_u,
                "right_u": leaf.panel.right_u,
                "depth": leaf.panel.depth,
                "accepted_estimate": "sum_of_two_child_panels",
                "score": _leaf_score(leaf, config, tolerances),
                "pairings": pairing_payload,
            }
        )
    return tuple(records)


def _provider_statistics(provider: _PointProvider) -> dict[str, Any]:
    names = (
        "cached_point_count",
        "unique_q_count",
        "certification_batches",
        "requested_q_evaluations",
        "new_q_evaluations",
        "cache_hit_q_evaluations",
    )
    return {name: int(getattr(provider, name, 0)) for name in names}


def _unresolved_result(
    config: AdaptiveRadialCasimirConfig,
    *,
    leaves: Sequence[_LeafEstimate],
    pairing_results: Mapping[str, Any],
    rounds: int,
    requested_q_keys: set[tuple[str, str]],
    unresolved: Sequence[Mapping[str, Any]],
    reason: str,
    provider: _PointProvider,
) -> AdaptiveRadialCasimirResult:
    if leaves:
        _, _, tolerances = _summarize(leaves, config)
        panels = _panel_records(leaves, config, tolerances)
    else:
        panels = ()
    return AdaptiveRadialCasimirResult(
        status="unresolved",
        config=config,
        radial_converged=False,
        all_microscopic_nodes_certified=not unresolved,
        pairing_results=dict(pairing_results),
        panel_records=panels,
        refinement_rounds=rounds,
        unique_microscopic_q_node_count=len(requested_q_keys),
        unresolved_points=tuple(unresolved),
        termination_reason=reason,
        provider_statistics=_provider_statistics(provider),
    )


def run_adaptive_radial_casimir(
    config: AdaptiveRadialCasimirConfig,
    *,
    provider: _PointProvider | None = None,
) -> AdaptiveRadialCasimirResult:
    """Adapt radial panels on a fixed finite ``u`` interval.

    The accepted estimate on each leaf is the sum of two child-panel rules; the
    parent-versus-children difference supplies the local error estimate.  Global
    convergence is required separately for every pairing and Matsubara contribution.
    """

    if not isinstance(config, AdaptiveRadialCasimirConfig):
        raise TypeError("config must be an AdaptiveRadialCasimirConfig")
    active_provider: _PointProvider = (
        CertifiedOuterQProvider(
            config.point_config,
            cache_path=config.point_cache_path,
        )
        if provider is None
        else provider
    )
    leaves: tuple[_LeafEstimate, ...] = ()
    pairing_results: dict[str, Any] = {}
    requested_q_keys: set[tuple[str, str]] = set()
    rounds = 0
    initial_panels = tuple(
        AdaptiveRadialPanel(left, right, 0)
        for left, right in zip(
            config.initial_panel_edges[:-1],
            config.initial_panel_edges[1:],
            strict=True,
        )
    )

    try:
        leaves, unresolved, reason = _evaluate_panels(
            initial_panels,
            provider=active_provider,
            config=config,
            requested_q_keys=requested_q_keys,
        )
        if reason is not None:
            return _unresolved_result(
                config,
                leaves=leaves,
                pairing_results=pairing_results,
                rounds=rounds,
                requested_q_keys=requested_q_keys,
                unresolved=unresolved,
                reason=reason,
                provider=active_provider,
            )

        while True:
            pairing_results, converged, tolerances = _summarize(leaves, config)
            if converged:
                return AdaptiveRadialCasimirResult(
                    status="adaptive_finite_partial",
                    config=config,
                    radial_converged=True,
                    all_microscopic_nodes_certified=True,
                    pairing_results=pairing_results,
                    panel_records=_panel_records(leaves, config, tolerances),
                    refinement_rounds=rounds,
                    unique_microscopic_q_node_count=len(requested_q_keys),
                    unresolved_points=(),
                    termination_reason="radial_tolerance_met",
                    provider_statistics=_provider_statistics(active_provider),
                )
            if rounds >= config.max_refinement_rounds:
                return _unresolved_result(
                    config,
                    leaves=leaves,
                    pairing_results=pairing_results,
                    rounds=rounds,
                    requested_q_keys=requested_q_keys,
                    unresolved=(),
                    reason="maximum_refinement_rounds_reached",
                    provider=active_provider,
                )
            eligible = [
                leaf for leaf in leaves if leaf.panel.depth < config.max_panel_depth
            ]
            if not eligible:
                return _unresolved_result(
                    config,
                    leaves=leaves,
                    pairing_results=pairing_results,
                    rounds=rounds,
                    requested_q_keys=requested_q_keys,
                    unresolved=(),
                    reason="maximum_panel_depth_reached",
                    provider=active_provider,
                )
            ranked = sorted(
                eligible,
                key=lambda leaf: (
                    -_leaf_score(leaf, config, tolerances),
                    leaf.panel.left_u,
                    leaf.panel.right_u,
                ),
            )
            selected = ranked[: config.refine_panels_per_round]
            selected_panels = {leaf.panel for leaf in selected}
            children = tuple(
                child for leaf in selected for child in leaf.panel.split()
            )
            new_estimates, unresolved, reason = _evaluate_panels(
                children,
                provider=active_provider,
                config=config,
                requested_q_keys=requested_q_keys,
            )
            if reason is not None:
                return _unresolved_result(
                    config,
                    leaves=leaves,
                    pairing_results=pairing_results,
                    rounds=rounds,
                    requested_q_keys=requested_q_keys,
                    unresolved=unresolved,
                    reason=reason,
                    provider=active_provider,
                )
            leaves = tuple(
                sorted(
                    [
                        leaf for leaf in leaves if leaf.panel not in selected_panels
                    ]
                    + list(new_estimates),
                    key=lambda leaf: leaf.panel,
                )
            )
            rounds += 1
    except (CertifiedPointCacheError, FixedCasimirExecutionError) as exc:
        return _unresolved_result(
            config,
            leaves=leaves,
            pairing_results=pairing_results,
            rounds=rounds,
            requested_q_keys=requested_q_keys,
            unresolved=({"reason": str(exc)},),
            reason="point_provider_failure",
            provider=active_provider,
        )


__all__ = [
    "AdaptiveOuterQPanelGrid",
    "AdaptiveRadialCasimirConfig",
    "AdaptiveRadialCasimirResult",
    "AdaptiveRadialPanel",
    "build_adaptive_outer_q_panel_grid",
    "run_adaptive_radial_casimir",
]
