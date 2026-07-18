"""Canonical full adaptive Casimir calculation surface.

The public production route is intentionally narrow:

``build_full_casimir_config`` -> ``run_full_casimir``

Lower-level radial, angular, cutoff and Matsubara controllers remain implementation
modules.  The historical fixed-grid controller is isolated in
:mod:`lno327.casimir.legacy` and is not re-exported from :mod:`lno327.casimir`.

The numerical result remains fail-closed: repository production readiness does not
change the physical authorization flag carried by the adaptive result.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, Sequence

from .adaptive_matsubara_tail import (
    AdaptiveMatsubaraCasimirConfig,
    AdaptiveMatsubaraCasimirResult,
    run_adaptive_matsubara_casimir,
)
from .fixed_chain import FixedCasimirConfig

FullCasimirConfig = AdaptiveMatsubaraCasimirConfig
FullCasimirResult = AdaptiveMatsubaraCasimirResult


def build_full_casimir_config(
    *,
    pairings: Sequence[str] = ("spm",),
    temperature_K: float = 10.0,
    separation_nm: float = 20.0,
    plate_angles_deg: tuple[float, float] = (0.0, 17.0),
    delta0_eV: float = 0.1,
    eta_eV: float = 1e-8,
    degeneracy: float = 1.0,
    N_candidates: Sequence[int] = (128, 192, 256),
    required_consecutive_passes: int = 2,
    workers: int = 0,
    parallel_mode: Literal["auto", "serial", "q", "context", "wave"] = "auto",
    memory_budget_gb: float = 0.0,
    max_context_workers: int = 0,
    cutoff_u_values: Sequence[float] = (6.0, 10.0, 14.0, 18.0, 24.0, 30.0, 36.0, 42.0),
    outer_tail_start_u: float = 24.0,
    outer_tail_window_shells: int = 3,
    outer_tail_ratio_max: float = 0.8,
    matsubara_cutoff_values: Sequence[int] = (1, 3, 7, 11, 15, 23, 31),
    matsubara_tail_start_n: int = 8,
    matsubara_tail_window_terms: int = 4,
    matsubara_tail_ratio_max: float = 0.8,
    total_free_energy_rtol: float = 5e-2,
    total_free_energy_atol_J_m2: float = 1e-10,
    max_total_microscopic_q_nodes: int = 250_000,
    max_total_microscopic_point_entries: int = 1_000_000,
    point_cache_path: Path | None = None,
) -> FullCasimirConfig:
    """Build the canonical nested adaptive configuration.

    External physical parameters are supplied by the caller.  The builder only wires
    them into the complete adaptive integration stack and applies the repository's
    conservative default ladders.  Advanced studies may construct the underlying
    dataclasses directly from their implementation modules.
    """

    pairing_tuple = tuple(str(value) for value in pairings)
    base = AdaptiveMatsubaraCasimirConfig()
    radial_base = base.outer_tail_config.joint_config.radial_config
    point = replace(
        radial_base.point_config,
        pairings=pairing_tuple,
        matsubara_indices=(0, 1),
        temperature_K=float(temperature_K),
        separation_nm=float(separation_nm),
        plate_angles_deg=tuple(float(value) for value in plate_angles_deg),
        delta0_eV=float(delta0_eV),
        eta_eV=float(eta_eV),
        degeneracy=float(degeneracy),
        N_candidates=tuple(int(value) for value in N_candidates),
        required_consecutive_passes=int(required_consecutive_passes),
        workers=int(workers),
        parallel_mode=parallel_mode,
        memory_budget_gb=float(memory_budget_gb),
        max_context_workers=int(max_context_workers),
        transverse_checkpoint_path=None,
    )
    radial = replace(
        radial_base,
        point_config=point,
        max_microscopic_q_nodes=int(max_total_microscopic_q_nodes),
        point_cache_path=None,
    )
    # SPM pilot data show angular errors two orders of magnitude below the
    # radial allocation.  D-wave keeps a more conservative angular reserve.
    radial_fraction = 0.85 if set(pairing_tuple) == {"spm"} else 0.75
    joint = replace(
        base.outer_tail_config.joint_config,
        radial_config=radial,
        radial_budget_fraction=radial_fraction,
        angular_budget_fraction=1.0 - radial_fraction,
        max_total_microscopic_q_nodes=int(max_total_microscopic_q_nodes),
    )
    outer = replace(
        base.outer_tail_config,
        joint_config=joint,
        cutoff_u_values=tuple(float(value) for value in cutoff_u_values),
        total_outer_rtol=float(total_free_energy_rtol),
        total_outer_atol_J_m2=float(total_free_energy_atol_J_m2),
        tail_start_u=float(outer_tail_start_u),
        tail_window_shells=int(outer_tail_window_shells),
        tail_ratio_max=float(outer_tail_ratio_max),
        max_total_microscopic_q_nodes=int(max_total_microscopic_q_nodes),
    )
    return replace(
        base,
        outer_tail_config=outer,
        matsubara_cutoff_values=tuple(int(value) for value in matsubara_cutoff_values),
        total_free_energy_rtol=float(total_free_energy_rtol),
        total_free_energy_atol_J_m2=float(total_free_energy_atol_J_m2),
        tail_start_n=int(matsubara_tail_start_n),
        tail_window_terms=int(matsubara_tail_window_terms),
        tail_ratio_max=float(matsubara_tail_ratio_max),
        max_total_microscopic_point_entries=int(max_total_microscopic_point_entries),
        point_cache_path=None if point_cache_path is None else Path(point_cache_path),
    )


def run_full_casimir(config: FullCasimirConfig) -> FullCasimirResult:
    """Run the single canonical adaptive outer-integration route."""

    if not isinstance(config, AdaptiveMatsubaraCasimirConfig):
        raise TypeError("config must be a FullCasimirConfig")
    return run_adaptive_matsubara_casimir(config)


__all__ = [
    "FullCasimirConfig",
    "FullCasimirResult",
    "build_full_casimir_config",
    "run_full_casimir",
]
