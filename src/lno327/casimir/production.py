"""Canonical full adaptive Casimir calculation surface."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .adaptive_matsubara_tail import (
    AdaptiveMatsubaraCasimirConfig,
    AdaptiveMatsubaraCasimirResult,
    run_adaptive_matsubara_casimir,
)

FullCasimirConfig = AdaptiveMatsubaraCasimirConfig
FullCasimirResult = AdaptiveMatsubaraCasimirResult

_TELEMETRY_SCHEMA = "certified-point-provider-telemetry-v1"
_TELEMETRY_INTEGER_FIELDS = (
    "certification_batches",
    "certification_failed_batches",
    "requested_q_evaluations",
    "new_q_evaluations",
    "cache_hit_q_evaluations",
    "requested_point_evaluations",
    "new_point_evaluations",
    "cache_hit_point_evaluations",
    "cache_save_count",
)
_TELEMETRY_FLOAT_FIELDS = (
    "certifier_wall_seconds",
    "certifier_reported_level_wall_seconds",
    "certifier_material_build_seconds",
    "certifier_context_wall_seconds",
    "cache_save_seconds",
)


def _safe_nonnegative_number(value: Any, *, integer: bool) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(numeric) or numeric < 0.0:
        return False
    return not integer or numeric.is_integer()


def _telemetry_payload_is_safe(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("schema") != _TELEMETRY_SCHEMA:
        return True
    for name in _TELEMETRY_INTEGER_FIELDS:
        if name in payload and not _safe_nonnegative_number(payload[name], integer=True):
            return False
    for name in _TELEMETRY_FLOAT_FIELDS:
        if name in payload and not _safe_nonnegative_number(payload[name], integer=False):
            return False
    records = payload.get("certifier_batch_records", [])
    return isinstance(records, list) and all(
        isinstance(record, Mapping) for record in records
    )


def _quarantine_invalid_telemetry(config: FullCasimirConfig) -> Path | None:
    """Remove only malformed, non-authoritative telemetry from the resume path.

    The certified-point cache remains untouched.  A quarantined sidecar is retained
    next to the cache for diagnosis and can never affect physical acceptance.
    """

    if config.point_cache_path is None:
        return None
    telemetry_path = Path(config.point_cache_path).with_suffix(".telemetry.json")
    if not telemetry_path.exists():
        return None
    try:
        payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if _telemetry_payload_is_safe(payload):
        return None
    quarantine_path = telemetry_path.with_suffix(telemetry_path.suffix + ".invalid")
    try:
        quarantine_path.unlink(missing_ok=True)
        telemetry_path.replace(quarantine_path)
    except OSError as exc:
        raise RuntimeError(
            f"cannot quarantine malformed telemetry sidecar {telemetry_path}: {exc}"
        ) from exc
    return quarantine_path


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
    logdet_rtol: float = 1.5e-3,
    logdet_atol: float = 1e-6,
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
    total_free_energy_rtol: float = 5e-3,
    total_free_energy_atol_J_m2: float = 1e-12,
    max_total_microscopic_q_nodes: int = 250_000,
    max_total_microscopic_point_entries: int = 1_000_000,
    certifier_q_batch_size: int = 512,
    point_cache_path: Path | None = None,
) -> FullCasimirConfig:
    """Build the canonical nested adaptive configuration."""

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
        logdet_rtol=float(logdet_rtol),
        logdet_atol=float(logdet_atol),
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
        certifier_q_batch_size=int(certifier_q_batch_size),
        point_cache_path=None if point_cache_path is None else Path(point_cache_path),
    )


def run_full_casimir(config: FullCasimirConfig) -> FullCasimirResult:
    """Run the single canonical adaptive outer-integration route."""

    if not isinstance(config, AdaptiveMatsubaraCasimirConfig):
        raise TypeError("config must be a FullCasimirConfig")
    _quarantine_invalid_telemetry(config)
    return run_adaptive_matsubara_casimir(config)


__all__ = [
    "FullCasimirConfig",
    "FullCasimirResult",
    "build_full_casimir_config",
    "run_full_casimir",
]
