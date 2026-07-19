"""Unique non-adaptive controller for the fixed microscopic Casimir chain.

The controller composes the production-owned transverse-point certifier and fixed
outer-Q reducer without importing :mod:`validation`.  It intentionally returns a
finite Matsubara partial result only.  No radial adaptivity, automatic outer cutoff,
or Matsubara-tail estimate is introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Literal, Mapping

import numpy as np

from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE

from .fixed_outer_q import (
    OuterQGridPlan,
    OuterQNodeManifest,
    aggregate_certified_outer_q,
    build_staged_grid_plan,
    build_union_node_manifest,
    compare_ladders,
)

DEFAULT_SHIFTS = ((0.5, 0.5), (0.25, 0.75), (0.75, 0.25))

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_ALLOWED_PAIRINGS = ("spm", "dwave")
_ALLOWED_PARALLEL_MODES = ("auto", "serial", "q", "context", "wave")


class FixedCasimirExecutionError(RuntimeError):
    """Raised when the production point-certification process cannot complete."""


@dataclass(frozen=True)
class FixedCasimirConfig:
    """Immutable configuration for the qualified fixed Casimir chain.

    Defaults reproduce the qualified ``spm``, ``n=0,1`` fixed reference settings.
    They do not authorize a full production Casimir result because the Matsubara tail
    remains outside this contract.
    """

    pairings: tuple[str, ...] = ("spm",)
    matsubara_indices: tuple[int, ...] = (0, 1)
    u_max_values: tuple[float, ...] = (6.0, 10.0, 14.0, 18.0, 24.0)
    radial_orders: tuple[int, ...] = (4, 6, 8)
    angular_orders: tuple[int, ...] = (4, 8)
    angular_offsets: tuple[float, ...] = (0.0, 0.5)
    N_candidates: tuple[int, ...] = (128, 192, 256)
    shifts: tuple[tuple[float, float], ...] = DEFAULT_SHIFTS
    plate_angles_deg: tuple[float, float] = (0.0, 17.0)
    required_consecutive_passes: int = 2
    workers: int = 0
    parallel_mode: Literal["auto", "serial", "q", "context", "wave"] = "auto"
    memory_budget_gb: float = 0.0
    max_context_workers: int = 0
    memory_safety_factor: float = 1.5
    fallback_context_bytes_per_point: float = 16_384.0
    canonical_block: int = 4096
    runtime_chunk: int = 16_384
    temperature_K: float = 10.0
    delta0_eV: float = 0.1
    eta_eV: float = 1e-8
    degeneracy: float = 1.0
    separation_nm: float = 20.0
    ward_tolerance: float = 1e-7
    ward_absolute_tolerance: float = 1e-12
    condition_max: float = 1e12
    static_energy_scale_eV: float = 1.0
    static_reality_tolerance: float = 1e-8
    static_longitudinal_tolerance: float = 1e-6
    static_mixing_tolerance: float = 1e-6
    static_passivity_tolerance: float = 1e-10
    logdet_rtol: float = 1e-3
    logdet_atol: float = 1e-6
    outer_rtol: float = 5e-2
    outer_atol_J_m2: float = 1e-10
    transverse_checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        pairings = tuple(dict.fromkeys(str(value) for value in self.pairings))
        if not pairings or any(value not in _ALLOWED_PAIRINGS for value in pairings):
            raise ValueError(f"pairings must be a nonempty subset of {_ALLOWED_PAIRINGS}")
        object.__setattr__(self, "pairings", pairings)

        indices = tuple(sorted(set(int(value) for value in self.matsubara_indices)))
        if not indices or any(value < 0 for value in indices):
            raise ValueError("matsubara_indices must be nonempty and non-negative")
        object.__setattr__(self, "matsubara_indices", indices)

        n_candidates = tuple(int(value) for value in self.N_candidates)
        if len(n_candidates) < 3:
            raise ValueError("N_candidates must contain at least three levels")
        if (
            tuple(sorted(set(n_candidates))) != n_candidates
            or any(value <= 0 or value % 2 for value in n_candidates)
        ):
            raise ValueError(
                "N_candidates must be strictly increasing unique positive even integers"
            )
        object.__setattr__(self, "N_candidates", n_candidates)

        shifts = tuple(
            tuple(float(component) for component in shift) for shift in self.shifts
        )
        if any(len(shift) != 2 for shift in shifts):
            raise ValueError("each shift must contain exactly two components")
        if len(shifts) < 2 or len(set(shifts)) != len(shifts):
            raise ValueError("at least two unique shifts are required")
        if not all(np.isfinite(value) for shift in shifts for value in shift):
            raise ValueError("all shifts must be finite")
        object.__setattr__(self, "shifts", shifts)

        angles = tuple(float(value) for value in self.plate_angles_deg)
        if len(angles) != 2 or not all(np.isfinite(value) for value in angles):
            raise ValueError("plate_angles_deg must contain two finite values")
        object.__setattr__(self, "plate_angles_deg", angles)

        for name, caster in (
            ("u_max_values", float),
            ("radial_orders", int),
            ("angular_orders", int),
            ("angular_offsets", float),
        ):
            object.__setattr__(
                self,
                name,
                tuple(caster(value) for value in getattr(self, name)),
            )
        if (
            len(self.u_max_values) < 2
            or len(self.radial_orders) < 2
            or len(self.angular_orders) < 2
            or len(self.angular_offsets) < 2
        ):
            raise ValueError("every fixed outer-Q audit ladder requires at least two values")
        build_staged_grid_plan(
            u_max_values=self.u_max_values,
            radial_orders=self.radial_orders,
            angular_orders=self.angular_orders,
            angular_offsets=self.angular_offsets,
        )

        if self.required_consecutive_passes <= 0:
            raise ValueError("required_consecutive_passes must be positive")
        if self.required_consecutive_passes >= len(self.N_candidates):
            raise ValueError("required_consecutive_passes leaves no usable N ladder")
        if self.workers < 0 or self.max_context_workers < 0:
            raise ValueError("worker controls must be non-negative")
        if self.parallel_mode not in _ALLOWED_PARALLEL_MODES:
            raise ValueError(f"parallel_mode must be one of {_ALLOWED_PARALLEL_MODES}")
        if not np.isfinite(self.memory_budget_gb) or self.memory_budget_gb < 0.0:
            raise ValueError("memory_budget_gb must be finite and non-negative")
        if not np.isfinite(self.memory_safety_factor) or self.memory_safety_factor < 1.0:
            raise ValueError("memory_safety_factor must be finite and at least one")
        if (
            not np.isfinite(self.fallback_context_bytes_per_point)
            or self.fallback_context_bytes_per_point <= 0.0
        ):
            raise ValueError("fallback_context_bytes_per_point must be finite and positive")
        if self.canonical_block <= 0 or self.runtime_chunk <= 0:
            raise ValueError("canonical_block and runtime_chunk must be positive")

        for name in ("temperature_K", "delta0_eV", "degeneracy", "separation_nm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.eta_eV) or self.eta_eV < 0.0:
            raise ValueError("eta_eV must be finite and non-negative")
        if not np.isfinite(self.condition_max) or self.condition_max <= 0.0:
            raise ValueError("condition_max must be finite and positive")
        for name in (
            "ward_tolerance",
            "ward_absolute_tolerance",
            "static_energy_scale_eV",
            "static_reality_tolerance",
            "static_longitudinal_tolerance",
            "static_mixing_tolerance",
            "static_passivity_tolerance",
            "logdet_rtol",
            "logdet_atol",
            "outer_rtol",
            "outer_atol_J_m2",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        checkpoint = self.transverse_checkpoint_path
        if checkpoint is not None:
            object.__setattr__(self, "transverse_checkpoint_path", Path(checkpoint))

    @property
    def separation_m(self) -> float:
        return float(self.separation_nm) * 1e-9

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairings": list(self.pairings),
            "matsubara_indices": list(self.matsubara_indices),
            "u_max_values": list(self.u_max_values),
            "radial_orders": list(self.radial_orders),
            "angular_orders": list(self.angular_orders),
            "angular_offsets": list(self.angular_offsets),
            "N_candidates": list(self.N_candidates),
            "shifts": [list(value) for value in self.shifts],
            "plate_angles_deg": list(self.plate_angles_deg),
            "required_consecutive_passes": int(self.required_consecutive_passes),
            "workers": int(self.workers),
            "parallel_mode": str(self.parallel_mode),
            "memory_budget_gb": float(self.memory_budget_gb),
            "max_context_workers": int(self.max_context_workers),
            "memory_safety_factor": float(self.memory_safety_factor),
            "fallback_context_bytes_per_point": float(
                self.fallback_context_bytes_per_point
            ),
            "canonical_block": int(self.canonical_block),
            "runtime_chunk": int(self.runtime_chunk),
            "temperature_K": float(self.temperature_K),
            "delta0_eV": float(self.delta0_eV),
            "eta_eV": float(self.eta_eV),
            "degeneracy": float(self.degeneracy),
            "separation_nm": float(self.separation_nm),
            "ward_tolerance": float(self.ward_tolerance),
            "ward_absolute_tolerance": float(self.ward_absolute_tolerance),
            "condition_max": float(self.condition_max),
            "static_energy_scale_eV": float(self.static_energy_scale_eV),
            "static_reality_tolerance": float(self.static_reality_tolerance),
            "static_longitudinal_tolerance": float(
                self.static_longitudinal_tolerance
            ),
            "static_mixing_tolerance": float(self.static_mixing_tolerance),
            "static_passivity_tolerance": float(self.static_passivity_tolerance),
            "logdet_rtol": float(self.logdet_rtol),
            "logdet_atol": float(self.logdet_atol),
            "outer_rtol": float(self.outer_rtol),
            "outer_atol_J_m2": float(self.outer_atol_J_m2),
            "transverse_checkpoint_path": (
                None
                if self.transverse_checkpoint_path is None
                else str(self.transverse_checkpoint_path)
            ),
        }


@dataclass(frozen=True)
class FixedCasimirResult:
    """Result of the unique fixed production controller."""

    status: Literal["finite_partial", "unresolved"]
    config: FixedCasimirConfig
    grid_plan: Mapping[str, Any]
    reference_spec_id: str
    reference_results: Mapping[str, Any]
    config_results: Mapping[str, Any]
    unresolved_microscopic_points: tuple[Mapping[str, Any], ...]
    ladder_comparisons: Mapping[str, Any]
    certification_payload: Mapping[str, Any]
    certification_stdout: str
    all_microscopic_nodes_certified: bool
    finite_partial_outer_q_integrals_available: bool
    candidate_outer_q_budget_established: bool

    @property
    def production_casimir_allowed(self) -> bool:
        return False

    @property
    def partial_sum_only(self) -> bool:
        return True

    @property
    def matsubara_tail_estimated(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "fixed-casimir-result-v1",
            "status": self.status,
            "production_casimir_allowed": False,
            "partial_sum_only": True,
            "matsubara_tail_estimated": False,
            "all_microscopic_nodes_certified": bool(
                self.all_microscopic_nodes_certified
            ),
            "finite_partial_outer_q_integrals_available": bool(
                self.finite_partial_outer_q_integrals_available
            ),
            "candidate_outer_q_budget_established": bool(
                self.candidate_outer_q_budget_established
            ),
            "config": self.config.as_dict(),
            "grid_plan": dict(self.grid_plan),
            "reference_spec_id": self.reference_spec_id,
            "reference_results": dict(self.reference_results),
            "config_results": dict(self.config_results),
            "unresolved_microscopic_points": [
                dict(value) for value in self.unresolved_microscopic_points
            ],
            "ladder_comparisons": dict(self.ladder_comparisons),
            "transverse_certification": dict(self.certification_payload),
        }


@dataclass(frozen=True)
class _CertificationRun:
    payload: Mapping[str, Any]
    stdout: str
    stderr: str
    command: tuple[str, ...]


def _thread_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in _THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _transverse_certification_command(
    config: FixedCasimirConfig,
    manifest: OuterQNodeManifest,
    output: Path,
    *,
    q_points_file: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lno327.casimir.fixed_transverse_point_certification",
    ]
    if q_points_file is None:
        for label, q in zip(manifest.labels, manifest.q_model, strict=True):
            command.extend(
                [
                    "--q-point",
                    label,
                    np.format_float_positional(float(q[0]), unique=True, trim="-"),
                    np.format_float_positional(float(q[1]), unique=True, trim="-"),
                ]
            )
    else:
        command.extend(["--q-points-file", str(q_points_file)])
    command.extend(["--pairings", *config.pairings])
    command.extend(
        [
            "--matsubara-indices",
            *[str(value) for value in config.matsubara_indices],
        ]
    )
    command.extend(["--N-candidates", *[str(value) for value in config.N_candidates]])
    for shift in config.shifts:
        command.extend(["--shift", repr(shift[0]), repr(shift[1])])
    command.extend(
        [
            "--plate-angles-deg",
            repr(config.plate_angles_deg[0]),
            repr(config.plate_angles_deg[1]),
            "--required-consecutive-passes",
            str(config.required_consecutive_passes),
            "--workers",
            str(config.workers),
            "--parallel-mode",
            config.parallel_mode,
            "--memory-budget-gb",
            repr(config.memory_budget_gb),
            "--max-context-workers",
            str(config.max_context_workers),
            "--memory-safety-factor",
            repr(config.memory_safety_factor),
            "--fallback-context-bytes-per-point",
            repr(config.fallback_context_bytes_per_point),
            "--canonical-block",
            str(config.canonical_block),
            "--runtime-chunk",
            str(config.runtime_chunk),
            "--temperature-K",
            repr(config.temperature_K),
            "--delta0-eV",
            repr(config.delta0_eV),
            "--eta-eV",
            repr(config.eta_eV),
            "--degeneracy",
            repr(config.degeneracy),
            "--separation-nm",
            repr(config.separation_nm),
            "--ward-tolerance",
            repr(config.ward_tolerance),
            "--ward-absolute-tolerance",
            repr(config.ward_absolute_tolerance),
            "--condition-max",
            repr(config.condition_max),
            "--static-energy-scale-eV",
            repr(config.static_energy_scale_eV),
            "--static-reality-tolerance",
            repr(config.static_reality_tolerance),
            "--static-longitudinal-tolerance",
            repr(config.static_longitudinal_tolerance),
            "--static-mixing-tolerance",
            repr(config.static_mixing_tolerance),
            "--static-passivity-tolerance",
            repr(config.static_passivity_tolerance),
            "--logdet-rtol",
            repr(config.logdet_rtol),
            "--logdet-atol",
            repr(config.logdet_atol),
            "--output",
            str(output),
        ]
    )
    return command


def _run_transverse_certifier(
    config: FixedCasimirConfig,
    manifest: OuterQNodeManifest,
    output: Path,
) -> _CertificationRun:
    q_points_file = output.with_name("q_points.json")
    q_points_payload = [
        {
            "label": str(label),
            "q_lab": [float(q[0]), float(q[1])],
        }
        for label, q in zip(manifest.labels, manifest.q_model, strict=True)
    ]
    q_points_file.write_text(
        json.dumps(q_points_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    command = _transverse_certification_command(
        config,
        manifest,
        output,
        q_points_file=q_points_file,
    )
    completed = subprocess.run(
        command,
        env=_thread_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FixedCasimirExecutionError(
            "production transverse-point certification failed with return code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedCasimirExecutionError(
            f"cannot read production transverse-point certification payload: {exc}"
        ) from exc
    if payload.get("schema") != "transverse-point-sweet-spot-v4":
        raise FixedCasimirExecutionError(
            "production transverse-point certification returned an unexpected schema"
        )
    return _CertificationRun(
        payload=payload,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=tuple(command),
    )


def _grid_plan_payload(
    plan: OuterQGridPlan,
    manifest: OuterQNodeManifest,
) -> dict[str, Any]:
    return {
        "reference_spec_id": plan.reference_spec_id,
        "reference_offset_fraction": float(plan.reference_offset_fraction),
        "ladders": {name: list(values) for name, values in plan.ladders.items()},
        "specs": [
            {
                "spec_id": spec.spec_id,
                "u_max": float(spec.u_max),
                "radial_panel_edges": list(spec.radial_panel_edges),
                "radial_panel_order": int(spec.radial_panel_order),
                "angular_order": int(spec.angular_order),
                "angular_offset_fraction": float(spec.angular_offset_fraction),
                "node_count": int(manifest.grids[spec.spec_id].node_count),
            }
            for spec in plan.specs
        ],
        "unique_microscopic_q_node_count": len(manifest.labels),
        "node_reuse_is_exact": True,
    }


def _candidate_outer_budget_established(
    *,
    all_certified: bool,
    comparisons: Mapping[str, Any],
    pairings: tuple[str, ...],
) -> bool:
    return bool(
        all_certified
        and all(
            bool(
                comparisons[ladder][pairing][
                    "all_passed" if ladder == "offset" else "final_transition_passed"
                ]
            )
            for ladder in comparisons
            for pairing in pairings
        )
    )


def _complete_run(
    config: FixedCasimirConfig,
    plan: OuterQGridPlan,
    manifest: OuterQNodeManifest,
    certification: _CertificationRun,
) -> FixedCasimirResult:
    payload = certification.payload
    config_results, unresolved = aggregate_certified_outer_q(
        sweet_spot_payload=payload,
        plan=plan,
        manifest=manifest,
        pairings=config.pairings,
        matsubara_indices=config.matsubara_indices,
        temperature_K=config.temperature_K,
    )
    all_certified = bool(
        payload.get("run_complete")
        and payload.get("all_requested_sweet_spots_established")
        and not unresolved
    )
    comparisons = compare_ladders(
        plan=plan,
        config_results=config_results,
        pairings=config.pairings,
        absolute_tolerance_J_m2=config.outer_atol_J_m2,
        relative_tolerance=config.outer_rtol,
    )
    reference_results = config_results[plan.reference_spec_id]["pairings"]
    finite_available = bool(
        all_certified
        and all(
            reference_results[pairing].get("status") == "integrated"
            for pairing in config.pairings
        )
    )
    candidate = _candidate_outer_budget_established(
        all_certified=all_certified,
        comparisons=comparisons,
        pairings=config.pairings,
    )
    return FixedCasimirResult(
        status="finite_partial" if finite_available else "unresolved",
        config=config,
        grid_plan=_grid_plan_payload(plan, manifest),
        reference_spec_id=plan.reference_spec_id,
        reference_results=reference_results,
        config_results=config_results,
        unresolved_microscopic_points=tuple(unresolved),
        ladder_comparisons=comparisons,
        certification_payload=payload,
        certification_stdout=certification.stdout,
        all_microscopic_nodes_certified=all_certified,
        finite_partial_outer_q_integrals_available=finite_available,
        candidate_outer_q_budget_established=candidate,
    )


def run_casimir(config: FixedCasimirConfig) -> FixedCasimirResult:
    """Run the unique fixed microscopic Casimir chain.

    The returned value is always fail-closed with
    ``production_casimir_allowed == False``.  A successful result has
    ``status == "finite_partial"`` and contains only the explicitly requested finite
    Matsubara indices.  No tail or automatic cutoff is inferred.
    """

    if not isinstance(config, FixedCasimirConfig):
        raise TypeError("config must be a FixedCasimirConfig")
    plan = build_staged_grid_plan(
        u_max_values=config.u_max_values,
        radial_orders=config.radial_orders,
        angular_orders=config.angular_orders,
        angular_offsets=config.angular_offsets,
    )
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    manifest = build_union_node_manifest(
        plan,
        separation_m=config.separation_m,
        lattice_a_x_m=material.lattice_a_x_m,
        lattice_a_y_m=material.lattice_a_y_m,
    )

    checkpoint = config.transverse_checkpoint_path
    if checkpoint is not None:
        certification = _run_transverse_certifier(config, manifest, checkpoint)
        return _complete_run(config, plan, manifest, certification)

    with TemporaryDirectory(prefix="lno327-fixed-casimir-") as temporary:
        output = Path(temporary) / "transverse_sweet_spot.json"
        certification = _run_transverse_certifier(config, manifest, output)
        return _complete_run(config, plan, manifest, certification)


__all__ = [
    "FixedCasimirConfig",
    "FixedCasimirExecutionError",
    "FixedCasimirResult",
    "run_casimir",
]
