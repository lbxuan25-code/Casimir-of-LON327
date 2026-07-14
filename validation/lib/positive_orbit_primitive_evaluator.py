"""Common complete-orbit positive-Matsubara primitive evaluator.

The microscopic execution path is identical for ``spm`` and ``dwave``.  The only
pairing-specific choice retained here is the post-integral phase-Hessian policy:
``q_independent`` for ``spm`` and ``nearest_neighbor_bond_metric`` for ``dwave``.
"""
from __future__ import annotations

from threading import Lock
from typing import Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.dwave_orbit_primitive_evaluator import (
    DWaveOrbitEvaluatorProfile,
    DWaveOrbitPrimitiveEvaluator,
)

PositiveOrbitEvaluatorProfile = DWaveOrbitEvaluatorProfile


class PositiveOrbitPrimitiveEvaluator(DWaveOrbitPrimitiveEvaluator):
    """Evaluate complete commensurate orbits for spm or d-wave pairing.

    All heavy local evaluation, fork-process execution, profiling aggregation, and
    lifecycle handling are inherited from the established batched evaluator.  This
    constructor only relaxes the historical d-wave-only validation and records the
    correct post-integral phase-Hessian policy.
    """

    def __init__(
        self,
        *,
        spec: object,
        ansatz: object,
        pairing: object,
        xi_eV_values: Sequence[float] | np.ndarray,
        temperature_K: float,
        eta_eV: float,
        nk: int,
        mx: int,
        my: int,
        process_workers: int = 1,
    ) -> None:
        xi_values = np.asarray(xi_eV_values, dtype=float)
        if xi_values.ndim != 1 or xi_values.size == 0:
            raise ValueError("xi_eV_values must be a nonempty one-dimensional array")
        if not np.isfinite(xi_values).all() or np.any(xi_values <= 0.0):
            raise ValueError("all xi_eV_values must be finite and positive")

        pairing_name = str(getattr(ansatz, "name", ""))
        if pairing_name not in {"spm", "dwave"}:
            raise ValueError("complete-orbit positive evaluator supports spm and dwave")
        if getattr(ansatz, "phase_vertex", None) != "bond_endpoint_gauge":
            raise ValueError("positive orbit evaluator requires bond_endpoint_gauge")
        if int(nk) <= 0 or (int(mx) == 0 and int(my) == 0):
            raise ValueError("nk must be positive and q grid indices must be nonzero")
        workers = int(process_workers)
        if workers <= 0:
            raise ValueError("process_workers must be positive")

        self.spec = spec
        self.ansatz = ansatz
        self.pairing = pairing
        self.pairing_name = pairing_name
        self.phase_hessian_policy = (
            "nearest_neighbor_bond_metric"
            if pairing_name == "dwave"
            else "q_independent"
        )
        self.xi_values = np.array(xi_values, copy=True)
        self.xi_values.setflags(write=False)
        self.q_model = (2.0 * np.pi / float(nk)) * np.asarray(
            [int(mx), int(my)], dtype=float
        )
        self.q_model.setflags(write=False)
        self.base_config = KuboConfig.from_kelvin(
            omega_eV=float(self.xi_values[0]),
            temperature_K=float(temperature_K),
            eta_eV=float(eta_eV),
            output_si=False,
        )
        # Primitive workspaces always use the unmodified q-independent saddle.  The
        # selected policy is pulled back only after the full transverse integral.
        self.options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
        self.process_workers = workers

        self._profile_lock = Lock()
        self._submit_lock = Lock()
        self._callbacks = 0
        self._complete_orbit_points = 0
        self._material_workspace_implementation = "not_evaluated"
        self._q_workspace_implementation = "not_evaluated"
        self._material_workspace_seconds = 0.0
        self._q_workspace_seconds = 0.0
        self._kubo_factor_seconds = 0.0
        self._kubo_contraction_seconds = 0.0
        self._primitive_packing_seconds = 0.0
        self._process_pool = None
        self._fork_guard_held = False
        self._closed = False

        if workers > 1:
            self._start_process_pool()

    def __enter__(self) -> "PositiveOrbitPrimitiveEvaluator":
        return self


__all__ = [
    "PositiveOrbitEvaluatorProfile",
    "PositiveOrbitPrimitiveEvaluator",
]
