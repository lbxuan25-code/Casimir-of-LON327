"""Cache-request identity construction without response integration orchestration.

This module translates a geometry-independent material-response configuration
into the exact TODO 3 cache identities needed by cache population or geometry
planning. It deliberately does not import the material response engine,
reflection, propagation, logdet, or outer integration.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.material_response_cache_identity import (
    MaterialResponseCacheIdentity,
)
from lno327.casimir.matsubara import matsubara_energy_eV
from lno327.casimir.microscopic_model import get_finite_q_microscopic_model
from lno327.electrodynamics.static_sheet import STATIC_LOCAL_BASIS
from lno327.response.arbitrary_q_formal_policy import PRIMITIVE_CONTRACT_VERSION
from lno327.response.arbitrary_q_material_cache import material_state_fingerprint
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

MATERIAL_RESPONSE_CACHE_REQUEST_SCHEMA = "material-response-cache-request-v1"

_REQUIRED_CONFIG_ATTRIBUTES = (
    "pairing_name",
    "temperature_K",
    "delta0_eV",
    "eta_eV",
    "microscopic_model_name",
    "material_policy",
    "convergence_policy",
    "required_consecutive_passes",
    "envelope_levels",
    "n_candidates",
    "shifts",
    "canonical_reduction_block_size",
)


def _require_response_config(config: object) -> object:
    missing = [name for name in _REQUIRED_CONFIG_ATTRIBUTES if not hasattr(config, name)]
    if missing:
        raise TypeError(
            "config does not implement the material response identity contract: "
            f"missing {missing}"
        )
    as_dict = getattr(config, "as_dict", None)
    if not callable(as_dict):
        raise TypeError("config must provide a callable as_dict()")
    payload = dict(as_dict())
    if payload.get("schema") != "material-response-engine-config-v1":
        raise TypeError("config uses an unsupported material response schema")
    return config


def _readonly_q(value: np.ndarray) -> np.ndarray:
    q = np.array(value, dtype=float, copy=True)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_crystal must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_crystal must be nonzero")
    q.setflags(write=False)
    return q


def _phase_hessian_policy(pairing_name: str) -> str:
    return "nearest_neighbor_bond_metric" if pairing_name == "dwave" else "q_independent"


def build_material_response_identity_context(config: object) -> Mapping[str, Any]:
    """Build the q/frequency-independent part of a response cache request once."""

    request = _require_response_config(config)
    model = get_finite_q_microscopic_model(str(request.microscopic_model_name))
    ansatz = model.build_ansatz(
        str(request.pairing_name),
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(float(request.delta0_eV))
    base_config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=float(request.temperature_K),
        eta_eV=float(request.eta_eV),
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    return MappingProxyType(
        {
            "schema": MATERIAL_RESPONSE_CACHE_REQUEST_SCHEMA,
            "material_state_fingerprint": material_state_fingerprint(
                spec=model.spec,
                ansatz=ansatz,
                pairing=pairing,
                config=base_config,
                options=options,
            ),
            "phase_hessian_policy": _phase_hessian_policy(
                str(request.pairing_name)
            ),
        }
    )


def build_material_response_cache_identity(
    config: object,
    *,
    q_crystal: np.ndarray,
    matsubara_index: int,
    context: Mapping[str, Any] | None = None,
) -> MaterialResponseCacheIdentity:
    """Build one exact geometry-free physical and certification identity."""

    request = _require_response_config(config)
    q = _readonly_q(q_crystal)
    index = int(matsubara_index)
    if index < 0:
        raise ValueError("matsubara_index must be non-negative")
    state = (
        build_material_response_identity_context(request)
        if context is None
        else MappingProxyType(dict(context))
    )
    if state.get("schema") != MATERIAL_RESPONSE_CACHE_REQUEST_SCHEMA:
        raise ValueError("identity context uses an unsupported schema")
    return MaterialResponseCacheIdentity(
        pairing_name=str(request.pairing_name),
        temperature_K=float(request.temperature_K),
        matsubara_index=index,
        xi_eV=matsubara_energy_eV(index, float(request.temperature_K)),
        q_crystal=q,
        microscopic_model_name=str(request.microscopic_model_name),
        material_state_fingerprint=str(state["material_state_fingerprint"]),
        response_policy_fingerprint=str(request.material_policy.fingerprint),
        primitive_contract_version=PRIMITIVE_CONTRACT_VERSION,
        phase_hessian_policy=str(state["phase_hessian_policy"]),
        basis=STATIC_LOCAL_BASIS if index == 0 else "crystal_xy",
        convergence_policy=request.convergence_policy.as_dict(),
        required_consecutive_passes=int(request.required_consecutive_passes),
        envelope_levels=int(request.envelope_levels),
        n_candidates=tuple(int(value) for value in request.n_candidates),
        shifts=tuple(
            tuple(float(component) for component in shift)
            for shift in request.shifts
        ),
        canonical_reduction_block_size=int(
            request.canonical_reduction_block_size
        ),
    )


__all__ = [
    "MATERIAL_RESPONSE_CACHE_REQUEST_SCHEMA",
    "build_material_response_cache_identity",
    "build_material_response_identity_context",
]
