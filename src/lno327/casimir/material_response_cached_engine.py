"""Read-through persistent-cache orchestration for material response ladders.

The TODO 2 engine remains cache-agnostic. This wrapper performs exact identity
preflight, loads established responses, evaluates only missing Matsubara
frequencies, and persists only successful response-level certifications.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.material_response_cache_identity import MaterialResponseCacheIdentity
from lno327.casimir.material_response_cache_store import (
    CachedCertifiedMaterialResponse,
    MaterialResponseCacheStore,
)
from lno327.casimir.material_response_engine import (
    MaterialFrequencyResult,
    MaterialResponseEngineConfig,
    evaluate_material_response_ladder,
)
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot
from lno327.casimir.matsubara import matsubara_energy_eV
from lno327.casimir.microscopic_model import get_finite_q_microscopic_model
from lno327.electrodynamics.static_sheet import STATIC_LOCAL_BASIS
from lno327.response.arbitrary_q_formal_policy import PRIMITIVE_CONTRACT_VERSION
from lno327.response.arbitrary_q_material_cache import material_state_fingerprint
from lno327.workflows.finite_q_engine import FiniteQEngineOptions

MATERIAL_RESPONSE_CACHED_ENGINE_SCHEMA = "material-response-cached-engine-v1"


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


def _identity_context(config: MaterialResponseEngineConfig) -> dict[str, Any]:
    model = get_finite_q_microscopic_model(config.microscopic_model_name)
    ansatz = model.build_ansatz(
        config.pairing_name,
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(config.delta0_eV)
    base_config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=config.temperature_K,
        eta_eV=config.eta_eV,
        output_si=False,
    )
    options = FiniteQEngineOptions(phase_hessian_policy="q_independent")
    return {
        "material_state_fingerprint": material_state_fingerprint(
            spec=model.spec,
            ansatz=ansatz,
            pairing=pairing,
            config=base_config,
            options=options,
        ),
        "phase_hessian_policy": _phase_hessian_policy(config.pairing_name),
    }


def build_material_response_identity_context(
    config: MaterialResponseEngineConfig,
) -> Mapping[str, Any]:
    """Build one reusable q/frequency-independent cache-identity context."""

    if not isinstance(config, MaterialResponseEngineConfig):
        raise TypeError("config must be a MaterialResponseEngineConfig")
    return MappingProxyType(_identity_context(config))


def build_material_response_cache_identity(
    config: MaterialResponseEngineConfig,
    *,
    q_crystal: np.ndarray,
    matsubara_index: int,
    context: Mapping[str, Any] | None = None,
) -> MaterialResponseCacheIdentity:
    """Build one exact geometry-free physical and certification identity."""

    if not isinstance(config, MaterialResponseEngineConfig):
        raise TypeError("config must be a MaterialResponseEngineConfig")
    q = _readonly_q(q_crystal)
    index = int(matsubara_index)
    if index < 0:
        raise ValueError("matsubara_index must be non-negative")
    state = _identity_context(config) if context is None else dict(context)
    return MaterialResponseCacheIdentity(
        pairing_name=config.pairing_name,
        temperature_K=config.temperature_K,
        matsubara_index=index,
        xi_eV=matsubara_energy_eV(index, config.temperature_K),
        q_crystal=q,
        microscopic_model_name=config.microscopic_model_name,
        material_state_fingerprint=state["material_state_fingerprint"],
        response_policy_fingerprint=config.material_policy.fingerprint,
        primitive_contract_version=PRIMITIVE_CONTRACT_VERSION,
        phase_hessian_policy=state["phase_hessian_policy"],
        basis=STATIC_LOCAL_BASIS if index == 0 else "crystal_xy",
        convergence_policy=config.convergence_policy.as_dict(),
        required_consecutive_passes=config.required_consecutive_passes,
        envelope_levels=config.envelope_levels,
        n_candidates=config.n_candidates,
        shifts=config.shifts,
        canonical_reduction_block_size=config.canonical_reduction_block_size,
    )


@dataclass(frozen=True)
class CachedMaterialFrequencyResult:
    matsubara_index: int
    xi_eV: float
    cache_identity: MaterialResponseCacheIdentity
    source: str
    snapshot: MaterialResponseSnapshot | None
    microscopic_result: MaterialFrequencyResult | None

    def __post_init__(self) -> None:
        index = int(self.matsubara_index)
        if index < 0:
            raise ValueError("matsubara_index must be non-negative")
        object.__setattr__(self, "matsubara_index", index)
        xi = float(self.xi_eV)
        if not np.isfinite(xi) or xi < 0.0:
            raise ValueError("xi_eV must be finite and non-negative")
        object.__setattr__(self, "xi_eV", xi)
        if self.cache_identity.matsubara_index != index or self.cache_identity.xi_eV != xi:
            raise ValueError("frequency result differs from cache identity")
        allowed = {
            "persistent_cache_hit",
            "microscopic_certified_and_persisted",
            "microscopic_certified_cache_disabled",
            "microscopic_unresolved_not_persisted",
        }
        if self.source not in allowed:
            raise ValueError("unsupported cached material frequency source")
        if self.source == "persistent_cache_hit" and self.microscopic_result is not None:
            raise ValueError("cache hits cannot carry a microscopic result")
        if self.snapshot is not None:
            if self.snapshot.xi_eV != xi:
                raise ValueError("snapshot xi differs from frequency result")
            if self.snapshot.frequency_sector != self.cache_identity.frequency_sector:
                raise ValueError("snapshot sector differs from cache identity")

    @property
    def established(self) -> bool:
        return self.snapshot is not None


@dataclass(frozen=True)
class CachedMaterialResponseEngineResult:
    config: MaterialResponseEngineConfig
    q_crystal: np.ndarray
    frequencies: Mapping[int, CachedMaterialFrequencyResult]
    metadata: Mapping[str, Any]
    schema: str = MATERIAL_RESPONSE_CACHED_ENGINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_RESPONSE_CACHED_ENGINE_SCHEMA:
            raise ValueError("unsupported cached engine result schema")
        object.__setattr__(self, "q_crystal", _readonly_q(self.q_crystal))
        values = dict(self.frequencies)
        if set(values) != set(self.config.matsubara_indices):
            raise ValueError("cached result keys differ from requested Matsubara indices")
        object.__setattr__(self, "frequencies", MappingProxyType(values))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def all_requested_established(self) -> bool:
        return all(result.established for result in self.frequencies.values())


def evaluate_material_response_ladder_cached(
    config: MaterialResponseEngineConfig,
    *,
    q_crystal: np.ndarray,
    cache: MaterialResponseCacheStore,
) -> CachedMaterialResponseEngineResult:
    """Load exact hits and evaluate only missing Matsubara frequencies."""

    if not isinstance(config, MaterialResponseEngineConfig):
        raise TypeError("config must be a MaterialResponseEngineConfig")
    if not isinstance(cache, MaterialResponseCacheStore):
        raise TypeError("cache must be a MaterialResponseCacheStore")
    q = _readonly_q(q_crystal)
    context = _identity_context(config)
    identities = {
        index: build_material_response_cache_identity(
            config,
            q_crystal=q,
            matsubara_index=index,
            context=context,
        )
        for index in config.matsubara_indices
    }

    results: dict[int, CachedMaterialFrequencyResult] = {}
    misses: list[int] = []
    for index in config.matsubara_indices:
        artifact = cache.get(identities[index])
        if artifact is None:
            misses.append(index)
        else:
            results[index] = CachedMaterialFrequencyResult(
                matsubara_index=index,
                xi_eV=identities[index].xi_eV,
                cache_identity=identities[index],
                source="persistent_cache_hit",
                snapshot=artifact.snapshot,
                microscopic_result=None,
            )

    microscopic_frequency_count = 0
    persisted_count = 0
    unresolved_count = 0
    if misses:
        miss_config = replace(config, matsubara_indices=tuple(misses))
        microscopic = evaluate_material_response_ladder(miss_config, q_crystal=q)
        microscopic_frequency_count = len(misses)
        for index in misses:
            frequency = microscopic.frequencies[index]
            if frequency.certification is None:
                unresolved_count += 1
                results[index] = CachedMaterialFrequencyResult(
                    matsubara_index=index,
                    xi_eV=frequency.xi_eV,
                    cache_identity=identities[index],
                    source="microscopic_unresolved_not_persisted",
                    snapshot=None,
                    microscopic_result=frequency,
                )
                continue
            artifact = CachedCertifiedMaterialResponse.from_certification(
                identity=identities[index],
                certification=frequency.certification,
            )
            if cache.mode == "populate":
                artifact = cache.put(artifact)
                source = "microscopic_certified_and_persisted"
                persisted_count += 1
            else:
                source = "microscopic_certified_cache_disabled"
            results[index] = CachedMaterialFrequencyResult(
                matsubara_index=index,
                xi_eV=frequency.xi_eV,
                cache_identity=identities[index],
                source=source,
                snapshot=artifact.snapshot,
                microscopic_result=frequency,
            )

    ordered = {index: results[index] for index in config.matsubara_indices}
    metadata = {
        "casimir_stage": "geometry_independent_material_response_cache_orchestration",
        "cache_schema": "material-response-cache-v1",
        "cache_mode": cache.mode,
        "cache_hits": int(
            sum(row.source == "persistent_cache_hit" for row in ordered.values())
        ),
        "cache_misses": int(len(misses)),
        "microscopic_frequency_count": int(microscopic_frequency_count),
        "persisted_frequency_count": int(persisted_count),
        "unresolved_frequency_count": int(unresolved_count),
        "misses_only_sent_to_microscopic_engine": True,
        "geometry_inputs_present": False,
        "microscopic_fallback_in_read_only_mode": False,
        "valid_for_casimir_input": False,
        "production_casimir_allowed": False,
    }
    return CachedMaterialResponseEngineResult(
        config=config,
        q_crystal=q,
        frequencies=ordered,
        metadata=metadata,
    )


__all__ = [
    "MATERIAL_RESPONSE_CACHED_ENGINE_SCHEMA",
    "CachedMaterialFrequencyResult",
    "CachedMaterialResponseEngineResult",
    "build_material_response_cache_identity",
    "build_material_response_identity_context",
    "evaluate_material_response_ladder_cached",
]
