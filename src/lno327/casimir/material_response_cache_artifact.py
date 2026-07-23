"""Immutable certified-response artifact independent of filesystem storage."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.casimir.material_response_certification import CertifiedMaterialResponse
from lno327.casimir.material_response_cache_errors import (
    MaterialResponseCacheIdentityError,
    UnsupportedMaterialResponseCacheSchema,
)
from lno327.casimir.material_response_cache_identity import MaterialResponseCacheIdentity
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot

MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA = "material-response-cache-artifact-v1"


def json_safe(value: Any) -> Any:
    """Convert an audit payload to deterministic JSON-safe primitive values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("cache manifest rejects non-finite floats")
        return scalar
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError("cache manifest rejects non-finite complex values")
        return {"complex_hex": [float(value.real).hex(), float(value.imag).hex()]}
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    raise TypeError(f"unsupported cache manifest value: {type(value).__name__}")


def _require_false(payload: Mapping[str, Any], name: str) -> None:
    if name not in payload:
        raise ValueError(f"certification evidence must record {name}")
    if bool(payload[name]):
        raise ValueError(f"TODO 3 artifact cannot claim {name}")


@dataclass(frozen=True)
class CachedCertifiedMaterialResponse:
    """One certified response plus the evidence needed for later audit."""

    identity: MaterialResponseCacheIdentity
    snapshot: MaterialResponseSnapshot
    working_N: int
    audit_N: int
    primary_shift: str
    establishment_mode: str
    certification_evidence: Mapping[str, Any]
    audit_provenance_by_shift: Mapping[str, Any]
    schema: str = MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA:
            raise UnsupportedMaterialResponseCacheSchema(self.schema)
        if not isinstance(self.identity, MaterialResponseCacheIdentity):
            raise TypeError("identity must be a MaterialResponseCacheIdentity")
        if not isinstance(self.snapshot, MaterialResponseSnapshot):
            raise TypeError("snapshot must be a MaterialResponseSnapshot")

        working = int(self.working_N)
        audit = int(self.audit_N)
        if working <= 0 or audit <= working:
            raise ValueError("cache artifact requires 0 < working_N < audit_N")
        object.__setattr__(self, "working_N", working)
        object.__setattr__(self, "audit_N", audit)

        primary = str(self.primary_shift)
        mode = str(self.establishment_mode)
        if not primary or not mode:
            raise ValueError("primary_shift and establishment_mode must be nonempty")
        object.__setattr__(self, "primary_shift", primary)
        object.__setattr__(self, "establishment_mode", mode)

        evidence = json_safe(dict(self.certification_evidence))
        provenance = json_safe(dict(self.audit_provenance_by_shift))
        if not isinstance(evidence, dict) or not isinstance(provenance, dict):
            raise TypeError("certification evidence and provenance must be mappings")
        for name in (
            "observable_error_budget_calibrated",
            "valid_for_casimir_input",
            "production_casimir_allowed",
        ):
            _require_false(evidence, name)
        if evidence.get("convergence_policy") != dict(self.identity.convergence_policy):
            raise MaterialResponseCacheIdentityError(
                "certification convergence policy differs from cache identity"
            )
        if int(evidence.get("required_consecutive_passes", -1)) != (
            self.identity.required_consecutive_passes
        ):
            raise MaterialResponseCacheIdentityError(
                "certification consecutive-pass policy differs from cache identity"
            )
        if primary not in provenance or len(provenance) < 2:
            raise ValueError(
                "certified cache artifact requires primary and independent audit-shift provenance"
            )
        object.__setattr__(self, "certification_evidence", MappingProxyType(evidence))
        object.__setattr__(
            self,
            "audit_provenance_by_shift",
            MappingProxyType(provenance),
        )

        if not self.snapshot.hard_physical_passed:
            raise ValueError("only hard-physical-passed responses may enter certified cache")
        sample_identity = self.snapshot.identity_payload
        expected_pairs = {
            "frequency_sector": self.identity.frequency_sector,
            "xi_eV_hex": float(self.identity.xi_eV).hex(),
            "q_crystal_hex": [float(value).hex() for value in self.identity.q_crystal],
            "material_state_fingerprint": self.identity.material_state_fingerprint,
            "response_policy_fingerprint": self.identity.response_policy_fingerprint,
            "primitive_contract_version": self.identity.primitive_contract_version,
            "phase_hessian_policy": self.identity.phase_hessian_policy,
            "basis": self.identity.basis,
        }
        for name, expected in expected_pairs.items():
            if sample_identity.get(name) != expected:
                raise MaterialResponseCacheIdentityError(
                    f"snapshot identity field {name!r} differs from cache identity"
                )
        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 3 cache artifacts cannot admit production")
        object.__setattr__(self, "production_casimir_allowed", False)

    @property
    def matsubara_index(self) -> int:
        return self.identity.matsubara_index

    @classmethod
    def from_certification(
        cls,
        *,
        identity: MaterialResponseCacheIdentity,
        certification: CertifiedMaterialResponse,
    ) -> "CachedCertifiedMaterialResponse":
        if not isinstance(certification, CertifiedMaterialResponse):
            raise TypeError("certification must be a CertifiedMaterialResponse")
        return cls(
            identity=identity,
            snapshot=MaterialResponseSnapshot.from_sample(
                certification.primary_response
            ),
            working_N=certification.working_N,
            audit_N=certification.audit_N,
            primary_shift=certification.primary_shift,
            establishment_mode=certification.establishment_mode,
            certification_evidence=certification.evidence,
            audit_provenance_by_shift=certification.evidence.get(
                "audit_provenance_by_shift", {}
            ),
        )


__all__ = [
    "MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA",
    "CachedCertifiedMaterialResponse",
    "json_safe",
]
