"""Canonical identity contract for persistent certified material responses."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.casimir.matsubara import matsubara_energy_eV

MATERIAL_RESPONSE_CACHE_IDENTITY_SCHEMA = "material-response-cache-identity-v1"
MATERIAL_RESPONSE_CACHE_SCHEMA = "material-response-cache-v1"
MATSUBARA_IDENTITY_CONVENTION = "finite-temperature-bosonic-matsubara-v1"


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _finite_float(value: float, name: str, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or (positive and scalar <= 0.0):
        relation = "positive" if positive else "finite"
        raise ValueError(f"{name} must be {relation}")
    return scalar


def _readonly_q(value: np.ndarray) -> np.ndarray:
    q = np.array(value, dtype=float, copy=True)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_crystal must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_crystal must be nonzero")
    q.setflags(write=False)
    return q


def _nonempty(value: object, name: str) -> str:
    text = str(value)
    if not text or text == "unspecified" or text == "None":
        raise ValueError(f"{name} must be specified")
    return text


def _validated_convergence_policy(value: Mapping[str, Any]) -> MappingProxyType:
    payload = dict(value)
    required = {
        "schema",
        "comparison_order",
        "relative_tolerance",
        "absolute_tolerance",
        "observable_error_budget_calibrated",
        "production_admission",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"convergence_policy is missing fields: {missing}")
    if payload["schema"] != "material-response-convergence-policy-v1":
        raise ValueError("unsupported convergence policy schema")
    if payload["comparison_order"] != "absolute_first_then_relative_fallback":
        raise ValueError("unsupported convergence comparison order")
    for name in ("relative_tolerance", "absolute_tolerance"):
        scalar = float(payload[name])
        if not np.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"convergence policy {name} must be finite and non-negative")
        payload[name] = scalar
    if bool(payload["observable_error_budget_calibrated"]):
        raise ValueError("TODO 3 identity cannot claim observable error calibration")
    if bool(payload["production_admission"]):
        raise ValueError("TODO 3 identity cannot claim production admission")
    payload["observable_error_budget_calibrated"] = False
    payload["production_admission"] = False
    canonical_json_bytes(payload)
    return MappingProxyType(payload)


@dataclass(frozen=True)
class MaterialResponseCacheIdentity:
    """Geometry-free exact identity of one certified response artifact.

    Distance, plate angle, laboratory momentum, outer quadrature state, worker
    count, runtime chunking, filesystem path, and telemetry are deliberately not
    representable by this type.
    """

    pairing_name: str
    temperature_K: float
    matsubara_index: int
    xi_eV: float
    q_crystal: np.ndarray
    microscopic_model_name: str
    material_state_fingerprint: str
    response_policy_fingerprint: str
    primitive_contract_version: str
    phase_hessian_policy: str
    basis: str
    convergence_policy: Mapping[str, Any]
    required_consecutive_passes: int
    envelope_levels: int
    schema: str = MATERIAL_RESPONSE_CACHE_IDENTITY_SCHEMA
    cache_schema: str = MATERIAL_RESPONSE_CACHE_SCHEMA
    matsubara_convention: str = MATSUBARA_IDENTITY_CONVENTION

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_RESPONSE_CACHE_IDENTITY_SCHEMA:
            raise ValueError("unsupported cache identity schema")
        if self.cache_schema != MATERIAL_RESPONSE_CACHE_SCHEMA:
            raise ValueError("unsupported material response cache schema")
        if self.matsubara_convention != MATSUBARA_IDENTITY_CONVENTION:
            raise ValueError("unsupported Matsubara identity convention")
        pairing = _nonempty(self.pairing_name, "pairing_name")
        object.__setattr__(self, "pairing_name", pairing)
        temperature = _finite_float(self.temperature_K, "temperature_K", positive=True)
        object.__setattr__(self, "temperature_K", temperature)
        index = int(self.matsubara_index)
        if index < 0:
            raise ValueError("matsubara_index must be non-negative")
        object.__setattr__(self, "matsubara_index", index)
        xi = _finite_float(self.xi_eV, "xi_eV")
        if xi < 0.0 or (index == 0 and xi != 0.0) or (index > 0 and xi <= 0.0):
            raise ValueError("matsubara_index and xi_eV sector are inconsistent")
        expected_xi = matsubara_energy_eV(index, temperature)
        if xi != expected_xi:
            raise ValueError("temperature_K, matsubara_index, and xi_eV are inconsistent")
        object.__setattr__(self, "xi_eV", xi)
        object.__setattr__(self, "q_crystal", _readonly_q(self.q_crystal))
        for name in (
            "microscopic_model_name",
            "material_state_fingerprint",
            "response_policy_fingerprint",
            "primitive_contract_version",
            "phase_hessian_policy",
            "basis",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "convergence_policy",
            _validated_convergence_policy(self.convergence_policy),
        )
        required = int(self.required_consecutive_passes)
        if required <= 0:
            raise ValueError("required_consecutive_passes must be positive")
        object.__setattr__(self, "required_consecutive_passes", required)
        envelope = int(self.envelope_levels)
        if envelope < 3:
            raise ValueError("envelope_levels must be at least three")
        object.__setattr__(self, "envelope_levels", envelope)

    @property
    def frequency_sector(self) -> str:
        return "zero_matsubara" if self.matsubara_index == 0 else "positive_matsubara"

    @property
    def certification_policy_payload(self) -> dict[str, Any]:
        return {
            "schema": "material-response-cache-certification-identity-v1",
            "convergence_policy": dict(self.convergence_policy),
            "required_consecutive_passes": self.required_consecutive_passes,
            "envelope_levels": self.envelope_levels,
            "algorithm": "response-adjacent-plus-complete-pairwise-envelope-v1",
        }

    @property
    def certification_policy_fingerprint(self) -> str:
        return canonical_sha256(self.certification_policy_payload)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cache_schema": self.cache_schema,
            "pairing_name": self.pairing_name,
            "temperature_K_hex": float(self.temperature_K).hex(),
            "matsubara_index": self.matsubara_index,
            "xi_eV_hex": float(self.xi_eV).hex(),
            "frequency_sector": self.frequency_sector,
            "q_crystal_hex": [float(value).hex() for value in self.q_crystal],
            "microscopic_model_name": self.microscopic_model_name,
            "material_state_fingerprint": self.material_state_fingerprint,
            "response_policy_fingerprint": self.response_policy_fingerprint,
            "primitive_contract_version": self.primitive_contract_version,
            "phase_hessian_policy": self.phase_hessian_policy,
            "basis": self.basis,
            "matsubara_convention": self.matsubara_convention,
            "certification_policy": self.certification_policy_payload,
            "certification_policy_fingerprint": (
                self.certification_policy_fingerprint
            ),
            "geometry_inputs_present": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MaterialResponseCacheIdentity":
        data = dict(payload)
        if data.get("schema") != MATERIAL_RESPONSE_CACHE_IDENTITY_SCHEMA:
            raise ValueError("unsupported cache identity payload schema")
        certification = dict(data["certification_policy"])
        identity = cls(
            pairing_name=data["pairing_name"],
            temperature_K=float.fromhex(data["temperature_K_hex"]),
            matsubara_index=int(data["matsubara_index"]),
            xi_eV=float.fromhex(data["xi_eV_hex"]),
            q_crystal=np.asarray(
                [float.fromhex(value) for value in data["q_crystal_hex"]],
                dtype=float,
            ),
            microscopic_model_name=data["microscopic_model_name"],
            material_state_fingerprint=data["material_state_fingerprint"],
            response_policy_fingerprint=data["response_policy_fingerprint"],
            primitive_contract_version=data["primitive_contract_version"],
            phase_hessian_policy=data["phase_hessian_policy"],
            basis=data["basis"],
            convergence_policy=certification["convergence_policy"],
            required_consecutive_passes=int(
                certification["required_consecutive_passes"]
            ),
            envelope_levels=int(certification["envelope_levels"]),
            cache_schema=data["cache_schema"],
            matsubara_convention=data["matsubara_convention"],
        )
        if identity.payload != data:
            raise ValueError("cache identity payload is not canonical")
        return identity


__all__ = [
    "MATERIAL_RESPONSE_CACHE_IDENTITY_SCHEMA",
    "MATERIAL_RESPONSE_CACHE_SCHEMA",
    "MATSUBARA_IDENTITY_CONVENTION",
    "MaterialResponseCacheIdentity",
    "canonical_json_bytes",
    "canonical_sha256",
]
