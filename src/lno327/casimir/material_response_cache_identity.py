"""Canonical identity contract for persistent certified material responses."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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


def _normalize_n_candidates(values: Sequence[int]) -> tuple[int, ...]:
    levels = tuple(int(value) for value in values)
    if (
        len(levels) < 2
        or tuple(sorted(set(levels))) != levels
        or any(value <= 0 or value % 2 for value in levels)
    ):
        raise ValueError(
            "n_candidates must be strictly increasing unique positive even integers"
        )
    return levels


def _normalize_shifts(
    values: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    shifts: list[tuple[float, float]] = []
    for raw in values:
        if len(raw) != 2:
            raise ValueError("every certification shift must contain two values")
        shift = (float(raw[0]), float(raw[1]))
        if not np.isfinite(shift).all() or any(value < 0.0 or value >= 1.0 for value in shift):
            raise ValueError("certification shifts must be finite and lie in [0, 1)")
        shifts.append(shift)
    normalized = tuple(shifts)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise ValueError("at least two unique certification shifts are required")
    return normalized


@dataclass(frozen=True)
class MaterialResponseCacheIdentity:
    """Exact physical and certification identity of one persisted response.

    Distance, plate angle, laboratory momentum, outer quadrature state, worker
    count, runtime chunking, filesystem path, and telemetry are deliberately not
    representable by this type. N/shift sampling and canonical reduction are
    included because they determine which response certification was requested.
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
    n_candidates: tuple[int, ...]
    shifts: tuple[tuple[float, float], ...]
    canonical_reduction_block_size: int
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
        object.__setattr__(self, "pairing_name", _nonempty(self.pairing_name, "pairing_name"))
        temperature = _finite_float(self.temperature_K, "temperature_K", positive=True)
        object.__setattr__(self, "temperature_K", temperature)
        index = int(self.matsubara_index)
        if index < 0:
            raise ValueError("matsubara_index must be non-negative")
        object.__setattr__(self, "matsubara_index", index)
        xi = _finite_float(self.xi_eV, "xi_eV")
        if xi < 0.0 or (index == 0 and xi != 0.0) or (index > 0 and xi <= 0.0):
            raise ValueError("matsubara_index and xi_eV sector are inconsistent")
        if xi != matsubara_energy_eV(index, temperature):
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
        object.__setattr__(self, "n_candidates", _normalize_n_candidates(self.n_candidates))
        object.__setattr__(self, "shifts", _normalize_shifts(self.shifts))
        reduction = int(self.canonical_reduction_block_size)
        if reduction <= 0:
            raise ValueError("canonical_reduction_block_size must be positive")
        object.__setattr__(self, "canonical_reduction_block_size", reduction)

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
            "n_candidates": list(self.n_candidates),
            "shifts_hex": [
                [float(component).hex() for component in shift] for shift in self.shifts
            ],
            "shift_order_semantics": "ordered_exact_periodic_bz_shifts-v1",
            "canonical_reduction_block_size": self.canonical_reduction_block_size,
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
            "certification_policy_fingerprint": self.certification_policy_fingerprint,
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
            required_consecutive_passes=int(certification["required_consecutive_passes"]),
            envelope_levels=int(certification["envelope_levels"]),
            n_candidates=tuple(int(value) for value in certification["n_candidates"]),
            shifts=tuple(
                tuple(float.fromhex(component) for component in shift)
                for shift in certification["shifts_hex"]
            ),
            canonical_reduction_block_size=int(
                certification["canonical_reduction_block_size"]
            ),
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
