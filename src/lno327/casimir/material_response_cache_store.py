"""Atomic filesystem store for certified material-response snapshots.

Artifacts are content-addressed by a geometry-free exact identity.  The store
uses an explicit JSON manifest plus NumPy arrays with ``allow_pickle=False``;
partial files, identity mismatches, corruption, and concurrent conflicts fail
closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from lno327.casimir.material_response_certification import CertifiedMaterialResponse
from lno327.casimir.material_response_cache_identity import (
    MATERIAL_RESPONSE_CACHE_SCHEMA,
    MaterialResponseCacheIdentity,
    canonical_json_bytes,
    canonical_sha256,
)
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetConductivityConversion,
    SheetResponseValidation,
)
from lno327.electrodynamics.static_sheet import (
    StaticSheetResponse,
    StaticSheetValidation,
)

MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA = "material-response-cache-artifact-v1"
MaterialResponseCacheMode = Literal["disabled", "populate", "read_only"]


class MaterialResponseCacheError(RuntimeError):
    """Base class for persistent response-cache failures."""


class MaterialResponseCacheMiss(MaterialResponseCacheError):
    pass


class MaterialResponseCacheReadOnlyError(MaterialResponseCacheError):
    pass


class MaterialResponseCacheIdentityError(MaterialResponseCacheError):
    pass


class MaterialResponseCacheCorruptionError(MaterialResponseCacheError):
    pass


class MaterialResponseCacheConflictError(MaterialResponseCacheError):
    pass


class UnsupportedMaterialResponseCacheSchema(MaterialResponseCacheError):
    pass


class MaterialResponseCacheLockError(MaterialResponseCacheError):
    pass


def _json_safe(value: Any) -> Any:
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
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported cache manifest value: {type(value).__name__}")


def _readonly_array(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _array_descriptor(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _validation_payload(snapshot: MaterialResponseSnapshot) -> dict[str, Any]:
    validation = snapshot.sheet_validation
    if snapshot.frequency_sector == "zero_matsubara":
        return {
            "kind": "static",
            "finite": bool(validation.finite),
            "ward_passed": bool(validation.ward_passed),
            "relative_imaginary_norm": float(validation.relative_imaginary_norm),
            "relative_longitudinal_gauge_residual": float(
                validation.relative_longitudinal_gauge_residual
            ),
            "relative_density_transverse_mixing": float(
                validation.relative_density_transverse_mixing
            ),
            "chi_bar": float(validation.chi_bar),
            "dbar_t": float(validation.dbar_t),
            "reality_tolerance": float(validation.reality_tolerance),
            "longitudinal_tolerance": float(
                validation.longitudinal_tolerance
            ),
            "mixing_tolerance": float(validation.mixing_tolerance),
            "passivity_tolerance": float(validation.passivity_tolerance),
        }
    return {
        "kind": "positive",
        "finite": bool(validation.finite),
        "relative_imaginary_norm": float(validation.relative_imaginary_norm),
        "relative_symmetry_residual": float(validation.relative_symmetry_residual),
        "minimum_symmetric_eigenvalue": float(
            validation.minimum_symmetric_eigenvalue
        ),
        "reality_tolerance": float(validation.reality_tolerance),
        "symmetry_tolerance": float(validation.symmetry_tolerance),
        "passivity_tolerance": float(validation.passivity_tolerance),
    }


def _conversion_payload(conversion: SheetConductivityConversion) -> dict[str, Any]:
    return {
        "unit_stage": conversion.unit_stage,
        "unit_label": conversion.unit_label,
        "normalization_status": conversion.normalization_status,
        "valid_for_casimir_input": bool(conversion.valid_for_casimir_input),
        "notes": list(conversion.notes),
    }


@dataclass(frozen=True)
class CachedCertifiedMaterialResponse:
    """One persisted response certification and its geometry-facing snapshot."""

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
        evidence = _json_safe(dict(self.certification_evidence))
        provenance = _json_safe(dict(self.audit_provenance_by_shift))
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


def _artifact_arrays(artifact: CachedCertifiedMaterialResponse) -> dict[str, np.ndarray]:
    snapshot = artifact.snapshot
    arrays: dict[str, np.ndarray] = {
        "q_crystal": np.asarray(snapshot.q_crystal, dtype=np.float64),
    }
    if snapshot.frequency_sector == "zero_matsubara":
        arrays["static_kernel_lt"] = np.asarray(
            snapshot.response.kernel_lt, dtype=np.complex128
        )
    else:
        arrays.update(
            {
                "sigma_model_xy": np.asarray(
                    snapshot.response.matrix_model, dtype=np.complex128
                ),
                "sigma_sheet_si_xy": np.asarray(
                    snapshot.response.matrix_sheet_si, dtype=np.complex128
                ),
                "sigma_tilde_xy": np.asarray(
                    snapshot.response.matrix_tilde, dtype=np.complex128
                ),
            }
        )
    return arrays


def _artifact_payload(
    artifact: CachedCertifiedMaterialResponse,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    snapshot = artifact.snapshot
    response_payload: dict[str, Any]
    if snapshot.frequency_sector == "zero_matsubara":
        response_payload = {
            "kind": "static",
            "chi_bar": float(snapshot.response.chi_bar),
            "dbar_t": float(snapshot.response.dbar_t),
            "energy_scale_eV": float(snapshot.response.energy_scale_eV),
            "degeneracy": float(snapshot.response.degeneracy),
            "basis": snapshot.response.basis,
        }
    else:
        response_payload = {
            "kind": "positive",
            "xi_eV": float(snapshot.response.xi_eV),
            "degeneracy": float(snapshot.response.degeneracy),
            "basis": snapshot.response.basis,
            "sheet_conversion": _conversion_payload(
                snapshot.response.sigma_sheet_si_xy
            ),
            "tilde_conversion": _conversion_payload(snapshot.response.sigma_tilde_xy),
        }
    return {
        "schema": artifact.schema,
        "cache_schema": MATERIAL_RESPONSE_CACHE_SCHEMA,
        "identity": artifact.identity.payload,
        "identity_sha256": artifact.identity.sha256,
        "snapshot": {
            "schema": snapshot.schema,
            "frequency_index": snapshot.frequency_index,
            "frequency_sector": snapshot.frequency_sector,
            "xi_eV_hex": float(snapshot.xi_eV).hex(),
            "identity": snapshot.identity_payload,
            "provenance": _json_safe(dict(snapshot.provenance)),
            "physical_audit": _json_safe(dict(snapshot.physical_audit)),
            "response": response_payload,
            "validation": _validation_payload(snapshot),
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
        "certification": {
            "working_N": artifact.working_N,
            "audit_N": artifact.audit_N,
            "primary_shift": artifact.primary_shift,
            "establishment_mode": artifact.establishment_mode,
            "evidence": _json_safe(dict(artifact.certification_evidence)),
            "audit_provenance_by_shift": _json_safe(
                dict(artifact.audit_provenance_by_shift)
            ),
            "status": "response_certified_diagnostic",
        },
        "arrays": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "safety": {
            "observable_error_budget_calibrated": False,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
    }


def _manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    envelope = {
        "payload": dict(payload),
        "payload_sha256": canonical_sha256(payload),
    }
    return canonical_json_bytes(envelope)


def _write_npz(path: Path, artifact: CachedCertifiedMaterialResponse) -> None:
    arrays = _artifact_arrays(artifact)
    payload = _artifact_payload(artifact, arrays)
    manifest = np.frombuffer(_manifest_bytes(payload), dtype=np.uint8).copy()
    with path.open("xb") as handle:
        np.savez_compressed(handle, manifest_json_utf8=manifest, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _load_manifest(archive: Any) -> dict[str, Any]:
    try:
        manifest_array = np.asarray(archive["manifest_json_utf8"], dtype=np.uint8)
        envelope = json.loads(manifest_array.tobytes().decode("utf-8"))
        payload = dict(envelope["payload"])
        if envelope["payload_sha256"] != canonical_sha256(payload):
            raise MaterialResponseCacheCorruptionError("manifest checksum mismatch")
        return payload
    except MaterialResponseCacheError:
        raise
    except Exception as exc:
        raise MaterialResponseCacheCorruptionError(
            "invalid material response cache manifest"
        ) from exc


def _validated_arrays(archive: Any, payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    descriptors = dict(payload["arrays"])
    expected_names = {"manifest_json_utf8", *descriptors}
    if set(archive.files) != expected_names:
        raise MaterialResponseCacheCorruptionError("cache array names differ from manifest")
    arrays: dict[str, np.ndarray] = {}
    for name, descriptor in descriptors.items():
        value = np.asarray(archive[name])
        if value.dtype.str != descriptor["dtype"]:
            raise MaterialResponseCacheCorruptionError(f"array {name} dtype mismatch")
        if list(value.shape) != list(descriptor["shape"]):
            raise MaterialResponseCacheCorruptionError(f"array {name} shape mismatch")
        if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != descriptor["sha256"]:
            raise MaterialResponseCacheCorruptionError(f"array {name} checksum mismatch")
        arrays[name] = _readonly_array(value, dtype=value.dtype)
    return arrays


def _conversion_from_payload(
    matrix: np.ndarray, payload: Mapping[str, Any]
) -> SheetConductivityConversion:
    return SheetConductivityConversion(
        tensor=ConductivityTensor(
            matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0]
        ),
        unit_stage=payload["unit_stage"],
        unit_label=payload["unit_label"],
        normalization_status=payload["normalization_status"],
        valid_for_casimir_input=bool(payload["valid_for_casimir_input"]),
        notes=tuple(payload["notes"]),
    )


def _snapshot_from_payload(
    payload: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> MaterialResponseSnapshot:
    data = dict(payload["snapshot"])
    q = arrays["q_crystal"]
    validation_payload = dict(data["validation"])
    response_payload = dict(data["response"])
    if data["frequency_sector"] == "zero_matsubara":
        validation = StaticSheetValidation(
            finite=bool(validation_payload["finite"]),
            ward_passed=bool(validation_payload["ward_passed"]),
            relative_imaginary_norm=float(
                validation_payload["relative_imaginary_norm"]
            ),
            relative_longitudinal_gauge_residual=float(
                validation_payload["relative_longitudinal_gauge_residual"]
            ),
            relative_density_transverse_mixing=float(
                validation_payload["relative_density_transverse_mixing"]
            ),
            chi_bar=float(validation_payload["chi_bar"]),
            dbar_t=float(validation_payload["dbar_t"]),
            reality_tolerance=float(validation_payload["reality_tolerance"]),
            longitudinal_tolerance=float(
                validation_payload["longitudinal_tolerance"]
            ),
            mixing_tolerance=float(validation_payload["mixing_tolerance"]),
            passivity_tolerance=float(validation_payload["passivity_tolerance"]),
        )
        response = StaticSheetResponse(
            kernel_lt=arrays["static_kernel_lt"],
            chi_bar=float(response_payload["chi_bar"]),
            dbar_t=float(response_payload["dbar_t"]),
            q_model=q,
            energy_scale_eV=float(response_payload["energy_scale_eV"]),
            degeneracy=float(response_payload["degeneracy"]),
            basis=response_payload["basis"],
            validation=validation,
            metadata={
                "source": "persistent_certified_material_response_cache",
                "conductivity_division_forbidden": True,
            },
        )
    else:
        validation = SheetResponseValidation(
            finite=bool(validation_payload["finite"]),
            relative_imaginary_norm=float(
                validation_payload["relative_imaginary_norm"]
            ),
            relative_symmetry_residual=float(
                validation_payload["relative_symmetry_residual"]
            ),
            minimum_symmetric_eigenvalue=float(
                validation_payload["minimum_symmetric_eigenvalue"]
            ),
            reality_tolerance=float(validation_payload["reality_tolerance"]),
            symmetry_tolerance=float(validation_payload["symmetry_tolerance"]),
            passivity_tolerance=float(validation_payload["passivity_tolerance"]),
        )
        model_matrix = arrays["sigma_model_xy"]
        model_tensor = ConductivityTensor(
            model_matrix[0, 0],
            model_matrix[1, 1],
            model_matrix[0, 1],
            model_matrix[1, 0],
        )
        response = PositiveMatsubaraSheetResponse(
            sigma_model_xy=model_tensor,
            sigma_sheet_si_xy=_conversion_from_payload(
                arrays["sigma_sheet_si_xy"], response_payload["sheet_conversion"]
            ),
            sigma_tilde_xy=_conversion_from_payload(
                arrays["sigma_tilde_xy"], response_payload["tilde_conversion"]
            ),
            q_model=q,
            xi_eV=float(response_payload["xi_eV"]),
            degeneracy=float(response_payload["degeneracy"]),
            basis=response_payload["basis"],
            metadata={"source": "persistent_certified_material_response_cache"},
        )
    return MaterialResponseSnapshot(
        frequency_index=int(data["frequency_index"]),
        frequency_sector=data["frequency_sector"],
        q_crystal=q,
        xi_eV=float.fromhex(data["xi_eV_hex"]),
        response=response,
        sheet_validation=validation,
        identity=data["identity"],
        provenance=data["provenance"],
        physical_audit=data["physical_audit"],
    )


def load_cached_certified_material_response(
    path: Path,
    *,
    expected_identity: MaterialResponseCacheIdentity | None = None,
) -> CachedCertifiedMaterialResponse:
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = _load_manifest(archive)
            if payload.get("schema") != MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA:
                raise UnsupportedMaterialResponseCacheSchema(str(payload.get("schema")))
            if payload.get("cache_schema") != MATERIAL_RESPONSE_CACHE_SCHEMA:
                raise UnsupportedMaterialResponseCacheSchema(
                    str(payload.get("cache_schema"))
                )
            identity = MaterialResponseCacheIdentity.from_payload(payload["identity"])
            if payload["identity_sha256"] != identity.sha256:
                raise MaterialResponseCacheIdentityError("manifest identity SHA mismatch")
            if path.suffix == ".npz" and path.stem != identity.sha256:
                raise MaterialResponseCacheIdentityError("cache filename SHA mismatch")
            if expected_identity is not None and identity.payload != expected_identity.payload:
                raise MaterialResponseCacheIdentityError("requested cache identity mismatch")
            arrays = _validated_arrays(archive, payload)
            snapshot = _snapshot_from_payload(payload, arrays)
            certification = dict(payload["certification"])
            return CachedCertifiedMaterialResponse(
                identity=identity,
                snapshot=snapshot,
                working_N=int(certification["working_N"]),
                audit_N=int(certification["audit_N"]),
                primary_shift=certification["primary_shift"],
                establishment_mode=certification["establishment_mode"],
                certification_evidence=certification["evidence"],
                audit_provenance_by_shift=certification[
                    "audit_provenance_by_shift"
                ],
            )
    except MaterialResponseCacheError:
        raise
    except Exception as exc:
        raise MaterialResponseCacheCorruptionError(
            f"failed to load cache artifact {path}"
        ) from exc


def _responses_compatible(
    left: MaterialResponseSnapshot,
    right: MaterialResponseSnapshot,
    *,
    identity: MaterialResponseCacheIdentity,
) -> bool:
    if left.frequency_sector != right.frequency_sector:
        return False
    policy = dict(identity.convergence_policy)
    absolute_tolerance = float(policy["absolute_tolerance"])
    relative_tolerance = float(policy["relative_tolerance"])
    if left.frequency_sector == "zero_matsubara":
        pairs = (
            (float(left.response.chi_bar), float(right.response.chi_bar)),
            (float(left.response.dbar_t), float(right.response.dbar_t)),
        )
        for first, second in pairs:
            absolute = abs(second - first)
            scale = max(abs(first), abs(second), np.finfo(float).tiny)
            if absolute > absolute_tolerance and absolute / scale > relative_tolerance:
                return False
        return True
    difference = float(
        np.linalg.norm(right.response.matrix_tilde - left.response.matrix_tilde, ord=2)
    )
    scale = max(
        float(np.linalg.norm(left.response.matrix_tilde, ord=2)),
        float(np.linalg.norm(right.response.matrix_tilde, ord=2)),
        np.finfo(float).tiny,
    )
    return bool(
        difference <= absolute_tolerance or difference / scale <= relative_tolerance
    )


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class MaterialResponseCacheStore:
    root: Path
    mode: MaterialResponseCacheMode = "populate"

    def __post_init__(self) -> None:
        root = Path(self.root)
        mode = str(self.mode)
        if mode not in {"disabled", "populate", "read_only"}:
            raise ValueError("cache mode must be disabled, populate, or read_only")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "mode", mode)

    def path_for(self, identity: MaterialResponseCacheIdentity) -> Path:
        digest = identity.sha256
        return (
            self.root
            / MATERIAL_RESPONSE_CACHE_SCHEMA
            / digest[:2]
            / digest[2:4]
            / f"{digest}.npz"
        )

    def get(
        self, identity: MaterialResponseCacheIdentity
    ) -> CachedCertifiedMaterialResponse | None:
        if self.mode == "disabled":
            return None
        path = self.path_for(identity)
        if not path.is_file():
            if self.mode == "read_only":
                raise MaterialResponseCacheMiss(
                    f"certified material response cache miss: {identity.sha256}"
                )
            return None
        return load_cached_certified_material_response(
            path, expected_identity=identity
        )

    def put(
        self, artifact: CachedCertifiedMaterialResponse
    ) -> CachedCertifiedMaterialResponse:
        if self.mode != "populate":
            raise MaterialResponseCacheReadOnlyError(
                f"cache mode {self.mode!r} does not permit writes"
            )
        final_path = self.path_for(artifact.identity)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{artifact.identity.sha256}.",
            suffix=".tmp",
            dir=final_path.parent,
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        temp_path.unlink()
        lock_path = final_path.with_suffix(".lock")
        lock_descriptor: int | None = None
        try:
            _write_npz(temp_path, artifact)
            load_cached_certified_material_response(
                temp_path, expected_identity=artifact.identity
            )
            try:
                lock_descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                os.write(
                    lock_descriptor,
                    f"pid={os.getpid()}\nidentity={artifact.identity.sha256}\n".encode(
                        "utf-8"
                    ),
                )
                os.fsync(lock_descriptor)
            except FileExistsError as exc:
                raise MaterialResponseCacheLockError(
                    f"cache identity lock already exists: {lock_path}"
                ) from exc

            if final_path.exists():
                existing = load_cached_certified_material_response(
                    final_path, expected_identity=artifact.identity
                )
                if not _responses_compatible(
                    existing.snapshot,
                    artifact.snapshot,
                    identity=artifact.identity,
                ):
                    raise MaterialResponseCacheConflictError(
                        "existing certified response conflicts with new response"
                    )
                return existing

            os.replace(temp_path, final_path)
            _fsync_directory(final_path.parent)
            return load_cached_certified_material_response(
                final_path, expected_identity=artifact.identity
            )
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA",
    "CachedCertifiedMaterialResponse",
    "MaterialResponseCacheConflictError",
    "MaterialResponseCacheCorruptionError",
    "MaterialResponseCacheError",
    "MaterialResponseCacheIdentityError",
    "MaterialResponseCacheLockError",
    "MaterialResponseCacheMiss",
    "MaterialResponseCacheMode",
    "MaterialResponseCacheReadOnlyError",
    "MaterialResponseCacheStore",
    "UnsupportedMaterialResponseCacheSchema",
    "load_cached_certified_material_response",
]
