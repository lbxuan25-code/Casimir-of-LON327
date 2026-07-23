"""Explicit NPZ/JSON codec for certified material-response artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lno327.casimir.material_response_cache_artifact import (
    MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA,
    CachedCertifiedMaterialResponse,
    json_safe,
)
from lno327.casimir.material_response_cache_errors import (
    MaterialResponseCacheCorruptionError,
    MaterialResponseCacheError,
    MaterialResponseCacheIdentityError,
    UnsupportedMaterialResponseCacheSchema,
)
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


def artifact_arrays(
    artifact: CachedCertifiedMaterialResponse,
) -> dict[str, np.ndarray]:
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


def artifact_payload(
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
            "xi_eV_hex": float(snapshot.response.xi_eV).hex(),
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
            "source_batch_frequency_index": snapshot.source_batch_frequency_index,
            "frequency_sector": snapshot.frequency_sector,
            "xi_eV_hex": float(snapshot.xi_eV).hex(),
            "identity": snapshot.identity_payload,
            "provenance": json_safe(dict(snapshot.provenance)),
            "physical_audit": json_safe(dict(snapshot.physical_audit)),
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
            "evidence": json_safe(dict(artifact.certification_evidence)),
            "audit_provenance_by_shift": json_safe(
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


def write_cache_artifact(
    path: Path, artifact: CachedCertifiedMaterialResponse
) -> None:
    arrays = artifact_arrays(artifact)
    payload = artifact_payload(artifact, arrays)
    manifest = np.frombuffer(_manifest_bytes(payload), dtype=np.uint8).copy()
    with Path(path).open("xb") as handle:
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
        checksum = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if checksum != descriptor["sha256"]:
            raise MaterialResponseCacheCorruptionError(f"array {name} checksum mismatch")
        arrays[name] = _readonly_array(value, dtype=value.dtype)
    return arrays


def _require_false(payload: Mapping[str, Any], name: str, owner: str) -> None:
    if name not in payload or bool(payload[name]):
        raise MaterialResponseCacheCorruptionError(
            f"{owner} safety flag {name!r} is absent or true"
        )


def _validate_logical_contract(payload: Mapping[str, Any]) -> None:
    snapshot = dict(payload["snapshot"])
    certification = dict(payload["certification"])
    safety = dict(payload["safety"])
    for name in ("valid_for_casimir_input", "production_casimir_allowed"):
        _require_false(snapshot, name, "snapshot")
    for name in (
        "observable_error_budget_calibrated",
        "valid_for_casimir_input",
        "production_casimir_allowed",
    ):
        _require_false(safety, name, "artifact")
    if certification.get("status") != "response_certified_diagnostic":
        raise MaterialResponseCacheCorruptionError(
            "cache artifact certification status is not diagnostic-certified"
        )
    sector = snapshot.get("frequency_sector")
    expected_kind = "static" if sector == "zero_matsubara" else "positive"
    if sector not in {"zero_matsubara", "positive_matsubara"}:
        raise MaterialResponseCacheCorruptionError("unsupported snapshot frequency sector")
    if dict(snapshot["response"]).get("kind") != expected_kind:
        raise MaterialResponseCacheCorruptionError("response kind differs from sector")
    if dict(snapshot["validation"]).get("kind") != expected_kind:
        raise MaterialResponseCacheCorruptionError("validation kind differs from sector")


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
            xi_eV=float.fromhex(response_payload["xi_eV_hex"]),
            degeneracy=float(response_payload["degeneracy"]),
            basis=response_payload["basis"],
            metadata={"source": "persistent_certified_material_response_cache"},
        )
    return MaterialResponseSnapshot(
        frequency_index=int(data["source_batch_frequency_index"]),
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
    artifact_path = Path(path)
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            payload = _load_manifest(archive)
            if payload.get("schema") != MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA:
                raise UnsupportedMaterialResponseCacheSchema(str(payload.get("schema")))
            if payload.get("cache_schema") != MATERIAL_RESPONSE_CACHE_SCHEMA:
                raise UnsupportedMaterialResponseCacheSchema(
                    str(payload.get("cache_schema"))
                )
            _validate_logical_contract(payload)
            identity = MaterialResponseCacheIdentity.from_payload(payload["identity"])
            if payload["identity_sha256"] != identity.sha256:
                raise MaterialResponseCacheIdentityError("manifest identity SHA mismatch")
            if artifact_path.suffix == ".npz" and artifact_path.stem != identity.sha256:
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
            f"failed to load cache artifact {artifact_path}"
        ) from exc


def responses_compatible(
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


__all__ = [
    "artifact_arrays",
    "artifact_payload",
    "load_cached_certified_material_response",
    "responses_compatible",
    "write_cache_artifact",
]
