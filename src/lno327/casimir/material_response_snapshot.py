"""Serializable geometry-facing snapshots of certified material responses.

The live ``MaterialResponseSample`` carries microscopic kernels and Ward objects
that are useful during certification but unnecessary for later geometry
assembly. This module defines a compact immutable boundary that preserves the
physical sheet response, validation result, identity, provenance, and audit
summary without pretending to reconstruct microscopic state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np

from lno327.casimir.material_response import (
    MATERIAL_RESPONSE_IDENTITY_SCHEMA,
    MaterialResponseSample,
)
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetResponseValidation,
)
from lno327.electrodynamics.static_sheet import (
    StaticSheetResponse,
    StaticSheetValidation,
)

MATERIAL_RESPONSE_SNAPSHOT_SCHEMA = "material-response-snapshot-v1"
SnapshotSheetResponse: TypeAlias = StaticSheetResponse | PositiveMatsubaraSheetResponse
SnapshotSheetValidation: TypeAlias = StaticSheetValidation | SheetResponseValidation


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _readonly_q(value: np.ndarray) -> np.ndarray:
    q = np.array(value, dtype=float, copy=True)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("q_crystal must be a finite vector with shape (2,)")
    if float(np.linalg.norm(q)) == 0.0:
        raise ValueError("q_crystal must be nonzero")
    q.setflags(write=False)
    return q


def _require_identity_fields(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "frequency_sector",
        "xi_eV_hex",
        "q_crystal_hex",
        "material_state_fingerprint",
        "response_policy_fingerprint",
        "primitive_contract_version",
        "phase_hessian_policy",
        "basis",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"material response identity is missing fields: {missing}")
    if payload["schema"] != MATERIAL_RESPONSE_IDENTITY_SCHEMA:
        raise ValueError("unsupported material response identity schema")
    for name in (
        "material_state_fingerprint",
        "response_policy_fingerprint",
        "primitive_contract_version",
        "phase_hessian_policy",
        "basis",
    ):
        value = str(payload[name])
        if not value or value == "unspecified" or value == "None":
            raise ValueError(f"material response identity field {name!r} is unspecified")


@dataclass(frozen=True)
class MaterialResponseSnapshot:
    """Compact immutable response that can be loaded without microscopic objects.

    ``source_batch_frequency_index`` is only the response's position inside the
    microscopic integration batch that produced it. It is deliberately not
    called a Matsubara index; the physical Matsubara index lives in the persistent
    cache identity.
    """

    source_batch_frequency_index: int
    frequency_sector: str
    q_crystal: np.ndarray
    xi_eV: float
    response: SnapshotSheetResponse
    sheet_validation: SnapshotSheetValidation
    identity: Mapping[str, Any]
    provenance: Mapping[str, Any]
    physical_audit: Mapping[str, Any]
    schema: str = MATERIAL_RESPONSE_SNAPSHOT_SCHEMA
    valid_for_casimir_input: bool = False
    production_casimir_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_RESPONSE_SNAPSHOT_SCHEMA:
            raise ValueError(f"schema must be {MATERIAL_RESPONSE_SNAPSHOT_SCHEMA!r}")
        source_index = int(self.source_batch_frequency_index)
        if source_index < 0:
            raise ValueError("source_batch_frequency_index must be non-negative")
        object.__setattr__(self, "source_batch_frequency_index", source_index)

        sector = str(self.frequency_sector)
        if sector not in {"zero_matsubara", "positive_matsubara"}:
            raise ValueError("unsupported frequency_sector")
        object.__setattr__(self, "frequency_sector", sector)
        q = _readonly_q(self.q_crystal)
        object.__setattr__(self, "q_crystal", q)
        xi = float(self.xi_eV)
        if not np.isfinite(xi) or xi < 0.0:
            raise ValueError("xi_eV must be finite and non-negative")
        object.__setattr__(self, "xi_eV", xi)

        identity = dict(self.identity)
        _require_identity_fields(identity)
        if identity["frequency_sector"] != sector:
            raise ValueError("identity frequency sector differs from snapshot")
        if identity["xi_eV_hex"] != xi.hex():
            raise ValueError("identity xi differs from snapshot")
        if list(identity["q_crystal_hex"]) != [float(value).hex() for value in q]:
            raise ValueError("identity q_crystal differs from snapshot")
        object.__setattr__(self, "identity", MappingProxyType(identity))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        audit = dict(self.physical_audit)
        if "hard_physical_passed" not in audit:
            raise ValueError("physical_audit must record hard_physical_passed")
        object.__setattr__(self, "physical_audit", MappingProxyType(audit))

        if not np.array_equal(np.asarray(self.response.q_model, dtype=float), q):
            raise ValueError("sheet response q_model differs from q_crystal")
        if sector == "zero_matsubara":
            if xi != 0.0:
                raise ValueError("zero_matsubara snapshot requires xi_eV == 0")
            if not isinstance(self.response, StaticSheetResponse):
                raise TypeError("zero_matsubara snapshot requires StaticSheetResponse")
            if not isinstance(self.sheet_validation, StaticSheetValidation):
                raise TypeError("zero_matsubara snapshot requires StaticSheetValidation")
        else:
            if xi <= 0.0:
                raise ValueError("positive_matsubara snapshot requires xi_eV > 0")
            if not isinstance(self.response, PositiveMatsubaraSheetResponse):
                raise TypeError(
                    "positive_matsubara snapshot requires PositiveMatsubaraSheetResponse"
                )
            if not isinstance(self.sheet_validation, SheetResponseValidation):
                raise TypeError(
                    "positive_matsubara snapshot requires SheetResponseValidation"
                )
            if float(self.response.xi_eV) != xi:
                raise ValueError("positive response xi differs from snapshot")

        if bool(self.valid_for_casimir_input):
            raise ValueError("TODO 3 snapshots cannot claim valid_for_casimir_input")
        if bool(self.production_casimir_allowed):
            raise ValueError("TODO 3 snapshots cannot admit production Casimir")
        object.__setattr__(self, "valid_for_casimir_input", False)
        object.__setattr__(self, "production_casimir_allowed", False)

    @property
    def identity_payload(self) -> dict[str, Any]:
        return dict(self.identity)

    @property
    def identity_fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.identity)).hexdigest()

    @property
    def hard_physical_passed(self) -> bool:
        return bool(
            self.physical_audit["hard_physical_passed"]
            and self.sheet_validation.passed
        )

    @property
    def primary_matrix(self) -> np.ndarray:
        if isinstance(self.response, StaticSheetResponse):
            matrix = np.diag(
                [float(self.response.chi_bar), float(self.response.dbar_t)]
            ).astype(complex)
        else:
            matrix = np.array(self.response.matrix_tilde, dtype=complex, copy=True)
        matrix.setflags(write=False)
        return matrix

    def diagnostics(self) -> dict[str, Any]:
        state = {
            "material_response_schema": self.schema,
            "material_identity": self.identity_payload,
            "material_identity_fingerprint": self.identity_fingerprint,
            "material_provenance": dict(self.provenance),
            "source_batch_frequency_index": self.source_batch_frequency_index,
            "frequency_sector": self.frequency_sector,
            "xi_eV": self.xi_eV,
            "q_crystal": self.q_crystal.tolist(),
            "material_hard_physical_passed": self.hard_physical_passed,
            "sheet_validation_passed": bool(self.sheet_validation.passed),
            "primary_norm": float(np.linalg.norm(self.primary_matrix)),
            "snapshot_loaded_without_microscopic_state": True,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        }
        state.update(dict(self.physical_audit))
        return state

    @classmethod
    def from_sample(cls, sample: MaterialResponseSample) -> "MaterialResponseSnapshot":
        if not isinstance(sample, MaterialResponseSample):
            raise TypeError("sample must be a MaterialResponseSample")
        identity = sample.identity_payload
        _require_identity_fields(identity)
        diagnostics = sample.diagnostics()
        audit_names = (
            "operator_ward_passed",
            "ward_passed",
            "ward_effective_mixed_ratio_max",
            "schur_condition_number",
            "sheet_validation_passed",
            "material_hard_physical_passed",
            "strict_static_ward_passed",
            "strict_static_hard_gate",
            "static_longitudinal_warning",
        )
        physical_audit = {
            "hard_physical_passed": bool(sample.hard_physical_passed),
            **{name: diagnostics[name] for name in audit_names if name in diagnostics},
        }
        return cls(
            source_batch_frequency_index=sample.frequency_index,
            frequency_sector=sample.frequency_sector,
            q_crystal=sample.q_crystal,
            xi_eV=sample.xi_eV,
            response=sample.response,
            sheet_validation=sample.sheet_validation,
            identity=identity,
            provenance=sample.provenance_payload,
            physical_audit=physical_audit,
        )


GeometryMaterialResponse: TypeAlias = MaterialResponseSample | MaterialResponseSnapshot


def require_geometry_material_response(value: object) -> GeometryMaterialResponse:
    if not isinstance(value, (MaterialResponseSample, MaterialResponseSnapshot)):
        raise TypeError(
            "material response must be a MaterialResponseSample or MaterialResponseSnapshot"
        )
    return value


__all__ = [
    "GeometryMaterialResponse",
    "MATERIAL_RESPONSE_SNAPSHOT_SCHEMA",
    "MaterialResponseSnapshot",
    "SnapshotSheetResponse",
    "SnapshotSheetValidation",
    "require_geometry_material_response",
]
