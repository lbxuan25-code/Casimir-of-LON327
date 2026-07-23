from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir import material_response_cache_store as cache_store_module
from lno327.casimir.material_response import (
    MATERIAL_RESPONSE_IDENTITY_SCHEMA,
    MaterialResponseSample,
)
from lno327.casimir.material_response_cache_identity import MaterialResponseCacheIdentity
from lno327.casimir.material_response_cache_store import (
    CachedCertifiedMaterialResponse,
    MaterialResponseCacheConflictError,
    MaterialResponseCacheCorruptionError,
    MaterialResponseCacheIdentityError,
    MaterialResponseCacheLockError,
    MaterialResponseCacheMiss,
    MaterialResponseCacheReadOnlyError,
    MaterialResponseCacheStore,
    load_cached_certified_material_response,
)
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot
from lno327.casimir.material_two_plate import (
    TwoPlateGeometryPolicy,
    assemble_two_plate_logdet,
)
from lno327.casimir.matsubara import matsubara_energy_eV
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetConductivityConversion,
    SheetResponseValidation,
)
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
)

DEFAULT_N_CANDIDATES = (64, 96, 128)
DEFAULT_SHIFTS = ((0.5, 0.5), (0.25, 0.75))
DEFAULT_REDUCTION_BLOCK_SIZE = 4096


def _shift_label(index: int, shift: tuple[float, float]) -> str:
    return f"shift_{index}:{shift[0].hex()}:{shift[1].hex()}"


def _ward() -> SimpleNamespace:
    side = SimpleNamespace(effective_mixed_ratio=0.0)
    return SimpleNamespace(
        passed=True,
        left=side,
        right=side,
        schur_condition_number=1.0,
    )


def _identity_payload(*, q: np.ndarray, xi: float, sector: str, basis: str) -> dict:
    return {
        "schema": MATERIAL_RESPONSE_IDENTITY_SCHEMA,
        "frequency_sector": sector,
        "xi_eV_hex": float(xi).hex(),
        "q_crystal_hex": [float(value).hex() for value in q],
        "material_state_fingerprint": "state-fingerprint",
        "response_policy_fingerprint": "response-policy-fingerprint",
        "primitive_contract_version": "primitive-v-test",
        "phase_hessian_policy": "q_independent",
        "basis": basis,
    }


def _sample(index: int) -> MaterialResponseSample:
    q = np.array([0.015, 0.025])
    xi = matsubara_energy_eV(index, 40.0)
    ward = _ward()
    if index == 0:
        validation = StaticSheetValidation(
            finite=True,
            ward_passed=True,
            relative_imaginary_norm=0.0,
            relative_longitudinal_gauge_residual=0.0,
            relative_density_transverse_mixing=0.0,
            chi_bar=1.0,
            dbar_t=2.0,
            reality_tolerance=1e-8,
            longitudinal_tolerance=1e-6,
            mixing_tolerance=1e-6,
            passivity_tolerance=1e-10,
        )
        response = StaticSheetResponse(
            kernel_lt=np.diag([-1.0, 0.0, -2.0]).astype(complex),
            chi_bar=1.0,
            dbar_t=2.0,
            q_model=q,
            energy_scale_eV=1.0,
            degeneracy=1.0,
            basis=STATIC_LOCAL_BASIS,
            validation=validation,
            metadata={"source": "test"},
        )
        sector = "zero_matsubara"
        strict = SimpleNamespace(passed=True)
        basis = STATIC_LOCAL_BASIS
    else:
        tensor = ConductivityTensor(0.4, 0.5, 0.02, 0.02)
        sheet = SheetConductivityConversion(
            tensor=tensor,
            unit_stage="sheet_conductivity",
            unit_label="test-sheet",
            normalization_status="test",
            valid_for_casimir_input=True,
            notes=("sheet",),
        )
        tilde = SheetConductivityConversion(
            tensor=tensor,
            unit_stage="reflection_dimensionless_conductivity",
            unit_label="test-tilde",
            normalization_status="test",
            valid_for_casimir_input=True,
            notes=("tilde",),
        )
        response = PositiveMatsubaraSheetResponse(
            sigma_model_xy=tensor,
            sigma_sheet_si_xy=sheet,
            sigma_tilde_xy=tilde,
            q_model=q,
            xi_eV=xi,
            degeneracy=1.0,
            basis="crystal_xy",
            metadata={"source": "test"},
        )
        validation = SheetResponseValidation(
            finite=True,
            relative_imaginary_norm=0.0,
            relative_symmetry_residual=0.0,
            minimum_symmetric_eigenvalue=0.38,
            reality_tolerance=1e-9,
            symmetry_tolerance=1e-9,
            passivity_tolerance=1e-10,
        )
        sector, strict, basis = "positive_matsubara", None, "crystal_xy"
    return MaterialResponseSample(
        frequency_index=0,
        frequency_sector=sector,
        q_crystal=q,
        xi_eV=xi,
        material_cache_fingerprint="grid-fingerprint",
        kernel=SimpleNamespace(q_model=q, xi_eV=xi),
        operator_ward=SimpleNamespace(passed=True),
        effective_ward=ward,
        strict_static_ward=strict,
        response=response,
        sheet_validation=validation,
        metadata={
            **_identity_payload(q=q, xi=xi, sector=sector, basis=basis),
            "post_integral_phase_hessian_policy": "q_independent",
            "grid_fingerprint": "grid-fingerprint",
            "grid": {"N": 64, "shift": [0.5, 0.5]},
            "canonical_reduction_block_size": DEFAULT_REDUCTION_BLOCK_SIZE,
        },
    )


def _convergence_policy() -> dict[str, object]:
    return {
        "schema": "material-response-convergence-policy-v1",
        "comparison_order": "absolute_first_then_relative_fallback",
        "relative_tolerance": 1e-3,
        "absolute_tolerance": 1e-6,
        "observable_error_budget_calibrated": False,
        "production_admission": False,
    }


def _cache_identity(
    index: int,
    *,
    q: np.ndarray | None = None,
    n_candidates: tuple[int, ...] = DEFAULT_N_CANDIDATES,
    shifts: tuple[tuple[float, float], ...] = DEFAULT_SHIFTS,
    reduction_block_size: int = DEFAULT_REDUCTION_BLOCK_SIZE,
) -> MaterialResponseCacheIdentity:
    vector = np.array([0.015, 0.025]) if q is None else np.asarray(q, dtype=float)
    return MaterialResponseCacheIdentity(
        pairing_name="spm",
        temperature_K=40.0,
        matsubara_index=index,
        xi_eV=matsubara_energy_eV(index, 40.0),
        q_crystal=vector,
        microscopic_model_name="symmetry_bdg_2band",
        material_state_fingerprint="state-fingerprint",
        response_policy_fingerprint="response-policy-fingerprint",
        primitive_contract_version="primitive-v-test",
        phase_hessian_policy="q_independent",
        basis=STATIC_LOCAL_BASIS if index == 0 else "crystal_xy",
        convergence_policy=_convergence_policy(),
        required_consecutive_passes=2,
        envelope_levels=3,
        n_candidates=n_candidates,
        shifts=shifts,
        canonical_reduction_block_size=reduction_block_size,
    )


def _audit_provenance(identity: MaterialResponseCacheIdentity) -> dict[str, object]:
    return {
        _shift_label(index, shift): {
            "grid": {
                "N": 96,
                "shift": list(shift),
                "shift_hex": [value.hex() for value in shift],
            },
            "canonical_reduction_block_size": identity.canonical_reduction_block_size,
        }
        for index, shift in enumerate(identity.shifts)
    }


def _artifact(index: int) -> CachedCertifiedMaterialResponse:
    identity = _cache_identity(index)
    primary = _shift_label(0, identity.shifts[0])
    return CachedCertifiedMaterialResponse(
        identity=identity,
        snapshot=MaterialResponseSnapshot.from_sample(_sample(index)),
        working_N=64,
        audit_N=96,
        primary_shift=primary,
        establishment_mode="strict_consecutive_adjacent",
        certification_evidence={
            "convergence_policy": dict(identity.convergence_policy),
            "required_consecutive_passes": identity.required_consecutive_passes,
            "oscillatory_envelope": {"levels": identity.envelope_levels},
            "observable_error_budget_calibrated": False,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
        audit_provenance_by_shift=_audit_provenance(identity),
    )


def _conflicting_artifact() -> CachedCertifiedMaterialResponse:
    artifact = _artifact(1)
    tensor = ConductivityTensor(0.8, 0.9, 0.02, 0.02)
    sheet = SheetConductivityConversion(
        tensor=tensor,
        unit_stage="sheet_conductivity",
        unit_label="test-sheet",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=("sheet",),
    )
    tilde = SheetConductivityConversion(
        tensor=tensor,
        unit_stage="reflection_dimensionless_conductivity",
        unit_label="test-tilde",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=("tilde",),
    )
    response = PositiveMatsubaraSheetResponse(
        sigma_model_xy=tensor,
        sigma_sheet_si_xy=sheet,
        sigma_tilde_xy=tilde,
        q_model=artifact.snapshot.q_crystal,
        xi_eV=artifact.snapshot.xi_eV,
        degeneracy=1.0,
        basis="crystal_xy",
        metadata={"source": "conflict-test"},
    )
    validation = SheetResponseValidation(
        finite=True,
        relative_imaginary_norm=0.0,
        relative_symmetry_residual=0.0,
        minimum_symmetric_eigenvalue=0.78,
        reality_tolerance=1e-9,
        symmetry_tolerance=1e-9,
        passivity_tolerance=1e-10,
    )
    return replace(
        artifact,
        snapshot=replace(
            artifact.snapshot,
            response=response,
            sheet_validation=validation,
        ),
    )


def test_identity_is_exact_and_geometry_free() -> None:
    identity = _cache_identity(1)
    payload_text = str(identity.payload)
    for forbidden in ("separation", "theta", "q_lab", "outer", "worker", "runtime_chunk"):
        assert forbidden not in payload_text
    shifted = _cache_identity(
        1,
        q=np.array([0.015, np.nextafter(0.025, 1.0)]),
    )
    assert shifted.sha256 != identity.sha256
    assert MaterialResponseCacheIdentity.from_payload(identity.payload).payload == identity.payload


def test_certification_sampling_policy_is_part_of_exact_identity() -> None:
    identity = _cache_identity(1)
    assert replace(identity, n_candidates=(64, 96, 160)).sha256 != identity.sha256
    assert replace(
        identity,
        shifts=((0.5, 0.5), (0.75, 0.25)),
    ).sha256 != identity.sha256
    assert replace(
        identity,
        canonical_reduction_block_size=2048,
    ).sha256 != identity.sha256


def test_identity_rejects_inconsistent_matsubara_triplet_and_unsafe_policy() -> None:
    identity = _cache_identity(1)
    with pytest.raises(ValueError, match="temperature_K, matsubara_index, and xi_eV"):
        replace(identity, temperature_K=41.0)
    unsafe = _convergence_policy()
    unsafe["production_admission"] = True
    with pytest.raises(ValueError, match="production admission"):
        replace(identity, convergence_policy=unsafe)


def test_artifact_rejects_audit_provenance_from_another_shift_policy() -> None:
    artifact = _artifact(1)
    bad = dict(artifact.audit_provenance_by_shift)
    label = next(iter(bad))
    row = dict(bad[label])
    row["grid"] = {"N": 96, "shift_hex": [0.0.hex(), 0.0.hex()]}
    bad[label] = row
    with pytest.raises(MaterialResponseCacheIdentityError, match="grid shift"):
        replace(artifact, audit_provenance_by_shift=bad)


@pytest.mark.parametrize("index", [0, 1])
def test_cache_roundtrip_is_readonly_and_geometry_equivalent(
    tmp_path: Path,
    index: int,
) -> None:
    artifact = _artifact(index)
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    loaded = store.put(artifact)
    assert loaded.snapshot.q_crystal.flags.writeable is False
    assert loaded.snapshot.primary_matrix.flags.writeable is False
    np.testing.assert_array_equal(
        loaded.snapshot.primary_matrix,
        artifact.snapshot.primary_matrix,
    )
    q = artifact.snapshot.q_crystal
    policy = TwoPlateGeometryPolicy(separation_m=100e-9)
    live = assemble_two_plate_logdet(
        _sample(index),
        _sample(index),
        q_lab=q,
        theta_1_rad=0.0,
        theta_2_rad=0.0,
        policy=policy,
    )
    replay = assemble_two_plate_logdet(
        loaded.snapshot,
        loaded.snapshot,
        q_lab=q,
        theta_1_rad=0.0,
        theta_2_rad=0.0,
        policy=policy,
    )
    assert replay.logdet == pytest.approx(live.logdet, rel=0.0, abs=1e-14)
    assert replay.metadata["microscopic_integration_performed"] is False


def test_read_only_miss_does_not_create_directories(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    store = MaterialResponseCacheStore(root, mode="read_only")
    with pytest.raises(MaterialResponseCacheMiss):
        store.get(_cache_identity(1))
    assert not root.exists()
    with pytest.raises(MaterialResponseCacheReadOnlyError):
        store.put(_artifact(1))


def test_truncated_file_fails_closed(tmp_path: Path) -> None:
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    artifact = _artifact(1)
    store.put(artifact)
    path = store.path_for(artifact.identity)
    path.write_bytes(path.read_bytes()[:32])
    with pytest.raises(MaterialResponseCacheCorruptionError):
        store.get(artifact.identity)


def test_filename_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    artifact = _artifact(1)
    store.put(artifact)
    source = store.path_for(artifact.identity)
    wrong = source.with_name("0" * 64 + ".npz")
    wrong.write_bytes(source.read_bytes())
    with pytest.raises(MaterialResponseCacheIdentityError):
        load_cached_certified_material_response(wrong)


def test_existing_identity_lock_is_not_silently_removed(tmp_path: Path) -> None:
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    artifact = _artifact(1)
    final_path = store.path_for(artifact.identity)
    final_path.parent.mkdir(parents=True)
    lock_path = final_path.with_suffix(".lock")
    lock_path.write_text("owner=other-process", encoding="utf-8")
    with pytest.raises(MaterialResponseCacheLockError):
        store.put(artifact)
    assert lock_path.exists()
    assert not final_path.exists()


def test_write_failure_cannot_leave_loader_visible_final_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    artifact = _artifact(1)
    final_path = store.path_for(artifact.identity)

    def fail_write(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(cache_store_module, "_write_npz", fail_write)
    with pytest.raises(OSError, match="simulated write failure"):
        store.put(artifact)
    assert not final_path.exists()
    assert list(final_path.parent.glob("*.tmp")) == []


def test_same_identity_conflict_never_overwrites_existing_artifact(tmp_path: Path) -> None:
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    original = _artifact(1)
    stored = store.put(original)
    path = store.path_for(original.identity)
    original_bytes = path.read_bytes()
    with pytest.raises(MaterialResponseCacheConflictError):
        store.put(_conflicting_artifact())
    assert path.read_bytes() == original_bytes
    reloaded = store.get(original.identity)
    np.testing.assert_array_equal(
        reloaded.snapshot.primary_matrix,
        stored.snapshot.primary_matrix,
    )
