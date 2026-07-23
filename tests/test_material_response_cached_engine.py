from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir import material_response_cached_engine as cached
from lno327.casimir.material_response_cache_store import CachedCertifiedMaterialResponse, MaterialResponseCacheMiss, MaterialResponseCacheStore
from lno327.casimir.material_response_engine import MaterialResponseEngineConfig
from lno327.casimir.material_response_snapshot import MaterialResponseSnapshot
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import PositiveMatsubaraSheetResponse, SheetConductivityConversion, SheetResponseValidation
from lno327.electrodynamics.static_sheet import STATIC_LOCAL_BASIS, StaticSheetResponse, StaticSheetValidation


def _config(indices: tuple[int, ...] = (0, 1)) -> MaterialResponseEngineConfig:
    return MaterialResponseEngineConfig(pairing_name="spm", temperature_K=40.0, matsubara_indices=indices, n_candidates=(64, 96, 128), required_consecutive_passes=1, envelope_levels=3)


def _sample_identity(identity) -> dict:
    return {
        "schema": "material-response-identity-v1",
        "frequency_sector": identity.frequency_sector,
        "xi_eV_hex": float(identity.xi_eV).hex(),
        "q_crystal_hex": [float(value).hex() for value in identity.q_crystal],
        "material_state_fingerprint": identity.material_state_fingerprint,
        "response_policy_fingerprint": identity.response_policy_fingerprint,
        "primitive_contract_version": identity.primitive_contract_version,
        "phase_hessian_policy": identity.phase_hessian_policy,
        "basis": identity.basis,
    }


def _snapshot(identity) -> MaterialResponseSnapshot:
    q = identity.q_crystal
    if identity.matsubara_index == 0:
        validation = StaticSheetValidation(
            finite=True, ward_passed=True, relative_imaginary_norm=0.0,
            relative_longitudinal_gauge_residual=0.0,
            relative_density_transverse_mixing=0.0, chi_bar=1.0, dbar_t=2.0,
            reality_tolerance=1e-8, longitudinal_tolerance=1e-6,
            mixing_tolerance=1e-6, passivity_tolerance=1e-10,
        )
        response = StaticSheetResponse(
            kernel_lt=np.diag([-1.0, 0.0, -2.0]).astype(complex),
            chi_bar=1.0, dbar_t=2.0, q_model=q, energy_scale_eV=1.0,
            degeneracy=1.0, basis=STATIC_LOCAL_BASIS, validation=validation,
            metadata={"source": "test"},
        )
    else:
        tensor = ConductivityTensor(0.4, 0.5, 0.02, 0.02)
        sheet = SheetConductivityConversion(tensor=tensor, unit_stage="sheet_conductivity", unit_label="test-sheet", normalization_status="test", valid_for_casimir_input=True, notes=())
        tilde = SheetConductivityConversion(tensor=tensor, unit_stage="reflection_dimensionless_conductivity", unit_label="test-tilde", normalization_status="test", valid_for_casimir_input=True, notes=())
        response = PositiveMatsubaraSheetResponse(sigma_model_xy=tensor, sigma_sheet_si_xy=sheet, sigma_tilde_xy=tilde, q_model=q, xi_eV=identity.xi_eV, degeneracy=1.0, basis="crystal_xy", metadata={"source": "test"})
        validation = SheetResponseValidation(finite=True, relative_imaginary_norm=0.0, relative_symmetry_residual=0.0, minimum_symmetric_eigenvalue=0.38, reality_tolerance=1e-9, symmetry_tolerance=1e-9, passivity_tolerance=1e-10)
    return MaterialResponseSnapshot(
        frequency_index=0, frequency_sector=identity.frequency_sector,
        q_crystal=q, xi_eV=identity.xi_eV, response=response,
        sheet_validation=validation, identity=_sample_identity(identity),
        provenance={"grid": {"n": 96}},
        physical_audit={"hard_physical_passed": True},
    )


def _artifact(identity) -> CachedCertifiedMaterialResponse:
    return CachedCertifiedMaterialResponse(
        identity=identity, snapshot=_snapshot(identity), working_N=64, audit_N=96,
        primary_shift="shift_0", establishment_mode="strict_consecutive_adjacent",
        certification_evidence={
            "convergence_policy": dict(identity.convergence_policy),
            "required_consecutive_passes": identity.required_consecutive_passes,
            "observable_error_budget_calibrated": False,
            "valid_for_casimir_input": False,
            "production_casimir_allowed": False,
        },
        audit_provenance_by_shift={
            "shift_0": {"grid": 96},
            "shift_1": {"grid": 96},
        },
    )


def _patch_identity_context(monkeypatch) -> None:
    monkeypatch.setattr(cached, "_identity_context", lambda config: {"material_state_fingerprint": "state-fingerprint", "phase_hessian_policy": "q_independent"})


def _patch_artifact_factory(monkeypatch) -> None:
    def factory(cls, *, identity, certification):
        return _artifact(identity)
    monkeypatch.setattr(CachedCertifiedMaterialResponse, "from_certification", classmethod(factory))


def test_warm_run_never_calls_microscopic_engine(tmp_path: Path, monkeypatch) -> None:
    _patch_identity_context(monkeypatch)
    _patch_artifact_factory(monkeypatch)
    calls: list[tuple[int, ...]] = []
    def cold_engine(config, *, q_crystal):
        calls.append(config.matsubara_indices)
        return SimpleNamespace(frequencies={index: SimpleNamespace(xi_eV=cached.matsubara_energy_eV(index, config.temperature_K), certification=object()) for index in config.matsubara_indices})
    monkeypatch.setattr(cached, "evaluate_material_response_ladder", cold_engine)
    config = _config()
    q = np.array([0.015, 0.025])
    cold = cached.evaluate_material_response_ladder_cached(config, q_crystal=q, cache=MaterialResponseCacheStore(tmp_path, mode="populate"))
    assert calls == [(0, 1)]
    assert cold.metadata["persisted_frequency_count"] == 2
    def forbidden(*args, **kwargs):
        raise AssertionError("warm cache hit called microscopic engine")
    monkeypatch.setattr(cached, "evaluate_material_response_ladder", forbidden)
    warm = cached.evaluate_material_response_ladder_cached(config, q_crystal=q, cache=MaterialResponseCacheStore(tmp_path, mode="read_only"))
    assert warm.all_requested_established is True
    assert warm.metadata["cache_hits"] == 2
    assert warm.metadata["microscopic_frequency_count"] == 0
    assert {row.source for row in warm.frequencies.values()} == {"persistent_cache_hit"}


def test_partial_hit_sends_only_misses_to_microscopic_engine(tmp_path: Path, monkeypatch) -> None:
    _patch_identity_context(monkeypatch)
    _patch_artifact_factory(monkeypatch)
    config = _config()
    q = np.array([0.015, 0.025])
    context = cached._identity_context(config)
    identity_zero = cached.build_material_response_cache_identity(config, q_crystal=q, matsubara_index=0, context=context)
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    store.put(_artifact(identity_zero))
    def miss_only_engine(miss_config, *, q_crystal):
        assert miss_config.matsubara_indices == (1,)
        return SimpleNamespace(frequencies={1: SimpleNamespace(xi_eV=cached.matsubara_energy_eV(1, 40.0), certification=object())})
    monkeypatch.setattr(cached, "evaluate_material_response_ladder", miss_only_engine)
    result = cached.evaluate_material_response_ladder_cached(config, q_crystal=q, cache=store)
    assert result.frequencies[0].source == "persistent_cache_hit"
    assert result.frequencies[1].source == "microscopic_certified_and_persisted"
    assert result.metadata["microscopic_frequency_count"] == 1


def test_read_only_miss_fails_before_microscopic_fallback(tmp_path: Path, monkeypatch) -> None:
    _patch_identity_context(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only cache miss attempted microscopic fallback")
    monkeypatch.setattr(cached, "evaluate_material_response_ladder", forbidden)
    with pytest.raises(MaterialResponseCacheMiss):
        cached.evaluate_material_response_ladder_cached(_config((1,)), q_crystal=np.array([0.015, 0.025]), cache=MaterialResponseCacheStore(tmp_path, mode="read_only"))


def test_unresolved_frequency_is_never_persisted(tmp_path: Path, monkeypatch) -> None:
    _patch_identity_context(monkeypatch)
    config = _config((1,))
    q = np.array([0.015, 0.025])
    def unresolved_engine(miss_config, *, q_crystal):
        return SimpleNamespace(
            frequencies={
                1: SimpleNamespace(
                    xi_eV=cached.matsubara_energy_eV(1, 40.0),
                    certification=None,
                )
            }
        )
    monkeypatch.setattr(cached, "evaluate_material_response_ladder", unresolved_engine)
    store = MaterialResponseCacheStore(tmp_path, mode="populate")
    result = cached.evaluate_material_response_ladder_cached(config, q_crystal=q, cache=store)
    row = result.frequencies[1]
    assert row.source == "microscopic_unresolved_not_persisted"
    assert row.established is False
    assert result.metadata["unresolved_frequency_count"] == 1
    assert result.metadata["persisted_frequency_count"] == 0
    assert not store.path_for(row.cache_identity).exists()
