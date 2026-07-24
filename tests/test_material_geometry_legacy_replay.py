from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir import material_geometry_legacy_replay as replay
from lno327.casimir.material_geometry import ReflectionGeometryPolicy
from lno327.casimir.material_geometry_plan import GeometryBatchPolicy
from lno327.casimir.material_geometry_qualification import GeometryEquivalenceReport
from lno327.casimir.material_response import MaterialResponsePolicy


def _identity(q: tuple[float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        pairing_name="spm",
        temperature_K=40.0,
        shifts=((0.5, 0.5), (0.25, 0.75), (0.75, 0.25)),
        xi_eV=0.021,
        matsubara_index=1,
        microscopic_model_name="symmetry_bdg_2band",
        material_state_fingerprint="state-fingerprint",
        response_policy_fingerprint="response-policy-fingerprint",
        primitive_contract_version="primitive-v-test",
        phase_hessian_policy="q_independent",
        basis="crystal_xy",
        certification_policy_fingerprint="certification-policy-fingerprint",
        q_crystal=np.asarray(q, dtype=float),
        canonical_reduction_block_size=4096,
    )


def _artifact(q: tuple[float, float]) -> SimpleNamespace:
    identity = _identity(q)
    return SimpleNamespace(
        identity=identity,
        working_N=64,
        primary_shift=f"shift_0:{float(0.5).hex()}:{float(0.5).hex()}",
    )


def _batch() -> SimpleNamespace:
    point_id = "q:n=1:angles=0"
    spec = SimpleNamespace(
        point_id=point_id,
        matsubara_index=1,
        q_lab=np.array([0.015, 0.025]),
        theta_1_rad=0.0,
        theta_2_rad=0.3,
        plate_1_requirement="a" * 64,
        plate_2_requirement="b" * 64,
    )
    config = SimpleNamespace(
        material_policy=MaterialResponsePolicy(),
        microscopic_model_name="symmetry_bdg_2band",
        pairing_name="spm",
        delta0_eV=0.1,
        temperature_K=40.0,
        eta_eV=1e-8,
        canonical_reduction_block_size=4096,
        runtime_chunk_size=1024,
    )
    return SimpleNamespace(
        points={point_id: SimpleNamespace(spec=spec)},
        preflight=SimpleNamespace(
            artifacts={
                spec.plate_1_requirement: _artifact((0.015, 0.025)),
                spec.plate_2_requirement: _artifact((0.021726, 0.019451)),
            }
        ),
        plan=SimpleNamespace(
            response_config=config,
            policy=GeometryBatchPolicy(
                reflection_policy=ReflectionGeometryPolicy()
            ),
        ),
    )


def _legacy_metadata() -> dict[str, object]:
    return {
        "material_state_fingerprint": "state-fingerprint",
        "primitive_contract_version": "primitive-v-test",
        "post_integral_phase_hessian_policy": "q_independent",
        "canonical_reduction_block_size": 4096,
    }


def test_real_replay_orchestrator_uses_one_working_N_primary_shift(monkeypatch) -> None:
    batch = _batch()
    monkeypatch.setattr(replay, "GeometryBatchResult", SimpleNamespace)

    fake_model = SimpleNamespace(
        spec=SimpleNamespace(name="spec"),
        build_ansatz=lambda pairing, phase_vertex: SimpleNamespace(
            name=pairing,
            phase_vertex=phase_vertex,
        ),
        build_pairing_params=lambda delta: SimpleNamespace(delta=delta),
    )
    monkeypatch.setattr(replay, "get_finite_q_microscopic_model", lambda name: fake_model)

    calls: dict[str, object] = {}

    def fake_grid(n, shift):
        calls["grid"] = (n, shift)
        return SimpleNamespace(n=n, shift=shift)

    def fake_material_cache(**kwargs):
        calls["material_cache"] = kwargs
        return SimpleNamespace(grid=kwargs["grid"])

    legacy_plate_1 = SimpleNamespace(metadata=_legacy_metadata())
    legacy_plate_2 = SimpleNamespace(metadata=_legacy_metadata())

    def fake_integrate(**kwargs):
        calls["integrate"] = kwargs
        return SimpleNamespace(
            plate_1=legacy_plate_1,
            plate_2=(legacy_plate_2,),
            q_lab=np.asarray(kwargs["q_lab"], dtype=float),
        )

    monkeypatch.setattr(replay, "build_periodic_bz_grid", fake_grid)
    monkeypatch.setattr(replay, "build_material_grid_cache", fake_material_cache)
    monkeypatch.setattr(replay, "integrate_two_plate_angle_batch", fake_integrate)

    def fake_qualify(batch_value, **kwargs):
        calls["qualify"] = (batch_value, kwargs)
        return GeometryEquivalenceReport(
            mode="legacy_vs_persisted_batch",
            point_id=kwargs["point_id"],
            comparisons={"logdet": {"passed": True}},
            passed=True,
            metadata={"diagnostic_only": True},
        )

    monkeypatch.setattr(replay, "qualify_matched_legacy_point", fake_qualify)
    result = replay.run_matched_legacy_geometry_replay(
        batch,
        point_id="q:n=1:angles=0",
        distance_m=100e-9,
    )

    assert result.report.passed is True
    assert result.working_N == 64
    assert result.primary_shift == (0.5, 0.5)
    assert calls["grid"] == (64, (0.5, 0.5))
    integrate = calls["integrate"]
    assert integrate["canonical_reduction_block_size"] == 4096
    assert integrate["xi_eV_values"] == (0.021,)
    assert integrate["theta_2_rad_values"] == (0.3,)
    _, qualification = calls["qualify"]
    assert qualification["legacy_n"] == 64
    assert qualification["legacy_frequency_index"] == 0
    assert result.metadata["primitive_contract_version"] == "primitive-v-test"
    assert result.metadata["phase_hessian_policy"] == "q_independent"
    assert result.metadata["all_material_and_numerical_contracts_matched"] is True
    assert result.metadata["n_ladder_search_performed"] is False
    assert result.metadata["response_cache_write_performed"] is False


def test_legacy_replay_rejects_mismatched_primitive_contract(monkeypatch) -> None:
    batch = _batch()
    monkeypatch.setattr(replay, "GeometryBatchResult", SimpleNamespace)
    fake_model = SimpleNamespace(
        spec=SimpleNamespace(name="spec"),
        build_ansatz=lambda pairing, phase_vertex: SimpleNamespace(
            name=pairing,
            phase_vertex=phase_vertex,
        ),
        build_pairing_params=lambda delta: SimpleNamespace(delta=delta),
    )
    monkeypatch.setattr(replay, "get_finite_q_microscopic_model", lambda name: fake_model)
    monkeypatch.setattr(
        replay,
        "build_periodic_bz_grid",
        lambda n, shift: SimpleNamespace(n=n, shift=shift),
    )
    monkeypatch.setattr(
        replay,
        "build_material_grid_cache",
        lambda **kwargs: SimpleNamespace(grid=kwargs["grid"]),
    )
    bad = _legacy_metadata()
    bad["primitive_contract_version"] = "wrong-primitive"
    legacy = SimpleNamespace(metadata=bad)
    monkeypatch.setattr(
        replay,
        "integrate_two_plate_angle_batch",
        lambda **kwargs: SimpleNamespace(
            plate_1=legacy,
            plate_2=(SimpleNamespace(metadata=_legacy_metadata()),),
            q_lab=np.asarray(kwargs["q_lab"], dtype=float),
        ),
    )

    with pytest.raises(ValueError, match="primitive_contract_version"):
        replay.run_matched_legacy_geometry_replay(
            batch,
            point_id="q:n=1:angles=0",
            distance_m=100e-9,
        )


def test_legacy_replay_rejects_policy_the_archived_helper_cannot_express(
    monkeypatch,
) -> None:
    batch = _batch()
    batch.plan.response_config.material_policy = replace(
        MaterialResponsePolicy(),
        positive_reality_tolerance=2e-9,
    )
    monkeypatch.setattr(replay, "GeometryBatchResult", SimpleNamespace)

    with pytest.raises(ValueError, match="positive_reality_tolerance"):
        replay.run_matched_legacy_geometry_replay(
            batch,
            point_id="q:n=1:angles=0",
            distance_m=100e-9,
        )
