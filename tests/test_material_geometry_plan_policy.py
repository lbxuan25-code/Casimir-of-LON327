from __future__ import annotations

from dataclasses import replace

import numpy as np

from lno327.casimir import material_geometry_plan as geometry_plan
from lno327.casimir.material_response_engine import MaterialResponseEngineConfig


def _config() -> MaterialResponseEngineConfig:
    return MaterialResponseEngineConfig(
        pairing_name="spm",
        temperature_K=40.0,
        matsubara_indices=(0, 1),
        n_candidates=(64, 96, 128),
        required_consecutive_passes=1,
        envelope_levels=3,
    )


def _plan(config: MaterialResponseEngineConfig):
    return geometry_plan.build_geometry_batch_plan(
        config,
        q_lab_points={
            "axis": np.array([0.02, 0.0]),
            "oblique": np.array([0.015, 0.025]),
        },
        angle_pairs_rad=((0.0, 0.0), (0.0, 0.31)),
        separations_m=(50e-9, 100e-9),
    )


def test_runtime_chunk_does_not_change_geometry_or_response_identity() -> None:
    base = _config()
    changed = replace(base, runtime_chunk_size=2048)
    first = _plan(base)
    second = _plan(changed)

    assert first.sha256 == second.sha256
    assert tuple(first.requirements) == tuple(second.requirements)
    assert [row.point_id for row in first.points] == [row.point_id for row in second.points]
    assert "runtime_chunk_size" not in str(first.identity_payload)


def test_geometry_plan_builds_material_identity_context_once(monkeypatch) -> None:
    calls = 0
    original = geometry_plan.build_material_response_identity_context

    def counted(config):
        nonlocal calls
        calls += 1
        return original(config)

    monkeypatch.setattr(
        geometry_plan,
        "build_material_response_identity_context",
        counted,
    )
    plan = _plan(_config())

    assert calls == 1
    assert len(plan.points) == 8
    assert len(plan.requirements) == 12


def test_distance_changes_plan_but_not_material_requirements() -> None:
    config = _config()
    first = geometry_plan.build_geometry_batch_plan(
        config,
        q_lab_points={"q": np.array([0.015, 0.025])},
        angle_pairs_rad=((0.0, 0.31),),
        separations_m=(50e-9, 100e-9),
    )
    second = geometry_plan.build_geometry_batch_plan(
        config,
        q_lab_points={"q": np.array([0.015, 0.025])},
        angle_pairs_rad=((0.0, 0.31),),
        separations_m=(80e-9, 200e-9),
    )

    assert first.sha256 != second.sha256
    assert tuple(first.requirements) == tuple(second.requirements)
    for key in first.requirements:
        assert first.requirements[key].identity.payload == second.requirements[key].identity.payload
