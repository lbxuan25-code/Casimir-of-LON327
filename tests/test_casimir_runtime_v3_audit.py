from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lno327.casimir import build_full_casimir_config
from lno327.casimir import fixed_transverse_point_engine as _engine
from lno327.casimir.adaptive_joint_q import (
    AdaptiveJointCasimirConfig,
    _offset_comparison,
    run_adaptive_joint_casimir,
)
from lno327.casimir.adaptive_outer_q import (
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialCasimirResult,
)
from lno327.casimir.certified_point_provider import CertifiedPointBatch
from lno327.casimir.fixed_chain import (
    FixedCasimirConfig,
    _transverse_certification_command,
)
from lno327.casimir.fixed_outer_q import OuterQNodeManifest


def test_negative_scientific_q_values_survive_cli_round_trip(tmp_path: Path) -> None:
    config = FixedCasimirConfig()
    values = np.asarray([[-1.25e-17, 2.5e-18]], dtype=float)
    manifest = OuterQNodeManifest(
        labels=("tiny_negative",),
        q_model=values,
        grids={},
        labels_by_spec={},
    )
    command = _transverse_certification_command(
        config,
        manifest,
        tmp_path / "output.json",
    )
    q_index = command.index("--q-point")
    assert "e" not in command[q_index + 2].lower()
    assert "e" not in command[q_index + 3].lower()
    parsed = _engine._parse_args(command[3:])
    assert parsed.q_points[0]["q_lab"] == pytest.approx(values[0], rel=0.0, abs=0.0)


def test_full_builder_uses_audited_v3_point_policy() -> None:
    config = build_full_casimir_config(pairings=("dwave",))
    point = config.outer_tail_config.joint_config.radial_config.point_config
    assert point.logdet_rtol == pytest.approx(1.5e-3)
    assert point.logdet_atol == pytest.approx(1e-6)
    assert config.certifier_q_batch_size == 384
    assert config.as_dict()["certifier_q_batch_size"] == 384


class _UnresolvedProvider:
    cached_point_count = 0
    unique_q_count = 0
    certification_batches = 0
    requested_q_evaluations = 0
    new_q_evaluations = 0
    cache_hit_q_evaluations = 0

    def count_new_q(self, q_model: np.ndarray) -> int:
        return len({(float(q[0]).hex(), float(q[1]).hex()) for q in q_model})

    def evaluate(self, q_model: np.ndarray) -> CertifiedPointBatch:
        count = self.count_new_q(q_model)
        self.requested_q_evaluations += count
        self.new_q_evaluations += count
        self.unique_q_count += count
        return CertifiedPointBatch(
            point_results=(),
            unresolved_points=(
                {
                    "pairing": "spm",
                    "n": 0,
                    "q_label": "synthetic",
                    "reason": "synthetic_unresolved_gate",
                },
            ),
            requested_q_count=count,
            new_q_count=count,
            cache_hit_q_count=0,
            certification_batches=0,
        )

    def primary_logdet(self, pairing: str, n: int, q) -> float:
        raise AssertionError("unresolved provider values must never be reduced")


def _small_joint_config() -> AdaptiveJointCasimirConfig:
    point = FixedCasimirConfig(
        pairings=("spm",),
        matsubara_indices=(0,),
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
    )
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 1.0),
        radial_order=1,
        angular_order=4,
        max_refinement_rounds=0,
    )
    return AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(4, 8),
        max_joint_iterations=2,
    )


def test_prefetch_unresolved_points_remain_microscopic_failures() -> None:
    result = run_adaptive_joint_casimir(
        _small_joint_config(),
        provider=_UnresolvedProvider(),
    )
    assert result.status == "unresolved"
    assert result.all_microscopic_nodes_certified is False
    assert "synthetic_unresolved_gate" in result.termination_reason
    assert "joint_runtime_failure" not in result.termination_reason


def _radial_result(
    config: AdaptiveRadialCasimirConfig,
    *,
    value: float,
    error: float,
) -> AdaptiveRadialCasimirResult:
    return AdaptiveRadialCasimirResult(
        status="adaptive_finite_partial",
        config=config,
        radial_converged=True,
        all_microscopic_nodes_certified=True,
        pairing_results={
            "spm": {
                "contributions_J_m2": [value],
                "estimated_radial_errors_J_m2": [error],
                "matsubara_indices": [0],
            }
        },
        panel_records=(),
        refinement_rounds=0,
        unique_microscopic_q_node_count=0,
        unresolved_points=(),
        termination_reason="radial_tolerance_met",
        provider_statistics={},
    )


def test_offset_error_bound_includes_independent_audit_radial_error() -> None:
    config = _small_joint_config()
    primary = _radial_result(config.radial_config, value=1.0, error=0.1)
    audit = _radial_result(config.radial_config, value=1.05, error=0.2)
    metrics, *_ = _offset_comparison(primary, audit, config)
    payload = metrics["spm"]
    assert payload["offset_differences_J_m2"] == pytest.approx([0.05])
    assert payload["estimated_offset_error_bounds_J_m2"] == pytest.approx([0.25])
