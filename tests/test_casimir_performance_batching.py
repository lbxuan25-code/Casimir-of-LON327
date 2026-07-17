from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lno327.casimir.adaptive_joint_q import (
    AdaptiveJointCasimirConfig,
    _provider_statistics,
    run_adaptive_joint_casimir,
)
from lno327.casimir.adaptive_outer_q import (
    AdaptiveRadialCasimirConfig,
    AdaptiveRadialCasimirResult,
)
from lno327.casimir.certified_point_provider import (
    CertifiedOuterQProvider,
    CertifiedPointBatch,
)
from lno327.casimir.fixed_chain import FixedCasimirConfig, _CertificationRun


def _radial_result(
    config: AdaptiveRadialCasimirConfig,
    *,
    value: float,
    error: float = 0.01,
) -> AdaptiveRadialCasimirResult:
    count = len(config.point_config.matsubara_indices)
    values = np.full(count, float(value), dtype=float)
    errors = np.full(count, float(error), dtype=float)
    pairing_results = {
        pairing: {
            "status": "integrated",
            "partial_free_energy_J_m2": float(np.sum(values)),
            "contributions_J_m2": values.tolist(),
            "outer_q_integrals_m_inv2": values.tolist(),
            "estimated_radial_errors_J_m2": errors.tolist(),
            "radial_tolerances_J_m2": [1.0] * count,
            "radial_channel_passed": [True] * count,
            "matsubara_indices": list(config.point_config.matsubara_indices),
            "prime_weights": [1.0] * count,
        }
        for pairing in config.point_config.pairings
    }
    return AdaptiveRadialCasimirResult(
        status="adaptive_finite_partial",
        config=config,
        radial_converged=True,
        all_microscopic_nodes_certified=True,
        pairing_results=pairing_results,
        panel_records=(),
        refinement_rounds=config.max_refinement_rounds,
        unique_microscopic_q_node_count=0,
        unresolved_points=(),
        termination_reason="radial_tolerance_met",
        provider_statistics={},
    )


class _BatchingProvider:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self._seen: set[tuple[str, str]] = set()
        self.cached_point_count = 0
        self.unique_q_count = 0
        self.certification_batches = 0
        self.requested_q_evaluations = 0
        self.new_q_evaluations = 0
        self.cache_hit_q_evaluations = 0

    @staticmethod
    def _keys(q_model: np.ndarray) -> set[tuple[str, str]]:
        array = np.asarray(q_model, dtype=float)
        return {
            (float(q[0]).hex(), float(q[1]).hex())
            for q in array
        }

    def count_new_q(self, q_model: np.ndarray) -> int:
        return len(self._keys(q_model) - self._seen)

    def evaluate(self, q_model: np.ndarray) -> CertifiedPointBatch:
        array = np.asarray(q_model, dtype=float)
        keys = self._keys(array)
        new = keys - self._seen
        self.events.append(("prefetch", int(array.shape[0]), len(new)))
        self.requested_q_evaluations += len(keys)
        self.new_q_evaluations += len(new)
        self.cache_hit_q_evaluations += len(keys) - len(new)
        self.certification_batches += int(bool(new))
        self._seen.update(keys)
        self.unique_q_count = len(self._seen)
        self.cached_point_count = len(self._seen)
        return CertifiedPointBatch(
            point_results=(),
            unresolved_points=(),
            requested_q_count=len(keys),
            new_q_count=len(new),
            cache_hit_q_count=len(keys) - len(new),
            certification_batches=int(bool(new)),
        )


class _ScriptedRadialRunner:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def __call__(
        self,
        config: AdaptiveRadialCasimirConfig,
        *,
        provider=None,
    ) -> AdaptiveRadialCasimirResult:
        self.events.append(
            (
                "run",
                int(config.angular_order),
                float(config.angular_offset_fraction),
            )
        )
        value = 1.0 if config.angular_order == 4 else 1.1
        if config.angular_offset_fraction == 0.0:
            value += 0.05
        return _radial_result(config, value=value)


def test_joint_controller_prefetches_initial_comparison_grids_as_one_batch() -> None:
    point = FixedCasimirConfig(pairings=("spm",), matsubara_indices=(0,))
    radial = AdaptiveRadialCasimirConfig(
        point_config=point,
        initial_panel_edges=(0.0, 2.0),
        radial_order=1,
        angular_order=4,
        max_refinement_rounds=0,
    )
    config = AdaptiveJointCasimirConfig(
        radial_config=radial,
        angular_orders=(4, 8),
        outer_rtol=0.0,
        outer_atol_J_m2=1.0,
        offset_rtol=0.0,
        offset_atol_J_m2=1.0,
        initial_radial_round_cap=0,
    )
    events: list[tuple[object, ...]] = []
    provider = _BatchingProvider(events)

    result = run_adaptive_joint_casimir(
        config,
        provider=provider,
        radial_runner=_ScriptedRadialRunner(events),
    )

    assert result.status == "adaptive_finite_partial"
    assert events[0] == ("prefetch", 36, 36)
    assert [event[0] for event in events] == [
        "prefetch",
        "run",
        "run",
        "run",
    ]
    assert provider.certification_batches == 1


def _point(pairing: str, label: str, n: int, value: float) -> dict:
    return {
        "pairing": pairing,
        "q_label": label,
        "n": n,
        "sweet_spot": {
            "status": "established",
            "working_N": 12,
            "audit_N": 16,
            "establishment_mode": "strict_consecutive_adjacent",
        },
        "history": [
            {
                "N": 16,
                "two_plate_logdet_cross_shift": {"passed": True},
                "shifts": {
                    "primary": {
                        "two_plate_logdet": value,
                        "hard_physical_passed": True,
                    }
                },
            }
        ],
    }


def test_provider_persists_certifier_and_cache_telemetry(
    tmp_path: Path,
) -> None:
    config = FixedCasimirConfig()

    def runner(current, manifest, output):
        payload = {
            "schema": "transverse-point-sweet-spot-v4",
            "execution_levels": [
                {
                    "level_wall_seconds": 1.25,
                    "pairings": {
                        "spm": [
                            {
                                "material_build_seconds": 0.4,
                                "context_wall_seconds": 0.8,
                            }
                        ]
                    },
                }
            ],
            "point_results": [
                _point(pairing, label, n, float(n + 1))
                for label in manifest.labels
                for pairing in current.pairings
                for n in current.matsubara_indices
            ],
        }
        return _CertificationRun(payload, "", "", ("python",))

    cache = tmp_path / "points.json"
    provider = CertifiedOuterQProvider(
        config,
        cache_path=cache,
        runner=runner,
    )
    batch = provider.evaluate(np.asarray([[0.1, 0.2]], dtype=float))
    stats = provider.performance_statistics()

    assert batch.all_established
    assert stats["certification_batches"] == 1
    assert stats["certifier_reported_level_wall_seconds"] == pytest.approx(
        1.25
    )
    assert stats["certifier_material_build_seconds"] == pytest.approx(0.4)
    assert stats["certifier_context_wall_seconds"] == pytest.approx(0.8)
    assert stats["certifier_wall_seconds"] >= 0.0
    assert stats["cache_save_count"] == 1
    assert stats["cache_file_bytes"] > 0
    assert len(stats["certifier_batch_records"]) == 1
    assert _provider_statistics(provider)["certifier_material_build_seconds"] == pytest.approx(
        0.4
    )

    telemetry_path = cache.with_suffix(".telemetry.json")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert telemetry["cache_save_count"] == 1
    assert telemetry["cache_file_bytes"] > 0


def test_process_runtime_dependencies_are_base_dependencies() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    project_section = text.split("[project]", 1)[1].split(
        "[project.scripts]",
        1,
    )[0]
    assert '"threadpoolctl>=3"' in project_section
    assert '"psutil>=5"' in project_section
