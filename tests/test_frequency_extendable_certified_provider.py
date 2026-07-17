from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lno327.casimir.certified_point_provider import (
    CertifiedPointCacheError,
    FrequencyExtendableCertifiedOuterQProvider,
)
from lno327.casimir.fixed_chain import FixedCasimirConfig, _CertificationRun


def _point(pairing: str, label: str, n: int, value: float) -> dict:
    return {
        "pairing": pairing,
        "q_label": label,
        "n": n,
        "sweet_spot": {
            "status": "established",
            "working_N": 192,
            "audit_N": 256,
            "establishment_mode": "strict_consecutive_adjacent",
        },
        "history": [
            {
                "N": 256,
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


def test_frequency_provider_certifies_only_new_matsubara_indices_and_persists(
    tmp_path: Path,
) -> None:
    base = FixedCasimirConfig(matsubara_indices=(0, 1))
    calls: list[tuple[int, ...]] = []

    def runner(current, manifest, output):
        calls.append(tuple(current.matsubara_indices))
        payload = {
            "schema": "transverse-point-sweet-spot-v4",
            "point_results": [
                _point(pairing, label, n, float(100 * n + index + 1))
                for index, label in enumerate(manifest.labels)
                for pairing in current.pairings
                for n in current.matsubara_indices
            ],
        }
        return _CertificationRun(payload, "", "", ("python",))

    cache = tmp_path / "frequency-points-v2.json"
    q = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    provider = FrequencyExtendableCertifiedOuterQProvider(
        base,
        cache_path=cache,
        runner=runner,
    )
    first = provider.evaluate(q)
    extended = replace(base, matsubara_indices=(0, 1, 2, 3))
    provider.reconfigure(extended)
    second = provider.evaluate(q)
    replay = provider.evaluate(q[::-1])

    assert first.all_established and second.all_established and replay.all_established
    assert calls == [(0, 1), (2, 3)]
    assert first.new_point_count == 4
    assert second.requested_point_count == 8
    assert second.new_point_count == 4
    assert second.cache_hit_point_count == 4
    assert replay.new_q_count == 0
    assert provider.cached_point_count == 8
    assert provider.unique_q_count == 2
    assert provider.new_point_evaluations == 8
    assert provider.cache_hit_point_evaluations == 12
    assert cache.is_file()

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("persisted Matsubara entries must not be recomputed")

    restored = FrequencyExtendableCertifiedOuterQProvider(
        extended,
        cache_path=cache,
        runner=forbidden_runner,
    )
    restored_result = restored.evaluate(q)
    assert restored_result.all_established
    assert restored_result.new_point_count == 0
    assert restored.primary_logdet("spm", 3, q[0]) == 301.0


def test_frequency_provider_rejects_nonfrequency_policy_changes(tmp_path: Path) -> None:
    base = FixedCasimirConfig(matsubara_indices=(0, 1))
    provider = FrequencyExtendableCertifiedOuterQProvider(base)
    provider.reconfigure(replace(base, matsubara_indices=(0, 1, 2)))

    with pytest.raises(CertifiedPointCacheError, match="fingerprint"):
        provider.reconfigure(replace(base, temperature_K=11.0))


def test_v1_and_frequency_extendable_cache_schemas_are_separate(tmp_path: Path) -> None:
    from lno327.casimir.certified_point_provider import CertifiedOuterQProvider

    cache = tmp_path / "points.json"
    base = FixedCasimirConfig()

    def runner(current, manifest, output):
        payload = {
            "schema": "transverse-point-sweet-spot-v4",
            "point_results": [
                _point(pairing, label, n, 1.0)
                for label in manifest.labels
                for pairing in current.pairings
                for n in current.matsubara_indices
            ],
        }
        return _CertificationRun(payload, "", "", ("python",))

    CertifiedOuterQProvider(base, cache_path=cache, runner=runner).evaluate(
        np.asarray([[0.1, 0.2]], dtype=float)
    )
    with pytest.raises(CertifiedPointCacheError, match="schema"):
        FrequencyExtendableCertifiedOuterQProvider(
            base,
            cache_path=cache,
            runner=runner,
        )
