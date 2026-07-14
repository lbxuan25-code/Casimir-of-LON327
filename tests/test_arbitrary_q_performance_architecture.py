from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from validation.commands.matsubara.arbitrary_q_performance_preflight import (
    _architecture,
)


def _response(q: tuple[float, float], *, shifted_calls: int = 2) -> object:
    return SimpleNamespace(
        q_model=np.asarray(q, dtype=float),
        operator_ward=SimpleNamespace(passed=True),
        profile=SimpleNamespace(
            runtime_chunk_count=1,
            q_workspace_build_count=1,
            shifted_eigensystem_build_count=shifted_calls,
            counterterm_add_count=1,
        ),
    )


def _task(
    plate_1: object,
    plate_2: tuple[object, ...],
    *,
    cache_hits: int,
) -> object:
    return SimpleNamespace(
        result=SimpleNamespace(
            plate_1=plate_1,
            plate_2=plate_2,
            response_cache_metadata={"hits": cache_hits},
        ),
        payload_bytes=128,
        worker_rss_bytes=1024,
        worker_pss_bytes=512,
    )


def test_audit_workload_accepts_zero_exact_response_cache_hits() -> None:
    row = _architecture(
        (
            _task(
                _response((0.03, 0.02)),
                (_response((0.034, 0.011)),),
                cache_hits=0,
            ),
        )
    )
    assert row["expected_exact_response_cache_hits"] == 0
    assert row["exact_response_cache_hits"] == 0
    assert row["response_cache_hit_count_matches_expected"] is True
    assert row["passed"] is True


def test_duplicate_plate_response_requires_one_exact_cache_hit() -> None:
    q = (0.03, 0.02)
    row = _architecture(
        (
            _task(
                _response(q),
                (_response(q), _response((0.034, 0.011))),
                cache_hits=1,
            ),
        )
    )
    assert row["expected_exact_response_cache_hits"] == 1
    assert row["exact_response_cache_hits"] == 1
    assert row["passed"] is True

    missing = _architecture(
        (
            _task(
                _response(q),
                (_response(q), _response((0.034, 0.011))),
                cache_hits=0,
            ),
        )
    )
    assert missing["response_cache_hit_count_matches_expected"] is False
    assert missing["passed"] is False


def test_shifted_eigensystem_count_must_match_q_workspace_builds_exactly() -> None:
    malformed = _architecture(
        (
            _task(
                _response((0.03, 0.02), shifted_calls=3),
                (_response((0.034, 0.011)),),
                cache_hits=0,
            ),
        )
    )
    assert malformed["shifted_eigh_counts_exact"] is False
    assert malformed["passed"] is False
