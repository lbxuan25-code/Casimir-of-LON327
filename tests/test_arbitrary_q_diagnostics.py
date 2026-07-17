from __future__ import annotations

from types import SimpleNamespace

import pytest

from validation.commands.matsubara.arbitrary_q_performance_smoke import (
    _all_responses,
    _timing_breakdown,
)
from validation.commands.matsubara.arbitrary_q_physics_smoke import _trend


def _profile(**overrides):
    values = {
        "q_workspace_seconds": 4.0,
        "kubo_factor_seconds": 2.0,
        "kubo_contraction_seconds": 2.0,
        "primitive_pack_seconds": 1.0,
        "operator_ward_seconds": 0.5,
        "accumulation_seconds": 0.5,
        "total_seconds": 12.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_timing_breakdown_reports_proportions_and_unattributed_time():
    result = _timing_breakdown((_profile(),))
    assert result["reported_total_seconds"] == pytest.approx(12.0)
    assert result["attributed_seconds"] == pytest.approx(10.0)
    assert result["unattributed_seconds"] == pytest.approx(2.0)
    fractions = result["timing_fraction_of_reported_total"]
    assert fractions["q_workspace_seconds"] == pytest.approx(1.0 / 3.0)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_parallel_response_collection_deduplicates_task_local_cache_hits():
    response = object()
    other = object()
    task = SimpleNamespace(
        result=SimpleNamespace(plate_1=response, plate_2=(response, other))
    )
    assert _all_responses((task,)) == (response, other)


def test_static_trend_reports_last_over_first_without_claiming_convergence():
    rows = []
    for scale in (2.0, 1.0):
        strict = {
            "primitive_residual_over_q": scale,
            "amplitude_defect_over_q": scale,
            "phase_defect_over_q": scale,
            "effective_direct_over_q": scale,
            "effective_residual_over_q": scale,
            "relative_longitudinal_gauge_residual": scale,
        }
        rows.append({"strict_static": strict})
    trend = _trend(rows)
    assert trend["relative_longitudinal_gauge_residual"]["last_over_first"] == pytest.approx(0.5)
    assert trend["relative_longitudinal_gauge_residual"]["nonincreasing"] is True
