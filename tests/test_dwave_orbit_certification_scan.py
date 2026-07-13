from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.commands.matsubara.dwave_orbit_certification_scan import (
    CaseMetrics,
    CaseSpec,
    TaskSpec,
    _fingerprint,
    _orbit_origin_count,
    _parse_case,
    _resolved_budget,
    _resume_match,
    _select_labels,
    _validate_cases,
)


def test_parse_case_levels_and_validation():
    assert _parse_case("axis:1:0:full") == CaseSpec("axis", 1, 0, "full")
    assert _parse_case("generic:2:1") == CaseSpec("generic", 2, 1, "screen")
    with pytest.raises(ValueError, match="q=\\(0,0\\)"):
        _parse_case("zero:0:0")
    with pytest.raises(ValueError, match="duplicate q coordinate"):
        _validate_cases((CaseSpec("a", 1, 0), CaseSpec("b", 1, 0)))


def test_auto_budget_accounts_for_complementary_odd_orbit_origin():
    odd = CaseSpec("odd", 1, 0)
    even = CaseSpec("even", 6, 4)
    assert _orbit_origin_count(odd, "auto") == 2
    assert _orbit_origin_count(even, "auto") == 1
    assert _orbit_origin_count(odd, "none") == 1
    assert _resolved_budget(
        0,
        nk=1256,
        case=odd,
        subgrid_average="auto",
        transverse_order=256,
    ) == 1256 * 2 * 256
    assert _resolved_budget(
        123,
        nk=1256,
        case=odd,
        subgrid_average="auto",
        transverse_order=256,
    ) == 123


def test_selection_keeps_mandatory_and_failures_then_adds_worst():
    cases = (
        CaseSpec("mandatory", 1, 0, "tight"),
        CaseSpec("failed", 1, 1),
        CaseSpec("hard", 2, 1),
        CaseSpec("easy", 3, 2),
    )
    metrics = {
        "mandatory": CaseMetrics("mandatory", True, True, True, 0.1, 0.01, 10.0),
        "failed": CaseMetrics("failed", True, False, False, 0.2, 0.02, 20.0),
        "hard": CaseMetrics("hard", True, True, True, 0.8, 0.03, 30.0),
        "easy": CaseMetrics("easy", True, True, True, 0.01, 0.001, 3.0),
    }
    assert _select_labels(
        cases,
        metrics,
        mandatory_attribute="mandatory_tight",
        top_k=1,
        include_failures=True,
    ) == ("mandatory", "failed", "hard")


def test_resume_requires_exact_fingerprint_and_child_json(tmp_path: Path):
    case = CaseSpec("case", 1, 0)
    output = tmp_path / "result.csv"
    fingerprint = _fingerprint({"a": 1})
    task = TaskSpec(
        case,
        "periodic_screen",
        output,
        ("python", "child"),
        fingerprint,
    )
    output.write_text("header\n", encoding="utf-8")
    task.output_json.write_text(
        json.dumps({"schema": "child_v1"}),
        encoding="utf-8",
    )
    task.manifest_path.write_text(
        json.dumps(
            {
                "schema": "dwave_positive_orbit_certification_task_v1",
                "state": "completed",
                "fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    assert _resume_match(task)
    task.manifest_path.write_text(
        json.dumps(
            {
                "schema": "dwave_positive_orbit_certification_task_v1",
                "state": "completed",
                "fingerprint": "different",
            }
        ),
        encoding="utf-8",
    )
    assert not _resume_match(task)
