from __future__ import annotations

from scripts.full_casimir.qualification_prepare import select_holdout_keys


def _entry(pairing: str, n: int, index: int) -> tuple[tuple[str, int, str, str], dict]:
    key = (pairing, n, float(index + 1).hex(), float(index + 2).hex())
    return key, {"point_result": {"sweet_spot": {"status": "established"}}}


def test_stopping_changes_are_sampled_not_all_mandatory(monkeypatch) -> None:
    entries = dict(_entry("spm" if i % 2 == 0 else "dwave", i % 3, i) for i in range(100))
    weighted = {
        key: {"weighted_error_contribution_J_m2": float(100 - i)}
        for i, key in enumerate(entries)
    }
    decisions = [
        {
            "identity": list(key),
            "source_status": "established",
            "target_status": "established",
            "decision_changed": True,
        }
        for key in entries
    ]
    monkeypatch.setattr(
        "scripts.full_casimir.qualification_prepare._local_uncertainty",
        lambda point: 1e-6,
    )

    selected, reasons = select_holdout_keys(
        entries=entries,
        weighted=weighted,
        projection_reports=({"decisions": decisions},),
        max_points=32,
    )

    assert len(selected) == 32
    assert all(
        "acceptance_status_changed_under_frozen_candidate" not in reasons.get(key, set())
        for key in selected
    )


def test_acceptance_status_changes_are_mandatory(monkeypatch) -> None:
    entries = dict(_entry("dwave", 1, i) for i in range(8))
    mandatory = list(entries)[:4]
    decisions = [
        {
            "identity": list(key),
            "source_status": "not_established" if key in mandatory else "established",
            "target_status": "established",
            "decision_changed": True,
        }
        for key in entries
    ]
    monkeypatch.setattr(
        "scripts.full_casimir.qualification_prepare._local_uncertainty",
        lambda point: 1e-6,
    )

    selected, reasons = select_holdout_keys(
        entries=entries,
        weighted={},
        projection_reports=({"decisions": decisions},),
        max_points=6,
    )

    assert set(mandatory).issubset(selected)
    assert all(
        "acceptance_status_changed_under_frozen_candidate" in reasons[key]
        for key in mandatory
    )


def test_mandatory_boundary_points_cannot_be_silently_truncated() -> None:
    entries = dict(_entry("dwave", 1, i) for i in range(5))
    decisions = [
        {
            "identity": list(key),
            "source_status": "not_established",
            "target_status": "established",
            "decision_changed": True,
        }
        for key in entries
    ]

    try:
        select_holdout_keys(
            entries=entries,
            weighted={},
            projection_reports=({"decisions": decisions},),
            max_points=4,
        )
    except ValueError as exc:
        assert "mandatory acceptance-boundary holdout count" in str(exc)
    else:
        raise AssertionError("mandatory boundary overflow must fail closed")
