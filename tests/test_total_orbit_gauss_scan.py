from __future__ import annotations

import csv
from pathlib import Path

import pytest

from validation.commands.matsubara.total_orbit_gauss_scan import (
    _metrics,
    _parse_args,
    _sector_acceptance,
)


def _write_rows(
    path: Path,
    *,
    static_response: float = 8e-4,
    positive_response: float = 8e-4,
    reflection: float = 2e-5,
    logdet: float = 3e-5,
    static_strict_ward: bool = True,
) -> None:
    rows = []
    for index in (0, 1, 2):
        rows.append(
            {
                "matsubara_index": index,
                "point_pipeline_passed": True,
                "ward_passed": True,
                "strict_static_ward_passed": (
                    static_strict_ward if index == 0 else False
                ),
                "sheet_validation_passed": True,
                "reflection_constructed": True,
                "logdet_passed": True,
                "reference_primary_response_relative": (
                    static_response if index == 0 else positive_response
                ),
                "reference_reflection_matrix_relative": reflection,
                "reference_logdet_relative": logdet,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path):
    return _metrics(
        path,
        static_strict=1e-3,
        static_soft=2e-3,
        sigma_strict=1e-3,
        sigma_soft=2e-3,
        observable=1e-3,
    )


def test_parser_exposes_independent_static_soft_thresholds() -> None:
    args = _parse_args(
        [
            "--case",
            "small:1:1",
            "--matsubara-indices",
            "0",
            "1",
            "--gauss-orders",
            "64",
            "96",
            "--panel-count",
            "16",
            "--strict-static-rtol",
            "8e-4",
            "--soft-static-rtol",
            "2.5e-3",
        ]
    )
    assert args.strict_static_rtol == pytest.approx(8e-4)
    assert args.soft_static_rtol == pytest.approx(2.5e-3)


def test_parser_rejects_reversed_static_thresholds() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--strict-static-rtol",
                "2e-3",
                "--soft-static-rtol",
                "1e-3",
            ]
        )


def test_static_response_may_soft_pass_without_softening_static_ward(
    tmp_path: Path,
) -> None:
    path = tmp_path / "static_soft.csv"
    _write_rows(path, static_response=1.6e-3, positive_response=8e-4)
    metrics = _read(path)
    assert metrics.physical_all
    assert metrics.observable_all
    assert not metrics.static_strict_all
    assert metrics.static_soft_all
    assert metrics.positive_strict_all
    assert not metrics.strict_all
    assert metrics.soft_all
    assert _sector_acceptance(metrics) == ("soft", "strict")


def test_static_ward_failure_remains_hard(tmp_path: Path) -> None:
    path = tmp_path / "static_ward_fail.csv"
    _write_rows(
        path,
        static_response=1.6e-3,
        positive_response=8e-4,
        static_strict_ward=False,
    )
    metrics = _read(path)
    assert not metrics.static_strict_all
    assert not metrics.static_soft_all
    assert not metrics.soft_all
    assert _sector_acceptance(metrics)[0] == "unresolved"


def test_static_and_positive_soft_can_coexist_with_observable_hard_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "both_soft.csv"
    _write_rows(path, static_response=1.5e-3, positive_response=1.7e-3)
    metrics = _read(path)
    assert metrics.static_soft_all
    assert metrics.positive_soft_all
    assert metrics.soft_all
    assert _sector_acceptance(metrics) == ("soft", "soft")

    failed = tmp_path / "observable_fail.csv"
    _write_rows(
        failed,
        static_response=1.5e-3,
        positive_response=1.7e-3,
        reflection=2e-3,
    )
    failed_metrics = _read(failed)
    assert not failed_metrics.observable_all
    assert not failed_metrics.soft_all
