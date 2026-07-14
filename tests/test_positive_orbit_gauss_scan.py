from __future__ import annotations

import csv
from pathlib import Path

import pytest

from validation.commands.matsubara.positive_orbit_gauss_scan import (
    _metrics,
    _parse_args,
)


def _write_rows(path: Path, *, sigma: float, reflection: float, logdet: float) -> None:
    rows = []
    for index in (1, 2):
        rows.append(
            {
                "matsubara_index": index,
                "point_pipeline_passed": True,
                "ward_passed": True,
                "sheet_validation_passed": True,
                "reflection_constructed": True,
                "logdet_passed": True,
                "reference_sigma_matrix_relative": sigma,
                "reference_reflection_matrix_relative": reflection,
                "reference_logdet_relative": logdet,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_scan_parser_keeps_one_method_order_ladder() -> None:
    args = _parse_args(
        [
            "--pairings",
            "spm",
            "dwave",
            "--case",
            "small:1:1",
            "--gauss-orders",
            "64",
            "96",
            "128",
            "--panel-count",
            "16",
        ]
    )
    assert args.pairings == ("spm", "dwave")
    assert args.gauss_orders == (64, 96, 128)
    assert args.panel_count == 16
    assert args.cases[0].label == "small"
    assert (args.cases[0].mx, args.cases[0].my) == (1, 1)


def test_scan_parser_rejects_exact_zero_matsubara() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--matsubara-indices", "0", "1"])


def test_scan_metrics_distinguish_strict_and_soft(tmp_path: Path) -> None:
    strict_path = tmp_path / "strict.csv"
    _write_rows(strict_path, sigma=8e-4, reflection=2e-5, logdet=3e-5)
    strict = _metrics(
        strict_path,
        sigma_strict=1e-3,
        sigma_soft=2e-3,
        observable=1e-3,
    )
    assert strict.physical_all
    assert strict.observable_all
    assert strict.strict_all
    assert strict.soft_all

    soft_path = tmp_path / "soft.csv"
    _write_rows(soft_path, sigma=1.6e-3, reflection=2e-5, logdet=3e-5)
    soft = _metrics(
        soft_path,
        sigma_strict=1e-3,
        sigma_soft=2e-3,
        observable=1e-3,
    )
    assert soft.physical_all
    assert soft.observable_all
    assert not soft.strict_all
    assert soft.soft_all


def test_scan_metrics_keep_observable_failure_hard(tmp_path: Path) -> None:
    path = tmp_path / "observable_fail.csv"
    _write_rows(path, sigma=8e-4, reflection=2e-3, logdet=3e-5)
    metrics = _metrics(
        path,
        sigma_strict=1e-3,
        sigma_soft=2e-3,
        observable=1e-3,
    )
    assert metrics.physical_all
    assert not metrics.observable_all
    assert not metrics.strict_all
    assert not metrics.soft_all
