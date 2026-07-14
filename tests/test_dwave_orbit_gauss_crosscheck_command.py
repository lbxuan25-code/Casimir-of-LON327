from __future__ import annotations

from pathlib import Path

import pytest

from validation.commands.matsubara.dwave_orbit_gauss_crosscheck import _parse_args


def test_gauss_crosscheck_allows_consecutive_order_mode_without_reference(
    tmp_path: Path,
) -> None:
    output = tmp_path / "crosscheck.csv"
    args = _parse_args(
        [
            "--nk",
            "24",
            "--mx",
            "1",
            "--my",
            "1",
            "--matsubara-indices",
            "1",
            "2",
            "--gauss-orders",
            "32",
            "48",
            "--panel-count",
            "16",
            "--transverse-workers",
            "4",
            "--transverse-task-size",
            "3",
            "--output",
            str(output),
        ]
    )

    assert args.reference_csv is None
    assert args.gauss_orders == [32, 48]
    assert args.panel_count == 16
    assert args.transverse_workers == 4
    assert args.transverse_task_size == 3
    assert args.output == output


def test_gauss_crosscheck_rejects_missing_explicit_reference(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(SystemExit):
        _parse_args(["--reference-csv", str(missing)])


def test_gauss_crosscheck_rejects_nonpositive_parallel_controls() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--transverse-workers", "0"])
    with pytest.raises(SystemExit):
        _parse_args(["--transverse-task-size", "0"])
