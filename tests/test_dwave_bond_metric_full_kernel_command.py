from __future__ import annotations

import json
from pathlib import Path
import sys

from validation.commands.ward.bond_metric_full_kernel import main


def test_bond_metric_full_kernel_command_writes_complete_audit(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "bond_metric_full_kernel.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m validation ward bond-metric-full-kernel",
            "--nk",
            "8",
            "--mx",
            "2",
            "--my",
            "0",
            "--chunk-size",
            "16",
            "--max-points",
            "64",
            "--ward-tolerance",
            "1",
            "--ward-absolute-tolerance",
            "1",
            "--condition-max",
            "1e16",
            "--output",
            str(output),
        ],
    )

    main()

    assert output.is_file()
    assert output.with_suffix(".summary.txt").is_file()
    json_path = output.with_suffix(".json")
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "dwave_bond_metric_full_kernel_audit_v1"
    assert payload["row"]["subgrid_count"] == 1
    assert payload["row"]["counterterm_changed_only_22"] is True
    assert payload["status"] == {
        "diagnostic_only": True,
        "projection_applied": False,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
    }
