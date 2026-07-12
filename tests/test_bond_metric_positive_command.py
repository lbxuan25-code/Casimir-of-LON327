from __future__ import annotations

import json
from pathlib import Path
import sys

from validation.commands.matsubara.bond_metric_positive import main


def test_bond_metric_positive_command_writes_outputs(tmp_path: Path, monkeypatch):
    output = tmp_path / "positive.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m validation matsubara bond-metric-positive",
            "--nks",
            "4",
            "6",
            "--matsubara-indices",
            "1",
            "--workers",
            "1",
            "--qx",
            "0.03",
            "--qy",
            "0.02",
            "--ward-tolerance",
            "1",
            "--ward-absolute-tolerance",
            "1",
            "--condition-max",
            "1e16",
            "--convergence-tolerance",
            "1",
            "--output",
            str(output),
        ],
    )

    main()

    assert output.is_file()
    assert output.with_suffix(".summary.txt").is_file()
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["schema"] == "dwave_bond_metric_positive_nk_convergence_v1"
    assert [row["nk"] for row in payload["rows"]] == [4, 6]
    for row in payload["rows"]:
        assert row["matsubara_index"] == 1
        assert row["phase_hessian_policy"] == "nearest_neighbor_bond_metric"
        assert row["valid_for_casimir_input"] is False
        assert "ward_effective_mixed_ratio_max" in row
        assert "sheet_validation_passed" in row
