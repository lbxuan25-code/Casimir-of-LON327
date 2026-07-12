from __future__ import annotations

import json
from pathlib import Path

from validation.__main__ import main as validation_main


def test_public_full_kernel_route_records_commensurate_cache(tmp_path: Path):
    output = tmp_path / "cached_full_kernel.csv"
    validation_main(
        [
            "ward",
            "bond-metric-full-kernel",
            "--nk",
            "4",
            "--mx",
            "2",
            "--my",
            "0",
            "--chunk-size",
            "8",
            "--max-points",
            "16",
            "--ward-tolerance",
            "1",
            "--ward-absolute-tolerance",
            "1",
            "--condition-max",
            "1e16",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["schema"] == "dwave_bond_metric_full_kernel_audit_v1"
    assert payload["optimization"]["commensurate_eigensystem_cache_enabled"] is True
    assert payload["optimization"]["cached_subgrid_count"] == 1
    assert payload["optimization"]["cached_eigensystem_count"] == 16
    assert payload["row"]["eigensystem_cache_enabled"] is True
    assert payload["row"]["counterterm_changed_only_22"] is True
