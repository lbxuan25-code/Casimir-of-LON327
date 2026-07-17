from __future__ import annotations

import json
from pathlib import Path
import pytest
from lno327.casimir.cli import execute_case


class _FakeConfig:
    def as_dict(self):
        return {"schema": "fake-config", "pairings": ["spm"]}


class _FakeResult:
    status = "adaptive_tail_bounded"
    termination_reason = "outer_and_matsubara_cutoff_tail_tolerances_met"
    matsubara_converged = True
    outer_tail_estimated = True
    matsubara_tail_estimated = True
    production_casimir_allowed = False
    selected_matsubara_cutoff = 15
    pairing_results = {"spm": {
        "status": "integrated_with_outer_and_matsubara_tail_bounds",
        "finite_matsubara_partial_J_m2": -1.0,
        "finite_matsubara_outer_error_bound_J_m2": 0.01,
        "estimated_matsubara_tail_bound_J_m2": 0.02,
        "estimated_total_error_J_m2": 0.03,
        "total_free_energy_tolerance_J_m2": 0.05,
        "matsubara_tail_ratio_envelope": 0.2,
        "matsubara_tail_decay_passed": True,
        "finite_matsubara_budget_passed": True,
        "matsubara_tail_budget_passed": True,
        "total_free_energy_budget_passed": True,
    }}
    cutoff_records = ({"selected_u_max": 42.0},)
    provider_statistics = {"cached_point_count": 12, "unique_q_count": 4}
    def as_dict(self):
        return {"schema": "adaptive-matsubara-casimir-result-v1", "status": self.status, "production_casimir_allowed": False}


def test_named_case_writes_one_deterministic_run_layout(tmp_path: Path) -> None:
    seen = {}
    def builder(**kwargs):
        seen.update(kwargs)
        return _FakeConfig()
    result = execute_case(case="spm_T10K_d20nm_theta17deg", output_root=tmp_path, resume=False,
                          config_builder=builder, runner=lambda config: _FakeResult(), pairings=("spm",))
    run = tmp_path / "spm_T10K_d20nm_theta17deg"
    assert result.matsubara_converged
    assert seen["point_cache_path"] == run / "cache" / "certified_points.json"
    assert {path.name for path in run.iterdir()} == {"config.json", "manifest.json", "result.json", "summary.json"}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["paths"]["point_cache"] == "cache/certified_points.json"
    assert summary["selected_matsubara_cutoff"] == 15
    assert summary["selected_u_max"] == 42.0
    assert summary["production_casimir_allowed"] is False


def test_existing_case_requires_explicit_resume(tmp_path: Path) -> None:
    (tmp_path / "case").mkdir()
    with pytest.raises(FileExistsError, match="--resume"):
        execute_case(case="case", output_root=tmp_path, resume=False,
                     config_builder=lambda **kwargs: _FakeConfig(), runner=lambda config: _FakeResult())


def test_case_name_rejects_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="case must start"):
        execute_case(case="../escape", output_root=tmp_path, resume=False,
                     config_builder=lambda **kwargs: _FakeConfig(), runner=lambda config: _FakeResult())
