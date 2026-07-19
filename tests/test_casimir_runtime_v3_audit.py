from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from lno327.casimir.adaptive_joint_q import run_adaptive_joint_casimir
from lno327.casimir.adaptive_matsubara_tail import AdaptiveMatsubaraCasimirConfig
from lno327.casimir.certified_point_provider import (
    CertifiedOuterQProvider,
    FrequencyExtendableCertifiedOuterQProvider,
    certified_point_policy_fingerprint,
)
from lno327.casimir.fixed_chain import (
    FixedCasimirConfig,
    FixedCasimirExecutionError,
    _CertificationRun,
    _run_transverse_certifier,
)
from lno327.casimir.fixed_transverse_point_engine import _parse_args
from lno327.casimir.production import build_full_casimir_config
from scripts.full_casimir.cache_migration import reassess_point
from scripts.full_casimir.config import (
    DEFAULT_CERTIFIER_Q_BATCH_SIZE,
    DEFAULT_LOGDET_RTOL,
    DEFAULT_RESERVED_LOGICAL_CPUS,
    DEFAULT_WORKER_CAP,
    PILOT_PROFILE,
    PROFILE_NAME,
)
from scripts.full_casimir.energy import _case_state


def _point(pairing: str, label: str, n: int, value: float) -> dict:
    return {
        "pairing": pairing,
        "q_label": label,
        "n": n,
        "sweet_spot": {"status": "established", "working_N": 192, "audit_N": 256},
        "history": [{
            "N": 256,
            "two_plate_logdet_cross_shift": {"passed": True},
            "shifts": {"primary": {"two_plate_logdet": value, "hard_physical_passed": True}},
        }],
    }


def _successful(current, manifest, output):
    payload = {
        "schema": "transverse-point-sweet-spot-v4",
        "point_results": [
            _point(pairing, label, n, 1.0)
            for label in manifest.labels
            for pairing in current.pairings
            for n in current.matsubara_indices
        ],
    }
    return _CertificationRun(payload, "", "", ("python",))


def test_q_points_file_accepts_negative_scientific_notation(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"label": "q", "q_lab": [-1.0e-17, 0.25]}]))
    args = _parse_args(["--q-points-file", str(path), "--N-candidates", "2", "4", "6"])
    assert args.q_points[0]["q_lab"][0] == -1.0e-17


def test_transverse_runner_uses_q_file_not_inline_negative_values(tmp_path: Path, monkeypatch) -> None:
    observed = {}
    def fake_run(command, **kwargs):
        observed["command"] = command
        q_path = Path(command[command.index("--q-points-file") + 1])
        assert json.loads(q_path.read_text())[0]["q_lab"][0] == -1.0e-17
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"schema": "transverse-point-sweet-spot-v4"}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr("lno327.casimir.fixed_chain.subprocess.run", fake_run)
    manifest = SimpleNamespace(labels=("q",), q_model=np.asarray([[-1e-17, 0.25]]))
    _run_transverse_certifier(FixedCasimirConfig(), manifest, tmp_path / "out.json")
    assert "--q-points-file" in observed["command"]
    assert "--q-point" not in observed["command"]


def test_scheduling_changes_do_not_invalidate_frequency_cache(tmp_path: Path) -> None:
    base = FixedCasimirConfig(matsubara_indices=(0, 1), workers=28, parallel_mode="q")
    provider = FrequencyExtendableCertifiedOuterQProvider(base, runner=_successful)
    changed = replace(base, matsubara_indices=(0, 1, 2), workers=26, memory_budget_gb=12.0)
    provider.reconfigure(changed)
    assert certified_point_policy_fingerprint(base) != certified_point_policy_fingerprint(
        replace(base, logdet_rtol=1.5e-3)
    )


def test_provider_rejects_missing_or_duplicate_certifier_rows() -> None:
    def missing(current, manifest, output):
        return _CertificationRun({"schema": "transverse-point-sweet-spot-v4", "point_results": []}, "", "", ())
    provider = CertifiedOuterQProvider(FixedCasimirConfig(matsubara_indices=(0,)), runner=missing)
    with pytest.raises(FixedCasimirExecutionError, match="omitted"):
        provider.evaluate(np.asarray([[0.1, 0.2]]))


def test_provider_telemetry_survives_resume_and_compact_summary(tmp_path: Path) -> None:
    cache = tmp_path / "points.json"
    provider = CertifiedOuterQProvider(
        FixedCasimirConfig(matsubara_indices=(0,)), cache_path=cache,
        runner=_successful, certifier_q_batch_size=1)
    provider.evaluate(np.asarray([[0.1, 0.2]]))
    restored = CertifiedOuterQProvider(
        FixedCasimirConfig(matsubara_indices=(0,)), cache_path=cache,
        runner=_successful, certifier_q_batch_size=1)
    assert restored.performance_summary()["certification_batches"] == 1
    assert "certifier_batch_records" not in restored.performance_summary()
    assert len(restored.performance_statistics()["certifier_batch_records"]) == 1


def test_runtime_v3_defaults_are_calibrated() -> None:
    config = build_full_casimir_config(pairings=("dwave",))
    point = config.outer_tail_config.joint_config.radial_config.point_config
    assert point.logdet_rtol == pytest.approx(1.5e-3)
    assert point.logdet_atol == pytest.approx(1e-6)
    assert config.certifier_q_batch_size == 512
    assert config.total_free_energy_rtol == pytest.approx(5e-3)
    assert config.total_free_energy_atol_J_m2 == pytest.approx(1e-12)
    assert DEFAULT_LOGDET_RTOL == pytest.approx(1.5e-3)
    assert DEFAULT_CERTIFIER_Q_BATCH_SIZE == 512
    assert DEFAULT_RESERVED_LOGICAL_CPUS == 6
    assert DEFAULT_WORKER_CAP == 26
    assert PILOT_PROFILE == "0deg_pilot_v3"
    assert PROFILE_NAME == "runtime_budget_v3"
    payload = config.as_dict()
    assert payload["per_term_outer_budget_policy"] == "active_term_count"


def test_case_state_recognizes_successful_summary_status(tmp_path: Path) -> None:
    run = tmp_path / "case"
    run.mkdir()
    reason = "outer_and_matsubara_cutoff_tail_tolerances_met"
    common = {
        "selected_matsubara_cutoff": 1,
        "production_casimir_allowed": False,
        "provider_statistics": {},
    }
    (run / "manifest.json").write_text(json.dumps({
        "schema": "full-casimir-run-manifest",
        "case": "case",
        "status": "completed",
        "termination_reason": reason,
    }))
    (run / "summary.json").write_text(json.dumps({
        "schema": "full-casimir-run-summary",
        "case": "case",
        "status": "adaptive_tail_bounded",
        "matsubara_converged": True,
        "termination_reason": reason,
        "pairings": {},
        **common,
    }))
    (run / "result.json").write_text(json.dumps({
        "schema": "adaptive-matsubara-casimir-result-v1",
        "status": "adaptive_tail_bounded",
        "matsubara_converged": True,
        "termination_reason": reason,
        "pairing_results": {},
        **common,
    }))
    assert _case_state(run) == "completed"


def test_reassessment_can_establish_point_under_calibrated_rtol() -> None:
    shifts = {
        "s0": {"two_plate_logdet": 1.0, "hard_physical_passed": True},
        "s1": {"two_plate_logdet": 1.0005, "hard_physical_passed": True},
    }
    history = []
    for N, offset in ((128, 0.0), (192, 0.0014), (256, 0.0028)):
        history.append({"N": N, "shifts": {
            key: {**value, "two_plate_logdet": value["two_plate_logdet"] + offset}
            for key, value in shifts.items()
        }})
    point = {"pairing": "dwave", "q_label": "q", "n": 1, "history": history,
             "sweet_spot": {"status": "not_established"}}
    strict = reassess_point(point, rtol=1e-3, atol=1e-6, required_consecutive_passes=2)
    relaxed = reassess_point(point, rtol=1.5e-3, atol=1e-6, required_consecutive_passes=2)
    assert strict["sweet_spot"]["status"] == "not_established"
    assert relaxed["sweet_spot"]["status"] == "established"


def test_matsubara_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="certifier_q_batch_size"):
        AdaptiveMatsubaraCasimirConfig(certifier_q_batch_size=0)
