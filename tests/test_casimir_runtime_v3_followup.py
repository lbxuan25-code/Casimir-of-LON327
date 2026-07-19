from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from lno327.casimir import cli, production
from lno327.casimir.fixed_transverse_point_cli import (
    validate_q_points_file_argument,
)
from lno327.casimir.production import (
    _quarantine_invalid_telemetry,
    build_full_casimir_config,
)
from lno327.casimir.strict_transverse_runner import run_strict_transverse_certifier
from scripts.full_casimir.energy import _case_state
from scripts.full_casimir.postprocess import collect_energy_points


def test_migrated_cache_only_directory_remains_a_valid_seed(tmp_path: Path) -> None:
    run = tmp_path / "case"
    cache = run / "cache" / "certified_points.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{}\n", encoding="utf-8")

    assert _case_state(
        run,
        expected_config={"new": "runtime-v3-policy"},
    ) == "cache_seeded"


def test_run_artifacts_without_config_are_rejected(tmp_path: Path) -> None:
    run = tmp_path / "case"
    run.mkdir()
    (run / "manifest.json").write_text("{}\n", encoding="utf-8")

    assert _case_state(
        run,
        expected_config={"new": "runtime-v3-policy"},
    ) == "configuration_mismatch"


def test_malformed_numeric_telemetry_is_quarantined_without_touching_cache(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache" / "certified_points.json"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"authoritative":"cache"}\n', encoding="utf-8")
    telemetry = cache.with_suffix(".telemetry.json")
    telemetry.write_text(
        json.dumps(
            {
                "schema": "certified-point-provider-telemetry-v1",
                "certification_batches": "not-an-integer",
                "certifier_batch_records": [],
            }
        ),
        encoding="utf-8",
    )
    config = build_full_casimir_config(point_cache_path=cache)

    quarantined = _quarantine_invalid_telemetry(config)

    assert quarantined == telemetry.with_suffix(".json.invalid")
    assert quarantined is not None and quarantined.is_file()
    assert not telemetry.exists()
    assert cache.read_text(encoding="utf-8") == '{"authoritative":"cache"}\n'


def test_q_points_file_requires_explicit_nonempty_string_label(
    tmp_path: Path,
) -> None:
    q_file = tmp_path / "q.json"
    q_file.write_text(
        json.dumps([{"q_lab": [-1.0e-17, 0.25]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nonempty string label"):
        validate_q_points_file_argument(["--q-points-file", str(q_file)])


def test_canonical_route_installs_strict_certifier_runner(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeProvider:
        def __init__(self, config, **kwargs):
            observed["runner"] = kwargs.get("runner")

    sentinel = object()
    monkeypatch.setattr(
        production,
        "FrequencyExtendableCertifiedOuterQProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        production,
        "run_adaptive_matsubara_casimir",
        lambda config, *, provider: sentinel,
    )

    assert production.run_full_casimir(build_full_casimir_config()) is sentinel
    assert observed["runner"] is run_strict_transverse_certifier


def test_postprocess_marks_truncated_result_unusable_instead_of_crashing(
    tmp_path: Path,
) -> None:
    profile = "corrupt_test"
    run = (
        tmp_path
        / "runs"
        / f"spm_T10K_d20nm_theta_p000deg_{profile}"
    )
    run.mkdir(parents=True)
    (run / "summary.json").write_text("{}\n", encoding="utf-8")
    (run / "manifest.json").write_text("{}\n", encoding="utf-8")
    (run / "config.json").write_text("{}\n", encoding="utf-8")
    (run / "result.json").write_text("{truncated", encoding="utf-8")

    points = collect_energy_points(run_root=tmp_path / "runs", profile=profile)

    assert len(points) == 1
    assert points[0].artifact_consistent is False
    assert points[0].usable is False


def test_execute_case_recovers_from_a_truncated_manifest(tmp_path: Path) -> None:
    case = "resume_case"
    run = tmp_path / case
    run.mkdir()
    config_payload = {"policy": "same"}
    (run / "config.json").write_text(
        json.dumps(config_payload),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text("{truncated", encoding="utf-8")

    class DummyConfig:
        def as_dict(self):
            return dict(config_payload)

    class DummyResult:
        status = "adaptive_tail_bounded"
        termination_reason = "done"
        matsubara_converged = True
        outer_tail_estimated = True
        matsubara_tail_estimated = True
        production_casimir_allowed = False
        selected_matsubara_cutoff = 1
        pairing_results = {}
        cutoff_records = ()
        provider_statistics = {}

        def as_dict(self):
            return {
                "schema": "adaptive-matsubara-casimir-result-v1",
                "status": self.status,
                "termination_reason": self.termination_reason,
                "matsubara_converged": self.matsubara_converged,
                "pairing_results": {},
            }

    result = cli.execute_case(
        case=case,
        output_root=tmp_path,
        resume=True,
        config_builder=lambda **kwargs: DummyConfig(),
        runner=lambda config: DummyResult(),
    )

    assert result.matsubara_converged
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["attempt_count"] == 1


def test_git_commit_lookup_is_pinned_to_package_checkout(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(returncode=0, stdout="abc123\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._git_commit() == "abc123"
    command = observed["command"]
    assert isinstance(command, list)
    assert command[:2] == ["git", "-C"]
    assert command[-2:] == ["rev-parse", "HEAD"]
