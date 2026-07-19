from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

from lno327.casimir import cli
from lno327.casimir.production import (
    _quarantine_invalid_telemetry,
    build_full_casimir_config,
)
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
