from __future__ import annotations

from pathlib import Path

from scripts.full_casimir.energy import _case_state


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
