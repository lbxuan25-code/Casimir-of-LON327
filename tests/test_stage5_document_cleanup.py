from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_removed_stage5_zh_notes_are_not_recreated():
    assert list((ROOT / "docs").rglob("stage5_*_zh.md")) == []


def test_tests_do_not_read_removed_stage5_notes():
    removed_path_prefix = "docs" + "/" + "notes" + "/" + "stage5"
    offenders: list[Path] = []
    for path in (ROOT / "tests").rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        if removed_path_prefix in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT))
    assert offenders == []
