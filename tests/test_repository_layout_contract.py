from __future__ import annotations

from pathlib import Path

from validation.commands.static.nk_scan import DEFAULT_OUTPUT


ROOT = Path(__file__).resolve().parents[1]


def test_retired_repository_trees_are_absent() -> None:
    retired = (
        "results",
        "sandbox",
        "validation/scripts",
        "outputs/casimir/finite_q_bdg_pipeline",
    )
    for relative in retired:
        assert not (ROOT / relative).exists(), relative


def test_active_validation_surface_is_grouped() -> None:
    validation_root = ROOT / "validation"
    assert (validation_root / "__main__.py").is_file()
    assert (validation_root / "commands/ward").is_dir()
    assert (validation_root / "commands/static").is_dir()
    assert (validation_root / "commands/matsubara").is_dir()
    assert (validation_root / "lib/finite_q_validation_models.py").is_file()
    assert (
        validation_root
        / "outputs/zero_matsubara/static_nk_convergence/README.md"
    ).is_file()


def test_validation_root_has_no_bare_runner_or_analyzer_modules() -> None:
    root_modules = {
        path.name
        for path in (ROOT / "validation").glob("*.py")
    }
    assert root_modules == {"__init__.py", "__main__.py"}


def test_static_scan_default_stays_under_validation_outputs() -> None:
    expected = Path(
        "validation/outputs/zero_matsubara/static_nk_convergence/raw/static_nk_scan.csv"
    )
    assert DEFAULT_OUTPUT == expected
