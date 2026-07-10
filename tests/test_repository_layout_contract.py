from __future__ import annotations

from pathlib import Path

from validation.run_static_nk_scan import DEFAULT_OUTPUT


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


def test_active_validation_surface_is_minimal() -> None:
    assert (ROOT / "validation/run_static_nk_scan.py").is_file()
    assert (ROOT / "validation/lib/finite_q_validation_models.py").is_file()
    assert (
        ROOT
        / "validation/outputs/zero_matsubara/static_nk_convergence/README.md"
    ).is_file()


def test_static_scan_default_stays_under_validation_outputs() -> None:
    expected = Path(
        "validation/outputs/zero_matsubara/static_nk_convergence/raw/static_nk_scan.csv"
    )
    assert DEFAULT_OUTPUT == expected
