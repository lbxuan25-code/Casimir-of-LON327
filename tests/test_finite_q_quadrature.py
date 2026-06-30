from __future__ import annotations

from pathlib import Path

import numpy as np

from lno327.finite_q_quadrature import FiniteQQuadratureOptions, finite_q_quadrature_points


def test_uniform_quadrature_metadata_and_weights() -> None:
    points, weights, metadata = finite_q_quadrature_points(
        np.array([0.01, 0.02]),
        FiniteQQuadratureOptions(integration_strategy="uniform", coarse_grid=4),
    )

    assert points.shape == (16, 2)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-14, rtol=0.0)
    assert metadata["integration_strategy"] == "uniform"
    assert metadata["num_cells_refined"] == 0
    assert metadata["num_cells_unrefined"] == 16
    assert metadata["parent_child_double_counting"] is False


def test_adaptive_level_zero_uses_final_cell_gauss_quadrature() -> None:
    points, weights, metadata = finite_q_quadrature_points(
        np.array([0.01, 0.02]),
        FiniteQQuadratureOptions(
            integration_strategy="best_available_adaptive",
            coarse_grid=4,
            adaptive_level=0,
            gauss_order=2,
            fermi_window_eV=1.0e9,
        ),
    )

    assert metadata["integration_strategy"] == "best_available_adaptive"
    assert metadata["num_cells_total"] == 16
    assert metadata["num_quadrature_points"] == 16 * 2 * 2
    assert points.shape == (64, 2)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-14, rtol=0.0)


def test_adaptive_level_is_recursive_four_way_refinement() -> None:
    q = np.array([0.01, 0.02])
    level1_points, level1_weights, level1_meta = finite_q_quadrature_points(
        q,
        FiniteQQuadratureOptions(
            integration_strategy="best_available_adaptive",
            coarse_grid=2,
            adaptive_level=1,
            gauss_order=1,
            fermi_window_eV=1.0e9,
        ),
    )
    level2_points, level2_weights, level2_meta = finite_q_quadrature_points(
        q,
        FiniteQQuadratureOptions(
            integration_strategy="best_available_adaptive",
            coarse_grid=2,
            adaptive_level=2,
            gauss_order=1,
            fermi_window_eV=1.0e9,
        ),
    )

    assert level1_meta["num_base_cells"] == 4
    assert level1_meta["num_flagged_base_cells"] == 4
    assert level1_meta["num_cells_total"] == 16
    assert level1_meta["num_cells_refined"] == 4
    assert level1_meta["num_quadrature_points"] == 16
    assert level1_points.shape == (16, 2)
    np.testing.assert_allclose(level1_weights.sum(), 1.0, atol=1e-14, rtol=0.0)

    assert level2_meta["num_base_cells"] == 4
    assert level2_meta["num_flagged_base_cells"] == 4
    assert level2_meta["num_cells_total"] == 64
    assert level2_meta["num_cells_refined"] == 20
    assert level2_meta["num_quadrature_points"] == 64
    assert level2_points.shape == (64, 2)
    np.testing.assert_allclose(level2_weights.sum(), 1.0, atol=1e-14, rtol=0.0)


def test_adaptive_metadata_records_stage4_15_semantics() -> None:
    q = np.array([0.013, 0.007])
    _points, _weights, metadata = finite_q_quadrature_points(
        q,
        FiniteQQuadratureOptions(
            integration_strategy="best_available_adaptive",
            coarse_grid=2,
            adaptive_level=1,
            gauss_order=1,
            fermi_window_eV=1.0e9,
        ),
    )

    assert metadata["q_model_used_for_quadrature"] == [float(q[0]), float(q[1])]
    assert "num_flagged_base_cells" in metadata
    assert metadata["validation_semantics"] == "stage4_15_build_adaptive_cells_and_quadrature_points_for_cells"
    assert metadata["parent_child_double_counting"] is False


def test_main_pipeline_and_quadrature_do_not_import_validation_scripts() -> None:
    for relative in (
        "scripts/casimir/finite_q_bdg_casimir_pipeline.py",
        "src/lno327/finite_q_quadrature.py",
    ):
        text = Path(relative).read_text(encoding="utf-8")
        assert "validation/scripts" not in text
        assert "validation.scripts" not in text
