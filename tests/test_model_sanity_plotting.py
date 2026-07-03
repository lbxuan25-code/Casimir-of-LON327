import json
from pathlib import Path

import numpy as np

from lno327.plotting import (
    plot_band_structure,
    plot_bdg_min_gap,
    plot_fermi_surface,
    plot_gap_texture,
)


def _assert_plot_and_metadata(path: Path):
    assert path.exists()
    metadata_path = path.with_suffix(".json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["sanity_plot_only"] is True
    assert metadata["not_casimir_input"] is True


def test_shared_plotting_backend_writes_png_and_metadata(tmp_path):
    nk = 11
    axis = np.linspace(-np.pi, np.pi, nk)
    kx_grid, ky_grid = np.meshgrid(axis, axis, indexing="ij")
    band_energies = np.stack(
        [
            np.cos(kx_grid) + np.cos(ky_grid),
            np.cos(kx_grid) - np.cos(ky_grid),
        ]
    )
    metadata = {"model": "unit_test", "nk": nk}

    band_path = tmp_path / "band.png"
    plot_band_structure(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0, 1.0], [0.2, 0.8], [0.4, 0.6]]),
        (0.0, 1.0, 2.0),
        ("G", "X", "M"),
        band_path,
        title="bands",
        metadata=metadata,
    )
    _assert_plot_and_metadata(band_path)

    fermi_path = tmp_path / "fermi.png"
    plot_fermi_surface(kx_grid, ky_grid, band_energies, fermi_path, title="fermi", metadata=metadata)
    _assert_plot_and_metadata(fermi_path)

    gap_path = tmp_path / "gap.png"
    plot_gap_texture(
        kx_grid,
        ky_grid,
        np.cos(kx_grid) - np.cos(ky_grid),
        gap_path,
        title="gap",
        fermi_contours=band_energies,
        metadata=metadata,
    )
    _assert_plot_and_metadata(gap_path)

    bdg_gap_path = tmp_path / "bdg_gap.png"
    plot_bdg_min_gap(kx_grid, ky_grid, np.abs(np.sin(kx_grid)), bdg_gap_path, title="bdg", metadata=metadata)
    _assert_plot_and_metadata(bdg_gap_path)


def test_old_model_sanity_plotting_entrypoints_are_removed():
    assert not Path("scripts/normal_state/inspect_band_structure.py").exists()
    assert not Path("scripts/pairing/inspect_gap_structure.py").exists()
    assert not Path("scripts/pairing/inspect_pairing_structure.py").exists()
