import json
import importlib.util
from pathlib import Path

import numpy as np

from lno327.plotting import (
    plot_band_structure,
    plot_fermi_surface,
    plot_fermi_surface_gap,
)


def _assert_plot_and_metadata(path: Path):
    assert path.exists()
    metadata_path = path.with_suffix(".json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["sanity_plot_only"] is True
    assert metadata["not_casimir_input"] is True
    return metadata


def _plot_model_sanity_module():
    script_path = Path("scripts/models/plot_model_sanity.py")
    spec = importlib.util.spec_from_file_location("plot_model_sanity", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    fermi_gap_path = tmp_path / "fermi_gap.png"
    summary = plot_fermi_surface_gap(
        kx_grid,
        ky_grid,
        np.stack([kx_grid**2 + ky_grid**2 - 1.0]),
        np.stack([kx_grid]),
        fermi_gap_path,
        title="fermi gap",
        metadata=metadata,
    )
    _assert_plot_and_metadata(fermi_gap_path)
    assert summary
    assert any(row["has_sign_change"] for row in summary)


def test_old_model_sanity_plotting_entrypoints_are_removed():
    assert not Path("scripts/normal_state/inspect_band_structure.py").exists()
    assert not Path("scripts/pairing/inspect_gap_structure.py").exists()
    assert not Path("scripts/pairing/inspect_pairing_structure.py").exists()


def test_generate_plots_records_fermi_gap_and_path_metadata(tmp_path):
    module = _plot_model_sanity_module()

    module.generate_plots(
        model_name="symmetry_bdg_2band",
        nk=21,
        channels=("spp",),
        plots=("band", "fermi", "fermi-gap"),
        output_root=tmp_path,
        path_points=5,
        gap_projection_gauge="anchor",
        gap_value_mode="real",
    )

    model_dir = tmp_path / "symmetry_bdg_2band"
    assert (model_dir / "band_structure" / "normal_bands.png").exists()
    assert (model_dir / "fermi_surface" / "normal_fermi_surface.png").exists()
    assert (model_dir / "fermi_gap" / "spp_real.png").exists()
    assert (model_dir / "fermi_gap" / "spp_summary.csv").exists()
    assert (model_dir / "fermi_gap" / "spp_summary.json").exists()
    assert not (model_dir / "gap_texture").exists()
    assert not (model_dir / "bdg_min_gap").exists()

    band_metadata = json.loads((model_dir / "band_structure" / "normal_bands.json").read_text())
    assert band_metadata["k_path"] == "Gamma-X-M-Gamma"
    assert band_metadata["path_points_per_segment"] == 5
    assert band_metadata["path_num_points"] == 16

    fermi_gap_metadata = json.loads((model_dir / "fermi_gap" / "spp_real.json").read_text())
    assert fermi_gap_metadata["plot"] == "fermi_gap"
    assert fermi_gap_metadata["domain"] == "fermi_surface"
    assert fermi_gap_metadata["projected_gap_sanity_quantity"] is True
    assert fermi_gap_metadata["gap_projection_gauge"] == "anchor"
    assert fermi_gap_metadata["gap_value_mode"] == "real"


def test_generate_plots_rejects_removed_plot_names(tmp_path):
    module = _plot_model_sanity_module()

    for plot_name in ("gap", "bdg-gap"):
        try:
            module.generate_plots(
                model_name="symmetry_bdg_2band",
                nk=7,
                channels=("spp",),
                plots=(plot_name,),
                output_root=tmp_path,
            )
        except ValueError as exc:
            assert "full-BZ gap texture and BdG min-gap plots are no longer main model sanity outputs" in str(exc)
        else:
            raise AssertionError(f"{plot_name} should have been rejected")
