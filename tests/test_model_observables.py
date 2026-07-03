import pytest

from lno327.models.registry import build_model_spec, get_observables_module


@pytest.mark.parametrize(
    ("model_name", "normal_shape", "bdg_shape", "gap_shape"),
    [
        ("lno327_four_orbital", (4,), (8,), (4,)),
        ("symmetry_bdg_2band", (2,), (4,), (2,)),
    ],
)
def test_model_observable_shapes(model_name, normal_shape, bdg_shape, gap_shape):
    spec = build_model_spec(model_name)
    observables = get_observables_module(model_name)
    channel = next(item.name for item in spec.channels() if item.name != "normal")
    kx, ky = 0.37, -0.22

    assert observables.normal_band_energies(kx, ky, spec).shape == normal_shape
    assert observables.bdg_energies(kx, ky, channel, spec).shape == bdg_shape
    assert observables.band_projected_gap(kx, ky, channel, spec, gauge="anchor").shape == gap_shape
    assert observables.band_projected_gap(kx, ky, channel, spec, gauge="raw").shape == gap_shape
    assert observables.min_positive_bdg_energy(kx, ky, channel, spec) >= 0.0


@pytest.mark.parametrize("model_name", ["lno327_four_orbital", "symmetry_bdg_2band"])
def test_model_observables_reject_unknown_projection_gauge(model_name):
    spec = build_model_spec(model_name)
    observables = get_observables_module(model_name)
    channel = next(item.name for item in spec.channels() if item.name != "normal")

    with pytest.raises(ValueError, match="gauge must be 'anchor' or 'raw'"):
        observables.band_projected_gap(0.37, -0.22, channel, spec, gauge="other")
