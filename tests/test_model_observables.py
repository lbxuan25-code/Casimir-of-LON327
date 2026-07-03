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
    assert observables.band_projected_gap(kx, ky, channel, spec).shape == gap_shape
    assert observables.min_positive_bdg_energy(kx, ky, channel, spec) >= 0.0
