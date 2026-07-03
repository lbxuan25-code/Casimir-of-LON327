from types import ModuleType

from lno327.models.registry import available_models, build_model_spec, get_observables_module


def test_available_models_include_both_model_packages():
    assert "lno327_four_orbital" in available_models()
    assert "symmetry_bdg_2band" in available_models()


def test_build_model_spec_constructs_supported_specs():
    for name in available_models():
        spec = build_model_spec(name)
        assert spec.metadata().name == name
        assert tuple(channel.name for channel in spec.channels())


def test_get_observables_module_returns_module():
    for name in available_models():
        module = get_observables_module(name)
        assert isinstance(module, ModuleType)
        assert hasattr(module, "normal_band_energies")
        assert hasattr(module, "band_projected_gap")
