"""Registry for available BdG model specs."""

from __future__ import annotations

from importlib import import_module

_MODEL_NAMES = ("lno327_four_orbital", "symmetry_bdg_2band")


def available_models() -> tuple[str, ...]:
    return _MODEL_NAMES


def build_model_spec(name: str):
    if name == "lno327_four_orbital":
        from lno327.models.lno327_four_orbital.spec import LNO327FourOrbitalSpec

        return LNO327FourOrbitalSpec()
    if name == "symmetry_bdg_2band":
        from lno327.models.symmetry_bdg_2band.spec import SymmetryBdG2BandSpec

        return SymmetryBdG2BandSpec()
    raise ValueError(f"unknown model: {name}")


def get_observables_module(name: str):
    if name not in _MODEL_NAMES:
        raise ValueError(f"unknown model: {name}")
    return import_module(f"lno327.models.{name}.observables")
