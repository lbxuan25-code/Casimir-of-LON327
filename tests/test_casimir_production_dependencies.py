"""Guards for dependencies owned by the production Casimir package."""
from __future__ import annotations

import numpy as np
import pytest

from lno327.casimir.matsubara import matsubara_energy_eV
from lno327.casimir.microscopic_model import (
    FiniteQMicroscopicModel,
    available_finite_q_microscopic_models,
    get_finite_q_microscopic_model,
)
from lno327.constants import KB_EV_PER_K


def test_production_matsubara_energy_helper_matches_exact_definition() -> None:
    assert matsubara_energy_eV(0, 10.0) == 0.0
    expected = 2.0 * np.pi * 3 * KB_EV_PER_K * 10.0
    assert matsubara_energy_eV(3, 10.0) == pytest.approx(
        expected,
        rel=0.0,
        abs=0.0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        matsubara_energy_eV(-1, 10.0)
    with pytest.raises(ValueError, match="positive"):
        matsubara_energy_eV(1, 0.0)


def test_production_finite_q_model_adapter_preserves_active_contract() -> None:
    assert available_finite_q_microscopic_models() == ("symmetry_bdg_2band",)
    model = get_finite_q_microscopic_model("symmetry_bdg_2band")
    assert isinstance(model, FiniteQMicroscopicModel)
    assert model.primary_model
    assert model.default_pairings == ("spm", "dwave")
    assert model.pairing_names == ("normal", "spm", "dwave")
    assert model.metadata()["valid_for_casimir_input"] is False
    assert "primary_validation_model" not in model.metadata()
    assert not hasattr(model, "primary_validation_model")
    assert model.build_ansatz("spm") is not None
    assert model.build_pairing_params(0.1) is not None


def test_unknown_production_microscopic_model_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown finite-q microscopic model"):
        get_finite_q_microscopic_model("lno327_four_orbital")
