from __future__ import annotations

import pytest

from lno327.casimir.microscopic_model import (
    FiniteQMicroscopicModel,
    available_finite_q_microscopic_models,
    get_finite_q_microscopic_model,
)


def test_active_microscopic_adapter_is_two_band_only() -> None:
    assert available_finite_q_microscopic_models() == ("symmetry_bdg_2band",)
    model = get_finite_q_microscopic_model("symmetry_bdg_2band")
    assert isinstance(model, FiniteQMicroscopicModel)
    assert model.primary_model
    assert model.default_pairings == ("spm", "dwave")
    assert model.pairing_names == ("normal", "spm", "dwave")
    assert model.metadata()["valid_for_casimir_input"] is False
    assert "primary_validation_model" not in model.metadata()


def test_retired_four_orbital_microscopic_adapter_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown finite-q microscopic model"):
        get_finite_q_microscopic_model("lno327_four_orbital")
