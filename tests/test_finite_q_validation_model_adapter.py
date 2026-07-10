from __future__ import annotations

import pytest

from validation.lib.finite_q_validation_models import (
    available_finite_q_validation_models,
    get_finite_q_validation_model,
)


def test_active_validation_adapter_is_two_band_only() -> None:
    assert available_finite_q_validation_models() == ("symmetry_bdg_2band",)
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    assert model.primary_validation_model
    assert model.default_pairings == ("spm", "dwave")
    assert model.pairing_names == ("normal", "spm", "dwave")


def test_retired_four_orbital_validation_adapter_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown finite-q validation model"):
        get_finite_q_validation_model("lno327_four_orbital")
