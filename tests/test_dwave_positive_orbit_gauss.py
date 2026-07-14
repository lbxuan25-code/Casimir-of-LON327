from __future__ import annotations

import numpy as np

from lno327.constants import KB_EV_PER_K
from validation.lib.dwave_positive_orbit_gauss import (
    integrate_dwave_positive_orbit_gauss,
)
from validation.lib.finite_q_validation_models import (
    get_finite_q_validation_model,
)


def test_fixed_gauss_reuses_batched_complete_orbit_evaluator() -> None:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        "dwave",
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(0.1)
    temperature = 10.0
    xi = np.asarray([2.0 * np.pi * KB_EV_PER_K * temperature])

    result = integrate_dwave_positive_orbit_gauss(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=temperature,
        eta_eV=1e-8,
        nk=4,
        mx=1,
        my=0,
        transverse_order=4,
        max_point_evaluations=64,
    )

    profile = result.evaluator_profile
    assert profile.callbacks == 4
    assert profile.complete_orbit_points == 16
    assert profile.q_workspace_implementation == "batched_model_capability"
    assert result.quadrature.transverse_evaluations == 4
    assert result.quadrature.point_evaluations == 16
    assert result.components[0].metadata["q_workspace_implementation"] == (
        "batched_model_capability"
    )
