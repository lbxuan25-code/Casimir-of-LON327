from __future__ import annotations

import numpy as np

from lno327.constants import KB_EV_PER_K
from validation.lib.dwave_positive_orbit_gauss import (
    integrate_dwave_positive_orbit_gauss,
)
from validation.lib.finite_q_validation_models import (
    get_finite_q_validation_model,
)


def _small_dwave_gauss(*, workers: int, task_size: int):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        "dwave",
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(0.1)
    temperature = 10.0
    xi = np.asarray([2.0 * np.pi * KB_EV_PER_K * temperature])

    return integrate_dwave_positive_orbit_gauss(
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
        transverse_workers=workers,
        transverse_task_size=task_size,
    )


def test_fixed_gauss_reuses_batched_complete_orbit_evaluator() -> None:
    result = _small_dwave_gauss(workers=1, task_size=1)

    profile = result.evaluator_profile
    assert profile.callbacks == 4
    # m=(1,0) requires two complementary origins, so every t has 2*nk points.
    assert profile.complete_orbit_points == 32
    assert profile.material_workspace_implementation == "batched_model_capability"
    assert profile.q_workspace_implementation == "batched_model_capability"
    assert result.quadrature.transverse_evaluations == 4
    assert result.quadrature.point_evaluations == 32
    assert result.components[0].metadata["q_workspace_implementation"] == (
        "batched_model_capability"
    )
    assert result.components[0].metadata["material_workspace_implementation"] == (
        "batched_model_capability"
    )


def test_threaded_dwave_gauss_matches_serial_primitive_integral() -> None:
    serial = _small_dwave_gauss(workers=1, task_size=1)
    threaded = _small_dwave_gauss(workers=2, task_size=2)

    np.testing.assert_array_equal(threaded.quadrature.value, serial.quadrature.value)
    assert threaded.evaluator_profile.callbacks == 4
    assert threaded.evaluator_profile.complete_orbit_points == 32
    assert threaded.quadrature.transverse_workers == 2
    assert threaded.quadrature.transverse_task_size == 2
    assert threaded.quadrature.transverse_task_count == 2
    assert threaded.quadrature.execution_strategy == (
        "threaded_transverse_nodes_ordered_parent_reduction"
    )
    assert threaded.components[0].metadata["fixed_gauss_transverse_workers"] == 2
    assert threaded.components[0].metadata["fixed_gauss_transverse_task_size"] == 2
