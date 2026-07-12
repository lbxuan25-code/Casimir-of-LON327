from __future__ import annotations

import numpy as np

from lno327 import KuboConfig
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.__main__ import resolve_command
from validation.commands.ward.bond_metric_full_kernel import (
    complementary_subgrid_origins,
)
from validation.lib.commensurate_dwave_bdg_cache import (
    CachedCommensurateDWaveContext,
    build_commensurate_dwave_bdg_cache,
)
from validation.lib.commensurate_periodic import CommensuratePeriodicGrid
from validation.lib.dwave_iterated_adaptive import (
    build_dwave_static_integrand_context as build_reference_context,
)
from validation.lib.dwave_iterated_adaptive_fast import (
    build_dwave_static_integrand_context as build_fast_context,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def _validation_state(q: np.ndarray):
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    options = FiniteQEngineOptions()
    return model, ansatz, pairing, config, options


def test_fast_primitive_chunk_matches_reference_pointwise_contract():
    q = np.asarray([0.11, 0.07])
    model, ansatz, pairing, config, options = _validation_state(q)
    points = np.asarray(
        [
            [-2.1, -1.4],
            [-0.3, 0.8],
            [1.2, 2.4],
            [2.8, -2.5],
        ],
        dtype=float,
    )

    reference = build_reference_context(
        model.spec, ansatz, q, config, pairing, options
    ).evaluate_complex(points)
    fast = build_fast_context(
        model.spec, ansatz, q, config, pairing, options
    ).evaluate_complex(points)

    assert reference.shape == fast.shape == (4, 48)
    difference = np.abs(fast - reference)
    flat_index = int(np.argmax(difference))
    index = np.unravel_index(flat_index, difference.shape)
    assert np.allclose(fast, reference, rtol=3e-12, atol=3e-13), (
        f"largest primitive mismatch at point={index[0]}, component={index[1]}: "
        f"fast={fast[index]!r}, reference={reference[index]!r}, "
        f"abs_difference={difference[index]:.16e}"
    )


def test_commensurate_cache_matches_fast_evaluator_for_even_and_odd_half_q():
    nk = 8
    for mx, my, expected_subgrids in ((2, 0, 1), (3, 2, 2)):
        q = (2.0 * np.pi / nk) * np.asarray([mx, my], dtype=float)
        model, ansatz, pairing, config, options = _validation_state(q)
        fast_context = build_fast_context(
            model.spec, ansatz, q, config, pairing, options
        )
        origins = complementary_subgrid_origins(mx, my, 0.5, 0.5)
        cache = build_commensurate_dwave_bdg_cache(
            fast_context,
            nk=nk,
            mx=mx,
            my=my,
            origins=origins,
            chunk_size=16,
            max_points=64,
        )
        assert len(cache.subgrids) == expected_subgrids
        assert cache.eigensystem_count == expected_subgrids * nk * nk

        for origin in origins:
            grid = CommensuratePeriodicGrid(
                nk=nk,
                mx=mx,
                my=my,
                shift_x=origin[0],
                shift_y=origin[1],
                max_points=64,
            )
            points = np.concatenate(list(grid.iter_point_chunks(16)), axis=0)
            expected = fast_context.evaluate_complex(points)
            actual = CachedCommensurateDWaveContext(
                cache, origin
            ).evaluate_complex(points)
            assert np.allclose(actual, expected, rtol=5e-12, atol=5e-13)


def test_public_ward_routes_use_optimized_modules():
    assert resolve_command("ward", "bond-metric-full-kernel").endswith(
        "bond_metric_full_kernel_fast"
    )
    assert resolve_command("ward", "bond-metric-family").endswith(
        "bond_metric_family_fast"
    )
