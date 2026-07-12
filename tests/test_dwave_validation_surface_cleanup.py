from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_retired_dwave_n0_routes_stay_absent():
    retired = (
        "validation/run_dwave_static_adaptive_scan.py",
        "validation/run_dwave_static_vector_adaptive_scan.py",
        "validation/run_dwave_static_periodic_multishift_scan.py",
        "validation/run_dwave_static_global_reference_scan.py",
        "validation/run_dwave_shift_spatial_diagnostic.py",
        "validation/run_dwave_shift_spectrum_diagnostic.py",
        "validation/run_dwave_shift_bandpair_diagnostic.py",
        "validation/run_dwave_shift_signed_reconstruction_diagnostic.py",
        "validation/commands/static/dwave_shift_batch.py",
        "validation/commands/static/dwave_shift_reference.py",
        "validation/commands/static/dwave_shift_budget.py",
        "validation/commands/static/dwave_iterated_adaptive.py",
        "validation/commands/static/dwave_iterated_adaptive_bond_metric.py",
        "validation/commands/static/dwave_adaptive_convergence.py",
        "validation/commands/static/bond_metric_nk_convergence.py",
        "validation/commands/static/bond_metric_shift_convergence.py",
        "validation/commands/ward/contract_audit.py",
        "validation/commands/ward/phase_column.py",
        "validation/commands/ward/phase_hessian.py",
        "validation/commands/ward/phase_hessian_family.py",
        "validation/commands/ward/average_subgrids.py",
        "validation/commands/ward/bond_metric_full_kernel_fast.py",
        "validation/commands/ward/bond_metric_family_fast.py",
        "validation/lib/iterated_adaptive.py",
        "validation/lib/dwave_iterated_adaptive.py",
        "validation/lib/dwave_iterated_adaptive_fast.py",
        "validation/lib/commensurate_dwave_bdg_cache.py",
    )
    assert not [path for path in retired if (ROOT / path).exists()]


def test_active_dwave_n0_commands_are_grouped():
    active = (
        "validation/commands/static/dwave_gauss_outer.py",
        "validation/commands/ward/commensurate.py",
        "validation/commands/ward/bond_metric_full_kernel.py",
        "validation/commands/ward/bond_metric_family.py",
        "validation/commands/matsubara/dwave_small_xi.py",
    )
    missing = [path for path in active if not (ROOT / path).is_file()]
    assert not missing
