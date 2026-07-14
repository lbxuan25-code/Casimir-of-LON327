from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import numpy as np

from lno327.response.arbitrary_q_formal_policy import QUALIFICATION_MATRIX_ID
from lno327.workflows.arbitrary_q_parallel import actual_threadpool_record
from validation.commands.matsubara import arbitrary_q_periodic_bz_qualification as module


def _args() -> Namespace:
    return Namespace(
        reference_nk=1256,
        matsubara_indices=(0, 1, 8),
        separation_nm=20.0,
        ward_tolerance=1e-7,
        ward_absolute_tolerance=1e-12,
        logdet_atol=1e-14,
        logdet_tolerance=3e-4,
        temperature_K=10.0,
        eta_eV=1e-8,
        delta0_eV=0.1,
    )


def _batch(name: str) -> object:
    return SimpleNamespace(
        theta_2_rad_values=np.asarray([0.0, np.deg2rad(17.0)]),
        plate_1=f"{name}:p1",
        plate_2=(f"{name}:p2-zero", f"{name}:p2-17"),
    )


def _context(name: str, n: int, shift: tuple[float, float]) -> object:
    return SimpleNamespace(
        pairing="spm",
        n=n,
        shift=shift,
        two_plate_batch=_batch(name),
    )


def test_two_plate_gate_uses_all_N_audit_and_paired_observables(monkeypatch) -> None:
    args = _args()
    contexts = (
        _context("n256", 256, (0.5, 0.5)),
        _context("n384", 384, (0.5, 0.5)),
        _context("n512", 512, (0.5, 0.5)),
    )
    audit_a = _context("audit-a", 512, (0.25, 0.75))
    audit_b = _context("audit-b", 512, (0.75, 0.25))

    model = SimpleNamespace(
        build_ansatz=lambda *_args, **_kwargs: object(),
        build_pairing_params=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(module, "get_finite_q_validation_model", lambda *_args: model)

    paired_counter = {"value": 0}

    def paired(first, second, **_kwargs):
        paired_counter["value"] += 1
        return f"paired-{paired_counter['value']}"

    monkeypatch.setattr(module, "paired_average_arbitrary_q_results", paired)
    values = {
        ("n256:p1", "n256:p2-17"): [1.0, 0.8, 0.6],
        ("n384:p1", "n384:p2-17"): [1.00010, 0.80008, 0.60006],
        ("n512:p1", "n512:p2-17"): [1.00011, 0.80009, 0.60007],
        ("audit-a:p1", "audit-a:p2-17"): [1.00012, 0.80010, 0.60008],
        ("audit-b:p1", "audit-b:p2-17"): [1.00010, 0.80008, 0.60006],
        ("paired-1", "paired-2"): [1.00011, 0.80009, 0.60007],
    }

    def states(_args, p1, p2):
        return [
            {"n": n, "logdet": value, "passed": True, "error": ""}
            for n, value in zip(args.matsubara_indices, values[(p1, p2)], strict=True)
        ]

    monkeypatch.setattr(module, "_two_plate_states", states)
    row = module._two_plate_row(
        args,
        pairing_name="spm",
        contexts=contexts,
        audit_a=audit_a,
        audit_b=audit_b,
    )
    assert row["passed"] is True
    assert paired_counter["value"] == 2
    for frequency in row["frequencies"]:
        assert len(frequency["primary_logdet_by_N"]) == 3
        assert len(frequency["N_refinement"]) == 2
        assert frequency["audit_a_vs_b"]["passed"] is True
        assert frequency["primary_final_vs_paired"]["passed"] is True
        assert frequency["all_primary_audit_and_paired_two_plate_physical"] is True


def test_two_plate_gate_fails_on_final_consumed_logdet_drift(monkeypatch) -> None:
    args = _args()
    contexts = (
        _context("n256", 256, (0.5, 0.5)),
        _context("n384", 384, (0.5, 0.5)),
        _context("n512", 512, (0.5, 0.5)),
    )
    audit_a = _context("audit-a", 512, (0.25, 0.75))
    audit_b = _context("audit-b", 512, (0.75, 0.25))
    model = SimpleNamespace(
        build_ansatz=lambda *_args, **_kwargs: object(),
        build_pairing_params=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(module, "get_finite_q_validation_model", lambda *_args: model)
    counter = {"value": 0}

    def paired(*_args, **_kwargs):
        counter["value"] += 1
        return f"paired-{counter['value']}"

    monkeypatch.setattr(module, "paired_average_arbitrary_q_results", paired)

    def states(_args, p1, p2):
        value = 1.0
        if (p1, p2) == ("audit-b:p1", "audit-b:p2-17"):
            value = 1.5
        return [
            {"n": n, "logdet": value, "passed": True, "error": ""}
            for n in args.matsubara_indices
        ]

    monkeypatch.setattr(module, "_two_plate_states", states)
    row = module._two_plate_row(
        args,
        pairing_name="spm",
        contexts=contexts,
        audit_a=audit_a,
        audit_b=audit_b,
    )
    assert row["passed"] is False
    assert any(not item["audit_a_vs_b"]["passed"] for item in row["frequencies"])


def test_q_coverage_is_discrete_and_never_overclaims_principal_domain() -> None:
    coverage = module._q_coverage(_args(), passed=True)
    assert coverage["qualification_matrix_id"] == QUALIFICATION_MATRIX_ID
    assert coverage["discrete_matrix_passed"] is True
    assert coverage["qualified_outer_q_envelope_established"] is False
    assert coverage["continuous_angle_coverage_established"] is False
    assert coverage["outer_tail_requirement_bound"] is False
    assert coverage["principal_supported_domain_is_not_claimed_as_qualified"] is True
    assert coverage["tested_q_max_norm"] < np.pi


def test_actual_threadpool_record_is_runtime_observation() -> None:
    rows, passed = actual_threadpool_record()
    assert rows
    assert isinstance(passed, bool)
    assert all("num_threads" in row for row in rows)
