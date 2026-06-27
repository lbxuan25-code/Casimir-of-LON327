# Finite-q diagnostic pipeline

The finite-q response path is currently diagnostic. It is for Ward-closure
debugging only and its outputs are not Casimir-ready.

## Module boundaries

- `model.py` owns the normal-state Hamiltonian and normal-state electromagnetic
  vertices.
- `pairing_ansatz.py` owns pairing inputs: mean pairing, collective vertices,
  phase-vertex convention, and the current gap-equation counterterm provider.
- `finite_q_engine.py` is the generic finite-q response calculator. It consumes
  a `PairingAnsatz` and does not branch on pairing names.
- `finite_q_diagnostics.py` is the diagnostic workflow layer. It builds an
  explicit ansatz, computes `bare_total`, `minus_schur`, and
  `amplitude_phase_schur`, then runs Ward checks on those matrices.
- `ward_validation.py` only checks Ward residuals. It must never repair or
  modify a response matrix.
- `casimir.py` remains separate and must not consume finite-q diagnostic
  responses.

## Diagnostic defaults

New finite-q diagnostics should specify the phase vertex explicitly:

```python
phase_vertex = "bond_endpoint_gauge"
current_vertex = "peierls"
collective_mode = "amplitude_phase"
collective_counterterm = "goldstone_gap_equation"
include_phase_phase_direct = True
```

The legacy finite-q wrapper keeps its historical defaults for compatibility, so
new code should not rely on wrapper defaults when debugging finite-q Ward
closure.

## Gating

All finite-q diagnostic reports must keep
`valid_for_casimir_input=False`. The response has not been promoted to a
production Casimir pipeline, and no finite-q diagnostic output should be passed
to Casimir formulas as validated input.

Response-level LSQ fitting, Ward-residual repair, or production fitting of the
finite-q response is not allowed. If a diagnostic fails a Ward check, the failure
should be reported as a diagnostic result rather than patched at the response
level.
