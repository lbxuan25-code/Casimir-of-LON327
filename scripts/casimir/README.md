# Casimir outer integration

This directory intentionally contains no executable full outer-integration pipeline yet.

The retired monolithic `finite_q_bdg_casimir_pipeline.py` was removed before the outer-integration handoff because it encoded superseded zero-RHS Ward, positive-frequency-to-zero extrapolation, and legacy TE/TM conventions. Git history remains the audit trail; do not restore or copy that implementation.

The next implementation must start from `docs/full_outer_integration_handoff.md` and remain a thin orchestration layer over the typed two-band library:

- exact zero-Matsubara density/stiffness sheet response;
- positive-Matsubara conductivity sheet response;
- common lab LT tangential-electric reflection basis;
- `lno327.casimir.lifshitz_integrand.passive_sheet_logdet`;
- explicit Matsubara prime weight, q/angle quadrature, convergence and sensitivity reports.

Until the handoff acceptance gates are completed:

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```
