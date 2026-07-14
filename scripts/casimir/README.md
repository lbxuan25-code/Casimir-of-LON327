# Casimir outer integration

This directory intentionally contains no executable full outer-integration pipeline yet.

The retired monolithic `finite_q_bdg_casimir_pipeline.py` was removed before the outer-integration handoff because it encoded superseded zero-RHS Ward, positive-frequency-to-zero extrapolation, and legacy TE/TM conventions. Git history remains the audit trail; do not restore or copy that implementation.

## Microscopic intake now available

The typed library now contains an implementation of `ArbitraryQPeriodicBZContract-v1` for the microscopic response needed by a future full outer integrator:

```text
fixed shifted N x N periodic BZ lattice
exact real q_crystal = R(-theta) q_lab
shared quadrature-independent primitive kernel
readonly q-independent material cache
streamed canonical reduction blocks
exact zero + positive Matsubara shared q eigensystems
normal Peierls operator-identity diagnostics
integrated RHS-aware Ward validation
q_lab + angle-batch persistent-fork execution
```

The qualified complete-orbit backend remains the commensurate-q reference and regression authority. It is not forced onto arbitrary q.

Implementation and validation contracts are documented in:

```text
docs/full_outer_integration_handoff.md
docs/arbitrary_q_periodic_bz_design.md
validation/README.md
```

The required order is:

```bash
python -m validation matsubara arbitrary-q-performance-preflight \
  --output validation/outputs/matsubara/arbitrary_q_performance_preflight/real_head.json

python -m validation matsubara arbitrary-q-periodic-bz-qualification \
  --performance-manifest \
    validation/outputs/matsubara/arbitrary_q_performance_preflight/real_head.json
```

The public qualification route rejects a missing, failed, or stale performance manifest before starting any expensive numerical work. Its manifest must have the exact current git head.

The performance command must produce a same-head real-hardware manifest before the expensive `N=256,384,512` numerical qualification is accepted. CI only establishes small deterministic architecture and regression contracts.

## Future full outer layer

After both arbitrary-q microscopic gates pass, the outer layer must remain thin orchestration over the typed library and provide:

- exact zero-Matsubara density/stiffness sheet response;
- positive-Matsubara conductivity sheet response;
- common lab LT tangential-electric reflection basis;
- `lno327.casimir.lifshitz_integrand.passive_sheet_logdet`;
- explicit Matsubara prime weight, q/angle quadrature, restartable caches, convergence and sensitivity reports;
- energy convergence before torque differentiation.

No full outer result may be treated as production input until the microscopic performance and numerical qualification manifests and the subsequent q/angle/Matsubara energy/torque convergence reports pass.

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```
