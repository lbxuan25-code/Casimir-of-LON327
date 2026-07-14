# Casimir outer integration

This directory intentionally contains no executable full outer-integration pipeline yet.

The retired monolithic `finite_q_bdg_casimir_pipeline.py` encoded superseded zero-RHS Ward, positive-frequency-to-zero extrapolation and legacy TE/TM conventions. Do not restore or copy it.

## Microscopic intake implementation

The typed library contains `ArbitraryQPeriodicBZContract-v2`:

```text
fixed shifted N x N full periodic BZ lattice
exact q_crystal = R(-theta) q_lab
no q rounding, wrapping or interpolation
shared primitive kernel with the commensurate reference
readonly MaterialGridCache-v2
streamed canonical reduction blocks
exact zero + positive Matsubara shared q workspace
q-workspace-integrated Peierls operator audit
integrated RHS-aware Ward validation
CrystalResponseCache-v2 with complete numerical-policy identity
q_lab + angle-batch persistent POSIX-fork execution
```

The complete-orbit backend remains the commensurate-q reference and regression authority.

Implementation and validation contracts:

```text
docs/full_outer_integration_handoff.md
docs/arbitrary_q_periodic_bz_design.md
validation/README.md
```

## Mandatory formal order

Set all BLAS/OpenMP thread counts to one, then run on the target WSL/Linux machine:

```bash
python -m validation matsubara arbitrary-q-performance-preflight \
  --pairings spm dwave \
  --N 128 \
  --q-tasks 8 \
  --workers 8 \
  --matsubara-indices 0 1 2 4 8 \
  --canonical-block-size 4096 \
  --runtime-chunk-sizes 4096 16384 \
  --output validation/outputs/matsubara/arbitrary_q_performance_preflight/real_head.json

python -m validation matsubara arbitrary-q-periodic-bz-qualification \
  --performance-manifest \
    validation/outputs/matsubara/arbitrary_q_performance_preflight/real_head.json \
  --pairings spm dwave \
  --N-values 256 384 512 \
  --reference-nk 1256 \
  --reference-order 384 \
  --workers 8 \
  --matsubara-indices 0 1 8 \
  --canonical-block-size 4096 \
  --runtime-chunk-size 16384
```

`ArbitraryQFormalPolicyV1` is fail-closed:

- looser CLI values can run only as explicitly nonformal diagnostics and cannot establish a formal pass;
- the performance manifest records its exact command, policy/config/hardware fingerprints, git head, actual BLAS threads, eigensystem counters, memory, IPC and parent overhead;
- the qualification route rejects missing, stale, forged, nonformal or execution-incompatible manifests before large computation;
- the numerical core itself cannot authorize outer integration; only the public same-head gate can promote `diagnostic_result_passed` to `qualified_for_diagnostic_outer_integration`.

The large-N qualification uses the same cached persistent q-task path measured by preflight. For each pairing it builds one cache per primary `N` plus one cache for each audit shift. Paired-shift averaging occurs at the linear packed-primitive level before any Schur, sheet, reflection or logdet operation.

The microscopic API currently rejects any `q_crystal` component outside `[-pi,pi]`. A future outer configuration must prove Q-cutoff/tail convergence while every rotated microscopic q remains inside that validated domain.

## Future full outer layer

After both microscopic gates pass, the outer layer must remain thin orchestration over the typed library and provide:

- exact zero-Matsubara density/stiffness sheet response;
- positive-Matsubara conductivity sheet response;
- common lab LT tangential-electric reflection basis;
- `lno327.casimir.lifshitz_integrand.passive_sheet_logdet`;
- explicit Matsubara prime weight, q/angle quadrature, restartable caches, convergence and sensitivity reports;
- compact worker-side logdet payloads with indexed ordered streaming reduction;
- energy convergence before torque differentiation.

No full outer result may be treated as production input until microscopic performance/numerical manifests and subsequent q/angle/Matsubara energy/torque convergence pass.

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```
