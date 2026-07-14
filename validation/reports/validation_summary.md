# Validation summary

## Current readiness

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

The finite-q microscopic response, amplitude/phase Schur contract, RHS-aware Ward validation, positive-Matsubara sheet construction, exact zero-Matsubara path, and reflection/logdet contracts are implemented and covered by tests. The unresolved blocker remains the common transverse/BZ integration reference for the full positive-Matsubara conductivity tensor.

## Active numerical paths

- deterministic full-period panel-adaptive controller with split-history error continuity;
- one common fixed/composite Gauss backend used as an independent offline reference;
- batched midpoint material and finite-q q workspaces;
- optional POSIX-fork transverse-node execution with original-node-order parent Kahan reduction;
- exact zero-Matsubara thermodynamic path and typed sheet/reflection contracts.

Rejected quadrature implementations are removed rather than kept as runnable archive paths.

## Difficult diagonal-point evidence

All reported fixed/composite Gauss points passed Ward and the complete physical pipeline. Reflection and the current zero-angle, 20 nm logdet were substantially more stable than the upstream conductivity tensor.

For composite `panel_count=16` at diagonal `(1,1)` and Matsubara indices `1,2`:

```text
C256-C224 sigma mixed ratios = 7.375, 5.773
C384-C320 sigma mixed ratios = 1.668, 1.362
C384-C320 reflection ratios  = 0.0060, 0.0048
C384-C320 logdet ratios      = 0.0166, 0.0130
```

Thus higher local Gauss order substantially reduced the full-sigma discrepancy but did not pass the predeclared `ratio <= 1` gate. The fixed-cut composite reference is not established; diagonal-mid and smooth-cut expansion remain skipped under the fail-fast contract.

## Performance evidence

Batched material/q workspaces reduced the local `(1,1)` callback from about `3.56 s/node` to about `0.32-0.37 s/node`, an approximately eleven-fold per-node speedup.

The first shared-thread transverse implementation did not accelerate the real `nk=1256` workload: C320+C384 used `237 s` wall and approximately one CPU core despite eight requested workers. It has therefore been replaced for d-wave Gauss evaluation by explicit POSIX-fork worker processes. Small tests prove serial/process primitive equality but are not treated as performance evidence; a real-`nk` serial/process A/B benchmark is required before another expensive scan.

GitHub Actions run `29300823582` passed completely on head `a118957af3537807311fedf7f0e79bb7fa0eac6e`, including the full repository suite and fork-process fixed-Gauss CLI smoke.

## Artifact and maintenance policy

Raw CSV, JSON, txt, log, figures, arrays, and intermediate validation outputs are local reproducible artifacts and are ignored by Git. The tree retains only active code, tests, README/command files, compact summaries, and status records. No q-specific fallback, symmetry reduction, or production-readiness claim is permitted.
