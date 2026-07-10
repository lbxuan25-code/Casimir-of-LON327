# Two-band d-wave exact-static nodal quadrature

This directory stores validation artifacts produced by

```bash
python -m validation.run_dwave_static_adaptive_scan
```

The runner addresses the failure of uniform midpoint and equal-cost four-shift
Gauss grids to converge the retained exact-static d-wave channels.  It is not a
new response formula and does not alter the Ward, collective, projection,
reflection, or Lifshitz contracts.

## Refinement indicator

For every Brillouin-zone cell the current `symmetry_bdg_2band` spec and d-wave
ansatz are sampled at `k`, `k-q/2`, and `k+q/2`.  A cell is refined when any of
the following is detected:

1. a low absolute BdG quasiparticle energy;
2. a normal-state Fermi-surface crossing together with a small or sign-changing
   d-wave pairing trace;
3. a low-energy near-degenerate transition between shifted `k-q/2` and
   `k+q/2` BdG states.

The thresholds are explicit CLI parameters and are recorded in CSV/JSON output.
The classifier does not call the retired four-orbital normal Hamiltonian.

## Integration contract

All final cells use tensor-product Gauss-Legendre quadrature.  Parent cells are
replaced by children, so there is no parent-child double counting.  Every point
and normalized weight is passed into one finite-q material workspace.  The
primitive EM, EM-collective, collective, contact, and Goldstone-counterterm
blocks are therefore integrated first, followed by one amplitude/phase Schur
complement.  Per-cell or per-shift effective kernels are never averaged.

## Validation outputs

Each row reports:

- refinement history and actual point count;
- mixed RHS-aware Ward and Schur diagnostics;
- raw longitudinal leakage, `chi_bar`, and `Dbar_T`;
- projection eligibility under the unchanged fail-closed ceiling;
- projected static reflection and a signed-logdet diagnostic at a configurable
  separation when projection is eligible;
- convergence relative to the row with the largest actual quadrature point
  count.

A converged physical result requires stability of `chi_bar`, `Dbar_T`, the
static reflection, and the diagnostic logdet.  Passing Ward alone is not a
quadrature convergence criterion, and the projection ceiling must not be
relaxed merely to admit unconverged nodal integrals.
