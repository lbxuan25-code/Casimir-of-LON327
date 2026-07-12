# d-wave bond-metric validation plan

## Status

The nearest-neighbour d-wave phase-Hessian policy is implemented as an explicit
core opt-in:

```python
phase_hessian_policy="nearest_neighbor_bond_metric"
```

It changes only the phase diagonal of the Hubbard--Stratonovich counterterm,

\[
K_{22}^{\mathrm{HS}}(q)=
\frac{\cos^2(q_x/2)+\cos^2(q_y/2)}{2}
K_{22}^{\mathrm{HS}}(0),
\]

and rebuilds the complete amplitude/phase Schur complement.  It remains
fail-closed and invalid for Casimir input.

## Why two static runners are required

A commensurate grid uses

\[
q=\frac{2\pi}{N_k}(m_x,m_y).
\]

It is the correct setting for exact finite-grid Ward closure because translation
by q is an index permutation.  Holding `(mx,my)` fixed while changing `Nk`,
however, changes the physical q and is **not** a fixed-q convergence study.

Therefore validation is split into two independent tasks.

### 1. Commensurate Ward and C4 family

Command:

```bash
python -m validation ward bond-metric-family ...
```

The default points cover axial, diagonal, generic, odd/even half-q parity, and
C4 partners:

```text
(2,0), (0,2), (2,2), (4,2), (2,4), (3,2), (2,3)
```

Odd integer components automatically trigger complementary half-step subgrid
averaging of the complete 48-component primitive vector before either Schur
complement is formed.

The hard exact-static gate requires all of

```text
primitive residual / |q|
amplitude defect / |q|
phase defect / |q|
effective direct contraction / |q|
effective Ward residual / |q|
relative longitudinal kernel norm
```

to lie below their explicit tolerances, with a regular inverse and bounded
collective condition number.  The generic mixed absolute-relative
`ward.passed` result is recorded but is not used as the production closure gate.

### 2. Fixed-physical-q observable convergence

Command:

```bash
python -m validation static bond-metric-nk-convergence ...
```

This keeps `(qx,qy)` fixed while changing `Nk`.  It is the correct runner for
convergence of

```text
chi_bar
Dbar_T
```

and also records the distance from q to the nearest integer grid translation.
That distance explains why exact commensurate Ward closure and fixed-q
observable convergence need not have the same finite-grid behaviour.

## Positive Matsubara validation

Command:

```bash
python -m validation matsubara bond-metric-positive ...
```

For each `Nk`, all requested positive Matsubara indices are evaluated in one
batched q-workspace contraction.  The validated chain is

```text
BdG response
-> RHS-aware effective Ward
-> positive-Matsubara sheet response
-> reflection matrix
-> passive signed logdet
```

The runner reports convergence of the sheet matrix, reflection matrix, and
single-point logdet against the finest grid.  It performs no q integration or
Matsubara sum.

## Current gating state

All three commands deliberately report

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

Promotion requires successful local runs over the selected q/direction/frequency
families and review of the generated CSV/JSON reports.  No longitudinal
projection is permitted as a substitute for microscopic closure.
