# Casimir production chain v1

## Status

The non-adaptive fixed microscopic Casimir chain is fully production-owned. Its
single public controller is:

```python
from lno327.casimir import FixedCasimirConfig, FixedCasimirResult, run_casimir

result: FixedCasimirResult = run_casimir(FixedCasimirConfig())
```

The controller returns an explicitly finite Matsubara partial result and remains
fail-closed for production authorization. This contract does **not** qualify the
infinite Matsubara tail and does not enable a complete production Casimir result.

## Dependency and ownership boundary

The allowed dependency direction is:

```text
validation -> src/lno327
```

Code below `src/lno327/` must never import the top-level `validation` package.

The former fixed-chain validation facades, legacy numerical copies, microscopic
outer-Q preflight command, and transverse-point compatibility command have been
removed. There is no alternate validation implementation or CLI route for this
calculation.

Generated files under `validation/outputs/` are local artifacts and are ignored by
version control. The reviewed golden fixture under `validation/references/casimir/`
is retained solely as regression evidence.

## Unique fixed calculation chain

```text
FixedCasimirConfig
  -> run_casimir(config)
  -> nested compound outer-Q grid plan
  -> exact union and reuse of microscopic q nodes
  -> production transverse-point certification process
  -> finite-q microscopic response
  -> sheet response and plate reflection matrices
  -> two-plate propagation and stable logdet
  -> certified primary-shift estimator
  -> fixed outer-Q reduction
  -> Matsubara prime weighting
  -> fixed ladder comparisons
  -> FixedCasimirResult(status="finite_partial" | "unresolved")
```

The controller performs these operations in this fixed order and does not silently
drop unresolved microscopic points.

## Fixed controller result contract

A successful controller result has:

```text
status = finite_partial
partial_sum_only = true
matsubara_tail_estimated = false
production_casimir_allowed = false
```

Any missing or uncertified microscopic point produces:

```text
status = unresolved
production_casimir_allowed = false
```

`FixedCasimirConfig` owns every fixed physical and numerical input used by the chain,
including the Matsubara indices, outer-Q ladders, transverse N ladder, shifts,
parallel policy, physical gates, and convergence tolerances.

## Outer-Q contract

There is one mathematical outer-Q convention:

```text
u = 2 Q d
Q = u / (2 d)
d^2Q/(2pi)^2 = u du dphi/(16 pi^2 d^2)
phi in [0, 2pi)
q_model = (a_x Q_x, a_y Q_y)
```

The full angular interval is retained. There is no crystal, mirror, plate-exchange,
or pairing symmetry reduction. Periodic angular nodes do not duplicate the `2pi`
endpoint. Gauss-Legendre radial nodes contain no explicit `Q=0` node.

A cutoff sequence such as:

```text
6, 10, 14, 18, 24
```

means cumulative panels:

```text
[0,6]
[0,6] + [6,10]
[0,6] + [6,10] + [10,14]
...
```

The radial order is the Gauss-Legendre order per panel. Increasing the cutoff adds
one panel and preserves all earlier nodes and weights exactly.

## Matsubara contract

```text
xi_n = 2 pi n k_B T / hbar
w_0 = 1/2
w_n = 1 for n >= 1
```

The prime weight is applied exactly once by the Matsubara free-energy reduction. The
helper converting `n,T` to `hbar*xi_n` in eV is owned exclusively by
`lno327.casimir.matsubara`.

## Microscopic and certification contract

The active symmetry-based two-band finite-q model adapter is owned by
`lno327.casimir.microscopic_model`. Constructing the model does not certify a point
and does not authorize Casimir input.

The fixed transverse-point numerical engine is owned by
`lno327.casimir.fixed_transverse_point_engine`. The universal adjacent-N,
cross-shift, oscillatory-envelope, and hard-physical acceptance controller is owned
by `lno327.casimir.fixed_transverse_point_certification`.

The complete composition is owned by `lno327.casimir.fixed_chain`. No validation
facade, legacy copy, subprocess route through `python -m validation`, or historical
output is required by the controller.

## Frozen numerical boundary

This cleanup does not:

- introduce adaptive radial integration;
- introduce an automatic outer-Q cutoff;
- infer a Matsubara cutoff or tail;
- change the N ladder, shifts, physical gates, or tolerances;
- change worker scheduling, task flattening, or summation order;
- enable `production_casimir_allowed`.

All future numerical extensions must preserve the fixed controller and its golden
contract until separately qualified.

## Golden fixed reference

For `spm`, `T=10 K`, `d=20 nm`, plate angles `0 deg` and `17 deg`, Matsubara indices
`n=0,1`, panel boundaries `6,10,14,18,24`, radial order 8 per panel, angular order 8,
and angular offset 0.5:

```text
reference spec: u24_p5_r8_a8_o0p5
partial free energy: -2.4696354012227892e-08 J/m^2
n=0 contribution:    -8.238520130188009e-09 J/m^2
n=1 contribution:    -1.6457833882039883e-08 J/m^2
working N: 192
audit N: 256
unresolved points: 0
```

This is a finite `n=0,1` partial result only. `production_casimir_allowed` remains
false until the Matsubara tail and final production qualification are complete.
