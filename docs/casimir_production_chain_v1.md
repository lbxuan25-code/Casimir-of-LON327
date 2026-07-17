# Casimir production chain v1

## Status

This document freezes the non-adaptive calculation chain migrated out of
`validation`. The complete fixed controller now exists at
`lno327.casimir.run_casimir`. It returns an explicitly finite Matsubara partial
result and remains fail-closed for production authorization.

This contract does **not** qualify the infinite Matsubara tail and does not enable a
full production Casimir result.

## Dependency direction

The allowed dependency direction is

```text
validation -> lno327.casimir
```

Code below `src/lno327/` must never import the top-level `validation` package.

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

There is one mathematical outer-Q contract:

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

## Fixed controller contract

The public production entry point is

```python
from lno327.casimir import FixedCasimirConfig, FixedCasimirResult, run_casimir

result: FixedCasimirResult = run_casimir(FixedCasimirConfig())
```

`FixedCasimirConfig` owns all fixed numerical and physical inputs used by the
qualified chain. Its defaults reproduce the qualified `spm`, `n=0,1` reference
settings.

The controller performs the following operations in one fixed order:

1. build the nested outer-Q ladder and deduplicated microscopic node manifest;
2. invoke `lno327.casimir.fixed_transverse_point_certification` directly as the
   production point-certification backend;
3. retain the established independent-process and single-thread BLAS/OpenMP
   environment;
4. retain checkpoint-after-each-completed-N behavior when a checkpoint path is
   supplied;
5. require established hard-physical and numerical certification at every requested
   microscopic point;
6. reduce the canonical primary-shift logdet values with the fixed outer-Q measure;
7. apply the Matsubara prime weight exactly once;
8. evaluate the fixed cutoff, radial, angular, and offset ladders;
9. return a finite-partial or unresolved result without inferring any missing tail.

A successful controller result has

```text
status = finite_partial
partial_sum_only = true
matsubara_tail_estimated = false
production_casimir_allowed = false
```

Any missing or uncertified microscopic point produces

```text
status = unresolved
production_casimir_allowed = false
```

The controller never silently drops unresolved points and never promotes a finite
partial sum to a complete Casimir result.

## Fixed nested radial contract

A cutoff sequence such as

```text
6, 10, 14, 18, 24
```

means cumulative panels

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

The prime weight is applied exactly once by the Matsubara free-energy reduction.
The helper converting `n,T` to `hbar*xi_n` in eV is owned by
`lno327.casimir.matsubara`; validation only re-exports it.

## Microscopic model and point-certification contract

The active symmetry-based two-band finite-q model adapter is owned by
`lno327.casimir.microscopic_model`. Constructing the model does not certify a point
and does not authorize Casimir input.

The fixed transverse-point numerical engine is owned by
`lno327.casimir.fixed_transverse_point_engine`. The universal adjacent-N,
cross-shift, oscillatory-envelope, and hard-physical acceptance controller is owned
by `lno327.casimir.fixed_transverse_point_certification`.

The full fixed composition is owned by `lno327.casimir.fixed_chain`. Validation may
retain historical command, report, and monkeypatch surfaces as compatibility layers,
but none of them is required by the production controller.

## Migration rule

This migration moves only already-qualified fixed-grid components into
`src/lno327/casimir`. It does not change numerical rules, introduce adaptivity,
alter worker scheduling, change microscopic convergence criteria, or clean
historical outputs.

Completed production ownership:

- Matsubara energy helper;
- active finite-q microscopic model adapter;
- fixed transverse-point numerical engine and universal certification controller;
- nested compound outer-Q planning and exact node reuse;
- certified-point reduction into finite Matsubara partial free energies;
- cutoff/radial/angular/offset ladder comparisons;
- unique `run_casimir(config) -> FixedCasimirResult` controller.

The fixed main chain must remain frozen while its controller-level regression and
CI evidence are established. Compatibility-facade, legacy-reference, and historical
output cleanup may proceed only after that evidence passes. Adaptive radial
integration, automatic outer cutoff, and Matsubara-tail inference remain separate
future qualification tasks.

## Golden fixed reference

For the qualified `spm`, `T=10 K`, `d=20 nm`, plate angles `0 deg` and `17 deg`,
Matsubara indices `n=0,1`, panel boundaries `6,10,14,18,24`, radial order 8 per
panel, angular order 8, and angular offset 0.5:

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
false until the Matsubara tail and the final production qualification are complete.
