# Casimir production chain v1

## Status

This document freezes the non-adaptive calculation chain used during migration out
of `validation`.  It does **not** qualify the infinite Matsubara tail and does not
enable a production Casimir result.

## Dependency direction

The allowed dependency direction is

```text
validation -> lno327.casimir
```

Code below `src/lno327/` must never import `validation`.

## Unique fixed calculation chain

```text
physical configuration
  -> Matsubara index and prime weight
  -> nested compound outer-Q grid
  -> SI Q to model q mapping
  -> finite-q microscopic response certification
  -> sheet response and plate reflection matrices
  -> two-plate propagation and stable logdet
  -> fixed outer-Q reduction
  -> finite Matsubara partial sum
```

There is one mathematical outer-Q contract:

```text
u = 2 Q d
Q = u / (2 d)
d^2Q/(2pi)^2 = u du dphi/(16 pi^2 d^2)
phi in [0, 2pi)
q_model = (a_x Q_x, a_y Q_y)
```

The full angular interval is retained.  There is no crystal, mirror, plate-exchange,
or pairing symmetry reduction.  Periodic angular nodes do not duplicate the `2pi`
endpoint.  Gauss-Legendre radial nodes contain no explicit `Q=0` node.

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

The radial order is the Gauss-Legendre order per panel.  Increasing the cutoff adds
one panel and preserves all earlier nodes and weights exactly.

## Matsubara contract

```text
xi_n = 2 pi n k_B T / hbar
w_0 = 1/2
w_n = 1 for n >= 1
```

The prime weight is applied exactly once by the Matsubara free-energy reduction.

## Migration rule

The first migration stage only moves already-qualified fixed-grid planning and
reduction into `src/lno327/casimir/fixed_outer_q.py`.  It does not change numerical
rules, introduce adaptivity, alter worker scheduling, change microscopic convergence
criteria, or clean historical outputs.

Validation facades may temporarily preserve private legacy helpers, but every public
outer-Q planning/reduction function used by validation must resolve to the production
module.

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

This is a finite `n=0,1` partial result only.  `production_casimir_allowed` remains
false until the complete main chain and Matsubara tail are certified.
