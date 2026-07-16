# Outer-q Casimir integration contract

## Scope

This layer integrates the already validated two-plate signed real logdet over the
physical in-plane vacuum wavevector. It does not evaluate the microscopic response,
choose the transverse Brillouin-zone working N, or estimate the infinite Matsubara
tail.

The fixed free-energy formula is

```text
F/A = k_B T sum_n' integral d^2Q/(2pi)^2 L_n(Q, phi)
```

where `L_n` is the common-lab two-plate Lifshitz logdet and the prime gives `n=0`
weight one half.

## Radial variable and measure

Use

```text
u = 2 Q d
Q = u/(2d)
```

so

```text
d^2Q/(2pi)^2 = u du dphi/(16 pi^2 d^2).
```

This variable makes the vacuum propagation scale explicit. The first implementation
uses Gauss-Legendre quadrature on a finite interval `u in [0,u_max]`. The tail is not
silently discarded: a microscopic preflight must compare an increasing `u_max`
ladder before a formal run.

## Angular domain

The quadrature always uses the full interval

```text
phi in [0, 2pi)
```

with an equal-weight periodic trapezoidal rule. No C4, mirror, plate-exchange, or
pairing symmetry reduction is assumed in the integration measure. A fractional
angular cell offset is part of the grid definition; the default is one half and an
offset-zero grid is retained as a cut/phase audit.

## Momentum units

The outer measure is defined in SI wavevector components `(Qx,Qy)` in `m^-1`. The
microscopic evaluator receives

```text
q_model = (a_x Q_x, a_y Q_y).
```

This is the exact inverse of the reflection layer conversion. The outer grid records
both representations for every node.

## Origin

Gauss-Legendre radial nodes lie strictly inside `(0,u_max)`, so no node occurs at
`Q=0`. The measure contains a factor `Q dQ`, and the origin is represented by the
radial quadrature limit rather than by a special LT-basis convention.

## Numerical qualification sequence

1. Analytic quadrature preflight:
   - exact disk measure for a constant integrand;
   - exact radial polynomial moments;
   - full-angle cancellation of a fourfold harmonic;
   - invariance under angular cut offset;
   - SI/model momentum round trip;
   - Matsubara `n=0` half weight and free-energy units.
2. Microscopic outer-q smoke:
   - low/high radial order;
   - low/high angular order;
   - angular offset audit;
   - increasing `u_max` ladder;
   - online point-specific transverse-N certification.
3. Matsubara partial-sum and tail qualification.
4. End-to-end free-energy and torque symmetry checks.

Passing the analytic preflight only permits stage 2. It does not authorize a
production Casimir result.

## Public command

```bash
python -m validation casimir outer-q-quadrature-preflight
```

Default exploratory contract:

```text
separation = 20 nm
u_max = 24
radial orders = 16 / 32
angular orders = 16 / 32
angular offset = 1/2, audited against 0
```

The `u_max` and orders are preflight defaults, not yet production budgets.
