# d-wave exact-static phase-Hessian scaling status

## Current evidence

The complete commensurate periodic audits at fixed integer direction
`(m_x,m_y)=(3,2)` produced:

| `N` | `|q|` | required counterterm shift | scalar bond shift | absolute mismatch |
|---:|---:|---:|---:|---:|
| 628 | 0.0360738006342 | 1.769411473420e-4 | 1.626547621185e-4 | 1.428638522349e-5 |
| 471 | 0.0480984008456 | 3.217221925937e-4 | 2.891500230442e-4 | 3.257216954955e-5 |
| 314 | 0.0721476012684 | 5.197648405220e-4 | 6.504975688578e-4 | 1.307327283359e-4 |

All three primitive Ward residuals divided by `|q|` remain at approximately
`1e-15`, so the observed defect is confined to the collective phase-column
Hessian rather than the Peierls/contact primitive identity.

## Corrected scaling interpretation

The original permissive classifier reported
`bond_metric_matches_leading_q2_geometry`.  That conclusion was not justified:

- global required-shift exponent: `1.53020397`;
- adjacent-point required exponents: `2.07823600`, `1.18305633`;
- bond metric exponent: `1.99972589`;
- bond-mismatch exponent: `3.20923367`.

The largest-q point is not in a clean common quadratic regime.  The classifier now
requires both the global required exponent and every adjacent-point exponent to be
stably quadratic before interpreting the residual exponent.  The current family is
therefore classified fail-closed as:

```text
not_in_clean_small_q_regime
```

A two-smallest-point even-power fit,

```text
y = a2 |q|^2 + a4 |q|^4,
```

gives the diagnostic estimates

```text
required shift a2  = 0.13199132
bond metric a2     = 0.125
bond mismatch a2   = 0.00699132
```

This suggests, but does not yet establish, that the scalar bond metric misses roughly
five percent of the leading quadratic phase curvature.  No fitted multiplier is
permitted to modify `K_etaeta`.

## Next discriminator: fixed-grid axial small-q family

Changing `N` changes both momentum and Brillouin-zone resolution.  The next test
holds the complete `628 x 628` periodic grid fixed and varies only the exact integer
translation `(m_x,m_y)=(1,0),(2,0),(3,0)`.  These points have
`|q| approximately 0.0100, 0.0200, 0.0300`, all below the original oblique point.

The tetragonal two-band model permits an isotropic scalar coefficient at quadratic
order; angular distinctions first affect higher-order terms.  The axial family can
therefore decide whether the leading coefficient approaches the scalar bond value
`1/8` without mixing in a change of grid resolution.

Use the reduced phase-column runner, which evaluates only the four primitive
averages required for this question and is regression-tested against the full
48-channel commensurate audit:

```bash
OUT="validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw"
mkdir -p "$OUT"

for MX in 1 2 3; do
  env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    python -m validation.run_dwave_static_commensurate_phase_column_audit \
      --nk 628 \
      --mx "$MX" \
      --my 0 \
      --shift-x 0.5 \
      --shift-y 0.5 \
      --chunk-size 1024 \
      --max-points 500000 \
      --temperature-K 10 \
      --delta0-eV 0.1 \
      --eta-eV 1e-8 \
      --output "$OUT/dwave_phase_column_n628_m${MX}_0_T10.json"
done

python -m validation.analyze_dwave_commensurate_phase_hessian_family \
  "$OUT/dwave_phase_column_n628_m1_0_T10.json" \
  "$OUT/dwave_phase_column_n628_m2_0_T10.json" \
  "$OUT/dwave_phase_column_n628_m3_0_T10.json" \
  --output "$OUT/dwave_phase_column_n628_axis_T10.scaling.json"
```

## Decision rule

- Stable `required shift ~ q^2` and `bond error ~ q^2` establishes that a scalar
  finite-q counterterm misses leading curvature; proceed to independent x/y bond
  amplitude and phase channels.
- Stable `required shift ~ q^2` and `bond error ~ q^4` supports the scalar bond
  metric only at leading order; derive the exact bond-resolved Hessian before any
  production change.
- Failure of the required shift to show stable quadratic scaling keeps the problem
  in numerical/grid convergence status.

All outputs remain diagnostic-only, unprojected, and invalid for Casimir input.
