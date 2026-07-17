# d-wave exact-static Ward contract audit

This directory contains fail-closed diagnostics for the finite-q exact-static d-wave longitudinal Ward chain. No command in this group applies a longitudinal projection or produces a Casimir-ready reference.

## Public command surface

```bash
python -m validation ward --help
```

Available commands:

```text
ward contract-audit
ward commensurate
ward phase-column
ward phase-hessian
ward phase-hessian-family
ward average-subgrids
```

The exact periodic target is

\[
u_L K_{\mathrm{eff}}=0,
\qquad
K_{\mathrm{eff}}u_R=0.
\]

For one finite quadrature, the source audit decomposes

\[
u_L K_{\mathrm{eff}}
=R_S-C_\eta K_{\eta\eta}^{-1}K_{\eta S}+r_{\mathrm{primitive}},
\]

and the corresponding right identity. The stored sources separate primitive EM closure, collective phase-column closure and the Schur projection into the effective longitudinal kernel.

## Full commensurate audit

```bash
OUT="validation/outputs/zero_matsubara/dwave_ward_contract_audit/raw"
mkdir -p "$OUT"

env \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python -m validation ward commensurate \
    --nk 628 \
    --mx 3 \
    --my 2 \
    --shift-x 0.5 \
    --shift-y 0.5 \
    --chunk-size 1024 \
    --max-points 500000 \
    --temperature-K 10 \
    --delta0-eV 0.1 \
    --eta-eV 1e-8 \
    --output "$OUT/dwave_commensurate_n628_m3_2_T10.csv"
```

Translation by `q=(2 pi/N)(mx,my)` is an exact index permutation. Translation by `q/2` remains on the same grid sublattice only when both integer shift components are even. Odd components require a complementary-origin check before interpreting the phase-Hessian residual.

## Reduced phase-column audit

```bash
python -m validation ward phase-column \
  --nk 628 \
  --mx 2 \
  --my 0 \
  --shift-x 0.5 \
  --shift-y 0.5 \
  --chunk-size 1024 \
  --max-points 500000 \
  --temperature-K 10 \
  --delta0-eV 0.1 \
  --eta-eV 1e-8 \
  --output "$OUT/dwave_phase_column_n628_m2_0_T10.json"
```

This evaluates only the left/right EM-phase contractions, the finite-q phase bubble and the q=0 Goldstone counterterm bubble. It uses the same BdG, Kubo, vertex and periodic integration conventions as the full audit.

## Existing-payload analysis

```bash
python -m validation ward phase-hessian \
  --input "$OUT/dwave_commensurate_n628_m3_2_T10.json" \
  --output "$OUT/dwave_commensurate_n628_m3_2_T10.phase_hessian.json"
```

```bash
python -m validation ward phase-hessian-family \
  "$OUT/point1.json" \
  "$OUT/point2.json" \
  "$OUT/point3.json" \
  --output "$OUT/phase_hessian_scaling.json"
```

## Complementary-subgrid average

For an odd shift component, run the same physical q on the complementary grid origin and average the raw phase-column pieces before forming the required multiplier:

```bash
python -m validation ward average-subgrids \
  "$OUT/dwave_commensurate_n628_m3_2_T10.json" \
  "$OUT/dwave_phase_column_n628_m3_2_sx0_T10.json" \
  --output "$OUT/dwave_phase_column_n628_m3_2_subgrid_average_T10.json"
```

The average is formed separately for the EM contraction, finite-q phase-bubble rotation and q=0 counterterm rotation. Averaging already-formed multipliers is not equivalent.

All raw CSV, JSON and logs remain diagnostic artifacts and are not valid Casimir inputs.
