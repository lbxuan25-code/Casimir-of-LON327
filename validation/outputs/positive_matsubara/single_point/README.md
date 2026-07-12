# Positive-Matsubara single-point validation

This directory records the numerical validation contract for one fixed nonzero in-plane momentum before the full Casimir q/angle/Matsubara quadrature is introduced.

## Scope

For every requested microscopic k-grid and positive bosonic Matsubara index

```text
xi_n_eV = 2*pi*n*k_B_eV_per_K*T_K,   n >= 1
```

`python -m validation matsubara positive-point` evaluates

```text
two-band BdG response
  -> amplitude/phase Schur effective kernel
  -> primitive crystal-xy RHS-aware Ward validation
  -> positive-Matsubara sheet conductivity
  -> common lab-LT tangential-E reflection
  -> passive signed single-point logdet.
```

The material and q workspaces are built once per k-grid. All requested Matsubara frequencies are then evaluated in one vectorized contraction.

## Gates

A row is marked `single_point_pipeline_passed` only when all of the following hold:

- the mixed absolute-relative Ward criterion passes;
- the collective Schur block uses normal inversion and satisfies the condition gate;
- the dimensionless sheet tensor is finite, real-symmetric, and passive;
- the lab-LT reflection is constructed without a nonphysical pole;
- the identical-sheet round-trip product admits the signed real logdet without a negative product eigenvalue or a branch crossing.

The CSV also reports convergence relative to the highest requested `nk` for the complete dimensionless sheet tensor, reflection matrix and signed single-point logdet.

## Non-goals

This command does not perform the radial q integral, azimuthal angle integral, primed Matsubara sum, or a Casimir energy, pressure or torque calculation. The diagnostic separation is not a production-distance claim.

## Initial command

```bash
python -m validation matsubara positive-point \
  --nks 96 128 160 176 \
  --matsubara-indices 1 \
  --workers 2 \
  --pairing spm \
  --qx 0.008320503 \
  --qy 0.005547002 \
  --temperature-K 10 \
  --delta0-eV 0.1 \
  --eta-eV 1e-8 \
  --ward-tolerance 1e-7 \
  --separation-nm 20 \
  --output validation/outputs/positive_matsubara/single_point/raw/spm_q001_n1_nk_scan.csv
```

The first calibration uses `n=1`, because it is the positive frequency closest to the exact-static sector and is therefore the most likely positive-Matsubara point to expose slow convergence or a low-frequency convention problem.
