# Stage 4.9 Ward residual regression after Kubo bubble audit

## Boundary

- no residual tuning
- no bubble formula change
- no conductivity / reflection / Casimir
- no claim of Ward closure unless `NUMERICALLY_CLOSED`

## Fixed Response Formula

$J=(\rho,-V_x,-V_y)$

$P=(\rho,V_x,V_y)$

$\Pi_{\mu\nu}^{4.8}=-\langle J_\mu P_\nu\rangle+\left\langle\delta J_\mu/\delta a_\nu\right\rangle$

## Configuration

mesh_size = 16; temperature_K = 30.0; matsubara_index = 1; omega_eV = 1.624329e-02; q_base = [0.02, 0.013]

## Stage 4.8 q-scaling

| q_scale | q_norm | left_error | right_error | max_error | max_norm |
| --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 2.385372e-02 | 1.505649e-02 | 1.849533e-02 | 1.849533e-02 | 1.012759e-02 |
| 5.000000e-01 | 1.192686e-02 | 7.663186e-03 | 9.218361e-03 | 9.218361e-03 | 5.000320e-03 |
| 2.500000e-01 | 5.963430e-03 | 3.839559e-03 | 4.596887e-03 | 4.596887e-03 | 2.489982e-03 |
| 1.250000e-01 | 2.981715e-03 | 1.920538e-03 | 2.296637e-03 | 2.296637e-03 | 1.243671e-03 |

## Historical Stage 4.7 q-scaling

historical Stage 4.7 diagnostic only

| q_scale | q_norm | left_error | right_error | max_error | max_norm |
| --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 2.385372e-02 | 1.756343e-02 | 1.756343e-02 | 1.756343e-02 | 1.903128e-02 |
| 5.000000e-01 | 1.192686e-02 | 8.804503e-03 | 8.804503e-03 | 8.804503e-03 | 9.481627e-03 |
| 2.500000e-01 | 5.963430e-03 | 4.404872e-03 | 4.404872e-03 | 4.404872e-03 | 4.739483e-03 |
| 1.250000e-01 | 2.981715e-03 | 2.202736e-03 | 2.202736e-03 | 2.202736e-03 | 2.369623e-03 |

## Left/Right Residual Decomposition

### Stage 4.8

| q_scale | left_density_source_abs | left_spatial_source_norm | right_density_observable_abs | right_spatial_observable_norm |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | 5.590219e-17 | 8.244565e-03 | 2.356935e-03 | 9.849514e-03 |
| 5.000000e-01 | 1.233194e-16 | 4.156746e-03 | 5.934558e-04 | 4.964979e-03 |
| 2.500000e-01 | 3.353462e-16 | 2.079762e-03 | 1.489673e-04 | 2.485522e-03 |
| 1.250000e-01 | 1.731500e-16 | 1.040007e-03 | 3.728324e-05 | 1.243112e-03 |

### Historical Stage 4.7

| q_scale | left_density_source_abs | left_spatial_source_norm | right_density_observable_abs | right_spatial_observable_norm |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | 5.649469e-17 | 1.903128e-02 | 5.652744e-17 | 1.903128e-02 |
| 5.000000e-01 | 1.232301e-16 | 9.481627e-03 | 1.232271e-16 | 9.481627e-03 |
| 2.500000e-01 | 3.353501e-16 | 4.739483e-03 | 3.353505e-16 | 4.739483e-03 |
| 1.250000e-01 | 1.730881e-16 | 2.369623e-03 | 1.730885e-16 | 2.369623e-03 |

## Longitudinal/Transverse Decomposition

### Stage 4.8

| q_scale | left_longitudinal_abs | left_transverse_abs | right_longitudinal_abs | right_transverse_abs |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | 8.244547e-03 | 1.692233e-05 | 9.849512e-03 | 5.531906e-06 |
| 5.000000e-01 | 4.156746e-03 | 8.446975e-07 | 4.964978e-03 | 1.642202e-06 |
| 2.500000e-01 | 2.079762e-03 | 8.404033e-08 | 2.485522e-03 | 2.206463e-07 |
| 1.250000e-01 | 1.040007e-03 | 9.928809e-09 | 1.243112e-03 | 2.798764e-08 |

### Historical Stage 4.7

| q_scale | left_longitudinal_abs | left_transverse_abs | right_longitudinal_abs | right_transverse_abs |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | 1.903127e-02 | 1.734973e-05 | 1.903127e-02 | 1.734973e-05 |
| 5.000000e-01 | 9.481627e-03 | 8.981241e-07 | 9.481627e-03 | 8.981241e-07 |
| 2.500000e-01 | 4.739483e-03 | 9.071873e-08 | 4.739483e-03 | 9.071873e-08 |
| 1.250000e-01 | 2.369623e-03 | 1.076361e-08 | 2.369623e-03 | 1.076361e-08 |

## Slope Table

| response | quantity | slope |
| --- | --- | --- |
| stage48_physical_observable_source | max_norm | 1.008273e+00 |
| stage48_physical_observable_source | left_norm | 9.959588e-01 |
| stage48_physical_observable_source | right_norm | 1.008273e+00 |
| stage48_physical_observable_source | left_spatial_source_norm | 9.959588e-01 |
| stage48_physical_observable_source | right_spatial_observable_norm | 9.956527e-01 |
| stage48_physical_observable_source | left_longitudinal_abs | 9.959579e-01 |
| stage48_physical_observable_source | right_longitudinal_abs | 9.956527e-01 |
| stage48_physical_observable_source | left_transverse_abs | 3.553434e+00 |
| stage48_physical_observable_source | right_transverse_abs | 2.577635e+00 |

| response | quantity | slope |
| --- | --- | --- |
| stage47_historical_observable_observable | max_norm | 1.001734e+00 |
| stage47_historical_observable_observable | left_norm | 1.001734e+00 |
| stage47_historical_observable_observable | right_norm | 1.001734e+00 |
| stage47_historical_observable_observable | left_spatial_source_norm | 1.001734e+00 |
| stage47_historical_observable_observable | right_spatial_observable_norm | 1.001734e+00 |
| stage47_historical_observable_observable | left_longitudinal_abs | 1.001733e+00 |
| stage47_historical_observable_observable | right_longitudinal_abs | 1.001733e+00 |
| stage47_historical_observable_observable | left_transverse_abs | 3.527105e+00 |
| stage47_historical_observable_observable | right_transverse_abs | 3.527105e+00 |

## Stage 4.8 vs Stage 4.7

| stage48_smallest_max_error | stage47_smallest_max_error | ratio_stage48_over_stage47 |
| --- | --- | --- |
| 2.296637e-03 | 2.202736e-03 | 1.042630e+00 |

Stage 4.8 source/observable split changes the residual but does not substantially reduce it.

## Final Diagnostic Status

- Stage 4.8: `ORDER_Q_RESIDUAL`
- Historical Stage 4.7: `ORDER_Q_RESIDUAL`

Dominance at smallest q for Stage 4.8: right / spatial / longitudinal.

## Next Step

Next: audit equal-time / commutator completion. After the Stage 4.13 bubble prefactor fix, do not tune signs further or introduce fitting coefficients.
