# Stage 4.18 Corrected full response Ward validation

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no source/observable change
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir

## Corrected Ward residual convention

$$R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu},$$

$$R_R[\mu]=i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.$$

The legacy right residual $i\Omega\Pi_{\mu0}+q_x\Pi_{\mu x}+q_y\Pi_{\mu y}$ is kept only as an old diagnostic comparison and is not a closure criterion.

## Analytic derivation summary

Stage 4.13 fixed the bubble sign. Stage 4.15 addressed the $C-K$ quadrature issue. Stage 4.17 found the right Ward diagnostic convention problem. Stage 4.18 does not alter the response formula; it only consolidates the residual diagnostic definition.

The asymmetric left/right signs follow from $J_i=-V_i$ and $P_i=V_i$, together with

$$G_+^{-1}-G_-^{-1}=i\Omega\rho-q_iV_i.$$

## Adaptive full-response setup

The validation reuses the Stage 4.16 adaptive Fermi-window points and weights. Bubble and direct contact use identical integration points and weights. The response remains $\Pi_{\mu\nu}=\Pi_{\mu\nu}^{bubble}+D_{\mu\nu}$ with corrected positive bubble prefactor and unchanged $D_{ij}=-\langle M_{ij}\rangle$.

## Corrected left/right Ward residuals

| q_scale | left_norm | right_norm | max_corrected | left_long | left_trans | right_long | right_trans | quad points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 2.907889e-07 | 2.907889e-07 | 2.907889e-07 | 2.541769e-07 | 1.412525e-07 | 2.541769e-07 | 1.412525e-07 | 649818 |
| 5.000000e-01 | 2.884635e-07 | 2.884635e-07 | 2.884635e-07 | 2.339902e-07 | 1.687002e-07 | 2.339902e-07 | 1.687002e-07 | 633294 |
| 2.500000e-01 | 4.139012e-07 | 4.139012e-07 | 4.139012e-07 | 4.085142e-07 | 6.656066e-08 | 4.085142e-07 | 6.656066e-08 | 625680 |
| 1.250000e-01 | 2.831614e-07 | 2.831614e-07 | 2.831614e-07 | 2.829208e-07 | 1.166894e-08 | 2.829208e-07 | 1.166894e-08 | 621630 |

## Legacy right residual comparison

| q_scale | corrected right | legacy right | legacy/corrected |
| --- | --- | --- | --- |
| 1.000000e+00 | 2.907889e-07 | 1.352015e-02 | 4.649472e+04 |
| 5.000000e-01 | 2.884635e-07 | 7.983478e-03 | 2.767587e+04 |
| 2.500000e-01 | 4.139012e-07 | 4.278246e-03 | 1.033639e+04 |
| 1.250000e-01 | 2.831614e-07 | 2.186036e-03 | 7.720106e+03 |

## Diagnostic decision

| quantity | status |
| --- | --- |
| corrected_ward_status | CORRECTED_WARD_NUMERICALLY_CLOSED |
| right_convention_status | RIGHT_WARD_CONVENTION_FIX_VALIDATED |
| max_corrected_norm | 4.139012e-07 |
| max_legacy_right_norm | 1.352015e-02 |
| dominant_remaining_channel | right_spatial_observable |
| likely_issue | PREVIOUS_RIGHT_RESIDUAL_WAS_DIAGNOSTIC_CONVENTION |

## Next step

Next: Stage 4.19 multi-parameter robustness scan before any conductivity/reflection/Casimir use.

Before conductivity/reflection/Casimir use, Stage 4.19 must perform a multi-parameter robustness scan.
