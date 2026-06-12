# Stage 4.16 Full response adaptive Ward diagnostic

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no source/observable change
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir

## Formula being tested

$$\Pi_{\mu\nu}=\Pi_{\mu\nu}^{bubble}+D_{\mu\nu},$$

with $J=(\rho,-V_x,-V_y)$, $P=(\rho,V_x,V_y)$, corrected positive bubble prefactor, and $D_{ij}=-\langle M_{ij}\rangle$.

## Adaptive quadrature summary

Stage 4.13 fixed the bubble sign. Stage 4.15 showed adaptive quadrature improves $C-K$. This stage applies the same point/weight strategy to the full physical response.

## Uniform reference comparison

| label | max max_norm |
| --- | --- |
| uniform_mesh_32 | 3.856895e-02 |
| uniform_mesh_64 | 9.985922e-03 |

## Ward residual versus refinement

| q_scale | level | max_norm | left_norm | right_norm | quad points |
| --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 4 | 1.352015e-02 | 2.907889e-07 | 1.352015e-02 | 649818 |
| 5.000000e-01 | 4 | 7.983478e-03 | 2.884635e-07 | 7.983478e-03 | 633294 |
| 2.500000e-01 | 4 | 4.278246e-03 | 4.139012e-07 | 4.278246e-03 | 625680 |
| 1.250000e-01 | 4 | 2.186036e-03 | 2.831614e-07 | 2.186036e-03 | 621630 |

## Longitudinal/transverse decomposition

| q_scale | left_long | left_trans | right_long | right_trans |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | 2.541769e-07 | 1.412525e-07 | 7.594954e-03 | 8.392795e-04 |
| 5.000000e-01 | 2.339902e-07 | 1.687002e-07 | 6.431449e-03 | 2.704554e-04 |
| 2.500000e-01 | 4.085142e-07 | 6.656066e-08 | 4.015938e-03 | 4.765851e-05 |
| 1.250000e-01 | 2.829208e-07 | 1.166894e-08 | 2.150110e-03 | 6.563041e-06 |

## Diagnostic decision

| quantity | status |
| --- | --- |
| adaptive_improvement_status | ADAPTIVE_FULL_RESPONSE_INCONCLUSIVE |
| refinement_convergence_status | FULL_RESPONSE_REFINEMENT_CONVERGING |
| closure_status | IMPROVED_BUT_NOT_CLOSED |
| dominant_remaining_channel | right_density_observable |
| likely_issue | FULL_RESPONSE_RESIDUAL_NOT_PRIMARILY_QUADRATURE |

If not closed, do not revert the Stage 4.13 bubble sign or change direct contact. Full Ward response numerical validation is required before conductivity/reflection/Casimir.

## Next step

Next: audit finite-q density vertex embedding, contact expectation routing, and response-level conventions.
