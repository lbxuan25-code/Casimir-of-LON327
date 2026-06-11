# Stage 4.13 Bubble prefactor sign fix regression

## Boundary

- bubble prefactor changed from negative to positive
- direct contact unchanged
- source/observable split unchanged
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir
- no Ward closure claim

## Changed formula

$$\Pi_{\mu\nu}^{bubble}=\sum_{k,m,n}\frac{f(E_m^-)-f(E_n^+)}{i\Omega+E_m^- -E_n^+}J_{\mu,mn}^{-+}P_{\nu,nm}^{+-}.$$

## Bubble sign regression

The corrected main bubble is expected to satisfy $R_L^{bubble}[j]\approx +C_j$.

## Direct contact regression

The direct term is unchanged and is expected to satisfy $R_L^{direct}[j]\approx -K_j$.

## Total residual bookkeeping

The total spatial-source residual is expected to satisfy $R_L^{total}[j]\approx C_j-K_j$.

| q_scale | direction | bubble=+C rel | direct=-K rel | total=C-K rel | |R_total| |
| --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | x | 1.577607e-15 | 1.365386e-15 | 3.301993e-15 | 6.921811e-03 |
| 1.000000e+00 | y | 2.012068e-14 | 7.001740e-16 | 1.452707e-14 | 4.478994e-03 |
| 5.000000e-01 | x | 2.020985e-14 | 4.551098e-16 | 1.219369e-14 | 3.485657e-03 |
| 5.000000e-01 | y | 5.063940e-14 | 3.500815e-16 | 3.179137e-14 | 2.264670e-03 |
| 2.500000e-01 | x | 1.811944e-14 | 1.517017e-15 | 8.980046e-15 | 1.743809e-03 |
| 2.500000e-01 | y | 7.731621e-15 | 5.834668e-16 | 5.795802e-15 | 1.133376e-03 |
| 1.250000e-01 | x | 2.276660e-14 | 6.068052e-16 | 1.368296e-14 | 8.719924e-04 |
| 1.250000e-01 | y | 1.900920e-14 | 5.834662e-16 | 1.339474e-14 | 5.667832e-04 |

## Diagnostic decision

| quantity | status |
| --- | --- |
| main_bubble_sign_status | MAIN_BUBBLE_MATCHES_PLUS_C |
| direct_contact_status | DIRECT_STILL_MATCHES_MINUS_K |
| total_bookkeeping_status | TOTAL_MATCHES_C_MINUS_K |
| likely_remaining_issue | C_MINUS_K_ROUTING_OR_CONTACT_EXPECTATION |
| max_total_residual_abs | 6.921811e-03 |

This remaining residual is not a bubble overall sign issue. It should be addressed by auditing C_j versus K_j routing, density q-convention, or contact thermal expectation.

## Next step

Next: rerun Stage 4.9-4.11 residual diagnostics with the corrected bubble sign. If an O(q) residual remains, audit C_j versus K_j routing, density q-convention, and contact thermal expectation.
