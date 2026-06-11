# Stage 4.11 Commutator sign and quadrature convergence audit

## Boundary

- no residual tuning
- no bubble formula change
- no main response path change
- no fitted contact
- no conductivity / reflection / Casimir
- no Ward closure claim

## Fixed formulas

$C_j^{(+q)}=\sum_k\operatorname{Tr}[(f(H_-)-f(H_+))V_j(k,q)]$.

$C_j^{(-q)}=\sum_k\operatorname{Tr}[(f(H_-)-f(H_+))V_j(k,-q)]$.

$K_j=q_i\langle M_{ij}\rangle$.

$R^{direct}_{L,j}$ should satisfy $R^{direct}_{L,j}=-K_j$.

## Bubble sign audit table

| q_scale | direction | best_candidate | best_rel | +C+ rel | -C+ rel | +C- rel | -C- rel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | x | MINUS_C_PLUS | 2.139214e-15 | 2.000000e+00 | 2.139214e-15 | 2.000000e+00 | 2.139214e-15 |
| 1.000000e+00 | y | MINUS_C_PLUS | 3.684408e-15 | 2.000000e+00 | 3.684408e-15 | 2.000000e+00 | 3.684408e-15 |
| 5.000000e-01 | x | MINUS_C_PLUS | 8.539079e-16 | 2.000000e+00 | 8.539079e-16 | 2.000000e+00 | 8.539079e-16 |
| 5.000000e-01 | y | MINUS_C_PLUS | 3.829834e-15 | 2.000000e+00 | 3.829834e-15 | 2.000000e+00 | 3.829834e-15 |
| 2.500000e-01 | x | MINUS_C_PLUS | 2.690944e-15 | 2.000000e+00 | 2.690944e-15 | 2.000000e+00 | 2.690944e-15 |
| 2.500000e-01 | y | MINUS_C_PLUS | 4.002633e-15 | 2.000000e+00 | 4.002633e-15 | 2.000000e+00 | 4.002633e-15 |
| 1.250000e-01 | x | MINUS_C_PLUS | 6.351972e-15 | 2.000000e+00 | 6.351972e-15 | 2.000000e+00 | 6.351972e-15 |
| 1.250000e-01 | y | MINUS_C_PLUS | 1.335386e-14 | 2.000000e+00 | 1.335386e-14 | 2.000000e+00 | 1.335386e-14 |

## Direct contact sign audit table

| q_scale | direction | direct_status | R_direct=-K rel | R_direct=+K rel |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | x | MATCH_R_DIRECT_EQUALS_MINUS_K | 9.462275e-16 | 2.000000e+00 |
| 1.000000e+00 | y | MATCH_R_DIRECT_EQUALS_MINUS_K | 4.852281e-16 | 2.000000e+00 |
| 5.000000e-01 | x | MATCH_R_DIRECT_EQUALS_MINUS_K | 1.576982e-16 | 2.000000e+00 |
| 5.000000e-01 | y | MATCH_R_DIRECT_EQUALS_MINUS_K | 3.639157e-16 | 2.000000e+00 |
| 2.500000e-01 | x | MATCH_R_DIRECT_EQUALS_MINUS_K | 4.730896e-16 | 2.000000e+00 |
| 2.500000e-01 | y | MATCH_R_DIRECT_EQUALS_MINUS_K | 6.065239e-16 | 2.000000e+00 |
| 1.250000e-01 | x | MATCH_R_DIRECT_EQUALS_MINUS_K | 7.884807e-16 | 2.000000e+00 |
| 1.250000e-01 | y | MATCH_R_DIRECT_EQUALS_MINUS_K | 1.091742e-15 | 2.000000e+00 |

## C_j vs K_j comparison

| q_scale | direction | C+ vs K rel | C- vs K rel | R_total abs |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | x | 6.168832e-01 | 6.168832e-01 | 3.971125e-02 |
| 1.000000e+00 | y | 6.399889e-01 | 6.399889e-01 | 2.701102e-02 |
| 5.000000e-01 | x | 6.208980e-01 | 6.208980e-01 | 2.000847e-02 |
| 5.000000e-01 | y | 6.332159e-01 | 6.332159e-01 | 1.332233e-02 |
| 2.500000e-01 | x | 6.254920e-01 | 6.254920e-01 | 1.009332e-02 |
| 2.500000e-01 | y | 6.288866e-01 | 6.288866e-01 | 6.604335e-03 |
| 1.250000e-01 | x | 6.272273e-01 | 6.272273e-01 | 5.063767e-03 |
| 1.250000e-01 | y | 6.280910e-01 | 6.280910e-01 | 3.297018e-03 |

## Mesh convergence table

| q_scale | direction | C+ status | C+ slope | C- status | C- slope | R_total slope |
| --- | --- | --- | --- | --- | --- | --- |
| 1.250000e-01 | x | NOT_CONVERGING_OR_INCONCLUSIVE | 4.511033e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 4.511033e-01 | 5.701548e-01 |
| 1.250000e-01 | y | NOT_CONVERGING_OR_INCONCLUSIVE | 4.532422e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 4.532422e-01 | 5.710557e-01 |
| 2.500000e-01 | x | NOT_CONVERGING_OR_INCONCLUSIVE | 4.457462e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 4.457462e-01 | 5.688234e-01 |
| 2.500000e-01 | y | NOT_CONVERGING_OR_INCONCLUSIVE | 4.541746e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 4.541746e-01 | 5.723478e-01 |
| 5.000000e-01 | x | NOT_CONVERGING_OR_INCONCLUSIVE | 4.288030e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 4.288030e-01 | 5.665188e-01 |
| 5.000000e-01 | y | NOT_CONVERGING_OR_INCONCLUSIVE | 4.598871e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 4.598871e-01 | 5.791097e-01 |
| 1.000000e+00 | x | NOT_CONVERGING_OR_INCONCLUSIVE | 3.711837e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 3.711837e-01 | 5.792266e-01 |
| 1.000000e+00 | y | NOT_CONVERGING_OR_INCONCLUSIVE | 4.430281e-01 | NOT_CONVERGING_OR_INCONCLUSIVE | 4.430281e-01 | 5.990379e-01 |

## Diagnostic decision

| quantity | status |
| --- | --- |
| bubble_sign_global_status | CONSISTENT_MATCH_MINUS_C_PLUS |
| direct_sign_global_status | MATCH_R_DIRECT_EQUALS_MINUS_K |
| quadrature_global_status | NOT_CONVERGING_OR_INCONCLUSIVE |
| global_Cplus_K_convergence_status | NOT_CONVERGING_OR_INCONCLUSIVE |
| global_Cminus_K_convergence_status | NOT_CONVERGING_OR_INCONCLUSIVE |
| likely_issue | BUBBLE_WARD_CONTRACTION_SIGN_CONVENTION |

## Next step

Next: audit Matsubara Fourier convention and left Ward contraction sign before modifying any vertex or contact term.
