# Stage 4.14 C_j versus K_j routing/contact audit

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no source/observable change
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir
- no Ward closure claim

## Analytic identity being tested

$$C_j=\sum_k\operatorname{Tr}[(f(H_-)-f(H_+))V_j(k,q)].$$

$$K_j=\sum_k\operatorname{Tr}[f(H(k))q_iM_{ij}(k,q)].$$

In the continuous BZ integral, the expected identity is $C_j=K_j$.

## Hamiltonian representation consistency

| quantity | value |
| --- | --- |
| status | H_REPRESENTATION_MATCH |
| max_abs_H_model_minus_H_hopping | 1.306054e-15 |
| max_rel_H_model_minus_H_hopping | 5.880581e-16 |

## Second-order Peierls identity

| quantity | value |
| --- | --- |
| status | SECOND_ORDER_IDENTITY_MATCH |
| max_abs_second_order_identity_error | 8.010558e-16 |
| max_rel_second_order_identity_error | 1.843120e-13 |

## Baseline C-K results

| q_scale | direction | |C-K| rel | |K-K_deltaV| rel | mesh_shift rel |
| --- | --- | --- | --- | --- |
| 1.000000e+00 | x | 6.053443e-01 | 3.060580e-16 | 6.053443e-01 |
| 1.000000e+00 | y | 6.026081e-01 | 2.684184e-15 | 6.026081e-01 |
| 5.000000e-01 | x | 6.096483e-01 | 3.036622e-15 | 6.096483e-01 |
| 5.000000e-01 | y | 6.093720e-01 | 8.293983e-15 | 6.093720e-01 |
| 2.500000e-01 | x | 6.099848e-01 | 6.979791e-15 | 6.099848e-01 |
| 2.500000e-01 | y | 6.099296e-01 | 1.018221e-15 | 6.099296e-01 |
| 1.250000e-01 | x | 6.100448e-01 | 2.579343e-15 | 6.100448e-01 |
| 1.250000e-01 | y | 6.100317e-01 | 2.512672e-14 | 6.100317e-01 |

## Mesh convergence

| q_scale | direction | finest mesh | finest rel | slope | status |
| --- | --- | --- | --- | --- | --- |
| 1.250000e-01 | x | 64 | 3.238206e-01 | -5.388195e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |
| 1.250000e-01 | y | 64 | 3.225496e-01 | -5.388907e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |
| 2.500000e-01 | x | 64 | 3.177806e-01 | -6.561589e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |
| 2.500000e-01 | y | 64 | 3.128382e-01 | -6.550934e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |
| 5.000000e-01 | x | 64 | 3.020902e-01 | -7.081047e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |
| 5.000000e-01 | y | 64 | 2.857820e-01 | -7.327530e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |
| 1.000000e+00 | x | 64 | 3.209726e-01 | -3.934965e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |
| 1.000000e+00 | y | 64 | 3.009020e-01 | -4.004907e-01 | NOT_CONVERGING_OR_INCONCLUSIVE |

## Commensurate q shift test

| mesh | q_label | direction | |C-K| rel |
| --- | --- | --- | --- |
| 8 | q_comm_x | x | 8.224259e-16 |
| 8 | q_comm_x | y | 1.014258e+00 |
| 8 | q_comm_y | x | 1.015844e+00 |
| 8 | q_comm_y | y | 1.151406e-15 |
| 12 | q_comm_x | x | 1.190908e-16 |
| 12 | q_comm_x | y | 1.111076e+00 |
| 12 | q_comm_y | x | 1.146623e+00 |
| 12 | q_comm_y | y | 3.568810e-16 |
| 16 | q_comm_x | x | 6.794150e-19 |
| 16 | q_comm_x | y | 4.453600e-01 |
| 16 | q_comm_y | x | 1.948447e-01 |
| 16 | q_comm_y | y | 9.425196e-16 |
| 24 | q_comm_x | x | 1.926870e-16 |
| 24 | q_comm_x | y | 1.030286e+00 |
| 24 | q_comm_y | x | 1.009149e+00 |
| 24 | q_comm_y | y | 2.118655e-15 |
| 32 | q_comm_x | x | 1.312973e-16 |
| 32 | q_comm_x | y | 1.027297e+00 |
| 32 | q_comm_y | x | 1.027349e+00 |
| 32 | q_comm_y | y | 1.291414e-18 |
| 48 | q_comm_x | x | 1.357120e-15 |
| 48 | q_comm_x | y | 1.005210e+00 |
| 48 | q_comm_y | x | 1.004455e+00 |
| 48 | q_comm_y | y | 3.877482e-16 |
| 64 | q_comm_x | x | 1.021331e-15 |
| 64 | q_comm_x | y | 7.094592e-01 |
| 64 | q_comm_y | x | 1.496995e+00 |
| 64 | q_comm_y | y | 3.829998e-16 |

## Temperature sweep

| temperature_K | direction | |C-K| rel |
| --- | --- | --- |
| 3.000000e+01 | x | 6.168832e-01 |
| 3.000000e+01 | y | 6.399889e-01 |
| 1.000000e+02 | x | 5.317741e-01 |
| 1.000000e+02 | y | 5.402236e-01 |
| 3.000000e+02 | x | 2.389431e-01 |
| 3.000000e+02 | y | 2.413716e-01 |
| 1.000000e+03 | x | 2.918534e-03 |
| 1.000000e+03 | y | 2.946478e-03 |

## Diagnostic decision

| quantity | status |
| --- | --- |
| H_representation_status | H_REPRESENTATION_MATCH |
| second_order_identity_status | SECOND_ORDER_IDENTITY_MATCH |
| q_base_CK_convergence_status | NOT_CONVERGING_OR_INCONCLUSIVE |
| commensurate_q_status | COMMENSURATE_SHIFT_NOT_CLOSE |
| temperature_sweep_status | TEMPERATURE_IMPROVES_CK_CONVERGENCE |
| likely_issue | LOW_TEMPERATURE_FERMI_SURFACE_QUADRATURE |

The remaining C-K mismatch should not be addressed by reverting the Stage 4.13 bubble sign fix or changing direct-contact signs.

## Next step

Improve Fermi-surface quadrature or use higher mesh before introducing new Ward terms.
