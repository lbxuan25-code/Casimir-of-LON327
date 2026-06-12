# Stage 4.15 Fermi-window adaptive quadrature

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir
- no Ward closure claim

## Purpose

Stage 4.13 fixed the bubble sign. Stage 4.14 pointed to low-temperature Fermi-surface quadrature. This stage checks whether Fermi-window adaptive quadrature improves $C_j-K_j$ using the same quadrature points and weights for both quantities.

## Adaptive final-level results

| q_scale | direction | level | |C-K| rel | cells | quad points |
| --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | x | 4 | 2.609208e-05 | 72202 | 649818 |
| 1.000000e+00 | y | 4 | 2.780084e-06 | 72202 | 649818 |
| 5.000000e-01 | x | 4 | 1.875259e-05 | 70366 | 633294 |
| 5.000000e-01 | y | 4 | 7.443490e-05 | 70366 | 633294 |
| 2.500000e-01 | x | 4 | 1.101750e-04 | 69520 | 625680 |
| 2.500000e-01 | y | 4 | 1.541137e-04 | 69520 | 625680 |
| 1.250000e-01 | x | 4 | 1.661059e-04 | 69070 | 621630 |
| 1.250000e-01 | y | 4 | 1.815119e-04 | 69070 | 621630 |

## Uniform reference comparison

| label | max |C-K| rel |
| --- | --- |
| uniform_mesh_32 | 6.399889e-01 |
| uniform_mesh_64 | 3.238206e-01 |
| adaptive_final | 1.815119e-04 |

## Temperature sanity

| temperature_K | max |C-K| rel |
| --- | --- |
| 3.000000e+01 | 2.609208e-05 |
| 1.000000e+02 | 1.360784e-05 |
| 3.000000e+02 | 3.667475e-04 |
| 1.000000e+03 | 1.106473e-05 |

## Diagnostic decision

| quantity | status |
| --- | --- |
| adaptive_improvement_status | ADAPTIVE_QUADRATURE_IMPROVES_CK |
| refinement_convergence_status | REFINEMENT_CONVERGING |
| temperature_sanity_status | TEMPERATURE_SANITY_CONFIRMED |
| likely_issue | FERMI_WINDOW_ADAPTIVE_QUADRATURE_CONFIRMED |

## Next step

Next: test the same adaptive quadrature strategy on the full Ward response diagnostic before any conductivity/reflection/Casimir use.
