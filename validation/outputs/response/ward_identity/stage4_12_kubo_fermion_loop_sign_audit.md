# Stage 4.12 Kubo bubble fermion-loop sign audit

## Boundary

- no residual tuning
- no bubble formula change to the main path
- no main response path change
- no direct contact change
- no conductivity / reflection / Casimir
- no Ward closure claim

## Analytic sign logic

$$\Pi^{bubble}=-\langle TJP\rangle_c.$$

$$\langle TJP\rangle_c=-\mathrm{Tr}[JGPG].$$

Therefore

$$\Pi^{bubble}=+\mathrm{Tr}[JGPG].$$

This audit compares the current negative band-sum bubble with a diagnostic-only candidate positive band-sum bubble.

## Ward bubble sign comparison

| q_scale | direction | neg=-C rel | neg=+C rel | pos=+C rel | pos=-C rel | status |
| --- | --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | x | 1.577607e-15 | 2.000000e+00 | 1.577607e-15 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| 1.000000e+00 | y | 2.012068e-14 | 2.000000e+00 | 2.012068e-14 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| 5.000000e-01 | x | 2.020985e-14 | 2.000000e+00 | 2.020985e-14 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| 5.000000e-01 | y | 5.063940e-14 | 2.000000e+00 | 5.063940e-14 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| 2.500000e-01 | x | 1.811944e-14 | 2.000000e+00 | 1.811944e-14 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| 2.500000e-01 | y | 7.731621e-15 | 2.000000e+00 | 7.731621e-15 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| 1.250000e-01 | x | 2.276660e-14 | 2.000000e+00 | 2.276660e-14 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| 1.250000e-01 | y | 1.900920e-14 | 2.000000e+00 | 1.900920e-14 | 2.000000e+00 | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |

## Total residual sign bookkeeping

| q_scale | direction | positive total vs C-K rel | negative total vs -(C+K) rel |
| --- | --- | --- | --- |
| 1.000000e+00 | x | 3.301993e-15 | 6.930574e-16 |
| 1.000000e+00 | y | 1.452707e-14 | 5.346257e-15 |
| 5.000000e-01 | x | 1.219369e-14 | 6.110493e-15 |
| 5.000000e-01 | y | 3.179137e-14 | 1.443440e-14 |
| 2.500000e-01 | x | 8.980046e-15 | 6.119262e-15 |
| 2.500000e-01 | y | 5.795802e-15 | 1.715672e-15 |
| 1.250000e-01 | x | 1.368296e-14 | 6.550873e-15 |
| 1.250000e-01 | y | 1.339474e-14 | 5.039492e-15 |

## Compressibility sanity check

| quantity | value |
| --- | --- |
| analytic_compressibility | -6.141087e-15 |
| positive_bubble_static_limit | -6.141087e-15 |
| negative_bubble_static_limit | 6.141087e-15 |
| compressibility_status | POSITIVE_BUBBLE_SIGN_MATCHES_COMPRESSIBILITY |

## Diagnostic decision

| quantity | status |
| --- | --- |
| ward_bubble_sign_global_status | POSITIVE_BUBBLE_GIVES_PLUS_C_AND_NEGATIVE_GIVES_MINUS_C |
| compressibility_status | POSITIVE_BUBBLE_SIGN_MATCHES_COMPRESSIBILITY |
| likely_issue | CURRENT_BUBBLE_OVERALL_SIGN_REVERSED |

## Next step

Next: prepare a narrow patch that flips the fermion-loop bubble prefactor in the main response path, then rerun Stage 4.9-4.11 diagnostics. Do not modify direct contact.
