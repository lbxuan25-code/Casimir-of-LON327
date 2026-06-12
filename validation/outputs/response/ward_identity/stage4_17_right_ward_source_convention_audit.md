# Stage 4.17 Right Ward source-side convention audit

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no source/observable change
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir

## Analytic source-side Ward identity

$$G_+^{-1}-G_-^{-1}=i\Omega\rho-q_iV_i.$$

Because $P_0=\rho$ and $P_i=V_i$, the natural right Ward source-side contraction is $i\Omega\Pi_{\mu0}-q_i\Pi_{\mu i}$.

## Candidate definitions

Candidates: plus/plus, plus/minus, minus/plus, and minus/minus in the omega/q signs. The predicted candidate is `R_right_plus_omega_minus_q`.

## Adaptive full-response setup

Uses Stage 4.16 adaptive full-response quadrature with corrected Stage 4.13 bubble sign and unchanged direct contact.

## Left Ward reference

| q_scale | left_norm |
| --- | --- |
| 1.000000e+00 | 2.907889e-07 |
| 5.000000e-01 | 2.884635e-07 |
| 2.500000e-01 | 4.139012e-07 |
| 1.250000e-01 | 2.831614e-07 |

## Right Ward candidate comparison

| q_scale | left | ++ | +- | -+ | -- | best |
| --- | --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 2.907889e-07 | 1.352015e-02 | 2.907889e-07 | 2.907889e-07 | 1.352015e-02 | R_right_plus_omega_minus_q |
| 5.000000e-01 | 2.884635e-07 | 7.983478e-03 | 2.884635e-07 | 2.884635e-07 | 7.983478e-03 | R_right_plus_omega_minus_q |
| 2.500000e-01 | 4.139012e-07 | 4.278246e-03 | 4.139012e-07 | 4.139012e-07 | 4.278246e-03 | R_right_plus_omega_minus_q |
| 1.250000e-01 | 2.831614e-07 | 2.186036e-03 | 2.831614e-07 | 2.831614e-07 | 2.186036e-03 | R_right_plus_omega_minus_q |

## Best candidate decision

| quantity | status |
| --- | --- |
| best_candidate_global | R_right_plus_omega_minus_q |
| right_source_sign_status | RIGHT_WARD_SOURCE_SIGN_CONFIRMED |
| closure_status | RIGHT_WARD_NUMERICALLY_CLOSED |
| dominant_remaining_channel | right_spatial_observable |
| likely_issue | RIGHT_WARD_DIAGNOSTIC_SIGN_CONVENTION |

## Diagnostic decision

Do not change the response formula from this diagnostic alone.

## Next step

Next: update diagnostic Ward residual convention docs/tests, then rerun full response validation without changing the response formula.
