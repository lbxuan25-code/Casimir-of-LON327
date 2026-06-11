# Stage 4.10 Equal-time / commutator completion audit

## Boundary

- does not modify bubble factor
- no residual tuning
- no conductivity / reflection / Casimir
- no Ward closure claim
- only audits equal-time / commutator completion

## Fixed response formula

$V_i=\delta H/\delta A_i$, $M_{ij}=\delta^2H/\delta A_i\delta A_j$.

$J=(\rho,-V_x,-V_y)$, $P=(\rho,V_x,V_y)$.

$\Pi_{\mu\nu}=-\langle J_\mu P_\nu\rangle+\langle\delta J_\mu/\delta a_\nu\rangle+E_{\mu\nu}^{ET}$.

Current code includes $D_{ij}=-\langle M_{ij}\rangle$ but no explicit $E^{ET}$ term.

## Ward identity output directory

`validation/outputs/response/ward_identity/`

## Second-order Peierls identity check

status = `MATCH`; max_abs_error = 3.258483e-16; max_rel_error = 4.236200e-14.

## Bubble/direct/total/missing residual decomposition

$R^{missing}=-(R^{bubble}+R^{direct})$.

| q_scale | left_bubble_norm | left_direct_norm | left_total_norm | left_missing_norm | right_total_norm | right_missing_norm |
| --- | --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 5.393391e-03 | 1.363791e-02 | 1.903128e-02 | 1.903128e-02 | 1.758498e-02 | 1.758498e-02 |
| 5.000000e-01 | 2.662441e-03 | 6.819187e-03 | 9.481627e-03 | 9.481627e-03 | 8.693674e-03 | 8.693674e-03 |
| 2.500000e-01 | 1.329860e-03 | 3.409622e-03 | 4.739483e-03 | 4.739483e-03 | 4.336282e-03 | 4.336282e-03 |
| 1.250000e-01 | 6.648081e-04 | 1.704815e-03 | 2.369623e-03 | 2.369623e-03 | 2.166838e-03 | 2.166838e-03 |

## Longitudinal/transverse decomposition

| q_scale | left_total_longitudinal_abs | left_total_transverse_abs | left_missing_longitudinal_abs | left_missing_transverse_abs | right_total_longitudinal_abs | right_missing_longitudinal_abs |
| --- | --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 1.903127e-02 | 1.734973e-05 | 1.903127e-02 | 1.734973e-05 | 1.742631e-02 | 1.742631e-02 |
| 5.000000e-01 | 9.481627e-03 | 8.981241e-07 | 9.481627e-03 | 8.981241e-07 | 8.673395e-03 | 8.673395e-03 |
| 2.500000e-01 | 4.739483e-03 | 9.071873e-08 | 4.739483e-03 | 9.071873e-08 | 4.333723e-03 | 4.333723e-03 |
| 1.250000e-01 | 2.369623e-03 | 1.076361e-08 | 2.369623e-03 | 1.076361e-08 | 2.166517e-03 | 2.166517e-03 |

## q-scaling slopes

| quantity | slope |
| --- | --- |
| total_max_norm | 1.001734e+00 |
| left_total_norm | 1.001734e+00 |
| right_total_norm | 1.006555e+00 |
| left_missing_norm | 1.001734e+00 |
| right_missing_norm | 1.006555e+00 |
| left_total_longitudinal_abs | 1.001733e+00 |
| right_total_longitudinal_abs | 1.002444e+00 |
| left_missing_longitudinal_abs | 1.001733e+00 |
| right_missing_longitudinal_abs | 1.002444e+00 |

## Conclusion table

| item | analytic_status | code_or_diagnostic_status | conclusion |
| --- | --- | --- | --- |
| V_i vertex-level Ward identity | derived in Stage 4.1B | covered by existing vertex tests | MATCH |
| second-order Peierls identity q_i M_ij = Delta V_j | derived from hopping formulas | MATCH | MATCH |
| direct derivative term D_ij=-<M_ij> | derived | included in direct component | MATCH |
| residual after bubble + direct | nonzero allowed before ET audit | Stage 4.10 result | UNRESOLVED |
| need for E_ET | not proven zero | DIRECT_TERM_LEAVES_ORDER_Q_RESIDUAL | UNRESOLVED |

## Diagnostic status

- second_order_identity_status = `MATCH`
- direct_completion_status = `DIRECT_TERM_LEAVES_ORDER_Q_RESIDUAL`
- direct term numerically closes Ward residual: `False`
- remaining residual order: `DIRECT_TERM_LEAVES_ORDER_Q_RESIDUAL`

## Next step

Next: derive and evaluate the explicit equal-time commutator E_ET. Do not change bubble signs or fit contact coefficients.
