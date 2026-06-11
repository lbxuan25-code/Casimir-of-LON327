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
| 1.000000e+00 | 5.393391e-03 | 1.363791e-02 | 8.244565e-03 | 8.244565e-03 | 1.012759e-02 | 1.012759e-02 |
| 5.000000e-01 | 2.662441e-03 | 6.819187e-03 | 4.156746e-03 | 4.156746e-03 | 5.000320e-03 | 5.000320e-03 |
| 2.500000e-01 | 1.329860e-03 | 3.409622e-03 | 2.079762e-03 | 2.079762e-03 | 2.489982e-03 | 2.489982e-03 |
| 1.250000e-01 | 6.648081e-04 | 1.704815e-03 | 1.040007e-03 | 1.040007e-03 | 1.243671e-03 | 1.243671e-03 |

## Longitudinal/transverse decomposition

| q_scale | left_total_longitudinal_abs | left_total_transverse_abs | left_missing_longitudinal_abs | left_missing_transverse_abs | right_total_longitudinal_abs | right_missing_longitudinal_abs |
| --- | --- | --- | --- | --- | --- | --- |
| 1.000000e+00 | 8.244547e-03 | 1.692233e-05 | 8.244547e-03 | 1.692233e-05 | 9.849512e-03 | 9.849512e-03 |
| 5.000000e-01 | 4.156746e-03 | 8.446975e-07 | 4.156746e-03 | 8.446975e-07 | 4.964978e-03 | 4.964978e-03 |
| 2.500000e-01 | 2.079762e-03 | 8.404033e-08 | 2.079762e-03 | 8.404033e-08 | 2.485522e-03 | 2.485522e-03 |
| 1.250000e-01 | 1.040007e-03 | 9.928809e-09 | 1.040007e-03 | 9.928809e-09 | 1.243112e-03 | 1.243112e-03 |

## q-scaling slopes

| quantity | slope |
| --- | --- |
| total_max_norm | 1.008273e+00 |
| left_total_norm | 9.959588e-01 |
| right_total_norm | 1.008273e+00 |
| left_missing_norm | 9.959588e-01 |
| right_missing_norm | 1.008273e+00 |
| left_total_longitudinal_abs | 9.959579e-01 |
| right_total_longitudinal_abs | 9.956527e-01 |
| left_missing_longitudinal_abs | 9.959579e-01 |
| right_missing_longitudinal_abs | 9.956527e-01 |

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

Next: derive and evaluate the explicit equal-time commutator E_ET. After the Stage 4.13 bubble prefactor fix, do not tune signs further or fit contact coefficients.
