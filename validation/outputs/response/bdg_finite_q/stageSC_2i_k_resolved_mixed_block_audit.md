# stageSC_2i_k_resolved_mixed_block_audit

- status: PARTIAL_STAGE2I_NO_CLEAR_K_BASIS
- diagnostic only: True
- formal Casimir ran: False
- production default modified: False

## K-resolved projection summary

| pairing | N | q | baseline | LSQ Ward | sum ok | best k basis | single residual | multi residual | family |
| ------- | -: | - | -------: | -------: | ------ | ------------ | --------------: | -------------: | ------ |
| dwave | 24 | (0.5235988,0) | 0.00087408578 | 1.2559842e-17 | True | basis_phi | 1 | 1 | unclear_basis_phi |
| dwave | 24 | (0.5235988,0.5235988) | 0.00083397642 | 1.4696624e-16 | True | basis_phi | 1 | 1 | unclear_basis_phi |
| dwave | 36 | (0.3490659,0) | 0.00063617515 | 6.0921378e-18 | True | basis_phi | 1 | 1 | unclear_basis_phi |
| dwave | 36 | (0.3490659,0.3490659) | 0.00063094848 | 1.8229414e-17 | True | basis_phi | 1 | 1 | unclear_basis_phi |
| onsite_s | 24 | (0.5235988,0) | 3.4483054e-16 | 3.469447e-18 | True | basis_partial_q_phi_x | 1 | 1 | partial_q_phi |
| spm | 24 | (0.5235988,0) | 4.8019605e-16 | 1.0062046e-18 | True | basis_phi | 1 | 1 | unclear_basis_phi |
| dwave_const_form | 24 | (0.5235988,0) | 2.8694333e-16 | 3.4179953e-17 | True | basis_phi | 1 | 1 | unclear_basis_phi |

## Top local eta2 residuals

### dwave N=24 q=(0.5235988,0)

| rank | k | abs | phase | partial_q_x | partial_q_y | phi | partial_k_x |
| ---: | - | --: | ----: | ----------: | ----------: | --: | ----------: |
| 1 | (4.97419,0) | 0.0016234278 | 1.5708 | 0.066987298 | 0 | 2.5 | 1.8660254 |
| 2 | (1.309,0) | 0.0016234278 | 1.5708 | 0.066987298 | 0 | 2.5 | 1.8660254 |
| 3 | (1.309,6.02139) | 0.0012825002 | 1.5708 | 0.066987298 | 0 | 2.4318517 | 1.8660254 |
| 4 | (1.309,0.261799) | 0.0012825002 | 1.5708 | 0.066987298 | 0 | 2.4318517 | 1.8660254 |
| 5 | (4.97419,0.261799) | 0.0012825002 | 1.5708 | 0.066987298 | 0 | 2.4318517 | 1.8660254 |

### dwave N=24 q=(0.5235988,0.5235988)

| rank | k | abs | phase | partial_q_x | partial_q_y | phi | partial_k_x |
| ---: | - | --: | ----: | ----------: | ----------: | --: | ----------: |
| 1 | (0,1.309) | 0.0016469897 | 1.5708 | 0.25881905 | 0.066987298 | 2.4318517 | 0 |
| 2 | (4.97419,0) | 0.0016469897 | 1.5708 | 0.066987298 | 0.25881905 | 2.4318517 | 1.8660254 |
| 3 | (1.309,0) | 0.0016469897 | 1.5708 | 0.066987298 | 0.25881905 | 2.4318517 | 1.8660254 |
| 4 | (0,4.97419) | 0.0016469897 | 1.5708 | 0.25881905 | 0.066987298 | 2.4318517 | 0 |
| 5 | (0.261799,1.309) | 0.0013116636 | 1.5708 | 0.25 | 0.066987298 | 2.3660254 | 0.5 |

### dwave N=36 q=(0.3490659,0)

| rank | k | abs | phase | partial_q_x | partial_q_y | phi | partial_k_x |
| ---: | - | --: | ----: | ----------: | ----------: | --: | ----------: |
| 1 | (5.06145,5.75959) | 0.00060235069 | 1.5708 | 0.059391175 | 0 | 2.405699 | 1.8508332 |
| 2 | (1.22173,5.75959) | 0.00060235069 | 1.5708 | 0.059391175 | 0 | 2.405699 | 1.8508332 |
| 3 | (1.22173,0.523599) | 0.00060235069 | 1.5708 | 0.059391175 | 0 | 2.405699 | 1.8508332 |
| 4 | (5.06145,0.523599) | 0.00060235069 | 1.5708 | 0.059391175 | 0 | 2.405699 | 1.8508332 |
| 5 | (1.22173,0.349066) | 0.00052743836 | 1.5708 | 0.059391175 | 0 | 2.5530334 | 1.8508332 |

### dwave N=36 q=(0.3490659,0.3490659)

| rank | k | abs | phase | partial_q_x | partial_q_y | phi | partial_k_x |
| ---: | - | --: | ----: | ----------: | ----------: | --: | ----------: |
| 1 | (1.22173,0.349066) | 0.00057995029 | 1.5708 | 0.059391175 | 0.16317591 | 2.5244813 | 1.8508332 |
| 2 | (0.349066,1.22173) | 0.00057995029 | 1.5708 | 0.16317591 | 0.059391175 | 2.5244813 | 0.67364818 |
| 3 | (5.93412,5.06145) | 0.00057995029 | 1.5708 | 0.16317591 | 0.059391175 | 2.5244813 | 0.67364818 |
| 4 | (5.06145,5.93412) | 0.00057995029 | 1.5708 | 0.059391175 | 0.16317591 | 2.5244813 | 1.8508332 |
| 5 | (5.75959,1.22173) | 0.00053718504 | 1.5708 | 0.15038373 | 0.059391175 | 2.3793852 | 0.98480775 |

## Human-readable conclusion

The LSQ mixed block still closes dwave in the quick k-resolved cases.
The k-resolved local eta2 residual sums back to the integrated residual within tolerance.
Dominant k-resolved basis families for dwave: ['unclear_basis_phi'].
The hopping-like / partial_k negative controls remain cancellation diagnostics, not production formulas.
This run does not claim a final analytic formula.
Formal Casimir input remains forbidden.
