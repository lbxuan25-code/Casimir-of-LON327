# stageSC_2j_longitudinal_ward_completion_audit

- status: PARTIAL_STAGE2J_LONGITUDINAL_IMPROVES_BUT_NOT_CLOSED
- diagnostic only: True
- formal Casimir ran: False
- production default modified: False
- right Ward convention tested: same_column_vector_[iomega,qx,qy,0,2iDelta0]
- hermitianized implementation: delta_right = delta_left.conjugate().T at the same finite-q response point

This stage tests only the longitudinal response-level Ward completion fixed by gauge invariance. It does not determine or claim the transverse microscopic mixed current.

Ward identity fixes only the longitudinal part: q_i delta K[V_i, eta_a] = -R_a. The minimum spatial completion used here is delta K[V_i, eta_a] = -q_i R_a / q^2; any transverse addition with q_i delta K_T[V_i, eta_a] = 0 remains undetermined.

## Case summary

| pairing | N | q | baseline | LSQ | best variant | best Schur | best full 5x5 | sign ambiguous |
| ------- | -: | - | -------: | --: | ------------ | ---------: | ------------: | -------------- |
| dwave | 24 | (0.5235988,0) | 0.00087408578 | 1.2559842e-17 | right_only_longitudinal | 0.00055798548 | 0.039442878 | False |
| dwave | 24 | (0.5235988,0.5235988) | 0.00083397642 | 1.4696624e-16 | left_only_longitudinal | 0.00025459645 | 0.039576448 | False |
| dwave | 36 | (0.3490659,0) | 0.00063617515 | 6.0921378e-18 | left_only_longitudinal | 0.00057359818 | 0.042939651 | False |
| dwave | 36 | (0.3490659,0.3490659) | 0.00063094848 | 1.8229414e-17 | left_only_longitudinal | 0.00026643857 | 0.039861327 | False |
| onsite_s | 24 | (0.5235988,0) | 3.4483054e-16 | 3.469447e-18 | both_longitudinal_hermitianized | 3.4543048e-16 | 0.041163262 | True |
| spm | 24 | (0.5235988,0) | 4.8019605e-16 | 1.0062046e-18 | left_only_longitudinal | 4.7569626e-16 | 0.03969515 | True |
| dwave_const_form | 24 | (0.5235988,0) | 2.8694333e-16 | 3.4179953e-17 | both_longitudinal_hermitianized | 2.8509106e-16 | 0.040295541 | True |

## Dwave variant detail

| N | q | variant | left full | right full | left eta cols | right eta rows | Schur Ward | delta L | delta R |
| -: | - | ------- | --------: | ---------: | ------------: | -------------: | ---------: | ------: | ------: |
| 24 | (0.5235988,0) | left_only_longitudinal | 4.0195427e-16 | 0.039442878 | 2.0328791e-20 | 0.0077291268 | 0.00055798548 | 0.01282469 | 0 |
| 24 | (0.5235988,0) | right_only_longitudinal | 0.006714992 | 0.039442878 | 0.006714992 | 5.4210109e-20 | 0.00055798548 | 0 | 0.014860712 |
| 24 | (0.5235988,0) | both_longitudinal_independent | 0.0011809236 | 0.039442878 | 2.0328791e-20 | 5.4210109e-20 | 0.0011809236 | 0.01282469 | 0.014860712 |
| 24 | (0.5235988,0) | both_longitudinal_hermitianized | 0.0010259752 | 0.039442878 | 2.0328791e-20 | 0.014444119 | 0.001115971 | 0.01282469 | 0.01282469 |
| 24 | (0.5235988,0) | both_longitudinal_opposite_sign | 0.013429984 | 0.039442878 | 0.013429984 | 0.015458254 | 0.0026040028 | 0.01282469 | 0.014860712 |
| 24 | (0.5235988,0) | density_plus_vector_min_norm | 0.001180493 | 0.039462466 | 6.9388939e-18 | 1.0842022e-19 | 0.001180493 | 0.012822352 | 0.014858003 |
| 24 | (0.5235988,0.5235988) | left_only_longitudinal | 3.1718371e-16 | 0.039576448 | 0 | 0.013764575 | 0.00025459645 | 0.01813685 | 0 |
| 24 | (0.5235988,0.5235988) | right_only_longitudinal | 0.013429984 | 0.039576448 | 0.013429984 | 3.0814879e-33 | 0.00025459645 | 0 | 0.018778878 |
| 24 | (0.5235988,0.5235988) | both_longitudinal_independent | 0.0010515361 | 0.039576448 | 0 | 3.0814879e-33 | 0.0010537319 | 0.01813685 | 0.018778878 |
| 24 | (0.5235988,0.5235988) | both_longitudinal_hermitianized | 0.0010259752 | 0.039576448 | 0 | 0.027194559 | 0.0010259752 | 0.01813685 | 0.01813685 |
| 24 | (0.5235988,0.5235988) | both_longitudinal_opposite_sign | 0.026859968 | 0.039576448 | 0.026859968 | 0.027529151 | 0.0023328247 | 0.01813685 | 0.018778878 |
| 24 | (0.5235988,0.5235988) | density_plus_vector_min_norm | 0.0010513444 | 0.039596039 | 3.0814879e-33 | 2.1684043e-19 | 0.0010531605 | 0.018135197 | 0.018777166 |
| 36 | (0.3490659,0) | left_only_longitudinal | 2.1991817e-16 | 0.042939651 | 3.0814879e-33 | 0.0049761838 | 0.00057359818 | 0.0088072413 | 0 |
| 36 | (0.3490659,0) | right_only_longitudinal | 0.0030743072 | 0.042939651 | 0.0030743072 | 1.540744e-33 | 0.00057359818 | 0 | 0.016124827 |
| 36 | (0.3490659,0) | both_longitudinal_independent | 0.0011404573 | 0.042939651 | 3.0814879e-33 | 1.540744e-33 | 0.0011455377 | 0.0088072413 | 0.016124827 |
| 36 | (0.3490659,0) | both_longitudinal_hermitianized | 0.00070457931 | 0.042939651 | 3.0814879e-33 | 0.008050491 | 0.0011471964 | 0.0088072413 | 0.0088072413 |
| 36 | (0.3490659,0) | both_longitudinal_opposite_sign | 0.0061486144 | 0.042939651 | 0.0061486144 | 0.0099523676 | 0.002248215 | 0.0088072413 | 0.016124827 |
| 36 | (0.3490659,0) | density_plus_vector_min_norm | 0.0011395221 | 0.042959819 | 6.9388939e-18 | 6.9388939e-18 | 0.0011430813 | 0.0088036295 | 0.016118214 |
| 36 | (0.3490659,0.3490659) | left_only_longitudinal | 1.2923815e-16 | 0.039861327 | 6.9388939e-18 | 0.0068016213 | 0.00026643857 | 0.01245532 | 0 |
| 36 | (0.3490659,0.3490659) | right_only_longitudinal | 0.0061486144 | 0.039861327 | 0.0061486144 | 6.9388939e-18 | 0.00027736289 | 0 | 0.014465746 |
| 36 | (0.3490659,0.3490659) | both_longitudinal_independent | 0.00077940839 | 0.039861327 | 6.9388939e-18 | 6.9388939e-18 | 0.00078338132 | 0.01245532 | 0.014465746 |
| 36 | (0.3490659,0.3490659) | both_longitudinal_hermitianized | 0.00070457931 | 0.039861327 | 6.9388939e-18 | 0.012950236 | 0.00070457931 | 0.01245532 | 0.01245532 |
| 36 | (0.3490659,0.3490659) | both_longitudinal_opposite_sign | 0.012297229 | 0.039861327 | 0.012297229 | 0.013603243 | 0.0018903837 | 0.01245532 | 0.014465746 |
| 36 | (0.3490659,0.3490659) | density_plus_vector_min_norm | 0.00077908869 | 0.039881503 | 6.9388939e-18 | 2.1684043e-19 | 0.00078244968 | 0.012452765 | 0.014462779 |

## Human-readable conclusion

The StageSC-2f LSQ mixed block remains the diagnostic reference for closing dwave.
Longitudinal response completion closes physical Schur Ward in at least one dwave case: False.
Full 5x5 left/right Ward closes in at least one dwave case: False.
Best Schur variants across dwave cases: {'left_only_longitudinal': 3, 'right_only_longitudinal': 1}.
Sign convention ambiguity suspected: False.
Control pairings present and monitored: ['dwave_const_form', 'onsite_s', 'spm'].
The transverse microscopic mixed current is still not determined by this audit.
Formal Casimir input remains forbidden.
This is a possible production-candidate direction only for the longitudinal response-level completion, not for a full microscopic mixed-current vertex.
