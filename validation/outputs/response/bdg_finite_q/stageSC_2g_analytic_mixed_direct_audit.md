# stageSC_2g_analytic_mixed_direct_audit

- status: PARTIAL_STAGE2G_LSQ_CLOSES_BUT_ANALYTIC_DOES_NOT
- diagnostic only: True
- formal Casimir ran: False
- production default modified: False
- least-squares candidate used as production formula: False
- best analytic candidates for dwave: ['analytic_right_only_plus']

## Candidate comparison

| pairing | N | q | baseline | lsq ref | best analytic | best name | etaeta | analytic+etaeta |
| ------- | -: | - | -------: | ------: | ------------: | --------- | -----: | ---------------: |
| onsite_s | 24 | (0.5235988,0) | 3.4483054e-16 | 3.469447e-18 | 3.4483054e-16 | analytic_right_only_plus | 3.4483054e-16 | 3.4483054e-16 |
| onsite_s | 24 | (0.5235988,0.5235988) | 1.7347238e-16 | 1.1796163e-16 | 1.7347238e-16 | analytic_right_only_plus | 1.7347238e-16 | 1.7347238e-16 |
| onsite_s | 36 | (0.3490659,0) | 1.2056624e-16 | 3.469447e-18 | 1.2056624e-16 | analytic_right_only_plus | 1.2056624e-16 | 1.2056624e-16 |
| onsite_s | 36 | (0.3490659,0.3490659) | 2.2898357e-16 | 5.2041848e-17 | 2.2898357e-16 | analytic_right_only_plus | 2.2898357e-16 | 2.2898357e-16 |
| onsite_s | 48 | (0.2617994,0) | 1.3292537e-16 | 8.1764085e-18 | 1.3292537e-16 | analytic_right_only_plus | 1.3292537e-16 | 1.3292537e-16 |
| onsite_s | 48 | (0.2617994,0.2617994) | 7.3420104e-17 | 5.2083824e-18 | 7.3420104e-17 | analytic_right_only_plus | 7.3420104e-17 | 7.3420104e-17 |
| spm | 24 | (0.5235988,0) | 4.8019605e-16 | 1.0062046e-18 | 4.8019605e-16 | analytic_right_only_plus | 4.8019605e-16 | 4.8019605e-16 |
| spm | 24 | (0.5235988,0.5235988) | 1.5352372e-16 | 3.036216e-17 | 1.5352372e-16 | analytic_right_only_plus | 1.5352372e-16 | 1.5352372e-16 |
| spm | 36 | (0.3490659,0) | 6.9612637e-17 | 3.0901934e-17 | 6.9612637e-17 | analytic_right_only_plus | 6.9612637e-17 | 6.9612637e-17 |
| spm | 36 | (0.3490659,0.3490659) | 8.8472066e-17 | 5.0307095e-17 | 8.8472066e-17 | analytic_right_only_plus | 8.8472066e-17 | 8.8472066e-17 |
| spm | 48 | (0.2617994,0) | 5.5803513e-17 | 3.469447e-18 | 5.5803513e-17 | analytic_right_only_plus | 5.5803513e-17 | 5.5803513e-17 |
| spm | 48 | (0.2617994,0.2617994) | 7.7196076e-17 | 5.9197607e-17 | 7.7196076e-17 | analytic_right_only_plus | 7.7196076e-17 | 7.7196076e-17 |
| dwave | 24 | (0.5235988,0) | 0.00087408578 | 1.2559842e-17 | 0.00087408578 | analytic_right_only_plus | 0.00067892515 | 0.00067892515 |
| dwave | 24 | (0.5235988,0.5235988) | 0.00083397642 | 1.4696624e-16 | 0.00083397642 | analytic_right_only_plus | 0.00056979513 | 0.00056979513 |
| dwave | 36 | (0.3490659,0) | 0.00063617515 | 6.0921378e-18 | 0.00063617515 | analytic_right_only_plus | 0.00051108807 | 0.00051108807 |
| dwave | 36 | (0.3490659,0.3490659) | 0.00063094848 | 1.8229414e-17 | 0.00063094848 | analytic_right_only_plus | 0.00045260193 | 0.00045260193 |
| dwave | 48 | (0.2617994,0) | 0.00047235104 | 1.84612e-17 | 0.00047235104 | analytic_right_only_plus | 0.00040304618 | 0.00040304618 |
| dwave | 48 | (0.2617994,0.2617994) | 0.00047358344 | 4.6847719e-17 | 0.00047358344 | analytic_right_only_plus | 0.00036762868 | 0.00036762868 |
| dwave_const_form | 24 | (0.5235988,0) | 2.8694333e-16 | 3.4179953e-17 | 2.8694333e-16 | analytic_right_only_plus | 2.8694333e-16 | 2.8694333e-16 |
| dwave_const_form | 24 | (0.5235988,0.5235988) | 3.7171872e-16 | 3.304283e-17 | 3.7171872e-16 | analytic_right_only_plus | 3.7171872e-16 | 3.7171872e-16 |
| dwave_const_form | 36 | (0.3490659,0) | 6.6520939e-17 | 4.7564177e-18 | 6.6520939e-17 | analytic_right_only_plus | 6.6520939e-17 | 6.6520939e-17 |
| dwave_const_form | 36 | (0.3490659,0.3490659) | 4.1653211e-17 | 3.6431649e-17 | 4.1653211e-17 | analytic_right_only_plus | 4.1653211e-17 | 4.1653211e-17 |
| dwave_const_form | 48 | (0.2617994,0) | 5.7879723e-17 | 1.167404e-17 | 5.7879723e-17 | analytic_right_only_plus | 5.7879723e-17 | 5.7879723e-17 |
| dwave_const_form | 48 | (0.2617994,0.2617994) | 1.1363659e-16 | 6.5061271e-17 | 1.1363659e-16 | analytic_right_only_plus | 1.1363659e-16 | 1.1363659e-16 |

## Human-readable conclusion

The analytic mixed direct formula was tested against the StageSC-2f least-squares reference.
The best dwave analytic candidates are ['analytic_right_only_plus'].
Max dwave best analytic Ward: 0.00087408578.
Max dwave LSQ reference Ward: 1.4696624e-16.
Left and right Ward components are stored separately for every analytic candidate in JSON.
onsite_s, spm, and dwave_const_form remain validation controls.
etaeta is retained as a secondary comparison and is not the primary tested block.
Formal Casimir input remains forbidden.
