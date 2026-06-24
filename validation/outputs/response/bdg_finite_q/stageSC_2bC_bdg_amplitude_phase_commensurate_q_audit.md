# stageSC_2bC_bdg_amplitude_phase_commensurate_q_audit

- overall status: PARTIAL_PASS_MATERIAL_DWAVE_BLOCKED
- formal Casimir ran: False
- bare_total_ward_max_abs is monitor-only for superconducting gauge-restored response.
- interpretation: onsite_s and spm pass; momentum-dependent dwave collective form-factor closure remains blocked.

## Commensurate-q AP Ward summary

| pairing | N | q | phase vertex | contact closure | bare Ward monitor | AP Ward | status |
| ------- | -: | - | ------------ | --------------: | ----------------: | ------: | ------ |
| onsite_s | 24 | (0.5235988,0) | midpoint | 1.6654438e-16 | 0.018075842 | 3.4483054e-16 | PASSED |
| onsite_s | 24 | (0.5235988,0.5235988) | midpoint | 3.8857911e-16 | 0.014245032 | 1.7347238e-16 | PASSED |
| onsite_s | 36 | (0.3490659,0) | midpoint | 1.1102313e-16 | 0.022027144 | 1.2056624e-16 | PASSED |
| onsite_s | 36 | (0.3490659,0.3490659) | midpoint | 1.6653346e-16 | 0.017997121 | 2.2898357e-16 | PASSED |
| onsite_s | 48 | (0.2617994,0) | midpoint | 2.7756721e-17 | 0.021234817 | 1.3292537e-16 | PASSED |
| onsite_s | 48 | (0.2617994,0.2617994) | midpoint | 8.3266942e-17 | 0.017649753 | 7.3420104e-17 | PASSED |
| spm | 24 | (0.5235988,0) | midpoint | 1.5367793e-18 | 0.012384532 | 4.8019605e-16 | PASSED |
| spm | 24 | (0.5235988,0.5235988) | midpoint | 2.2205108e-16 | 0.010161768 | 1.5352372e-16 | PASSED |
| spm | 36 | (0.3490659,0) | midpoint | 2.2204475e-16 | 0.016432457 | 6.9612637e-17 | PASSED |
| spm | 36 | (0.3490659,0.3490659) | midpoint | 2.4980018e-16 | 0.012305762 | 8.8472066e-17 | PASSED |
| spm | 48 | (0.2617994,0) | midpoint | 2.7756648e-17 | 0.015354519 | 5.5803513e-17 | PASSED |
| spm | 48 | (0.2617994,0.2617994) | midpoint | 2.7756613e-17 | 0.013417687 | 7.7196076e-17 | PASSED |
| dwave | 24 | (0.5235988,0) | symmetric_kpm | 1.1102977e-16 | 0.0063496717 | 0.00087408445 | FAILED |
| dwave | 24 | (0.5235988,0) | midpoint | 1.1102977e-16 | 0.0063496717 | 3.4802022e-05 | MONITOR |
| dwave | 24 | (0.5235988,0.5235988) | symmetric_kpm | 1.6653979e-16 | 0.0045138882 | 0.00083398815 | FAILED |
| dwave | 24 | (0.5235988,0.5235988) | midpoint | 1.6653979e-16 | 0.0045138882 | 3.1924587e-16 | PASSED |
| dwave | 36 | (0.3490659,0) | symmetric_kpm | 1.6653367e-16 | 0.0086264138 | 0.00063619755 | FAILED |
| dwave | 36 | (0.3490659,0) | midpoint | 1.6653367e-16 | 0.0086264138 | 7.6091027e-06 | MONITOR |
| dwave | 36 | (0.3490659,0.3490659) | symmetric_kpm | 2.2204461e-16 | 0.0063681954 | 0.00063095811 | FAILED |
| dwave | 36 | (0.3490659,0.3490659) | midpoint | 2.2204461e-16 | 0.0063681954 | 1.2316671e-16 | PASSED |
| dwave | 48 | (0.2617994,0) | symmetric_kpm | 2.7757233e-17 | 0.0083779385 | 0.0004723741 | FAILED |
| dwave | 48 | (0.2617994,0) | midpoint | 2.7757233e-17 | 0.0083779385 | 1.7987882e-05 | MONITOR |
| dwave | 48 | (0.2617994,0.2617994) | symmetric_kpm | 8.326675e-17 | 0.006588706 | 0.00047359299 | FAILED |
| dwave | 48 | (0.2617994,0.2617994) | midpoint | 8.326675e-17 | 0.006588706 | 7.7196066e-17 | PASSED |

## onsite_s/spm pass summary

| pairing | best N | best q | best AP Ward | contact closure | status |
| ------- | -----: | ------ | -----------: | --------------: | ------ |
| onsite_s | 48 | (0.2617994,0.2617994) | 7.3420104e-17 | 3.8857911e-16 | PASSED |
| spm | 48 | (0.2617994,0) | 5.5803513e-17 | 2.4980018e-16 | PASSED |

## dwave form-factor comparison

| N | q | phase vertex | AP Ward | contact closure | condition number | status |
| -: | - | ------------ | ------: | --------------: | ---------------: | ------ |
| 24 | (0.5235988,0) | symmetric_kpm | 0.00087408445 | 1.1102977e-16 | 1.1436222 | FAILED |
| 24 | (0.5235988,0) | midpoint | 3.4802022e-05 | 1.1102977e-16 | 1.1694077 | MONITOR |
| 24 | (0.5235988,0.5235988) | symmetric_kpm | 0.00083398815 | 1.6653979e-16 | 1.0308108 | FAILED |
| 24 | (0.5235988,0.5235988) | midpoint | 3.1924587e-16 | 1.6653979e-16 | 1.0377939 | PASSED |
| 36 | (0.3490659,0) | symmetric_kpm | 0.00063619755 | 1.6653367e-16 | 1.2863475 | FAILED |
| 36 | (0.3490659,0) | midpoint | 7.6091027e-06 | 1.6653367e-16 | 1.3088269 | MONITOR |
| 36 | (0.3490659,0.3490659) | symmetric_kpm | 0.00063095811 | 2.2204461e-16 | 1.0645481 | FAILED |
| 36 | (0.3490659,0.3490659) | midpoint | 1.2316671e-16 | 2.2204461e-16 | 1.0716474 | PASSED |
| 48 | (0.2617994,0) | symmetric_kpm | 0.0004723741 | 2.7757233e-17 | 1.3670474 | FAILED |
| 48 | (0.2617994,0) | midpoint | 1.7987882e-05 | 2.7757233e-17 | 1.3903675 | MONITOR |
| 48 | (0.2617994,0.2617994) | symmetric_kpm | 0.00047359299 | 8.326675e-17 | 1.1221717 | FAILED |
| 48 | (0.2617994,0.2617994) | midpoint | 7.7196066e-17 | 8.326675e-17 | 1.1316348 | PASSED |

## Overall conclusion

| overall status | interpretation | next step |
| -------------- | -------------- | --------- |
| PARTIAL_PASS_MATERIAL_DWAVE_BLOCKED | onsite_s and spm pass; momentum-dependent dwave collective form-factor closure remains blocked. | Derive/audit the gauge-covariant dwave collective vertex and counterterm. |
