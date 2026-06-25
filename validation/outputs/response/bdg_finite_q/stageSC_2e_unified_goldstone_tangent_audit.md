# stageSC_2e_unified_goldstone_tangent_audit

## Section 1: Goldstone dimension statement

For all tested spm/dwave/onsite_s pairings, if only total charge U(1) is spontaneously broken, the Goldstone manifold dimension is one. Bond-resolved internal modes are not additional Goldstone modes.

## Section 2: pairing reconstruction

| pairing | reconstruction max | status |
| ------- | -----------------: | ------ |
| onsite_s | 0 | PASSED |
| spm | 0 | PASSED |
| dwave | 6.9388939e-18 | PASSED |

## Section 3: exact Goldstone tangent operator Ward

| pairing | q | normalization | operator Ward | status |
| ------- | - | ------------: | ------------: | ------ |
| onsite_s | [[0.01, 0.0], [0.01, 0.01]] | 1 | 6.0194905e-16 | PASSED |
| spm | [[0.01, 0.0], [0.01, 0.01]] | 1 | 6.0194905e-16 | PASSED |
| dwave | [[0.01, 0.0], [0.01, 0.01]] | 1 | 6.0194905e-16 | PASSED |

## Section 4: old prescriptions comparison

| pairing | q | midpoint Ward | symmetric_kpm Ward | exact tangent Ward |
| ------- | - | ------------: | -----------------: | -----------------: |
| onsite_s | (0.01,0) | 6.0194905e-16 | 6.0194905e-16 | 6.0194905e-16 |
| onsite_s | (0.01,0.01) | 6.0194905e-16 | 6.0194905e-16 | 6.0194905e-16 |
| spm | (0.01,0) | 6.0194905e-16 | 6.0194905e-16 | 6.0194905e-16 |
| spm | (0.01,0.01) | 6.0194905e-16 | 6.0194905e-16 | 6.0194905e-16 |
| dwave | (0.01,0) | 1.9553287e-06 | 6.0194905e-16 | 6.0194905e-16 |
| dwave | (0.01,0.01) | 1.9553287e-06 | 6.0194905e-16 | 6.0194905e-16 |

## Section 5: commensurate-q restored Ward

| pairing |  N | q | contact closure | bare Ward monitor | restored Ward | status |
| ------- | -: | - | --------------: | ----------------: | ------------: | ------ |
| onsite_s | 24 | (0.5235988,0) | 1.6654438e-16 | 0.018075842 | 3.4483054e-16 | PASSED |
| onsite_s | 24 | (0.5235988,0.5235988) | 3.8857911e-16 | 0.014245032 | 1.7347238e-16 | PASSED |
| onsite_s | 36 | (0.3490659,0) | 1.1102313e-16 | 0.022027144 | 1.2056624e-16 | PASSED |
| onsite_s | 36 | (0.3490659,0.3490659) | 1.6653346e-16 | 0.017997121 | 2.2898357e-16 | PASSED |
| onsite_s | 48 | (0.2617994,0) | 2.7756721e-17 | 0.021234817 | 1.3292537e-16 | PASSED |
| onsite_s | 48 | (0.2617994,0.2617994) | 8.3266942e-17 | 0.017649753 | 7.3420104e-17 | PASSED |
| spm | 24 | (0.5235988,0) | 1.5367793e-18 | 0.012384532 | 4.8019605e-16 | PASSED |
| spm | 24 | (0.5235988,0.5235988) | 2.2205108e-16 | 0.010161768 | 1.5352372e-16 | PASSED |
| spm | 36 | (0.3490659,0) | 2.2204475e-16 | 0.016432457 | 6.9612637e-17 | PASSED |
| spm | 36 | (0.3490659,0.3490659) | 2.4980018e-16 | 0.012305762 | 8.8472066e-17 | PASSED |
| spm | 48 | (0.2617994,0) | 2.7756648e-17 | 0.015354519 | 5.5803513e-17 | PASSED |
| spm | 48 | (0.2617994,0.2617994) | 2.7756613e-17 | 0.013417687 | 7.7196076e-17 | PASSED |
| dwave | 24 | (0.5235988,0) | 1.1102977e-16 | 0.0063496717 | 0.00087408578 | FAILED |
| dwave | 24 | (0.5235988,0.5235988) | 1.6653979e-16 | 0.0045138882 | 0.00083397642 | FAILED |
| dwave | 36 | (0.3490659,0) | 1.6653367e-16 | 0.0086264138 | 0.00063617515 | FAILED |
| dwave | 36 | (0.3490659,0.3490659) | 2.2204461e-16 | 0.0063681954 | 0.00063094848 | FAILED |
| dwave | 48 | (0.2617994,0) | 2.7757233e-17 | 0.0083779385 | 0.00047235104 | FAILED |
| dwave | 48 | (0.2617994,0.2617994) | 8.326675e-17 | 0.006588706 | 0.00047358344 | FAILED |

## Section 6: dwave failure diagnosis if any

| diagnosis | evidence | next step |
| --------- | -------- | --------- |
| COUNTERTERM_BLOCKED | Exact Goldstone operator Ward and old-basis projection pass; the remaining Schur failure is not fixed by adding bond-resolved Goldstones. This is not an additional Goldstone mode. | Audit the nonlocal pairing counterterm block before adding internal modes. |

## Overall

- status: PARTIAL_PASS_DWAVE_SCHUR_BLOCKED
- formal Casimir ran: False
- production default modified: False
- interpretation: dwave exact tangent passes operator Ward, but restored Schur Ward fails.
