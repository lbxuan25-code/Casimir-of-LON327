# stageSC_2d_pairing_bond_collective_vertex_audit

## Table 1: pairing reconstruction

| pairing | reconstruction max | status |
| ------- | -----------------: | ------ |
| onsite_s | 0 | PASSED |
| spm | 0 | PASSED |
| dwave | 6.9388939e-18 | PASSED |

## Table 2: operator Ward by phase vertex

| pairing | phase vertex | operator Ward max | status |
| ------- | ------------ | ----------------: | ------ |
| onsite_s | midpoint | 6.0194905e-16 | PASSED |
| onsite_s | symmetric_kpm | 6.0194905e-16 | PASSED |
| onsite_s | bond_endpoint_gauge | 6.0194905e-16 | PASSED |
| spm | midpoint | 6.0194905e-16 | PASSED |
| spm | symmetric_kpm | 6.0194905e-16 | PASSED |
| spm | bond_endpoint_gauge | 6.0194905e-16 | PASSED |
| dwave | midpoint | 1.9553287e-06 | FAILED |
| dwave | symmetric_kpm | 6.0194905e-16 | PASSED |
| dwave | bond_endpoint_gauge | 6.0194905e-16 | PASSED |

## Table 3: collective basis projection

| pairing | num channels | projection rel residual | status |
| ------- | -----------: | ----------------------: | ------ |
| onsite_s | 1 | 6.6613381e-16 | PASSED |
| spm | 1 | 0 | PASSED |
| dwave | 1 | 2.183361e-16 | PASSED |

## Table 4: commensurate-q Schur Ward

| pairing | phase vertex |  N | q | contact closure | AP Ward | status |
| ------- | ------------ | -: | - | --------------: | ------: | ------ |
| onsite_s | bond_endpoint_gauge | 24 | (0.5235988,0) | 1.6654438e-16 | 3.4483054e-16 | PASSED |
| onsite_s | bond_endpoint_gauge | 24 | (0.5235988,0.5235988) | 3.8857911e-16 | 1.7347238e-16 | PASSED |
| onsite_s | bond_endpoint_gauge | 36 | (0.3490659,0) | 1.1102313e-16 | 1.2056624e-16 | PASSED |
| onsite_s | bond_endpoint_gauge | 36 | (0.3490659,0.3490659) | 1.6653346e-16 | 2.2898357e-16 | PASSED |
| onsite_s | bond_endpoint_gauge | 48 | (0.2617994,0) | 2.7756721e-17 | 1.3292537e-16 | PASSED |
| onsite_s | bond_endpoint_gauge | 48 | (0.2617994,0.2617994) | 8.3266942e-17 | 7.3420104e-17 | PASSED |
| spm | bond_endpoint_gauge | 24 | (0.5235988,0) | 1.5367793e-18 | 4.8019605e-16 | PASSED |
| spm | bond_endpoint_gauge | 24 | (0.5235988,0.5235988) | 2.2205108e-16 | 1.5352372e-16 | PASSED |
| spm | bond_endpoint_gauge | 36 | (0.3490659,0) | 2.2204475e-16 | 6.9612637e-17 | PASSED |
| spm | bond_endpoint_gauge | 36 | (0.3490659,0.3490659) | 2.4980018e-16 | 8.8472066e-17 | PASSED |
| spm | bond_endpoint_gauge | 48 | (0.2617994,0) | 2.7756648e-17 | 5.5803513e-17 | PASSED |
| spm | bond_endpoint_gauge | 48 | (0.2617994,0.2617994) | 2.7756613e-17 | 7.7196076e-17 | PASSED |
| dwave | bond_endpoint_gauge | 24 | (0.5235988,0) | 1.1102977e-16 | 0.00087408578 | FAILED |
| dwave | bond_endpoint_gauge | 24 | (0.5235988,0.5235988) | 1.6653979e-16 | 0.00083397642 | FAILED |
| dwave | bond_endpoint_gauge | 36 | (0.3490659,0) | 1.6653367e-16 | 0.00063617515 | FAILED |
| dwave | bond_endpoint_gauge | 36 | (0.3490659,0.3490659) | 2.2204461e-16 | 0.00063094848 | FAILED |
| dwave | bond_endpoint_gauge | 48 | (0.2617994,0) | 2.7757233e-17 | 0.00047235104 | FAILED |
| dwave | bond_endpoint_gauge | 48 | (0.2617994,0.2617994) | 8.326675e-17 | 0.00047358344 | FAILED |

## Table 5: overall conclusion

| status | interpretation | next step |
| ------ | -------------- | --------- |
| PARTIAL_PASS_DWAVE_COUNTERTERM_BLOCKED | dwave reconstruction/operator/projection pass, but the amplitude-phase Ward Schur residual remains above threshold. | Audit the nonlocal pairing counterterm block before changing production defaults. |
