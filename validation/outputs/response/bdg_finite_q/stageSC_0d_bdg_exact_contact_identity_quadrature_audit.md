# stageSC_0d_bdg_exact_contact_identity_quadrature_audit

- total status: FAILED
- formal Casimir ran: False
- diagnostic only: True
- dominant failure: shift_invariant_quadrature
- best interpretation: contact_formula_passed_but_shift_invariant_quadrature_closure_failed

## Part A: pointwise exact contact identity

| pairing | j | normal identity max | BdG identity max | status |
| ------- | - | ------------------: | ----------------: | ------ |
| onsite_s | x | 3.3400563e-16 | 3.3400563e-16 | PASSED |
| onsite_s | y | 1.1885983e-16 | 1.1885983e-16 | PASSED |
| spm | x | 3.3400563e-16 | 3.3400563e-16 | PASSED |
| spm | y | 1.1885983e-16 | 1.1885983e-16 | PASSED |
| dwave | x | 3.3400563e-16 | 3.3400563e-16 | PASSED |
| dwave | y | 1.1885983e-16 | 1.1885983e-16 | PASSED |

## Part B: shift-invariant contact closure

| pairing | q | Vx closure | Vy closure | max | status |
| ------- | - | ---------: | ---------: | --: | ------ |
| onsite_s | (0.26179939, 0) | 0.0026007727 | 3.7137222e-18 | 0.0026007727 | FAILED |
| onsite_s | (0.26179939, 0.26179939) | 0.0029555849 | 0.0029555849 | 0.0029555849 | FAILED |
| spm | (0.26179939, 0) | 0.0041686796 | -3.2191398e-18 | 0.0041686796 | FAILED |
| spm | (0.26179939, 0.26179939) | 0.0048681076 | 0.0048681076 | 0.0048681076 | FAILED |
| dwave | (0.26179939, 0) | 0.0046518094 | 2.9443517e-18 | 0.0046518094 | FAILED |
| dwave | (0.26179939, 0.26179939) | 0.0049248637 | 0.0049248637 | 0.0049248637 | FAILED |

## Part C: joint small-q/finer-grid diagnostic

q_model = 2*pi/N, so this is a joint small-q/finer-grid diagnostic, not fixed-q convergence.

| N | q=2pi/N | onsite_s max spatial closure |
| -: | ------: | ---------------------------: |
| 12 | 0.52359878 | 0.021031851 |
| 18 | 0.34906585 | 0.0032364745 |
| 24 | 0.26179939 | 0.0026007727 |
| 36 | 0.17453293 | 0.00081460813 |

## Part D: fixed-q quadrature convergence

| N | fixed q=0.01 | onsite_s max spatial closure |
| -: | -----------: | ---------------------------: |
| 12 | 0.01 | 0.0033892047 |
| 18 | 0.01 | 0.0035823167 |
| 24 | 0.01 | 0.001288786 |
| 36 | 0.01 | 0.0016731326 |

### Fixed-q results for all pairings

| pairing | N | max spatial closure | status |
| ------- | -: | ------------------: | ------ |
| onsite_s | 12 | 0.0033892047 | FAILED |
| onsite_s | 18 | 0.0035823167 | FAILED |
| onsite_s | 24 | 0.001288786 | FAILED |
| onsite_s | 36 | 0.0016731326 | FAILED |
| spm | 12 | 0.0037219777 | FAILED |
| spm | 18 | 0.0062304843 | FAILED |
| spm | 24 | 0.0028826581 | FAILED |
| spm | 36 | 0.0017357667 | FAILED |
| dwave | 12 | 0.0032485896 | FAILED |
| dwave | 18 | 0.0029546594 | FAILED |
| dwave | 24 | 0.0017993343 | FAILED |
| dwave | 36 | 0.0016664479 | FAILED |

## Interpretation

The pointwise contact formula passes, while one or more integrated closures fail. This does not establish a missing contact term; quadrature/contact closure implementation and shifted-grid/response-assembly consistency remain unresolved.
