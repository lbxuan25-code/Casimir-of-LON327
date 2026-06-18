# stageSC_4_bdg_q0_limit_audit

- status: PASSED
- quick: True
- cases: 4

## Summary

| key | value |
| --- | --- |
| q_scaling_table | `[{"q_model_magnitude": 0.05, "local_comparison_abs": 0.01928786296994261, "local_comparison_relative": 0.2068906317762049}, {"q_model_magnitude": 0.02, "local_comparison_abs": 0.0029896057243429825, "local_comparison_relative": 0.032067908095103945}, {"q_model_magnitude": 0.01, "local_comparison_abs": 0.0007443060964462481, "local_comparison_relative": 0.007983775017927937}, {"q_model_magnitude": 0.005, "local_comparison_abs": 0.000185886396440559, "local_comparison_relative": 0.0019939043562327715}]` |
| smallest_q_abs | 0.000185886 |
| largest_q_abs | 0.0192879 |
| monotonic_decreasing | True |
| phase_vertex | symmetric_kpm |
| phase_phase_direct_included | True |
| phase_phase_direct_convention | plus |

## Q Scaling

| q_model | abs diff | relative diff |
| --- | ---: | ---: |
| 0.05 | 0.0192879 | 0.206891 |
| 0.02 | 0.00298961 | 0.0320679 |
| 0.01 | 0.000744306 | 0.00798378 |
| 0.005 | 0.000185886 | 0.0019939 |

## Monitors
- none

## Case Diagnostics

### Case 1
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.0192879
- local_comparison_relative: 0.206891
### Case 2
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.00298961
- local_comparison_relative: 0.0320679
### Case 3
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.000744306
- local_comparison_relative: 0.00798378
### Case 4
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.000185886
- local_comparison_relative: 0.0019939
