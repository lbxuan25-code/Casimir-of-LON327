# stageSC_4_bdg_q0_limit_audit

- status: PASSED
- quick: True
- cases: 4

## Summary

| key | value |
| --- | --- |
| q_scaling_table | `[{"q_model_magnitude": 0.05, "local_comparison_abs": 0.011197028438760612, "local_comparison_relative": 0.12010455960421046, "bare_relative": 0.017612534087126718, "phase_only_relative": 0.2068906317762049, "amplitude_phase_relative": 0.12010455960421046}, {"q_model_magnitude": 0.02, "local_comparison_abs": 0.0016901726562229037, "local_comparison_relative": 0.01812958175831873, "bare_relative": 0.0025506888208404375, "phase_only_relative": 0.032067908095103945, "amplitude_phase_relative": 0.01812958175831873}, {"q_model_magnitude": 0.01, "local_comparison_abs": 0.00041917540480849037, "local_comparison_relative": 0.004496271280080182, "bare_relative": 0.0006289036800000697, "phase_only_relative": 0.007983775017927937, "amplitude_phase_relative": 0.004496271280080182}, {"q_model_magnitude": 0.005, "local_comparison_abs": 0.00010458571906766717, "local_comparison_relative": 0.0011218352974820373, "bare_relative": 0.0001566853768796811, "phase_only_relative": 0.0019939043562327715, "amplitude_phase_relative": 0.0011218352974820373}]` |
| smallest_q_abs | 0.000104586 |
| largest_q_abs | 0.011197 |
| monotonic_decreasing | True |
| phase_vertex | symmetric_kpm |
| phase_phase_direct_included | True |
| phase_phase_direct_convention | plus |

## Q Scaling

| q_model | abs diff | relative diff | bare relative | phase-only relative | amplitude-phase relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.011197 | 0.120105 | 0.0176125 | 0.206891 | 0.120105 |
| 0.02 | 0.00169017 | 0.0181296 | 0.00255069 | 0.0320679 | 0.0181296 |
| 0.01 | 0.000419175 | 0.00449627 | 0.000628904 | 0.00798378 | 0.00449627 |
| 0.005 | 0.000104586 | 0.00112184 | 0.000156685 | 0.0019939 | 0.00112184 |

## Monitors
- none

## Case Diagnostics

### Case 1
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.011197
- local_comparison_relative: 0.120105
### Case 2
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.00169017
- local_comparison_relative: 0.0181296
### Case 3
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.000419175
- local_comparison_relative: 0.00449627
### Case 4
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- local_comparison_abs: 0.000104586
- local_comparison_relative: 0.00112184
