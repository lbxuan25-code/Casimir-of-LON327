# FS-Adaptive High-Precision Summary

Command:

```bash
python3 scripts/benchmark_normal_fs_adaptive_integration.py --nk-list 48 64 80 96 --eta-list 2e-4 1e-4 --matsubara-list 1 2 --temperature 30 --refine-factor-list 4 6 8 --fs-window-factor 1.0 --sampling uniform multishift_average fs_adaptive --shift-grid 4 --output-prefix outputs/archive/normal_state/fs_adaptive_integration/data/fs_adaptive_high_precision
```

Result: no code changes. The full high-precision run completed with 80 rows.

Key diagnostics:

| sampling | n | eta | refine | last-two Nk change | symmetry max |
|---|---:|---:|---:|---:|---:|
| uniform | 1 | 1e-4 | 1 | 0.333967 | 1.74e-14 |
| multishift_average | 1 | 1e-4 | 1 | 0.052172 | 6.38e-15 |
| fs_adaptive | 1 | 1e-4 | 4 | 0.012999 | 1.91e-14 |
| fs_adaptive | 1 | 1e-4 | 6 | 0.014234 | 2.18e-14 |
| fs_adaptive | 1 | 1e-4 | 8 | 0.007684 | 5.12e-14 |
| fs_adaptive | 2 | 1e-4 | 8 | 0.007554 | 3.96e-14 |
| fs_adaptive | 1 | 2e-4 | 8 | 0.007683 | 5.63e-14 |
| fs_adaptive | 2 | 2e-4 | 8 | 0.007554 | 3.97e-14 |

Assessment:

- `fs_adaptive` reaches the 2% last-two-Nk threshold for `refine_factor = 4, 6, 8`.
- `refine_factor = 8` is the most stable in this run.
- At `Nk=96`, r=6 -> r=8 changes are about 1.17-1.18%, but r=4 -> r=6 changes are about 7.19-7.25%, so r=4 is not enough.
- Eta sensitivity is below 5%: max eta relative change is 0.00605.
- C4 diagnostics remain below 1e-8.
- Uniform and multishift controls do not reach the 2% last-two-Nk threshold.
- This is still a normal-response numerical benchmark, not a Casimir result.
