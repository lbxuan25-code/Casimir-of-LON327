# FS-Adaptive Final Narrow-Range Check

Command:

```bash
python3 validation/scripts/numerical_stability/benchmark_normal_fs_adaptive_integration.py --nk-list 80 96 112 --eta-list 1e-4 --matsubara-list 1 2 --temperature 30 --refine-factor-list 8 10 --fs-window-factor 1.0 --sampling uniform multishift_average fs_adaptive --shift-grid 4 --output-prefix validation/outputs/archive/normal_state/fs_adaptive_integration/data/fs_adaptive_final_check
```

No code changes.

| sampling | n | refine_factor | Nk last-two change | r=8->10 change at Nk=112 | symmetry max |
|---|---:|---:|---:|---:|---:|
| uniform | 1 | 1 | 1.392354 | nan | 1.74e-14 |
| uniform | 2 | 1 | 1.375592 | nan | 1.94e-14 |
| multishift_average | 1 | 1 | 0.027593 | nan | 6.38e-15 |
| multishift_average | 2 | 1 | 0.027381 | nan | 4.42e-15 |
| fs_adaptive | 1 | 8 | 0.004545 | 6.64e-06 | 8.84e-14 |
| fs_adaptive | 1 | 10 | 0.000292 | 6.64e-06 | 7.59e-14 |
| fs_adaptive | 2 | 8 | 0.004432 | 3.39e-06 | 7.12e-14 |
| fs_adaptive | 2 | 10 | 0.000378 | 3.39e-06 | 1.93e-14 |

Assessment:

- `fs_adaptive` satisfies the 2% Nk=96->112 last-two threshold for both n=1 and n=2 at r=8 and r=10.
- The r=8->10 change at Nk=112 is far below 2% for both n=1 and n=2.
- C4 diagnostics remain below 1e-8.
- Uniform and multishift controls are useful references but do not pass the 2% threshold here.
- Classification: `fs_adaptive_confirmed_candidate` for normal low-Matsubara response sampling.
- This remains a response-layer numerical benchmark, not a Casimir result.
