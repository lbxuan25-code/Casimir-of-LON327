#!/usr/bin/env bash
set -euo pipefail

# Normal finite-q current-current kernel convergence.
# Outputs under validation/outputs/response/normal_finite_q_kernel_convergence/data
# are regenerated artifacts and are ignored by Git.
python validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py \
  --matsubara-n-list 1 2 4 8 \
  --temperature 30 \
  --q-list 0 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 \
  --q-angle-list 0 pi/8 pi/4 '3*pi/8' pi/2 \
  --nk-list 16 24 32 \
  --output-prefix validation/outputs/response/normal_finite_q_kernel_convergence/data/normal_finite_q_kernel_convergence_full
