#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

OUT="validation/outputs/zero_matsubara/static_nk_convergence/raw"
mkdir -p "$OUT"

set -o pipefail
/usr/bin/time -v env \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  python -m validation.run_static_nk_scan \
    --nks 8 12 16 24 \
    --workers 1 \
    --pairing spm \
    --qx 0.03 \
    --qy 0.02 \
    --temperature-K 10 \
    --delta0-eV 0.1 \
    --eta-eV 1e-8 \
    --ward-tolerance 1e-7 \
    --output "$OUT/spm_pilot_serial.csv" \
  2>&1 | tee "$OUT/spm_pilot_serial.log"
