#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python scripts/casimir/finite_q_bdg_casimir_pipeline.py \
  --pairings normal spm dwave \
  --distances-nm 20 30 50 80 100 150 200 \
  --angles-deg 0 15 30 45 60 75 90 \
  --n-max 16 \
  --q-num 24 \
  --phi-num 12 \
  --temperature-K 30 \
  --delta0-eV 0.04 \
  --n0-policy extrapolate \
  --num-workers 1 \
  --resume \
  --output-dir outputs/casimir/finite_q_bdg_pipeline

# Server example:
# nohup bash outputs/casimir/finite_q_bdg_pipeline/command.sh \
#   > outputs/casimir/finite_q_bdg_pipeline/logs/nohup.log 2>&1 &
#
# tail -f outputs/casimir/finite_q_bdg_pipeline/logs/nohup.log
# tail -f outputs/casimir/finite_q_bdg_pipeline/logs/run.log
