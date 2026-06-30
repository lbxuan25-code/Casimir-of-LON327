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
  --eta-eV 1e-10 \
  --integration-strategy best_available_adaptive \
  --coarse-grid 32 \
  --adaptive-level 5 \
  --gauss-order 5 \
  --fermi-window-eV 0.12 \
  --q-specific-adaptive-grid \
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
#
# Shard example:
# python scripts/casimir/finite_q_bdg_casimir_pipeline.py \
#   --task-shard-index 0 --task-shard-count 4 --resume \
#   --output-dir outputs/casimir/finite_q_bdg_pipeline
#
# Plot/finalize after shards:
# python scripts/casimir/finite_q_bdg_casimir_pipeline.py \
#   --plot-only --output-dir outputs/casimir/finite_q_bdg_pipeline
