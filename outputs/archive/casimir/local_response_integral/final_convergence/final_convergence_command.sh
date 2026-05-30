#!/usr/bin/env bash
set -euo pipefail

python scripts/run_casimir_local_convergence_final.py --kinds normal spm dwave --distance 5e-08 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --matsubara-max-list 4 8 16 24 --kparallel-num-list 32 64 96 --kparallel-max-factor-list 20 40 60 --phi-num-list 32 64 96 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --output-prefix outputs/archive/casimir/local_response_integral/final_convergence/data/final_local_convergence
