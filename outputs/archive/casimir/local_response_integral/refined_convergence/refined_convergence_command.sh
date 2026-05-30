#!/usr/bin/env bash
set -euo pipefail

python scripts/refine_casimir_local_convergence_blockers.py --kinds normal spm dwave --distance 5e-08 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --energy-theta-list 0 --torque-check-theta-list 0 0.7853981634 1.5707963268 --u-max-list 20 40 60 80 --du 0.5 --matsubara-max-list 24 32 48 64 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --phi-num 32 --output-prefix outputs/archive/casimir/local_response_integral/refined_convergence/data/refined_local_convergence --cache-dir outputs/cache/casimir_local_response/response_tensors --use-response-cache
