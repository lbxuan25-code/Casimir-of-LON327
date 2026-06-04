#!/usr/bin/env bash
set -euo pipefail

python validation/scripts/casimir/benchmark_casimir_local_response_distance_scan.py --kinds normal spm dwave --distance-list 3e-08 5e-08 7.5e-08 1e-07 1.5e-07 2e-07 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --matsubara-max 64 --u-max 80 --du 0.5 --phi-num 32 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --cache-dir validation/cache/casimir_local_response/response_tensors --output-prefix validation/outputs/casimir/local_response_integral/distance_scan/data/distance_scan --use-response-cache --include-toy-anisotropic-control --rebuild-response-cache --progress
