#!/usr/bin/env bash
set -euo pipefail

# q=0 response definition alignment
python validation/scripts/bdg_finite_q/q0_bdg_response_alignment.py normal --nk 3
python validation/scripts/bdg_finite_q/q0_bdg_response_alignment.py spm --nk 3
python validation/scripts/bdg_finite_q/q0_bdg_response_alignment.py dwave --nk 3

# small-q Ward residual scan
python validation/scripts/bdg_finite_q/finite_q_ward_scan.py --pairings onsite_s spm dwave --nk 3 --q-values 0.005 0.01 0.02

# dwave pairing reconstruction and endpoint-gauge tangent diagnostic
python validation/scripts/bdg_finite_q/dwave_pairing_tangent_diagnostics.py

# Goldstone counterterm and eta2-normalization diagnostic
python validation/scripts/bdg_finite_q/goldstone_counterterm_diagnostics.py --pairings onsite_s spm dwave --nk 3
