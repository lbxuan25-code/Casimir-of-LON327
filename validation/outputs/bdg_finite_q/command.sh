#!/usr/bin/env bash
set -euo pipefail

# unified q=0 response definition alignment
python validation/scripts/bdg_finite_q/q0_bdg_response_alignment.py --pairings normal onsite_s spm dwave --explain --json-output validation/outputs/bdg_finite_q/q0_status.json --nk 3

# small-q Ward residual scan
python validation/scripts/bdg_finite_q/finite_q_ward_scan.py --pairings onsite_s spm dwave --nk 3 --q-values 0.005 0.01 0.02 --q0-status-json validation/outputs/bdg_finite_q/q0_status.json --json-output validation/outputs/bdg_finite_q/ward_scan_status.json

# normal-state finite-q response/operator Ward residual audit
python validation/scripts/bdg_finite_q/normal_finite_q_ward_audit.py --nk 3 --q-values 0.005 0.01 0.02 --json-output validation/outputs/bdg_finite_q/normal_finite_q_ward_audit.json

# dwave pairing reconstruction and endpoint-gauge tangent diagnostic
python validation/scripts/bdg_finite_q/dwave_pairing_tangent_diagnostics.py

# Goldstone counterterm and eta2-normalization diagnostic
python validation/scripts/bdg_finite_q/goldstone_counterterm_diagnostics.py --pairings onsite_s spm dwave --nk 3

python validation/scripts/bdg_finite_q/summarize_validation.py
