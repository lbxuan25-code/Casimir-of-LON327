#!/usr/bin/env bash
set -euo pipefail

# BdG finite-q diagnostic stages. These commands regenerate ignored stage outputs.
# Run selectively; the full sequence can be expensive.

python validation/scripts/response/stageSC_1_bdg_finite_q_bare_kernel_audit.py
python validation/scripts/response/stageSC_2b_bdg_amplitude_phase_gauge_restoration_audit.py
python validation/scripts/response/stageSC_3_bdg_normal_limit_audit.py
python validation/scripts/response/stageSC_4_bdg_q0_limit_audit.py
python validation/scripts/response/stageSC_5_bdg_reflection_input_audit.py

# Optional deeper diagnostics:
# python validation/scripts/response/stageSC_2k_gauge_covariant_collective_package_audit.py
