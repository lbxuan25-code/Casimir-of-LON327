#!/usr/bin/env bash
set -euo pipefail

# Conductivity conversion diagnostic chain.
# Generated stage outputs remain ignored artifacts.

python validation/scripts/response/stage5_1b_bilayer_sheet_conductivity_convention.py
python validation/scripts/response/stage5_2_bilayer_sheet_conductivity_sanity_scan.py
python validation/scripts/response/stage5_3b_bilayer_sheet_conductivity_offdiag_convergence_audit.py
python validation/scripts/response/stage5_4a_conductivity_unit_conversion.py
python validation/scripts/response/stage5_4b_convert_model_conductivity_to_si_sheet.py
