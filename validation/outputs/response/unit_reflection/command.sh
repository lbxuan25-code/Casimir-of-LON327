#!/usr/bin/env bash
set -euo pipefail

# Unit conversion and reflection-input diagnostic chain.
# These scripts regenerate ignored stage outputs; run selectively for audits.

python validation/scripts/response/stage5_1b_bilayer_sheet_conductivity_convention.py
python validation/scripts/response/stage5_2_bilayer_sheet_conductivity_sanity_scan.py
python validation/scripts/response/stage5_3b_bilayer_sheet_conductivity_offdiag_convergence_audit.py
python validation/scripts/response/stage5_4a_conductivity_unit_conversion.py
python validation/scripts/response/stage5_4b_convert_model_conductivity_to_si_sheet.py
python validation/scripts/response/stage5_5b_reflection_input_tensor_formatter.py
python validation/scripts/response/stage5_6_te_tm_reflection_adapter.py

# Optional scaffold/prototype checks:
# python validation/scripts/response/stage5_8_casimir_integrand_prototype.py
# python validation/scripts/response/stage5_9_casimir_grid_planning_scaffold.py
# python validation/scripts/response/stage5_10_toy_casimir_integration_convergence_audit.py
