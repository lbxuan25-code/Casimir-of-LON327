#!/usr/bin/env bash
set -euo pipefail

# Ward / response convention diagnostic chain.
# Outputs under validation/outputs/response/ward_identity and related legacy
# directories are regenerated artifacts and are ignored by Git policy.

python validation/scripts/response/stage4_13_bubble_sign_fix_regression.py
python validation/scripts/response/stage4_17_right_ward_source_convention_audit.py
python validation/scripts/response/stage4_18_corrected_full_response_ward_validation.py
python validation/scripts/response/stage4_19_multi_parameter_ward_robustness_scan.py
python validation/scripts/response/stage4_20_targeted_refinement_scan.py

# Optional convention scans:
# python validation/scripts/response/verify_response_level_ward_conventions.py
# python validation/scripts/response/diagnose_normal_ward_identity.py
