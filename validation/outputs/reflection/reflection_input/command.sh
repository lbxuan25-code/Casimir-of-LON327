#!/usr/bin/env bash
set -euo pipefail

# Reflection input and adapter diagnostic chain.
# Generated stage outputs remain ignored artifacts.

python validation/scripts/response/stage5_5b_reflection_input_tensor_formatter.py
python validation/scripts/response/stage5_6_te_tm_reflection_adapter.py

# Optional scaffold/prototype checks:
# python validation/scripts/response/stage5_8_casimir_integrand_prototype.py
# python validation/scripts/response/stage5_10_toy_casimir_integration_convergence_audit.py
