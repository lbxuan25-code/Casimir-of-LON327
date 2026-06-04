# Normal finite-q current-current response diagnostic

这是 normal-state finite-q current-current diagnostic，不是完整 finite-q conductivity，
也不是 Casimir input。q=0 行使用现有 local normal Kubo reference fallback；
q!=0 行使用 shifted-state midpoint velocity approximation。

run_command = `python validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py --omega-list 1e-4 1e-3 1e-2 --q-list 0 1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 --q-angle-list 0 0.7853981633974483 --nk-list 16 24 32 48 --temperature 30 --eta 1e-4 --output-prefix validation/outputs/response/normal_finite_q_response_refinement/data/normal_finite_q_response_refinement`
quick_mode=False
finite_momentum_resolved=True
normal_state=True
current_current_only=True
midpoint_vertex_approximation=True
not_peierls_exact_vertex=True
ward_identity_not_yet_checked=True
not_final_casimir_input=True

## Quick diagnostic status
- q=0 maximum relative mismatch to local fallback: 0
- smallest sampled nonzero q: 1e-06
- maximum relative mismatch at smallest nonzero q: 0.999971
- maximum C4 covariance error: 1.94096e-13
- all q!=0 response components finite: True
- angular samples per q: 2
- q-to-zero continuity established by this run: False

## Boundary
- current-current-only response is not gauge closed;
- Ward identity has not been checked;
- midpoint vertex is not a Peierls-exact finite-q vertex;
- q=0 local fallback and q!=0 bare bubble are reported without empirical rescaling;
- quick-mode harmonic coefficients use only two angles and are smoke-level diagnostics;
- this script does not modify BdG, Casimir, or reflection-matrix logic;
- no final finite-q conductivity or Casimir conclusion is claimed.
