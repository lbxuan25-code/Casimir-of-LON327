# Normal finite-q current-current response diagnostic

这是 normal-state finite-q current-current diagnostic，不是完整 finite-q conductivity，
也不是 Casimir input。q=0 行仍使用现有 public local normal sigma fallback；
q!=0 行使用 shifted-state midpoint velocity approximation。

public local sigma 是 conductivity-level reference，包含 intraband/omega；
finite-q shifted-state result 是 current-current kernel-level quantity。因此
relative_error_to_public_local_sigma 只作辅助对照，不作为 q-to-zero 闭合判据。
新的主判据是 q_to_zero_kernel_relative_error：omega>0 使用 local interband
kernel，omega=0 使用 local static kernel。

run_command = `python validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py --omega-list 0 1e-6 1e-5 1e-4 --q-list 0 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 --q-angle-list 0 pi/8 pi/4 3pi/8 pi/2 --nk-list 16 24 32 --temperature 30 --eta 1e-4`
quick_mode=False
finite_momentum_resolved=True
normal_state=True
current_current_only=True
midpoint_vertex_approximation=True
not_peierls_exact_vertex=True
ward_identity_not_yet_checked=True
not_final_casimir_input=True

## Quick diagnostic status
- q=0 maximum relative mismatch to public local sigma fallback: 0
- smallest sampled nonzero q: 0.0001
- maximum kernel-relative mismatch at smallest nonzero q: 2.53769
- maximum public-sigma mismatch at smallest nonzero q: 0.999959
- maximum C4 covariance error: 4.71068e-13
- all q!=0 response components finite: True
- angular samples per q: 5
- q-to-zero kernel continuity established by this run: False

## Boundary
- current-current-only response is not gauge closed;
- Ward identity has not been checked;
- midpoint vertex is not a Peierls-exact finite-q vertex;
- public sigma and kernel-level references are both reported without empirical rescaling;
- quick-mode harmonic coefficients use only two angles and are smoke-level diagnostics;
- this script does not modify BdG, Casimir, or reflection-matrix logic;
- no final finite-q conductivity or Casimir conclusion is claimed.
