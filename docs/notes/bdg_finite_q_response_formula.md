# BdG finite-q response formula notes

This note records the response-layer convention used before any Casimir
postprocessing. The downstream Casimir module consumes response tensors; it
must not repair, rescale, or reinterpret the BdG response.

## Fixed chain

For positive bosonic Matsubara index `n >= 1`,

```text
H_BdG(k)
-> G_BdG(k, i omega)
-> current vertices J_i and contact vertices gamma_ij
-> K_para(i xi, q) + K_dia(q) + optional K_collective/Ward
-> K_total(i xi, q)
-> Sigma_SC(i xi, q) = K_total(i xi, q) / xi
-> reflection matrix R(i xi, q)
-> Casimir free energy
-> torque tau = -dF/dtheta
```

`K_para` is the current-current bubble. It is not a conductivity and is not
`Sigma_SC`.

`K_dia` is the diamagnetic/contact term. The current finite-q implementation
does not have a genuine finite-q contact kernel; diagnostics therefore mark
`finite_q_dia_status="q0_fallback_only"` when the local q=0 BdG contact term is
used as a temporary fallback.

`K_total = K_para + K_dia (+ K_collective)` is the only kernel that may be
divided by `xi` to form `Sigma_SC` for `n >= 1`.

`K_collective/Ward` is not implemented yet. Diagnostics therefore mark
`ward_status="not_closed"`. Any response with `ward_status!="closed"` must also
set `valid_for_casimir_input=False`.

## Current q=0 checks

The q=0 diagnostic compares layers in this order:

1. `K_para_q0` against local BdG `K_para`.
2. `K_total_q0` against local BdG `K_total`.
3. `Sigma_SC_q0 = K_total_q0 / xi` against local `K_total / xi`.

No empirical scaling is allowed. If a layer does not align, the diagnostic keeps
the failure status and reports the relative error.

## n=0 boundary

`Sigma_SC = K_total / xi` is only used for positive Matsubara terms. The `n=0`
sector remains unresolved unless an explicit static response model is added.

## Casimir boundary

The current Casimir code is a downstream consumer. It should accept only
response objects that are already physically closed and tagged
`valid_for_casimir_input=True`. Present finite-q BdG benchmark outputs are
diagnostic only and are not final Casimir torque conclusions.
