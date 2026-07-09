# Normal contact Ward control note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_normal_contact_ward_control.py
```

It is not a production convention proposal.

## Purpose

The pairing-contact missing audit showed that the contact mismatch is strong for dwave but much weaker for spm. The delta0=0 control could not be run through the BdG pairing path because the pairing form factor is undefined at delta0=0.

This audit provides a true normal-state control. It bypasses:

```text
BdG pairing ansatz
phase vertices
collective channels
Schur completion
BdG Nambu 1/2 prefactor
```

and tests only:

```text
normal Hamiltonian H(k)
normal Peierls vector vertices V_i(k,q)
normal Peierls contact vertices M_ij(k,q)
normal density/current Kubo bubble
normal diamagnetic contact
```

## Ward checks

At the vertex level it checks

```text
qx Vx + qy Vy = H(k+q/2) - H(k-q/2)
```

At the response level it forms

```text
K_total = K_bubble + K_contact
```

with source/observable convention

```text
observable = (rho, -Vx, -Vy)
source     = (rho, +Vx, +Vy)
```

and reports the normal Ward contractions

```text
left  = (i xi, +qx, +qy) K
right = K (i xi, -qx, -qy)^T
```

for `bubble`, `contact`, and `total`.

It also reports the diagnostic scalar that compares the Ward-required normal contact contraction to the implemented normal contact contraction:

```text
contact_required = - bubble_contraction
alpha = argmin || contact_required - alpha contact_current ||
```

This scalar is diagnostic only and must not be used as a production coefficient.

## Interpretation

If the normal total Ward residual is small and `alpha ~= 1`, the normal-state Peierls bubble/contact sector is likely consistent. Then the dwave BdG mismatch should be pursued as a superconducting pairing/gauge-completion issue.

If the normal total Ward residual is large or `alpha` differs strongly from 1, the normal Peierls contact or normal response convention is also suspect.

If the vertex identity is poor, the problem lies before response integration, in the normal Peierls vertex/contact construction or model hopping representation.

## Non-goal

This audit does not test d-wave pairing contact. It only establishes whether the normal-state Peierls sector is a valid control baseline.
