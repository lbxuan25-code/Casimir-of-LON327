# Finite-q static gauge contract: analytic audit

Status: **analytic audit draft**. This note does not change any production gate and does not establish Casimir readiness.

## 1. Scope

The current finite-q response stack contains two distinct checks:

1. an RHS-aware primitive/effective Ward closure check on the microscopic quadrature;
2. an exact-static longitudinal gate that assumes the Schur-effective electromagnetic kernel has a pure-gauge longitudinal zero mode.

The purpose of this audit is to determine whether these two contracts are analytically compatible in the continuum Brillouin-zone limit.

The central question is not whether the existing Ward residual is small. It is whether the exact continuum target is

\[
 u_L K_{\mathrm{eff}}=0,
 \qquad
 K_{\mathrm{eff}}u_R=0,
\]

or instead a nonzero effective RHS.

## 2. Current field and routing conventions

Primitive electromagnetic order:

\[
S=(A_0,A_x,A_y).
\]

Collective order:

\[
\eta=(\eta_1,\eta_2)
\]

with amplitude `eta1` and phase `eta2`. The finite-q routing is

\[
k_-=k-q/2,
\qquad
k_+=k+q/2.
\]

The current Ward vectors are

\[
u_L=(+i\xi,q_x,q_y),
\qquad
u_R=(-i\xi,q_x,q_y),
\]

and

\[
w_L=w_R=(0,-2i\Delta_0).
\]

The Schur-effective kernel is

\[
K_{\mathrm{eff}}
 =K_{SS}-K_{S\eta}K_{\eta\eta}^{-1}K_{\eta S}.
\]

## 3. Exact Peierls momentum is the bare lattice momentum q

For a hopping representation

\[
h(k)=\sum_R t_R e^{ik\cdot R},
\]

the implemented Peierls vector vertex is

\[
V_i(k,q)
 =\sum_R iR_i t_R e^{ik\cdot R}
   \operatorname{sinc}\!\left(\frac{q\cdot R}{2}\right).
\]

Therefore

\[
\begin{aligned}
q_iV_i(k,q)
&=\sum_R i(q\cdot R)t_R e^{ik\cdot R}
  \frac{\sin(q\cdot R/2)}{q\cdot R/2}\\
&=\sum_R 2i\sin(q\cdot R/2)t_R e^{ik\cdot R}\\
&=h(k+q/2)-h(k-q/2).
\end{aligned}
\]

Thus the exact gauge momentum of the present Peierls convention is the bare vector

\[
(q_x,q_y),
\]

not a replacement such as \(2\sin(q_i/2)\). The sinc factor has already converted the line-integrated Peierls phase into an identity contracted by the bare q.

Consequently the current xy-to-LT rotation built from \(\hat q\) is the correct candidate gauge direction for this implementation.

## 4. Exact finite-q phase tangent for the current pairing ansatz

Write the mean pairing as

\[
\Delta(k)=\Delta_0\phi(k).
\]

The implemented phase form factor is

\[
\phi_q(k)=\frac{\phi(k-q/2)+\phi(k+q/2)}{2},
\]

and the phase vertex is

\[
\Gamma_{\eta_2}(k,q)
=
\begin{pmatrix}
0&i\phi_q(k)\\
-i\phi_q^\dagger(k)&0
\end{pmatrix}.
\]

Let

\[
\Delta_\pm=\Delta(k\pm q/2).
\]

Then

\[
2i\Delta_0\Gamma_{\eta_2}
=
\begin{pmatrix}
0&-(\Delta_++\Delta_-)\\
\Delta_+^\dagger+\Delta_-^\dagger&0
\end{pmatrix}.
\]

Combining this with the Peierls identity gives the exact BdG operator identity

\[
q_i\Gamma_i^{\mathrm{BdG}}(k,q)
+2i\Delta_0\Gamma_{\eta_2}(k,q)
=
H_{\mathrm{BdG}}(k_+)\tau_3
-\tau_3H_{\mathrm{BdG}}(k_-).
\]

This is an algebraic identity before any Kubo or Brillouin-zone integration.

### 4.1 d-wave decomposition

For

\[
f_d(k)=\frac{\cos k_x-\cos k_y}{2},
\]

the endpoint-average form factor is

\[
\begin{aligned}
f_{d,q}(k)
&=\frac{f_d(k+q/2)+f_d(k-q/2)}{2}\\
&=\frac{1}{2}
\left[
\cos k_x\cos\frac{q_x}{2}
-\cos k_y\cos\frac{q_y}{2}
\right].
\end{aligned}
\]

Introduce

\[
f_s(k)=\frac{\cos k_x+\cos k_y}{2},
\quad
 a=\cos(q_x/2),
\quad
 b=\cos(q_y/2).
\]

Then

\[
f_{d,q}(k)
=\frac{a+b}{2}f_d(k)
 +\frac{a-b}{2}f_s(k).
\]

The apparent extended-s component is therefore already contained in the implemented q-dependent phase tangent. It is fixed by the single local gauge parameter and is not, by itself, evidence that an independent extended-s collective field is required to represent the gauge orbit.

This does not prove that the two-channel collective model is complete for all physical pairing fluctuations. It does show that the first-order gauge tangent used by the current Ward identity is not missing the endpoint-induced s-like component.

## 5. Gauge-invariant effective-action theorem

Let the exact Euclidean effective action be

\[
\Gamma[A,\eta],
\]

with combined field coordinate

\[
\Phi=(A,\eta).
\]

For an infinitesimal gauge parameter \(\alpha\), write the tangent at the homogeneous background as

\[
\delta\Phi=g\alpha,
\qquad
g=(u,w).
\]

Gauge invariance gives

\[
\Gamma_{,a}g^a=0.
\]

Differentiate with respect to \(\Phi^b\):

\[
\Gamma_{,ba}g^a
+\Gamma_{,a}\,\partial_b g^a=0.
\]

At a stationary saddle,

\[
\Gamma_{,a}=0,
\]

so the exact Hessian obeys

\[
K_{ba}g^a=0.
\]

In block form, the right identities are

\[
K_{SS}u_R+K_{S\eta}w_R=0,
\]

\[
K_{\eta S}u_R+K_{\eta\eta}w_R=0.
\]

The left identities are

\[
u_LK_{SS}+w_LK_{\eta S}=0,
\]

\[
u_LK_{S\eta}+w_LK_{\eta\eta}=0.
\]

For nonzero q, if \(K_{\eta\eta}\) is invertible under the selected physical Schur branch, the second right identity gives

\[
w_R=-K_{\eta\eta}^{-1}K_{\eta S}u_R.
\]

Substitution into the first gives

\[
K_{\mathrm{eff}}u_R=0.
\]

Similarly,

\[
u_LK_{\mathrm{eff}}=0.
\]

Therefore, conditional on the current response blocks being the Hessian of a complete gauge-invariant action evaluated at a stationary saddle, the exact Schur-effective electromagnetic kernel is transverse for every q and Matsubara frequency.

At exact zero Matsubara frequency,

\[
u=(0,q_x,q_y),
\]

so in the LT basis this theorem requires

\[
K_{0L}=K_{L0}=K_{LL}=K_{LT}=K_{TL}=0.
\]

The static longitudinal projection contract is therefore analytically justified only if the current microscopic blocks satisfy the hypotheses of this theorem.

## 6. Meaning of the current RHS-aware identity

The current validator constructs a quadrature-dependent vector

\[
R_S^{(Q)}
=R_{\mathrm{equal}}^{(Q)}
 -R_{\Delta v}^{(Q)}
 +R_{qM}^{(Q)}
\]

and checks

\[
u_LK_{SS}^{(Q)}+w_LK_{\eta S}^{(Q)}
=R_S^{(Q)}
\]

with the corresponding right identity.

The superscript \((Q)\) is essential. A finite quadrature is generally not invariant under an arbitrary momentum translation:

\[
Q[F(k+q/2)]\ne Q[F(k-q/2)].
\]

The equal-time and contact terms retain exactly this translation defect so that the operator identity still closes on the same microscopic nodes and weights.

For the exact periodic Brillouin-zone integral,

\[
\int_{\mathrm{BZ}}F(k+a)\,dk
=\int_{\mathrm{BZ}}F(k)\,dk,
\]

and a complete gauge-invariant Hessian must recover the homogeneous identities of section 5.

The working analytic hypothesis is therefore

\[
\lim_{Q\to\int_{\mathrm{BZ}}}R_S^{(Q)}=0.
\]

This statement still requires an explicit trace-level derivation for all three external components under the present observable/source signs. It should not be replaced by the stronger and currently unsupported statement that a nonzero finite-q RHS is a physical continuum target.

## 7. Algebraic redundancy of the current effective residual

The current validator defines

\[
C_L=u_LK_{S\eta}+w_LK_{\eta\eta}
\]

and

\[
P_L=R_S-C_LK_{\eta\eta}^{-1}K_{\eta S}.
\]

It compares

\[
E_L=u_LK_{\mathrm{eff}}
\]

with \(P_L\). Direct expansion gives

\[
\begin{aligned}
E_L-P_L
&=u_LK_{\mathrm{eff}}-R_S
 +C_LK_{\eta\eta}^{-1}K_{\eta S}\\
&=u_LK_{SS}+w_LK_{\eta S}-R_S.
\end{aligned}
\]

Thus the effective residual equals the primitive residual algebraically. The effective check is a useful Schur bookkeeping check, but it is not an independent test that

\[
u_LK_{\mathrm{eff}}=0.
\]

The same conclusion holds on the right.

## 8. Provisional analytic findings

The first pass of the analytic audit gives the following results.

1. The correct Peierls gauge momentum is the bare q used by the current LT rotation.
2. The endpoint-average phase vertex exactly supplies the finite-q BdG gauge tangent.
3. The apparent extended-s component of a d-wave endpoint tangent is already included in that q-dependent vertex.
4. A complete gauge-invariant Hessian at a stationary saddle has a homogeneous combined Ward zero mode, and its Schur-effective electromagnetic kernel is transverse.
5. The current effective Ward residual is algebraically dependent on the primitive RHS-aware residual and does not independently establish transversality.
6. The current finite-quadrature RHS should be treated as a translation/contact quadrature defect until a trace-level continuum derivation proves otherwise.

The audit therefore does **not** currently support the claim that the exact n=0 physical longitudinal sector may be nonzero.

## 9. Remaining analytic obligations

Before changing code or production gates, the following derivations must be completed.

### 9.1 Trace-level continuum cancellation of the external RHS

Starting from the exact operator Ward identity and the current Kubo divided-difference convention, derive separately for external columns

\[
A_0,
\quad A_x,
\quad A_y
\]

that

\[
R_{\mathrm{equal}}-R_{\Delta v}+R_{qM}
\]

is a periodic Brillouin-zone translation difference and therefore integrates to zero.

### 9.2 Collective-column identity

Derive

\[
u_LK_{S\eta}+w_LK_{\eta\eta}=0
\]

and its right counterpart including both:

- the fermionic collective bubble;
- the Hubbard-Stratonovich / Goldstone counterterm.

This must show explicitly where stationarity or the gap equation enters.

### 9.3 Response-Hessian equivalence

Verify that the current observable/source convention

\[
J=(\rho,-V_x,-V_y),
\qquad
P=(\rho,V_x,V_y)
\]

and the direct contact signs produce the same block Hessian, up to the documented fixed row/column sign metric, that appears in the gauge-invariant effective-action theorem.

### 9.4 Static consequence

Only after 9.1-9.3 are proved may the project retain the production interpretation

\[
\text{raw longitudinal}=\text{quadrature leakage toward an exact zero target}.
\]

Until then:

```text
production_reference_established = False
projection_eligible = False
valid_for_casimir_input = False
```
