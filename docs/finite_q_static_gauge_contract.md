# Finite-q static gauge contract: analytic audit

Status: **phase-1 analytic audit complete**. This note does not change a production gate and does not establish Casimir readiness. It fixes the continuum target that the subsequent numerical audit must test.

## 1. Scope and conclusion

The current finite-q response stack contains two distinct checks:

1. an RHS-aware primitive/effective Ward closure check on one microscopic quadrature;
2. an exact-static longitudinal gate that assumes the Schur-effective electromagnetic kernel has a pure-gauge longitudinal zero mode.

The analytic question is whether the exact periodic Brillouin-zone target is

\[
 u_L K_{\mathrm{eff}}=0,
 \qquad
 K_{\mathrm{eff}}u_R=0,
\]

or a nonzero effective RHS.

The result of this audit is:

\[
\boxed{
\text{for the present Peierls convention, endpoint phase tangent, and a stationary}
\atop
\text{gauge-invariant amplitude/phase action, the exact continuum target is zero.}
}
\]

The finite-quadrature RHS used by the validator is a translation/contact defect that preserves the operator identity on the chosen nodes and weights. It is not the physical continuum longitudinal response.

## 2. Current field and routing conventions

Primitive electromagnetic order:

\[
S=(A_0,A_x,A_y).
\]

Collective order:

\[
\eta=(\eta_1,\eta_2),
\]

where `eta1` is the Cartesian amplitude fluctuation and `eta2` is the Cartesian phase-direction fluctuation. The finite-q routing is

\[
k_-=k-q/2,
\qquad
k_+=k+q/2.
\]

The code Ward vectors are

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

not a replacement such as \(2\sin(q_i/2)\). The sinc factor has already converted the line-integrated Peierls phase into an identity contracted by the bare q. Consequently the current xy-to-LT rotation built from \(\hat q\) uses the correct gauge direction.

The corresponding second-order Peierls identity is

\[
q_i M_{ij}(k,q)
 =\Gamma_j(k+q/2,-q)\tau_3
  -\tau_3\Gamma_j(k-q/2,-q),
\]

with \(M_{ij}=\delta^2H/\delta A_i\delta A_j\).

## 4. Exact finite-q phase tangent

Write the mean pairing as

\[
\Delta(k)=\Delta_0\phi(k).
\]

The implemented finite-q collective form factor is

\[
\phi_q(k)=\frac{\phi(k-q/2)+\phi(k+q/2)}{2},
\]

and the Cartesian phase-direction vertex is

\[
\Gamma_{\eta_2}(k,q)
=
\begin{pmatrix}
0&i\phi_q(k)\\
-i\phi_q^\dagger(k)&0
\end{pmatrix}.
\]

Let \(\Delta_\pm=\Delta(k\pm q/2)\). Then

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

This identity holds before a Kubo sum or a Brillouin-zone integral.

### 4.1 d-wave endpoint decomposition

For

\[
f_d(k)=\frac{\cos k_x-\cos k_y}{2},
\]

the endpoint average is

\[
\begin{aligned}
f_{d,q}(k)
&=\frac{f_d(k+q/2)+f_d(k-q/2)}{2}\\
&=\frac12\left[
\cos k_x\cos\frac{q_x}{2}
-\cos k_y\cos\frac{q_y}{2}
\right].
\end{aligned}
\]

With

\[
f_s(k)=\frac{\cos k_x+\cos k_y}{2},
\qquad
a=\cos(q_x/2),
\qquad
b=\cos(q_y/2),
\]

one has

\[
f_{d,q}(k)
=\frac{a+b}{2}f_d(k)
 +\frac{a-b}{2}f_s(k).
\]

The endpoint-induced extended-s-looking component is already present in the implemented q-dependent gauge tangent. It is fixed by the single local gauge parameter and does not require an independent extended-s collective coordinate merely to represent the first-order gauge orbit.

## 5. Gauge-invariant effective-action theorem

Let the exact Euclidean effective action be

\[
\Gamma[A,\eta],
\qquad
\Phi=(A,\eta).
\]

For an infinitesimal gauge parameter \(\alpha\), write

\[
\delta\Phi=g\alpha,
\qquad
g=(u,w).
\]

Gauge invariance gives

\[
\Gamma_{,a}g^a=0.
\]

Differentiation with respect to \(\Phi^b\) gives

\[
\Gamma_{,ba}g^a
+\Gamma_{,a}\,\partial_b g^a=0.
\]

At a stationary saddle, \(\Gamma_{,a}=0\), so the exact Hessian obeys

\[
K_{ba}g^a=0.
\]

In block form,

\[
K_{SS}u_R+K_{S\eta}w_R=0,
\qquad
K_{\eta S}u_R+K_{\eta\eta}w_R=0,
\]

and

\[
u_LK_{SS}+w_LK_{\eta S}=0,
\qquad
u_LK_{S\eta}+w_LK_{\eta\eta}=0.
\]

If \(K_{\eta\eta}\) is invertible on the selected physical Schur branch,

\[
w_R=-K_{\eta\eta}^{-1}K_{\eta S}u_R,
\]

and therefore

\[
K_{\mathrm{eff}}u_R=0,
\qquad
u_LK_{\mathrm{eff}}=0.
\]

At exact zero Matsubara frequency,

\[
u=(0,q_x,q_y),
\]

so the LT representation requires

\[
K_{0L}=K_{L0}=K_{LL}=K_{LT}=K_{TL}=0.
\]

The remaining sections map the current implementation to the hypotheses of this theorem.

## 6. Proof I: continuum cancellation of the external RHS

### 6.1 Bubble contraction

Let the source-side primitive vertices be

\[
P_b=(\tau_3,\Gamma_x,\Gamma_y),
\]

and let the row-side vertices use the fixed sign metric discussed in section 8. The operator identity can be written at Green-function level as

\[
u_L\!\cdot J+w_L\!\cdot\Lambda
=G_+^{-1}\tau_3-\tau_3G_-^{-1}.
\]

Contracting the left index of the fermion bubble and cancelling the adjacent inverse propagators gives the equal-time term

\[
R^{\mathrm{equal}}_b
=\frac12\int_{\mathrm{BZ}}\!dk\,
\operatorname{Tr}\!\left[
\tau_3F_-P_b^\dagger
-F_+\tau_3P_b^\dagger
\right],
\]

where

\[
F_\pm=f(H_{\mathrm{BdG}}(k\pm q/2)).
\]

This is the trace form of the band-sum quantity named `equal_forward` by the current validator.

### 6.2 Density column

For \(b=0\), \(P_0=\tau_3\) and \(\tau_3^2=1\). Hence

\[
R^{\mathrm{equal}}_0
=\frac12\int_{\mathrm{BZ}}dk\,
\left[
\operatorname{Tr}F_--\operatorname{Tr}F_+
\right]=0
\]

by periodic translation invariance of the exact BZ integral. The `delta_v_mid` and `qM_mid` vectors have no density component, so

\[
R_{S,0}=0.
\]

### 6.3 Spatial columns: equal-time and contact cancellation

For a spatial source column \(j\), shift the integration variable separately in the two equal-time terms. Using

\[
\Gamma_j(k,q)^\dagger=\Gamma_j(k,-q),
\]

one obtains

\[
R^{\mathrm{equal}}_j
=\frac12\int_{\mathrm{BZ}}dk\,
\operatorname{Tr}\!\left[
F(k)
\left(
\Gamma_j(k+q/2,-q)\tau_3
-\tau_3\Gamma_j(k-q/2,-q)
\right)
\right].
\]

The second-order Peierls identity therefore gives

\[
R^{\mathrm{equal}}_j
=\frac12\int_{\mathrm{BZ}}dk\,
\operatorname{Tr}\left[F(k)q_iM_{ij}(k,q)\right].
\]

The implementation defines the direct block as

\[
K^{\mathrm{direct}}_{ij}
=-\frac12\int_{\mathrm{BZ}}dk\,
\operatorname{Tr}\left[F(k)M_{ij}(k,q)\right],
\]

so its Ward contraction is

\[
R_{qM,j}=q_iK^{\mathrm{direct}}_{ij}=-R^{\mathrm{equal}}_j.
\]

Thus

\[
R^{\mathrm{equal}}_j+R_{qM,j}=0
\]

in the exact periodic integral.

### 6.4 Vanishing of the vertex-translation term

The remaining implementation term is

\[
R_{\Delta v,j}
=\frac12\int_{\mathrm{BZ}}dk\,
\operatorname{Tr}\left[
F(k)X_j(k,q)
\right],
\]

with

\[
X_j(k,q)
=\Gamma_j(k+q/2,q)-\Gamma_j(k-q/2,q).
\]

The BdG Hamiltonian and this difference vertex obey particle-hole covariance

\[
\mathcal C H(k)\mathcal C^{-1}=-H(-k),
\qquad
\mathcal C X_j(k,q)\mathcal C^{-1}=X_j(-k,q),
\]

for the present real, inversion-symmetric two-band hopping and even-parity pairing ansatz. Since the BdG Fermi level is zero,

\[
f(-H)=1-f(H).
\]

Pairing the points \(k\) and \(-k\) in the BZ gives

\[
R_{\Delta v,j}
=\frac14\int_{\mathrm{BZ}}dk\,
\operatorname{Tr}X_j(k,q).
\]

Finally,

\[
\int_{\mathrm{BZ}}dk\,\operatorname{Tr}X_j(k,q)
=
\int_{\mathrm{BZ}}dk\,
\left[
\operatorname{Tr}\Gamma_j(k+q/2,q)
-\operatorname{Tr}\Gamma_j(k-q/2,q)
\right]=0
\]

by periodic translation invariance. Hence

\[
R_{\Delta v,j}=0.
\]

Combining the density and spatial columns yields

\[
\boxed{
R_S^{(\mathrm{continuum})}
=R_{\mathrm{equal}}-R_{\Delta v}+R_{qM}=0.
}
\]

### 6.5 Why the finite-quadrature RHS is nonzero

A finite quadrature \(Q\) is generally not invariant under arbitrary momentum translation:

\[
Q[F(k+a)]\ne Q[F(k)].
\]

The variable shifts used in sections 6.2-6.4 therefore cannot be made exactly on an arbitrary finite set of nodes. The code retains the resulting equal-time, vertex-translation, and contact terms on the same nodes and weights, so the operator identity closes as

\[
u_LK_{SS}^{(Q)}+w_LK_{\eta S}^{(Q)}=R_S^{(Q)}.
\]

The correct interpretation is

\[
\boxed{
R_S^{(Q)}=\text{finite-quadrature translation/contact defect},
\qquad
\lim_{Q\to\int_{\mathrm{BZ}}}R_S^{(Q)}=0.
}
\]

## 7. Proof II: collective-column identity and the gap equation

### 7.1 Cartesian collective coordinates

Write the complex pairing amplitude as

\[
\Delta=(\Delta_0+\eta_1)+i\eta_2.
\]

Under a local U(1) transformation,

\[
\Delta\longrightarrow e^{2i\alpha}\Delta.
\]

To first order in \(\alpha\), the collective gauge vector is

\[
g_\eta(\eta_1,\eta_2)
=
\left(
-2\eta_2,
2(\Delta_0+\eta_1)
\right)
\]

up to the fixed Fourier-i convention represented in code by

\[
w=(0,-2i\Delta_0).
\]

Its derivatives at the real homogeneous background are nonzero:

\[
\partial_{\eta_1}g_\eta=(0,2),
\qquad
\partial_{\eta_2}g_\eta=(-2,0).
\]

### 7.2 Differentiated Ward identity

Before imposing stationarity,

\[
K_{ba}g^a=-\Gamma_{,a}\partial_b g^a.
\]

For the two collective columns this gives, up to the same fixed Fourier-i signs,

\[
u K_{S\eta_1}+wK_{\eta\eta_1}
\propto \Gamma_{,\eta_2},
\]

and

\[
u K_{S\eta_2}+wK_{\eta\eta_2}
\propto \Gamma_{,\eta_1}.
\]

The amplitude-column identity therefore uses the vanishing phase tadpole, while the phase-column identity uses the amplitude saddle equation. At a real stationary saddle,

\[
\Gamma_{,\eta_2}=0,
\qquad
\Gamma_{,\eta_1}=0,
\]

so

\[
\boxed{
 u_LK_{S\eta}+w_LK_{\eta\eta}=0,
 \qquad
 K_{\eta S}u_R+K_{\eta\eta}w_R=0.
}
\]

This identifies precisely where the gap equation enters: it is required for the collective **phase column** of the Hessian Ward identity.

### 7.3 Hubbard-Stratonovich curvature implemented by the counterterm

For one complex pairing channel, the local Hubbard-Stratonovich action has the U(1)-symmetric quadratic form

\[
\Gamma_{\mathrm{HS}}
=\frac{c_g}{2}\int
\left[(\Delta_0+\eta_1)^2+\eta_2^2\right].
\]

Its collective Hessian is

\[
K_{\eta\eta}^{\mathrm{HS}}=c_g I_2.
\]

The amplitude saddle equation fixes \(c_g\). Equivalently, gauge invariance implies the zero-momentum phase Goldstone condition

\[
K_{22}(q=0,\xi=0)
=\Pi_{22}(0,0)+c_g=0,
\]

and hence

\[
c_g=-\Pi_{22}(0,0).
\]

The current `goldstone_gap_equation` counterterm does exactly this:

\[
K_{\eta\eta}^{\mathrm{counterterm}}
=-\Pi_{22}^{(Q)}(0,0)I_2.
\]

Using the same scalar on the amplitude and phase diagonals is not an arbitrary degeneracy assumption. It is the Hessian of a U(1)-symmetric Cartesian HS term; the physical amplitude mass remains nonzero because \(\Pi_{11}\ne\Pi_{22}\).

The adaptive integrand evaluates the q=0 phase bubble density and the finite-q response primitives on one shared vector quadrature. Consequently its counterterm is the discrete version of the same gap-equation curvature rather than an independently sampled correction.

The present validation model infers \(c_g\) from the selected \(\Delta_0\); it does not provide an independent microscopic pairing interaction. This is sufficient for the gauge-closure audit, but it remains a validation-model construction rather than a material gap-equation prediction.

## 8. Proof III: response-Hessian equivalence and sign metric

Let

\[
P=(\tau_3,\Gamma_x,\Gamma_y)
\]

be the source-side electromagnetic vertices and define

\[
D=\operatorname{diag}(1,-1,-1).
\]

The implementation uses row-side observables

\[
J=DP=(\tau_3,-\Gamma_x,-\Gamma_y).
\]

Let \(\mathcal H\) denote the source-coordinate Hessian of the effective action, with source vertices \(P\) on both external legs. The fermion bubble assembled by the code satisfies

\[
K_{SS}^{\mathrm{bubble,code}}
=D\mathcal H_{SS}^{\mathrm{bubble}}.
\]

The source-coordinate Hessian contains the direct Peierls term

\[
\mathcal H_{ij}^{\mathrm{direct}}
=+\frac12\int dk\,\operatorname{Tr}[F(k)M_{ij}(k,q)].
\]

Multiplication by the spatial row sign in \(D\) gives

\[
(D\mathcal H^{\mathrm{direct}})_{ij}
=-\frac12\int dk\,\operatorname{Tr}[F(k)M_{ij}(k,q)],
\]

which is exactly the `direct = -<M>` convention used by the response engine.

The complete block mapping is therefore

\[
K_{SS}^{\mathrm{code}}=D\mathcal H_{SS},
\qquad
K_{S\eta}^{\mathrm{code}}=D\mathcal H_{S\eta},
\]

\[
K_{\eta S}^{\mathrm{code}}=\mathcal H_{\eta S},
\qquad
K_{\eta\eta}^{\mathrm{code}}=\mathcal H_{\eta\eta}.
\]

Because \(D^2=1\), the Schur complement obeys

\[
K_{\mathrm{eff}}^{\mathrm{code}}
=D\mathcal H_{\mathrm{eff}}.
\]

For the source-coordinate Hessian, take the right gauge vector

\[
g_R=(-i\xi,q_x,q_y)=u_R
\]

and the reciprocal left covector

\[
g_L=(+i\xi,-q_x,-q_y).
\]

Then

\[
K_{\mathrm{eff}}^{\mathrm{code}}u_R
=D\mathcal H_{\mathrm{eff}}g_R=0.
\]

On the left,

\[
g_L\mathcal H_{\mathrm{eff}}=0
\]

implies

\[
(g_LD)K_{\mathrm{eff}}^{\mathrm{code}}=0.
\]

But

\[
g_LD=(+i\xi,q_x,q_y)=u_L.
\]

Thus the apparently asymmetric current/source signs and the left/right Matsubara signs are exactly the fixed row-metric representation of one gauge-invariant Hessian. They do not change the zero-longitudinal continuum target.

## 9. Algebraic redundancy of the current effective residual

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

Thus the effective residual equals the primitive residual algebraically. The effective check verifies Schur bookkeeping but is not an independent transversality test. The same conclusion holds on the right.

## 10. Completed analytic findings

The phase-1 audit establishes:

1. the exact Peierls gauge momentum is the bare q used by the LT rotation;
2. the endpoint-average phase vertex supplies the exact finite-q BdG gauge tangent;
3. its d-wave endpoint-induced s-like component is already included;
4. the exact periodic-BZ external RHS vanishes column by column;
5. the collective-column identity is homogeneous at a stationary saddle, with the amplitude gap equation entering the phase column;
6. the current scalar-identity Goldstone counterterm is the Cartesian U(1)-symmetric HS curvature fixed by that gap equation;
7. the observable/source sign convention is a fixed row metric multiplying the source-coordinate Hessian;
8. the exact Schur-effective electromagnetic kernel is transverse;
9. the current RHS-aware effective residual is algebraically dependent on primitive closure and does not independently test this transversality.

Therefore

\[
\boxed{
\text{the exact n=0 longitudinal target remains zero.}
}
\]

The current numerical `raw_longitudinal` is not reinterpreted as a physical nonzero longitudinal response. It is a measure of failure to reach the continuum homogeneous identity, potentially amplified by cancellation after the Schur complement.

## 11. Consequence for the next audit phase

The next phase must no longer ask whether the zero target is correct. It must identify why the converged-looking physical channels and the longitudinal cancellation converge at different rates.

A dedicated diagnostic must report, for each quadrature and orientation,

\[
R_S^{(Q)},
\qquad
C_\eta^{(Q)}=uK_{S\eta}+wK_{\eta\eta},
\]

\[
R_{\mathrm{eff}}^{(Q)}
=R_S^{(Q)}-C_\eta^{(Q)}K_{\eta\eta}^{-1}K_{\eta S},
\qquad
uK_{\mathrm{eff}},
\]

and the corresponding right-side quantities. It must also record the separate

\[
R_{\mathrm{equal}},
\quad
R_{\Delta v},
\quad
R_{qM}
\]

terms and their order/orientation convergence.

The decisive numerical test is

\[
R_S^{(Q)}\to0,
\qquad
C_\eta^{(Q)}\to0,
\qquad
uK_{\mathrm{eff}}\to0
\]

under a complete periodic quadrature. A small RHS-aware residual alone is insufficient because it only proves equality to the finite-quadrature defect.

Until that numerical phase passes:

```text
production_reference_established = False
projection_eligible = False
valid_for_casimir_input = False
```
