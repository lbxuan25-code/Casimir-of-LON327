# d-wave finite-q bond-phase Hessian derivation

## 1. Scope

This note derives the finite-q Hubbard--Stratonovich curvature required by the
current d-wave collective phase vertex.  The target is the exact multiplier

\[
g_{\rm bond}(q)
=\frac{1}{2}\left[\cos^2\!\left(\frac{q_x}{2}\right)
+\cos^2\!\left(\frac{q_y}{2}\right)\right].
\]

The derivation is independent of the fermionic band indices.  The orbital part
of the symmetry two-band d-wave gap is proportional to the identity, so only
the nearest-neighbour bond geometry is relevant.

No production counterterm is changed in this note.  The result first fixes the
phase-sector convention that a later full-kernel audit must implement and test.

## 2. Bond representation of the current d-wave form factor

The implemented form factor is

\[
\phi_d(k)=\frac{1}{2}(\cos k_x-\cos k_y).
\]

Introduce the bond basis

\[
F(k)=\frac{1}{2}
\begin{pmatrix}
\cos k_x\\
\cos k_y
\end{pmatrix},
\qquad
d=
\begin{pmatrix}
1\\
-1
\end{pmatrix}.
\]

Then

\[
\phi_d(k)=F(k)^T d,
\qquad
\Delta(k)=\Delta_0 F(k)^T d.
\]

Equivalently, the uniform saddle has x- and y-bond amplitudes with equal
magnitude and opposite sign.

## 3. Local gauge transformation of a bond field

Let a nearest-neighbour pair field on a bond centred at `R` be
\(\Delta_\mu(R)\), with endpoints \(R\mp\hat\mu/2\).  Under

\[
c_r\longrightarrow e^{i\alpha_r}c_r,
\]

the bond field transforms as

\[
\Delta_\mu(R)
\longrightarrow
\exp\left\{i\left[\alpha(R-\hat\mu/2)
+\alpha(R+\hat\mu/2)\right]\right\}\Delta_\mu(R).
\]

For one Fourier component

\[
\alpha(R)=\alpha_q e^{iq\cdot R},
\]

the endpoint phase sum is

\[
\alpha(R-\hat\mu/2)+\alpha(R+\hat\mu/2)
=2\alpha_q e^{iq\cdot R}\cos(q_\mu/2).
\]

Define

\[
D(q)=
\begin{pmatrix}
\cos(q_x/2)&0\\
0&\cos(q_y/2)
\end{pmatrix}.
\]

The bond-space gauge tangent through the d-wave saddle is therefore

\[
t(q)=D(q)d
=
\begin{pmatrix}
\cos(q_x/2)\\
-\cos(q_y/2)
\end{pmatrix}.
\]

At q=0, \(t(0)=d\).  For generic q, the tangent is not parallel to the static
d-wave direction.

## 4. Match to the implemented eta2 vertex

The current endpoint-average form factor is

\[
\begin{aligned}
\phi_q(k)
&=\frac{1}{2}\left[\phi_d(k-q/2)+\phi_d(k+q/2)\right]\\
&=\frac{1}{2}\left[
\cos k_x\cos(q_x/2)-\cos k_y\cos(q_y/2)
\right]\\
&=F(k)^Tt(q).
\end{aligned}
\]

Thus the implemented phase vertex

\[
\Gamma_2(k,q)=i\phi_q(k)
\]

is exactly the full bond gauge tangent contracted into the bond form-factor
basis.  It is not a fixed scalar d-wave phase vertex.

The primitive Ward vector uses

\[
w_2=-2i\Delta_0.
\]

Consequently,

\[
w_2\Gamma_2(k,q)
=(-2i\Delta_0)iF(k)^Tt(q)
=2\Delta_0F(k)^Tt(q),
\]

while

\[
\Delta(k-q/2)+\Delta(k+q/2)
=2\Delta_0F(k)^Tt(q).
\]

Therefore the implemented \(\Gamma_2\) and \(w_2\) reproduce the anomalous
finite-q Ward source exactly.

## 5. Pullback of the bond HS metric

Use the minimal gauge-completed nearest-neighbour HS action with an isotropic
bond metric in the x/y bond basis.  In the imaginary bond sector its quadratic
part is

\[
S_{\rm HS}^{(2)}
=\frac{1}{2}\sum_q p(-q)^T C_b I_2\,p(q),
\]

where \(p=(p_x,p_y)^T\) denotes imaginary bond fluctuations and \(C_b\) is the
single-bond curvature.

The scalar eta2 coordinate used by the current vertex is the q-dependent
embedding

\[
p(q)=t(q)\eta_2(q).
\]

Pulling the bond metric back to this scalar coordinate gives

\[
K_{22}^{\rm HS}(q)
=C_b\,t(q)^Tt(q)
=C_b\left[\cos^2(q_x/2)+\cos^2(q_y/2)\right].
\]

At q=0,

\[
K_{22}^{\rm HS}(0)=C_b d^Td=2C_b.
\]

Let the q=0 Goldstone counterterm inferred by the code be

\[
C_0\equiv K_{22}^{\rm HS}(0)=-\Pi_{22}(0,0).
\]

Then the exact finite-q phase counterterm required by the same bond action is

\[
\boxed{
K_{22}^{\rm HS}(q)
=C_0\,g_{\rm bond}(q)
}
\]

with

\[
\boxed{
g_{\rm bond}(q)
=\frac{t(q)^Tt(q)}{d^Td}
=\frac{1}{2}\left[\cos^2(q_x/2)+\cos^2(q_y/2)\right].}
\]

This result is exact for the nearest-neighbour bond geometry; it is not a
small-q fit.

## 6. Why the metric is not the square of an averaged cosine

Introduce the extended-s bond direction

\[
s=
\begin{pmatrix}
1\\
1
\end{pmatrix}
\]

and define

\[
\bar c=\frac{\cos(q_x/2)+\cos(q_y/2)}{2},
\qquad
\delta c=\frac{\cos(q_x/2)-\cos(q_y/2)}{2}.
\]

The gauge tangent decomposes as

\[
t(q)=\bar c\,d+\delta c\,s.
\]

Hence

\[
\frac{t(q)^Tt(q)}{d^Td}
=\bar c^2+\delta c^2
=\frac{\cos^2(q_x/2)+\cos^2(q_y/2)}{2}.
\]

A projection onto the static d-wave direction alone would retain only
\(\bar c^2\).  It would discard the extended-s component of the local gauge
tangent and is therefore not a complete finite-q gauge orbit.

This also explains why the numerically required Hessian is the average of
squares rather than the square of the average.

## 7. Small-q expansion

Using

\[
\cos^2(q_\mu/2)=1-\frac{q_\mu^2}{4}+\frac{q_\mu^4}{48}+O(q_\mu^6),
\]

one obtains

\[
g_{\rm bond}(q)
=1-\frac{q_x^2+q_y^2}{8}
+\frac{q_x^4+q_y^4}{96}
+O(q^6).
\]

The leading shift is isotropic and has coefficient 1/8, matching the clean
even-shift commensurate result.

## 8. Full bond-space Ward reduction

Let \(K_{Sp}\) and \(K_{pp}\) be the electromagnetic--bond-phase and
bond-phase--bond-phase blocks in the full x/y bond basis.  The full bond Ward
vector is

\[
w_b(q)=-2i\Delta_0 t(q).
\]

The bond-space collective identity is

\[
u_LK_{Sp}+w_b(q)^TK_{pp}=0.
\]

The scalar phase vertex is the restriction to the gauge tangent:

\[
K_{S2}=K_{Sp}t(q),
\qquad
K_{22}=t(q)^TK_{pp}t(q).
\]

Multiplying the full identity by \(t(q)\) on the right gives

\[
u_LK_{S2}+(-2i\Delta_0)K_{22}=0,
\]

which is exactly the scalar collective phase-column identity used by the
current Ward audit.  The q-independent scalar Ward coefficient is consistent
because all q dependence has been placed in the tangent vertex and in the
pulled-back Hessian.

## 9. Relation to the commensurate numerical result

For the subgrid-averaged `N=628`, `(m_x,m_y)=(3,2)` audit,

```text
required multiplier    = 0.9998373452379
bond metric multiplier = 0.9998373452379
bond multiplier error  = 1.08e-37
bond defect / |q|      = 9.68e-36
```

The equality at the component-averaged level is the numerical realization of

\[
K_{22}^{\rm HS}(q)=g_{\rm bond}(q)K_{22}^{\rm HS}(0).
\]

The opposite single-subgrid offsets are a half-q quadrature alias and are not
an additional physical Hessian term.

## 10. What is and is not fixed by this derivation

The phase-sector result is fixed:

\[
K_{22}^{\rm counterterm}(q)
=g_{\rm bond}(q)K_{22}^{\rm counterterm}(0).
\]

For a real/imaginary bond metric there is no HS amplitude--phase cross term, so

\[
K_{12}^{\rm HS}=K_{21}^{\rm HS}=0.
\]

The amplitude diagonal is not fixed by the Ward identity.  The current code
uses the same q-dependent form factor for eta1 and eta2.  If eta1 is literally
defined by the same bond embedding \(a(q)=t(q)\eta_1(q)\), then the same
pullback gives

\[
K_{11}^{\rm HS}(q)=g_{\rm bond}(q)K_{11}^{\rm HS}(0).
\]

However, a physical d-wave amplitude mode is more naturally the fixed bond
direction \(a(q)=d\eta_1(q)\), for which its HS curvature remains q-independent.
The present Ward evidence constrains only the phase row/column and does not
justify changing the amplitude counterterm.  The first production-facing
implementation should therefore modify only the phase counterterm and keep the
amplitude convention explicitly diagnostic until it is separately derived.

## 11. General bond metric

If the microscopic bond HS metric is a nontrivial matrix \(M_b\), the exact
normalized phase multiplier is

\[
g_M(q)=\frac{t(q)^TM_bt(q)}{d^TM_bd}.
\]

The observed equality with \(g_{\rm bond}\) corresponds to the isotropic
nearest-neighbour metric \(M_b\propto I_2\).  A pure rank-one d-channel metric
would instead produce a square-of-average-cosines factor and would not match
the commensurate audit.

## 12. Required implementation audit

Before promotion, add a diagnostic-only finite-q counterterm policy that:

1. computes the q=0 Goldstone scalar `C0` on the existing quadrature;
2. changes only `counterterm[1,1]` to `g_bond(q) * C0`;
3. leaves `counterterm[0,0]` unchanged;
4. rebuilds the complete 2x2 collective block and Schur complement;
5. verifies left and right primitive/effective Ward closure without projection;
6. checks that the amplitude-column residual is not enlarged;
7. reports the changes in `chi_bar`, `Dbar_T`, the collective condition number,
   and the complete local-LT kernel;
8. remains fail-closed and invalid for Casimir input until the result is repeated
   on half-q-compatible or complementary-subgrid-averaged quadratures.
