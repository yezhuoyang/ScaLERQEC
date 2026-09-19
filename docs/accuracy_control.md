# Automatic accuracy control for general-noise profiles

The experimental general-noise interface can now choose the sample allocation
and stopping time from an accuracy request. Users specify tolerances and a
confidence level instead of a number of samples per weight. A run either meets
that request on the specified p grid or explicitly returns an unresolved result.
This interface is local research code, separate from the released uniform-SID
S-curve estimator. It does not certify S-curve extrapolation.

```python
from scalerqec.Stratified import LinearNoiseModel

# The circuit contains probabilities c*p0; decoder is fixed for the whole curve.
model = LinearNoiseModel(circuit, reference_p=0.01)
result = model.sample_until_accuracy(
    decoder,
    probabilities=[0.001, 0.002, 0.005, 0.01],
    relative_error=0.10,
    confidence=0.99,
    max_seconds=60,
    seed=123,
)

print(result.status, result.reason)
for estimate in result.estimates:
    print(estimate.p, estimate.ler, estimate.lower, estimate.upper)

if result.converged:
    polynomial = result.to_polynomial()
    polynomial.save("accurate_curve.npz")
    values = polynomial([0.001, 0.002, 0.005, 0.01])
```

`max_shots` (default one million) and `max_seconds` (default 60) are resource
limits, not accuracy settings. `exact_budget` (default 100,000 histories) limits
automatic exact enumeration separately. The time limit is checked between
complete work units; setup or one work unit can exceed it. The sampler does
not automatically increase these caps indefinitely. A budget-limited result
has useful bounds but **has not established the requested accuracy**.
`to_polynomial()` rejects such a result unless the caller explicitly passes
`allow_unconverged=True`. Exported metadata retains the status and requested grid.
These results cannot currently be resumed; rerunning with a larger cap restarts
sampling. A fixed seed reproduces the work up to a time-dependent cutoff.

## What the accuracy statement means

For preselected values p_1,...,p_M, confidence 1-alpha, relative tolerance r,
and absolute tolerance a, the statistical contract is

\[
\Pr\{\text{the run declares success and some }j\text{ has }
|\widehat L(p_j)-L(p_j)|>a+rL(p_j)\}\leq\alpha.
\]

The guarantee holds despite adaptive allocation and stopping, under the model
and independent-sampling assumptions below. It is a bound on the probability
of a false success, not a statement that every run finishes, or a conditional
coverage claim given that a run finishes. Confidence is simultaneous over the
entire requested grid. It does not cover newly chosen p values or the continuum
between grid points, even though the exported polynomial can be evaluated there.
Selecting a decoder or the grid after inspecting these samples is outside the
contract. Repeatedly rerunning seeds until a preferred answer appears is also
outside the contract.

Relative accuracy at L=0 requires proving zero exactly when a=0. Observing no
failures in random samples cannot do this. An absolute tolerance allows a useful
upper bound near zero. For example, `absolute_error=1e-8, relative_error=0.1`
requests error at most 1e-8 + 0.1 L, not relative 10% at arbitrarily small L.
The estimate and bounds use floating-point probability arithmetic, not formally
verified interval arithmetic. Statistical confidence cannot certify a circuit,
decoder, implementation, or noise assumption as bug-free.

## Construction and proof

Let h be a complete categorical fault history of the supported linear Pauli
noise model. Let W(h) count the inserted single-qubit Pauli factors and F(h)
indicate a logical failure under a fixed deterministic decoder. In particular,
XX has W=2; measurement-record flips and heralded identity outcomes can have
W=0. Histories at different locations remain distinct even if their errors
later cancel. Write Z_w(p)=Pr_p(W=w) and

\[
L(p)=\sum_w L_w(p),\qquad
L_w(p)=\sum_{h:W(h)=w}F(h)P_p(h).
\]

This is the same weight partition as the earlier profiler. We retain enough
history likelihood information to reweight within a stratum; a scalar failure
rate per weight alone would be insufficient for general nonuniform noise.

### Fixed mixtures improve overlap across p

Before sampling, select anchor probabilities from the requested grid and include
the original, interior reference p_0. For up to eight grid values all are anchors;
larger grids use five grid quantiles plus p_0. In each stratum discard anchors
with Z_w(a)=0 and give the remaining anchors equal positive weights pi_a. Draw
an anchor at random, then draw h from its exact conditional distribution.

\[
q_w(h)=\sum_a\pi_a\frac{P_a(h)}{Z_w(a)},\quad W(h)=w,\qquad
X_{w,p}(h)=F(h)\frac{P_p(h)}{q_w(h)}.
\]

Full reference support implies E_q[X_{w,p}]=L_w(p). Mixtures are fixed before
observing outcomes; only the allocation between strata is adaptive. Component
samples may be processed together, but confidence checks occur only after the
whole IID mixture checkpoint is complete.

Every included target anchor has the simple deterministic bound

\[
0\leq X_{w,p}\leq B_{w,p}=Z_w(p)/\pi_p.
\]

For a target that is not an anchor, any interior anchor a gives

\[
B_{w,p}=\frac{Z_w(a)}{\pi_a}
\max_{h:W(h)=w}\frac{P_p(h)}{P_a(h)}.
\]

A max-product dynamic program computes this maximum, including no-error
probabilities and conditional survival in E/ELSE chains. It is not an observed
sample maximum. Normalize Y=X/B into [0,1]. Strata with Z_w(p)=0 contribute zero.
The implementation detects individual-channel support underflow and rejects
that numerical configuration rather than treating those outcomes as impossible.
If products underflow and produce an unverified zero LER interval at positive p,
the run returns `numerical_limit` with an uninformative [0,1] bound instead of
certifying zero. Exact zero requires exhaustive support checks; numerical zero
alone is insufficient even after enumeration if positive polynomial terms remain.

### Confidence remains valid at the stopping time

Stratum w is checked only at n=256,512,1024,... samples, indexed by k=0,1,2,... .
For target j and checkpoint k, assign

\[
\delta_{w,j,k}=\frac{\alpha}
{M(w+1)(w+2)(k+1)(k+2)}.
\]

Both series over w and k telescope to one, so the sum of all failure budgets
is alpha, including checkpoints and strata never actually visited. At each
checkpoint the two-sided empirical Bernstein radius for Y is

\[
\rho=\sqrt{\frac{2s_Y^2\log(4/\delta_{w,j,k})}{n}}
+\frac{7\log(4/\delta_{w,j,k})}{3(n-1)}.
\]

Here s_Y^2 is the unbiased sample variance. This follows by applying Theorem 4
of [Maurer and Pontil (2009)](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf)
to both tails. Intersect B[mean(Y)-rho,mean(Y)+rho] with [0,Z_w(p)] and all
earlier intervals for this same stratum and target. An interval contradiction
returns `inconsistent_bounds`, never success. A union bound establishes that
all these intervals cover simultaneously with probability at least 1-alpha.
Adaptive allocation or choosing a stopping checkpoint does not change this event.

Unvisited strata retain [0,Z_w(p)]. The total omitted weight mass is computed
by positive accumulation and contributes [0,tail], without assuming zero failure
above the sampled cutoff. The cutoff grows automatically when that tail is the
dominant uncertainty. Exact small-stratum enumeration replaces its interval by
its known contribution; deciding to enumerate based on earlier observations
does not create sampling uncertainty in the exhaustive result.

Sum stratum intervals and the tail to obtain [ell_j,u_j]. Stop successfully only
if, for every j,

\[
\max\{\widehat L(p_j)-\ell_j,\ u_j-\widehat L(p_j)\}
\leq a+r\ell_j.
\]

On the simultaneous coverage event, L(p_j)>=ell_j, proving the stated accuracy
contract. In particular, a zero empirical variance or a large observed ESS is
not a stopping certificate. Very rare, unseen failures still have an upper bound.

### The output remains a polynomial

For each history the original model provides sufficient statistics K(h),M_g(h):

\[
R_p(h)=\frac{P_p(h)}{P_{p_0}(h)}=
(p/p_0)^{K(h)}\prod_g
\left(\frac{1-c_gp}{1-c_gp_0}\right)^{M_g(h)}.
\]

K is the number of activated categorical factors, not Pauli weight. The
conditional mixture denominator d_w(h)=q_w(h)/P_{p_0}(h) is independent of the
evaluation p. Each sampled failure therefore contributes
R_p(h)/(n_w d_w(h)); each exactly enumerated failure contributes
P_{p_0}(h)R_p(h). Combining equal sufficient statistics gives a positive factored
polynomial with fixed coefficients. No p-specific resampling or curve fitting is
needed. The stopped estimator need not be unbiased at its random stopping time;
the sequential confidence proof, rather than an unbiasedness assertion, controls
its error. Omitted contributions remain bounded and need not be identically zero.

## Scope and practical limits

The supported model and frontend restrictions remain those in
[the general-noise derivation](general_noise_math.md). In particular, the
decoder is fixed, channel parameters are proportional to one p, and fault
responses must have the deterministic Pauli semantics validated by the parser.
An arbitrary Stim syntax extension is not automatically supported by this method.
The tests include Stim circuits and circuits generated through StabIR.

Automatic allocation prioritizes the largest remaining interval contribution
relative to the requested accuracy. Small strata are enumerated if their history
count fits the exact budget and is cheap compared with the next sampling batch.
This can remove the enormous cost of statistically bounding a low-weight stratum
with no failures. Enumeration still grows combinatorially; mixtures do not
eliminate failures that are themselves rare within a stratum. The bounds are
deliberately conservative. No finite universal sample count or universal speedup
over direct Monte Carlo is promised.

See [the validation results](accuracy_control_validation.md) for resolved cases,
unresolved cases, and comparisons against independent exact oracles and Stim.
