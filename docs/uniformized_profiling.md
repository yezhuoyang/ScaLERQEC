# A fixed polynomial for general linear Pauli noise

This is an experimental, exact-in-expectation construction, implemented in
`scalerqec.Stratified.uniformized`. It removes the conditional-weight dynamic
program and the fitted S-curve. It does **not** promise a speedup for every
code, decoder, noise family, or requested accuracy. Sampling uncertainty
remains, and an exhausted budget is never reported as an accuracy certificate.

## Model and weight convention

Fix the circuit, its noise coefficients, and the decoder. Each categorical
Pauli channel has outcome probabilities `c_a p`, with total activation rate
`c = sum_a c_a`; conditional on activation, its mark has probability `c_a/c`.
DEPOLARIZE2 is one such channel with 15 marks. Each E/ELSE branch instead has
its own conditional activation probability `c_r p`. Pre-sample all branch
trials independently and use the first successful branch. Sampling unused
later branches does not change the physical history distribution.

Measurement flips, heralded identity outcomes, and other supported classical
noise retain their original semantics. The physical Pauli weight W counts
nonidentity Pauli factors at their original locations: XI has weight 1, XX
has weight 2, and faults at different times count separately even if their
effects cancel. Measurement flips and heralded identity outcomes have W=0.

Let the M primitive trials be the positive-rate categorical activations and
positive-rate conditional branches. Zero-rate trials can be omitted. Set
`C = max_r c_r`. The physical domain is the model's domain, contained in
`0 <= p <= 1/C`. Identity annotations may further restrict that domain.
For M=0, the polynomial is constant.

## Uniformization and the polynomial identity

Replace each primitive trial by independent variables

    B_r ~ Bernoulli(C p), A_r ~ Bernoulli(c_r/C), U_r = B_r A_r.

Then `P(U_r=1)=c_r p`, and the U_r remain independent. Categorical marks,
first-success branch selection, and Clifford propagation map these variables
to exactly the original physical history H. This mapping is independent of p.

Let T=sum_r B_r. Conditional on T=t, the selected B locations are a uniform
t-subset of the M trials; thinning and marking remain independent of p. Define

    b_t     = P(decoder failure | T=t),
    a_{w,t} = P(decoder failure AND W=w | T=t).

These are fixed coefficients, with `b_t=sum_w a_{w,t}`. By conditioning,

    L(p)   = sum_{t=0}^M b_t binom(M,t) (C p)^t (1-C p)^(M-t),
    L_w(p) = sum_{t=0}^M a_{w,t} binom(M,t) (C p)^t (1-C p)^(M-t).

This is a Bernstein-form polynomial, not an interpolation or S-curve fit.
Its degree is at most M. The coefficients exist independently of the sampler;
their estimates remain statistical. A decoder retuned with p changes the
failure function and generally invalidates reuse of this single polynomial.

T is an **auxiliary** count, not a new definition of Pauli weight. Thinned
trials, unused ELSE trials, and weight-zero errors mean T can exceed W;
two-qubit and correlated Paulis mean W can exceed T. The implementation stores
the joint (T,W) profile. For uniform single-qubit depolarization, T=W and this
reduces to ordinary weight profiling. For general noise, a scalar conditional
failure probability indexed only by W is still insufficient.

A minimal example shows why the extra coordinate is necessary. Put independent
X faults of probabilities p and 2p on two qubits, and let failure mean that the
second qubit is flipped. Then
`P(failure | W=1) = 2(1-p)/(3-4p)`, which depends on p. In contrast, with
`C=2, M=2`, the latent coefficients are exactly `(b_0,b_1,b_2)=(0,1/2,1)`.
They give `L(p)=2p`. The joint polynomials retain the requested weight:
`L_1(p)=2p(1-p)` and `L_2(p)=2p^2`. This is an augmentation of the physical
weight profile, not an assumption that its conditional failure rate is fixed.

Example: one DEPOLARIZE2(p), with the Z measurement of its second qubit as the
logical observable and a zero decoder, has `M=C=1`, `b_0=0`, `b_1=8/15`.
Its failure coefficient splits into `a_{1,1}=2/15` and `a_{2,1}=6/15`.
An E(a p) Z followed by ELSE(b p) X gives an X-failure probability
`b p (1-a p)`; treating the ELSE branch as an independent applied fault would
give the wrong polynomial. The sampler explicitly selects only the first
accepted branch.

## One sampling run, full-support reweighting

Write `B_t(p)=binom(M,t)(Cp)^t(1-Cp)^(M-t)`. Preselect p anchors p_j including
all requested targets and the model's interior reference p. Use a fixed equal
mixture

    rho(t) = (1/J) sum_j B_t(p_j).

Draw T from rho, draw its uniform subset and marks, and decode once. For
every target p, `Y_p = F B_T(p)/rho(T)` has mean L(p). Full support comes from
the interior reference component. There is no unaccounted weight tail.
For a fixed sample count N, the exported polynomial is

    Lhat(p) = sum_t [failures_t / (N rho(t))] B_t(p).

Replacing failures_t by failures_{t,w} gives the joint-weight contribution.
The stored coefficients may exceed 1: these are importance estimates, not
empirical conditional Bernoulli frequencies. Do not clip them. The positive
factored representation avoids expanding high-degree alternating power
coefficients. Exported polynomials can be evaluated at unrequested p values,
but those values do not acquire the finite-grid accuracy certificate.

## Sequential accuracy control

For a requested target present in the mixture, `0 <= Y_p <= J`, because
`rho(t) >= B_t(p)/J`. Thus `X_p=Y_p/J` lies in [0,1]. The following bounded-mean
Chernoff argument includes unobserved failures; it does not use a normal
approximation or assume X_p is Bernoulli:

1. Convexity gives `exp(lambda*x) <= 1-x+x*exp(lambda)` for x in [0,1].
2. For IID X with mean mu, the product MGF is bounded by that of N independent
   Bernoulli(mu) trials.
3. Optimizing Chernoff's inequality gives either one-sided probability at
   most `exp(-N kl(xbar,mu))`.
4. Invert `N kl(xbar,mu) <= log(2/delta)` to obtain a two-sided interval.

The implementation also calculates the unbiased sample variance of X and
uses the two-sided empirical Bernstein radius
`sqrt(2 s^2 log(4/delta)/N) + 7 log(4/delta)/(3(N-1))`, from
[Maurer and Pontil, Theorem 4](https://arxiv.org/abs/0907.3740).
It splits each checkpoint's error budget equally between this bound and the
KL bound before intersecting them; selecting the smaller bound without that
split would not justify the claimed coverage. Centered histogram sums compute
the variance without subtracting nearly equal raw moments.

At deterministic batch checkpoint k=0,1,... and grid coordinate j, allocate
`delta_{j,k}=(1-confidence)/[G(k+1)(k+2)]`. Summing over k and G targets gives
at most `1-confidence`. Within each checkpoint, use the split described above.
Intersect the intervals across checkpoints. Stopping
when the requested accuracy is achieved therefore preserves simultaneous
coverage, under the stated model and IID assumptions. If the point estimate
is v and interval is [l,u], stop only when

    max(v-l,u-v) <= absolute_error + relative_error*l.

On the coverage event this bounds the error relative to the true LER. Zero
observed failures still give a positive upper bound. The point estimator is
unbiased at fixed N; optional stopping does not preserve that unbiasedness.
The certificate is statistical, for a finite grid, not an exact-arithmetic
proof or a confidence band over every real p. Floating-point arithmetic and
finite-state pseudorandom sampling retain their usual numerical limitations.

For a fixed finite target grid with positive LERs, bounded observations obey
the strong law and these interval widths tend to zero as the sample count
increases (`log(k)/N -> 0`). On the simultaneous coverage event, the requested
positive relative tolerance will therefore eventually be met if resources are
unlimited. If a target's true LER is zero, a positive absolute tolerance or a
separate exact zero-failure proof is needed to guarantee termination. This is
a convergence result; it does not supply a small universal sample bound.

The defensive-mixture bound is standard; see
[He and Owen, optimal mixture weights](https://artowen.su.domains/reports/optwtsmis.pdf).
Bounded-mean MGF methods and stronger sequential alternatives are discussed by
[Waudby-Smith and Ramdas](https://arxiv.org/abs/2010.09686).
The elementary proof above specifies the bound actually implemented.

## Memory and efficiency limits

Conditional T sampling selects a uniform subset, thins, marks, and propagates
sparse fault responses. It requires no M-by-weight suffix table. The sampler
stores linear-size trial metadata and sparse response columns, and bounds
temporary event gathers. The current response compiler still stores dense
fault response arrays; this is a separate large-circuit memory cost. Decoding
can dominate runtime, especially for large BP+OSD examples. Time limits are
cooperative between batches, not process deadlines. Predeclared batch sizes
start at 1 and double up to the requested maximum; this avoids an expensive
first large decoder call without making sample sizes data-dependent. One
decoder call or response compilation can still exceed the wall-clock budget.

This proposal includes ordinary Monte Carlo as a component and bounds its
importance weights. It does not automatically find rare failure patterns.
In the worst case, `Var(Y_p) <= J L(p)-L(p)^2`; consequently the method can
require work comparable to Monte Carlo. Unequal rates can add many null
latent trials. One should compare time to a *certified* accuracy target, not
just throughput, observed failure counts, or plausible point estimates.

A universally cheap relative-error estimator for an arbitrary black-box
failure function is impossible without additional structure. For ordinary
IID sampling, the chance of missing an event of probability epsilon is
`(1-epsilon)^N`; distinguishing it from zero requires order
`log(1/delta)/epsilon` samples. More generally, an arbitrary failure marked at
one unknown history among H equally likely histories requires order H oracle
queries to find in the worst case, even with adaptive proposals. Importance
sampling helps when the proposal exploits failure structure; uniformization
does not prove such structure for every decoder.

## Interface

```python
from scalerqec.Stratified import LinearNoiseModel

model = LinearNoiseModel(stim_circuit, reference_p=0.01)
# Or LinearNoiseModel.from_stabcode(code, reference_p=0.01).
result = model.sample_bernstein_profile(
    fixed_decoder, [0.001, 0.003, 0.01],
    relative_error=0.1, confidence=0.99,
    max_shots=1_000_000, max_seconds=120, seed=123,
)
print(result.status, result.estimates)
if result.converged:
    polynomial = result.to_polynomial()
    polynomial.save("ler_polynomial.npz")
    print(polynomial([0.001, 0.002, 0.003, 0.01]))
```

`to_polynomial(allow_unconverged=True)` permits exploratory export with explicit
uncertified metadata. `weight_polynomial(w)` returns the estimated joint
failure contribution at the original Pauli weight; it does not claim a
separate per-weight accuracy guarantee. `UniformizedSampler.sample` exposes
conditional T sampling and optional original factor outcomes for independent
auditing.

This shares `LinearNoiseModel`'s existing Stim parser and exact binary response
compiler. It supports that parser's linear categorical Pauli, correlated/ELSE,
heralded, and measurement noise, and deterministic ideal detector/observable
circuits, including StabIR circuits compiled to Stim. It does not claim every
future Stim noise instruction, nonlinear p dependence, coherent non-Pauli
noise, or arbitrary nondeterministic ideal observables. Unsupported cases
must be rejected explicitly instead of assigned an approximate noise law.

The implementation also rejects positive categorical marks whose cumulative
probabilities lose floating-point support, and an excessively large p-grid
likelihood table. These are numerical/resource limits, not changes to the
mathematical identity. Current measurements, including unresolved cases, are
in the [validation report](../experiment_results/uniformized_final_validation/results.md).
