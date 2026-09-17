# Reusable profiles for a one-parameter, nonuniform noise family

Status: research implementation, September 2026. The decoder is fixed throughout
a profile. Weight counts nonidentity single-qubit Pauli factors at their original
noise locations: XI has weight 1; XX has weight 2. Errors at different times are
counted separately, even when their effects cancel.

## Why one conditional LER per weight is insufficient

Let independent noise location j have outcomes a, with probabilities

    P_j(I; p) = 1 - c_j p,
    P_j(a; p) = c_ja p,     c_j = sum_{a != I} c_ja.

The valid interval is 0 <= p <= min_j 1/c_j. Write F(h) for failure of the
fixed decoder on fault history h, W(h) for its Pauli weight, and K(h) for the
number of activated channels. Then

    P_p(h) = p^K(h) product_active c_ja product_inactive (1-c_j p).

K is likelihood bookkeeping, **not the stratification weight**.

Even for single-qubit errors, q_w(p) = P_p(F=1 | W=w) need not be constant.
For two independent X errors of rates p and 2p, with failure defined by the
first qubit's flip,

    q_1(p) = (1-2p)/(3-4p).

It equals about 0.331 at p=0.01 and 0.273 at p=0.2. Reweighting a single
stored q_1 by the weight probability therefore gives the wrong answer.

DEPOLARIZE2 makes the obstruction stronger. Its weight generating polynomial is

    g(z;p) = (1-cp) + (2/5)cp z + (3/5)cp z^2.

There are six weight-one and nine weight-two outcomes among the 15 nonidentity
Paulis. Weight two can be caused by one XX channel outcome, of order p, or by
two independent single-qubit errors, of order p^2. Equal Pauli weight does not
imply equal likelihood. The uniform DEPOLARIZE1 case is special: K=W and every
history in a stratum has the same p dependence.

## Exact reweighting within Pauli-weight strata

Choose an interior reference p0 and let

    Z_w(p0) = P_p0(W=w),
    Q_w(h) = P_p0(h | W=w),
    R_p(h) = P_p(h)/P_p0(h).

Provided the proposal covers every target history,

    LER(p) = sum_w Z_w(p0) E_{Q_w}[ F(h) R_p(h) ].                 (1)

Proof: expand each conditional expectation as a finite sum. Its denominator
Z_w(p0) cancels, and P_p0(h) cancels against the likelihood ratio. The remaining
terms are precisely sum_h F(h) P_p(h).

For fixed independent sample counts n_w, an unbiased estimate of (1) is

    Lhat(p) = sum_w Z_w(p0)/n_w sum_{i=1}^{n_w} F(h_wi) R_p(h_wi).

Its estimated sampling variance is sum_w Z_w(p0)^2 s_w(p)^2/n_w, where s_w^2
is the ordinary unbiased sample variance of F R in that stratum. Do not
self-normalize the likelihood ratios: that introduces finite-sample bias.
Zero observed failures and zero estimated variance do not prove zero LER.

This profile is a fixed *collection of weighted histories*, rather than one
scalar conditional failure probability per weight. Evaluation needs no new
circuit sampling or decoding. A compact sufficient record for each history is
its failure indicator, W, K, and the number M_g of inactive locations at each
distinct total rate coefficient c_g:

    log R_p = K log(p/p0)
              + sum_g M_g [log(1-c_g p)-log(1-c_g p0)].           (2)

For independent categorical channels the within-channel outcome coefficients
cancel. A different decoder or different noise coefficients generally requires
more information or a new profile. A profile does not magically support
arbitrary changes to the noise model.

Equivalently, the saved estimate is a fixed sum of terms
`A_h F(h) p^K(h) product_g (1-c_g p)^M_g(h)`, with p-independent A_h. This
does recover a reusable polynomial *estimator*. Expanding it into monomial
coefficients is unnecessary and can cause severe cancellation. It is not an
exact deterministic calculation of every coefficient of the true LER.

The implemented polynomial export groups all histories sharing `(W,K,M)`.
Let n_ws be the number of sampled histories in such a group and f_ws the
number of failures. A group contributes

    [Z_w(p0) f_ws / n_w] (p/p0)^K_s
        product_g [(1-c_g p)/(1-c_g p0)]^M_sg.

The factor in square brackets is a fixed estimated coefficient. Terms with
equal `(K,M)` can then be combined across Pauli weights. This compression does
not fit a curve or change the estimator. Retaining both n_ws and f_ws also
reproduces the original sample variance and importance-weight ESS. Thus the
weighted profile supplies an entire polynomial, not merely a table of p values.

There is a useful invariance behind this: two histories with the same `(W,K,M)`
have a probability ratio independent of p. Their relative outcome probabilities
and the fixed decoder's conditional failure probability within that refined
group are therefore p-independent. Weight alone lacks this property in the
general model. The histogram estimator above handles the random group counts
without assuming that every possible refined group has been observed.

The polynomial degree is at most the number of attempted noise trials, not the
Pauli weight. For independent categorical locations it is at most the number
of locations. The degree may be smaller after coefficient cancellation.

`GeneralNoiseProfile.to_polynomial()` exports this representation, callable on
scalar or array p. `power_coefficients()` and `to_sympy()` expose the polynomial
explicitly. The default degree guard on expansion prevents accidental costly
conversion; the factored representation supports higher degrees directly.

Values at different p share samples and are correlated. For p_a and p_b,
their estimated covariance is sum_w Z_w(p0)^2 times the sample covariance of
F R_pa and F R_pb in stratum w, divided by n_w. Threshold fits must account for
this dependence; points on one curve are not independent experiments.

## Exact conditional sampling, without a mixing assumption

For a factor with possible outcomes a and weights v_a, form
g_j(z;p) = sum_a P_j(a;p) z^v_a. Then Z_w(p) is coefficient w of product_j g_j.
A backward dynamic program computes

    H_j(t) = sum_a P_j(a;p0) H_{j+1}(t-v_a),
    H_J(0)=1, H_J(t!=0)=0.

At location j, given remaining weight t, draw outcome a with probability
P_j(a;p0) H_{j+1}(t-v_a)/H_j(t). The factors telescope to Q_w. This is an exact
conditional sampler up to floating-point arithmetic, with no Markov-chain
convergence assumption. Outcomes of the same weight may be grouped for speed.

If only strata S are sampled, the estimator is unbiased for their contribution.
The omitted contribution obeys the deterministic bound

    0 <= LER(p)-LER_S(p) <= P_p(W not in S).

Report this missing probability mass separately from sampling error. It is not
valid to call a truncated estimate unbiased for the full LER. No S-curve is
needed for this identity; extrapolation, if added, retains model uncertainty.

## Stim semantics and the weight convention

Keep mutually exclusive Pauli outcomes categorical. In particular, sampling a
detector error model as if its mechanisms were independent is not an exact
substitute for Stim circuit noise, especially for heralded channels.

An E/ELSE_CORRELATED_ERROR chain is one categorical factor: branch i has
probability c_i p product_{k<i}(1-c_k p), and no branch has probability
product_k(1-c_k p). Equation (2) still applies if M counts only attempted,
unsuccessful branches; skipped branches are not inactive trials. K is one for
a selected branch and zero otherwise. Branch Pauli support determines W.

HERALDED_PAULI_CHANNEL_1 can activate an identity outcome. Such an outcome has
K=1 and W=0. The herald record is part of the simulated circuit. Similarly,
Stim's measurement-probability argument flips a **classical record**, and
contains no inserted Pauli. Under the user's literal Pauli-factor definition
its weight is zero. Explicit X_ERROR before a Z measurement has weight one.
These descriptions can yield the same measurement statistics but different
weight profiles; this is expected because a profile includes its noise
representation. Weight zero is not assumed to have zero failure probability.

Stim's deterministic Clifford operations, record-controlled Paulis, annotations,
and repeat syntax can be delegated to Stim. The experimental implementation
uses a noiseless measurement-to-detector converter and forced channel outcomes
to construct response columns, checking deterministic ideal detectors and
observables. It does not substitute the old restricted QEPG parser. A StabIR
program enters through its existing compiled Stim circuit, preserving its
locations and chosen noise model.

Circuits without deterministic ideal detectors/logical parities do not define
the same QEC failure experiment and are rejected. Measurement-conditioned
noise outside Stim's supported operations is outside this model. Sweep bits
use Stim's default zero assignment. Repeated blocks are expanded; very large
circuits can exceed the memory budget. Unknown future noise instructions must
raise an error, never be silently discarded.

## Limits and verification requirements

An unbiased identity is not an efficiency guarantee. A proposal too far from
the target can have extreme likelihood ratios and poor effective sample size.
Use several reference proposals or narrower p ranges when diagnostics require
it; multiple-importance sampling is a possible subsequent improvement. Very
rare failures within a stratum can still be missed. Ordinary standard errors
are asymptotic estimates, not rigorous confidence bounds.

Required checks: exact enumeration of small nonuniform circuits; forced
multi-fault comparisons against Stim; exact stratum weights; boundary p and
zero-rate channels; categorical exclusivity and DEPOLARIZE2 support counts;
heralded/conditional semantics; fixed-decoder Monte Carlo comparisons; saved
profile round trips; and an independent native QEPG fault-response audit.

References: [ScaLER paper](https://arxiv.org/pdf/2602.04921),
[Stim gate reference](https://github.com/quantumlib/Stim/blob/main/doc/gates.md),
[Owen, Monte Carlo, importance sampling](https://artowen.su.domains/mc/Ch-var-is.pdf).
