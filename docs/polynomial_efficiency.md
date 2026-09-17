# Polynomial profiles: correctness and measured efficiency

Follow-up: the [broader validation](general_noise_broad_validation.md) tests
additional codes and sizes and finds both substantial rare-event benefits and
circuit-level slowdowns. It also identifies overconfident sample standard
errors when rare failures inside a weight stratum are unobserved. The timings
below establish reuse speed, not a general advantage over direct Monte Carlo.

This follow-up makes the general-noise profile an explicit reusable polynomial
and speeds up both profiling and evaluation. Pauli weight is unchanged. There
is no fit to a grid of p values and no S-curve extrapolation in this estimator.

## What the profile now produces

For each distinct likelihood record s, the profile stores its Pauli weight,
number of activated channels K_s, inactive-rate counts M_sg, occurrence count,
and failure count. Combining records gives the fixed polynomial estimate

    Lhat(p) = sum_s A_s (p/p0)^K_s
                   product_g [(1-c_g p)/(1-c_g p0)]^M_sg.

A_s is fixed by the profiling experiment; it does not change when p changes.
The polynomial degree counts probability factors, not Pauli weight. Its
coefficients are Monte Carlo estimates of the sampled-weight contribution.
The original profile retains the sampling SE and omitted-weight bound.

```python
from scalerqec.Stratified import GeneralNoiseProfile, LERPolynomial

profile = GeneralNoiseProfile.load("surface_profile.npz")
polynomial = profile.to_polynomial()

# A symbolic polynomial, not a table of evaluated points:
print(polynomial.to_sympy())
polynomial.save("surface_polynomial.npz")

loaded = LERPolynomial.load("surface_polynomial.npz")
ler_values = loaded([0.001, 0.002, 0.005])
estimates_with_uncertainty = profile.curve([0.001, 0.002, 0.005])

# Ordinary powers, for small or explicitly opted-in expansions:
if polynomial.degree <= 100:
    coefficients = polynomial.power_coefficients(precision=50)
    print(polynomial.to_sympy(expanded=True))
```

The factored form is preferable numerically. Expanding a high-degree polynomial
can create enormous coefficients whose contributions nearly cancel. The
coefficient export uses Decimal arithmetic and a default degree guard, but
additional digits cannot remove statistical uncertainty or make ordinary
floating-point evaluation of an ill-conditioned expansion safe.

The existing `Scaler` workflow is integrated too:
`scaler.get_profile().to_polynomial()` exports its uniform-SID weighted spectrum.
In that case the polynomial is the original Bernstein sum
`sum_w q_w binom(N,w) p^w (1-p)^(N-w)`. Its fitted entries and extrapolation bias
are preserved in the export's provenance. General-noise profiling uses the
richer history likelihoods; the two noise assumptions are not interchangeable.

## The 13.4% error is corrected

The value 0.00219230 came from an intentionally tested **unsupported** formula:
reuse one conditional failure probability per weight from p0=0.2, then multiply
by the new weight masses. Nonuniform channels make that formula biased.

The supported profile and its polynomial export retain the missing likelihood
information. For the same 50,000 profiling trials, their common polynomial is

    Lhat(p) = 2.53869333333333 p
            - 2.10493333333333 p^2
            - 0.120386666666668 p^3.

The exact toy-circuit polynomial is

    L(p) = (38/15) p - (32/15) p^2.

At p=0.001:

| Method | LER |
|---|---:|
| Unsupported weight-only formula | 0.00219230081 |
| Corrected exported polynomial | 0.00253658828 |
| Exact polynomial | 0.00253120000 |

The corrected estimate is 0.21% above truth, with estimated SE 0.0000348602
(about 1.38% of truth). Its discrepancy is only 0.155 SE. The estimated cubic
term illustrates coefficient uncertainty: the exact circuit's cubic term
cancels, while finite sampling does not force that cancellation. Tests compare
the polynomial against the profile throughout the valid interval, and against
exact enumeration at six p values including both very low noise and the upper
domain boundary. We did not adjust coefficients to force agreement with the
known answer.

Old profiles containing only one scalar failure probability per weight cannot
be repaired by a conversion formula: the likelihood information was discarded.
They require new profiling with the general-noise sampler. The new richer
profiles can be converted to polynomials without additional circuit trials.

## Measured reuse speed

These measurements use the same saved histories as the preceding validation,
on the local Windows machine, for 1,001 p values from 0.001 through 0.02.
The raw baseline repeats the previous history-by-history evaluator. The
compressed evaluator returns the same LER, SE, ESS, and omitted-weight bound.
Polynomial-only evaluation returns LER without those diagnostics.

| Profile | Raw histories with diagnostics | Compressed profile with diagnostics | Polynomial values only |
|---|---:|---:|---:|
| Stim surface, d=3, r=3 | 31.71 s | 1.65 s (**19.3x**) | 0.292 s |
| StabIR surface, d=3, r=2 | 32.57 s | 0.576 s (**56.5x**) | 0.0915 s |

The Stim surface's 200,000 histories compress to 6,378 likelihood records and
1,613 polynomial terms, with degree at most 197. The StabIR surface's 216,000
histories compress to 2,526 records and 625 terms, with degree at most 141.
The standalone polynomial files are about 15 KB and 7 KB, respectively. Export
itself took 0.015 s and 0.0044 s. Loading a polynomial does not reconstruct a
circuit, compile fault responses, or load a decoder.

Separately, caching conditional-sampler tables and updating only affected fault
responses reduced median sampling time for 63,000 d=3 surface trials from
5.44 s to 3.05 s (about 1.78x). This comparison used the Stim d=3, r=3
family at p0=0.02, 3,000 trials per weight from 0 through 20, and seed 42.
All generated weights, failure indicators,
activation counts, and miss counts were identical for the fixed seed. The
timing comparison is recorded in
`experiment_results/polynomial_efficiency/sampler_comparison.json`.

## End-to-end rare-event example

For the exactly solvable three-qubit repetition example with DEPOLARIZE2(0.8p)
and DEPOLARIZE1(0.2p), at p=0.0001 the true LER is 2.13339022e-5.

| Method | Trials | Wall time | Standard error |
|---|---:|---:|---:|
| Pauli-weight profile | 48,000 | 0.163 s | 2.18038e-7 (estimated) |
| Actual Stim Monte Carlo | 10,000,000 | 0.185 s | 1.46060e-6 (computed from known truth) |

The profile obtains about 1.02% relative SE versus 6.85% for direct Monte Carlo
at similar measured runtime. Matching the profile's estimated SE would require
about 449 million ordinary MC shots using the known Bernoulli variance. That
shot count is a calculation, not an experiment we ran. This small circuit is
a favorable example; it does not establish a universal performance advantage.

## What is still required for broad efficiency

* A good reference proposal over the requested p range. Poor overlap can cause
  high variance despite fast polynomial evaluation. Several fixed proposals
  are a possible next improvement; they must preserve a p-independent proposal
  if a reusable polynomial is required.
* Sampling effort concentrated on strata relevant to the requested accuracy.
  The current equal-per-weight allocation is conservative and can waste work.
* A treatment of unsampled weights. The current method gives a probability-mass
  bound. Extrapolating a weight spectrum is still a separate modeling choice
  with possible systematic error. High-weight samples alone do not determine
  unsampled low-weight failure probabilities.

The useful conclusion is narrower than “always faster”: profiling can produce
a complete estimated polynomial, and its reuse is now much cheaper without
changing the estimate. There is an actual rare-event speed/precision benefit
on the tested example. Larger-code and broader-range efficiency remain open
engineering and statistical work.

## Reproduction and checks

Run `python benchmark/general_noise_validation.py`, then
`python benchmark/polynomial_efficiency.py`. Raw numerical results and the
comparison figure are in `experiment_results/polynomial_efficiency/`.
The benchmark checks raw and compressed results before reporting timings.

The complete suite passed 729 tests with two existing skips after the main
implementation. The final uniform-SID export integration then passed all 68
profile/export tests. The 115 general-noise/polynomial tests also passed with
Stim 1.16. These checks used local Windows builds; remote CI has not been run
for this follow-up.

New tests cover the entire polynomial interval, symbolic and Decimal
coefficient export, endpoints, extremely small reference probabilities,
zero polynomials, serialization without a circuit/decoder, invalid inputs,
and equality of compressed and raw uncertainty calculations. The new
polynomial module has 100% line coverage; the general-noise module has 97%
in the focused test run. CI requires at least 95% for each.
