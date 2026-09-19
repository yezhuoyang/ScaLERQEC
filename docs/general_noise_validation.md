# General-noise feasibility and correctness report

Local research validation, September 16, 2026. The code is on branch
`codex/general-noise-profiling`; this experimental API has not been published.

## Mathematical conclusion

The desired reuse is possible with Pauli weight as the stratum, but a single
conditional failure probability per weight is insufficient. Within-stratum
history probabilities change with p. Store each sampled history's failure
indicator and compact likelihood counts, then apply unnormalized stratified
importance sampling. [Full derivation and counterexamples](general_noise_math.md).

For the small repetition example below, direct enumeration also gives

    LER(p) = 4a/15 + (8a/15)(2b/3),    a=0.8p, b=0.2p.

The order-p term comes from a single two-qubit channel causing two physical
Pauli errors. Treating physical weight two as automatically order p^2 would
miss it.

A separate exact counterexample makes the failure of weight-only reweighting
visible: DEPOLARIZE2(p) on qubits 0 and 1, followed by X_ERROR(2p) on qubit 0
and X_ERROR(3p) on qubit 1, with failure defined by the qubit-0 measurement.
The exact LER is `(38p-32p^2)/15`. Reusing only the conditional weight LERs
measured at p0=0.2 gives the following at p=0.001:

| Method | LER |
|---|---:|
| Exact enumeration | 0.00253120 |
| Weight-only reweighting | 0.00219230 (**13.4% too low**) |
| History-likelihood profile | 0.00253659 ± 0.00003486 SE |
| Actual Stim Monte Carlo, one million shots | 0.00253700 ± 0.00005030 SE |

This example was also checked at p=0.01 and 0.1. It specifically demonstrates
why retaining additional likelihood statistics is necessary.

## Numerical comparison against actual Stim circuit sampling

Every case used one profile at p0=0.02 and one fixed decoder for its entire p
sweep. Every comparison point used 1,000,000 independent Stim shots. The
surface-code decoder was PyMatching constructed at p=0.005. No decoder was
retuned during the sweep and no S-curve was fitted.

| Circuit and noise | Profile trials | Tested p | Largest difference / combined SE |
|---|---:|---|---:|
| Three-qubit repetition; DEPOLARIZE2(0.8p) on a pair and DEPOLARIZE1(0.2p) on the third qubit | 48,000 | 0.0001–0.02, five points | 1.42 |
| Stim rotated surface code, d=3, r=3; gate depolarization p, round data depolarization 0.2p, readout flips 2p, reset flips 0.5p | 200,000 | 0.001–0.02, five points | 1.27 |
| StabIR surface code, d=3, r=2; SI1000-style noise | 216,000 | 0.001–0.02, five points | 1.42 |

At p=0.005, the two surface-code comparisons were:

| Case | Reused profile LER ± 1 SE | Stim Monte Carlo LER ± 1 SE |
|---|---:|---:|
| Stim surface | 0.0143553 ± 0.0003787 | 0.0140650 ± 0.0001178 |
| StabIR surface | 0.0320642 ± 0.0007764 | 0.0317270 ± 0.0001753 |

The largest omitted-weight probability mass across each sweep was about
1.85e-8 for the Stim surface profile and 8.74e-11 for the StabIR profile.
The repetition profile covered all weights. These tail bounds are separate
from sampling uncertainty. All 18 comparisons, including the three additional
counterexample points above, were within 1.43 combined
estimated standard errors. This is supporting numerical evidence, not a proof
of universal accuracy or calibrated confidence intervals.

The final local profile sampling times were 0.024 s, 7.15 s, and 5.51 s respectively,
excluding response compilation. Evaluating the five points from saved profiles
took 0.008 s, 0.051 s, and 0.081 s. Response compilation took 0.47 s and 0.39 s
for the two surface cases. Timings are observations from this Windows machine,
not portable performance promises. Stim Monte Carlo was faster for these small
surface examples: the prototype establishes correctness and reuse, not a
universal speedup. Reweighting cost and proposal quality still need optimization.

Full values and timings: `experiment_results/general_noise_validation/results.json`.
Figure: `experiment_results/general_noise_validation/comparison.png`.
Reproduction: [benchmark/general_noise_validation.py](../benchmark/general_noise_validation.py).

## Confirmed implementation defects fixed

* The legacy nonuniform mapper attached DEPOLARIZE2 only to a later matching
  CX/CZ pair. Errors before measurement or other intervening operations could
  disappear. Both operands now map independently to their next primitive
  source, with exact marginalization if a reset discards an operand.
* Repeated two-qubit channels were added as probabilities. They now remain
  independent categorical events, preserving cancellation and correlation.
* Pending noise before CZ could be placed after the target's basis change.
  Mapping now follows the primitive decomposition and preserves the noise's
  original effect.
* Unsupported legacy noise syntax could silently disappear. Unsupported cases
  now raise explicitly; PAULI_CHANNEL_1 mapping was added.
* X_ERROR was used for X-basis reset/readout, where it commutes with the basis
  and can have no effect. Those operations now use Z_ERROR.
* Gate-specific disable flags were ignored for several gate types. They now
  control the requested operation, including CZ separately from CX.
* Coalesced overlapping two-qubit gates received noise after the entire group.
  Noise is now interleaved with individual gate operations.
* Measure-reset operations now receive the reset fault as well as readout noise.
* SI1000's configured idle rate was unused. It now applies to inactive circuit
  qubits in each completed TICK-delimited layer; the interval convention is
  documented.
* The native matrix-returning API printed entire circuits and matrices, and the
  history-returning API unnecessarily copied nested vectors. Removed the
  unconditional output and copies; rebuilt the extension before testing.
* The new estimator also has regressions for two numerical pitfalls: loss of
  tiny tail probabilities through subtraction from one, and overflow of a raw
  likelihood ratio whose probability-weighted contribution is finite. Tails are
  accumulated directly and likelihood moments are evaluated with log scaling.

## Verification scope

The final complete test suite passed **708 tests with two existing skips**.
The detailed results are recorded in
`experiment_results/general_noise_validation/test-output.txt`.
The new estimator has 96% line coverage, and CI now requires at least 95% for
this module. A separately rebuilt native Debug test executable passed its
bitset, matrix, and RNG checks. All **144 new tests** also passed under Stim 1.16
(the complete suite used Stim 1.15). Changed-file lint and workflow validation
passed. These are local Windows results; the updated cross-platform CI has not
been run remotely for this branch.

The new native audit compares every X/Y/Z propagation column of 12 seeded
circuits with forced Stim faults, for both Python and C++. It then checks 960
returned native histories against Stim at weights 0, 1, 2, half the locations,
and all locations. A further 60,000 trials check all 27 weight-two configurations
of a three-location circuit for the correct uniform distribution.

The general-noise tests include exact enumeration, the analytic DEPOLARIZE2
example, conditional-likelihood normalization, categorical and correlated
channels, herald records, basis measurements, MPP, record feedback, multiple
observables, the installed unitary gate catalog, StabIR integration, boundary
parameters, malformed profile data, saved-profile reuse, and tiny positive
truncation tails. Regression tests exercise both native and Python nonuniform
backends, including the complete joint detector/logical distribution.

No finite test suite establishes that all Python and C++ code is bug-free.
The new checks provide direct evidence about the subspace outputs, rather than
only agreement of average LER estimates. The general-noise estimator remains
experimental pending broader code-distance studies, proposal optimization,
and review of its statistical diagnostics.
