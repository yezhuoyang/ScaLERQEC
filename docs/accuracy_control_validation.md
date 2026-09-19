# Validation of automatic accuracy control

Local results, September 16, 2026. The new controller meets its requested
accuracy in all 42 small exact-oracle configurations and in 100 sampling-only
replicates. It also explicitly refuses to certify larger cases whose bounds
remain too wide. This validates useful cases and the stopping contract; it is
not a proof of a bug-free implementation, arbitrary-noise support, or universal
efficiency. [Method, API, and confidence proof](accuracy_control.md).

## Exact-oracle matrix

The same 42 configurations as the earlier uncertainty audit are used: five-qubit,
Steane, Shor, and repetition codes of sizes 3, 5, and 7, crossed with seven noise
families. Those families are uniform single-qubit depolarization, nonuniform
single-qubit Pauli channels, mixed single-/two-qubit depolarization, biased
two-qubit Pauli channels, correlated E/ELSE chains, heralded Pauli noise, and
readout noise. Reference p_0=0.05; each run requests a single polynomial accurate
at p=0.001,0.01,0.05 with **10% relative error and 99% simultaneous confidence**.

All 42 runs converged. All 126 reported intervals contained the independent
exact answer, and all 126 estimates had less than 10% actual relative error.
The largest actual error was 5.067%. This observed coverage rate is a regression
result, not an empirical proof that coverage is exactly 99%. Values from one
polynomial are correlated.

| Code | Noise families passed | Largest actual relative error across the grid | Random samples used, range over families | Exact histories checked, range |
|---|---:|---:|---:|---:|
| Five-qubit | 7/7 | 4.019% | 0–1,280 | 32–2,620 |
| Steane | 7/7 | 1.728% | 0–10,240 | 128–9,662 |
| Shor | 7/7 | 3.308% | 0–35,072 | 512–48,896 |
| Repetition 3 | 7/7 | 2.841% | 0 | 8–376 |
| Repetition 5 | 7/7 | 5.067% | 0–5,376 | 32–12,826 |
| Repetition 7 | 7/7 | 0.585% | 0–38,912 | 128–66,379 |

Resource caps were 200,000 random samples, 100,000 enumerated histories, and
eight seconds of controller time per case. Sampling allocation, exact-stratum
selection, and stopping were automatic. The independent oracle defines a fixed
lookup decoder at p_0; it is not supplied to the controller to assess accuracy.
The stopping rule uses only noise probabilities, sampled outcomes, exhaustive
small-stratum checks, and their bounds. Oracle truth is computed independently
using Pauli commutation and syndrome-state probability convolution. Some runs
use no random samples; they still bound unenumerated contributions rather than
requiring complete enumeration of the whole circuit.

### Previously missed rare failures

Both repetition-7 cases that previously had an approximately 79% underestimate
at p=0.001 are now resolved automatically. Their exact LER is
5.144735353694415e-13 (up to roundoff between equivalent models).

| Noise | New estimate | Reported interval at p=0.001 | Random samples | Exact histories |
|---|---:|---:|---:|---:|
| Nonuniform Pauli | 5.144734705684228e-13 | [5.144734705684233e-13, 5.144744076934234e-13] | 3,072 | 9,094 |
| Heralded Pauli | 5.144675507065506e-13 | [5.143808340712020e-13, 5.145945193125709e-13] | 15,872 | 45,320 |

The first estimate differs from its reported lower bound in its final floating
digits because polynomial evaluation and interval accumulation use different
summation orders. This is numerical roundoff, not uncertainty calibration.
Both runs fully enumerate weight three, revealing the rare failures the old
5,000-samples-per-weight profiles missed. They also satisfy the requested
accuracy at the other two p values. This improvement depends on affordable
exact checks of small strata; it does not imply that arbitrary large circuits
can be resolved with these sample counts.

The earlier two-qubit counterexample also has a regression: the controller
recovers L(p)=38p/15-32p^2/15, giving 0.0025312 at p=0.001 instead of the
incorrect 0.00219230 obtained by reusing scalar conditional weight failure rates.

## Entirely stochastic stopping

A separate test disables all exact enumeration and repeats the adaptive run
for 100 seeds. The circuit has measurement-record flip probability p, so its
independent exact LER is p and its Pauli weight is zero. The 11 requested p
values range from 0.02 to 0.30. Mixtures have fewer anchors than target values,
exercising the likelihood maximum bound for targets between anchors.

Each run requests absolute error 0.005 plus 15% relative error, at 99%
simultaneous confidence. All 100 runs converged, using 32,768–65,536 random
samples. All 1,100 reported intervals covered truth; all estimates met the
requested error criterion. The largest actual error was 0.219 times the allowed
error. This is intentionally a tractable sampling-only calibration example,
not evidence of a general rare-event speed advantage.

## Circuit-level comparison with actual Stim sampling

Seven selected cases cover nonuniform and biased noise, Stim and StabIR frontends,
repetition codes and surface codes up to distance seven. They use the same
fixed PyMatching decoder at p=0.003 as the previous matrix. Each controller run
requests 10% relative accuracy at 99% simultaneous confidence on
p=0.001,0.003,0.01, with an eight-second controller cap, a 200,000-random-sample cap,
and a separate 100,000-history exact cap.

| Case | Entire requested grid certified? | Random samples | Exact histories | Controller seconds |
|---|---|---:|---:|---:|
| Stim repetition 3, nonuniform depolarization | Yes | 151,552 | 7,042 | 4.68 |
| Stim repetition 5, biased Pauli | No: time limit | 38,144 | 67,951 | 12.32 |
| Stim rotated-Z surface 3, nonuniform depolarization | No: time limit | 77,312 | 660 | 8.14 |
| Stim rotated-Z surface 5, biased Pauli | No: time limit | 5,120 | 3,450 | 8.26 |
| Stim rotated-Z surface 7, nonuniform depolarization | No: time limit | 2,048 | 1 | 8.05 |
| StabIR surface 3, SD6 | No: time limit | 161,024 | 562 | 10.12 |
| StabIR repetition 3, SI1000 | Yes | 153,600 | 9,886 | 4.31 |

The time cap is cooperative: one complete work unit can overrun it. These times
exclude construction of the model and decoder. Including setup, the distance-7
surface run took 118.56 seconds; its bounds were still unusably wide. This is
an outstanding performance and variance problem, not a successful estimate.
Zero point estimates from unresolved runs must not be interpreted as zero LER.
For example, the surface-7 interval at p=0.001 was [0,0.2808], while Stim observed
22 failures in two million shots. Export is blocked by default for all five
unresolved runs. Some individual p values can pass while the full grid fails.

The initial comparison uses two million actual Stim shots at each of the 21
points, totaling **42 million shots**. Every profile interval overlaps the
corresponding 95% exact binomial interval for MC. Overlap is only a consistency
check: it does not establish 10% accuracy, especially when either interval is
wide. Two additional independent 20-million-shot runs check the successful
repetition-code cases at p=0.001 more precisely. Their counts and intervals are
stored as `mc_confirmation` in the raw results. Total actual MC work is
**82 million shots**; no extrapolated shot counts are included in that total.

| Successful case at p=0.001 | Profile estimate | Independent MC failures / shots | MC 95% interval |
|---|---:|---:|---:|
| Stim repetition 3 | 4.81741e-5 | 985 / 20,000,000 | [4.62221e-5, 5.24242e-5] |
| StabIR repetition 3 | 1.33611e-4 | 2,722 / 20,000,000 | [1.31035e-4, 1.41311e-4] |

Both profile estimates fall inside these independent MC intervals. The MC
intervals are pointwise comparisons; the controller's confidence statement
uses its own simultaneous bounds over the preselected grid.

These experiments are accuracy validation, not a matched-precision speed
benchmark. They do not overturn the earlier finding that the current Python
general-noise prototype loses to Stim MC on many circuit-level workloads.

## Regression tests and reproduction

The new automated tests cover exact enumeration against independent Pauli
oracles, complete-history mixture normalization and likelihood bounds, the two
rare-failure regressions, polynomial coefficients and serialization, stochastic
optional stopping across seeds, zero observed failures, weight-zero faults,
tail expansion, endpoints, tiny reference probabilities, invalid controls,
underflow rejection, time/memory caps, and refusal to export unresolved results.
A late-review regression also prevents a false zero certificate at p=1e-200:
repetition-3 has positive LER 4p^2/3-16p^3/27, below the floating-point range.
Both omitted-history and fully-enumerated polynomial underflow return
`numerical_limit`; a genuinely exhaustive proof of zero still succeeds.
The final full suite passes **861 tests**, with two pre-existing skips. All
**51 new tests** also pass with Stim 1.16 in the separate release environment.
The main environment uses Python 3.14.3, NumPy 2.4.4, Stim 1.15, and
PyMatching 2.4. New-module statement coverage is 97%; CI enforces at least 95%.
Coverage measures exercised lines, not mathematical correctness.

```sh
python -m benchmark.accuracy_control_validation
python -m pytest tests/test_stratified/test_adaptive.py
python -m pytest tests
```

The benchmark also accepts `--suite oracles`, `--suite repeated`, or
`--suite circuits`. Raw results are in
`experiment_results/accuracy_control/{oracles,repeated,circuits}.json`.
Time-limited runs can have different sample counts on another machine.
See [the tests](../tests/test_stratified/test_adaptive.py) and
[the benchmark](../benchmark/accuracy_control_validation.py).
