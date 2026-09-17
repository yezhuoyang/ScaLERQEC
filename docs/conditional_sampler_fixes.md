# Conditional sampler and allocation fixes

The demonstrated conditional-probability underflow and p-grid starvation bugs
are fixed. This does **not** establish 10% accuracy for every large-code run.
The implementation remains experimental and has not been published to PyPI.

## Changes

- Conditional suffix probabilities and proposal normalizers stay in log space.
  A representable likelihood ratio no longer requires its numerator and
  denominator to survive separate exponentiation. Fixed profiles, reloads,
  confidence bounds, and polynomial export preserve these log normalizers.
- Sampling skips identity runs by inverting their exact conditional survival
  function. Measurement flips and heralded identity outcomes still count as
  nonzero categorical outcomes even though their Pauli weight is zero. Fault
  responses are bit-packed during propagation. The definition of weight has
  not changed: XX contributes two Pauli factors at its original location.
- The controller rotates among unresolved p targets. An initial sampling pass
  also accounts for the **combined** mass of unsampled weights, preventing many
  individually small weights from leaving a substantially low partial curve.
  The confidence bounds, fixed mixture proposals, and doubling checkpoints
  retain their statistical meaning.
- Response compilation checks log support, so a mathematically possible ELSE
  outcome is not discarded because its ordinary probability underflows.
- The independent replay audit also checks log support. The intermediate audit
  predicate could otherwise skip precisely the underflowed stratum being tested.
  Replay counts are now recorded separately for each weight.
- A legacy symbolic test used unseeded adaptive MC with about twenty failures,
  then converted differences above 25% into an expected failure. It now checks
  an independent exact distribution obtained from Stim fault responses, followed
  by seeded, fixed-budget MC and a binomial test. The exact check is not waived.

The sampling identity and its computational costs are derived in
[general_noise_math.md](general_noise_math.md#stable-conditional-sampling-and-automatic-allocation).

## Distance-13 result

Same d13 Z-memory circuit, nonuniform single/two-qubit depolarization, fixed
PyMatching decoder, p grid `[0.001, 0.003, 0.01]`, requested relative error 10%,
confidence 99%, and a five-minute profiling budget:

| Run | Random samples | Profile seconds | Estimate at p=0.01 | Certified 10% on the grid? |
|---|---:|---:|---:|---|
| Preserved baseline | 13,568 | 306.7 | 0 | No |
| Log sampling and target rotation, before broader initial coverage | 801,536 | 301.0 | 0.0789853 | No |
| With broader initial coverage | 705,792 | 301.5 | 0.0897092 | No |
| Independent historical Stim MC at this p | 100,000 | 69.0 | 0.0900400 | Reference only |

The MC reference observed 9,004 failures and its matrix-adjusted confidence
interval is `[0.0864465, 0.0937245]`. The revised point estimate differs by
about **0.37%** and lies inside that interval. The final profiler interval is
still `[0, 0.430301]`: this is point agreement, **not** a 10% accuracy certificate.
The low-p points also remain unresolved. The polynomial is an estimated
sampled-weight contribution, and export still requires explicitly allowing an
unconverged result.

The revised run collected approximately **52 times** as many samples under a
similar time cap. This measures observed progress on this circuit, not a
controlled speedup benchmark: local concurrent workloads differed. Compilation,
audits, and decoding costs are recorded separately. It does not establish an
advantage over direct MC across the whole p grid.

The previous two-qubit reweighting counterexample remains a regression:
`L(p) = 38 p / 15 - 32 p^2 / 15`, so `L(0.001) = 0.0025312`.

## Validation scope

The follow-up matrix contains 11 configurations: d13 surface Z/nonuniform and
X/biased noise; d13 StabIR SD6 and SI1000; HGP58/correlated, HGP180/nonuniform,
HGP245/biased; color d7/nonuniform; BB72/nonuniform; and BB144 with nonuniform
and biased noise. See the
[complete rerun results](../experiment_results/conditional_fix_validation/results.md)
and [machine-readable summary](../experiment_results/conditional_fix_validation/summary.json).
The preserved [56-case baseline](large_code_validation.md) supplies the broader
size/noise matrix and independently sampled MC references. Circuit hashes and
fixed-decoder descriptions are checked before those references are reused.

New sampler regressions test complete small-circuit conditional distributions
at four p values, log-binomial normalizers and uniform subsets for 1,070 and
3,000 locations, actual K/miss counts, zero-weight outcomes, ELSE branches,
fixed-profile serialization, p-grid coverage, and RNG endpoints. The large-code
audits replay actual sampled histories in independent Stim circuits. Separate
moment checks compare draws with an independent dynamic program at the
predeclared seven-standard-error threshold; these are selected-moment checks,
not a proof of every joint distribution.

The final full suite reports **938 passed and two existing skips**. A separate
Stim 1.16 run passed 99 selected regressions. Coverage is 96.8% for the accuracy
controller, 97.3% for the conditional sampler, 97.4% for the general-noise
estimator, and 100% for confidence bounds and polynomial export. All 13 saved
polynomial files reload and reproduce their recorded grid values.

The exact low-weight BB144 histories must be checked as well as the typical
weights: root probabilities that exponentiate to zero are never treated as
impossible. The final records list per-weight replay counts. An intermediate
run that checked only typical weight is retained in `before_log_audit_fix` and
is **not** counted as an underflow regression pass. Its third direct-MC point
was stopped after more than fifteen minutes of total case time; the first two
completed fixed-budget MC points remain available.

BB fixtures expose one defined logical observable, not full-block LER. The HGP
fixtures use ideal syndrome measurements; they do not establish circuit-level
QLDPC performance or independently prove code distance. Passing any finite
matrix does not establish universal correctness.

## Remaining limits

Only **one of the 11** rerun profiles met the requested grid accuracy within
its budget: HGP58 with correlated-ELSE noise. The other ten remain unresolved.

Suffix tables still cost O(number of factors times weight cutoff) memory for
each proposal. The BB144 circuits have 83,520 factors and still encounter
profiling time or table-memory limits. Their conditional sampler can now be
audited at formerly underflowed weights; this is separate from obtaining a
useful automatically certified LER curve.

The per-stratum simultaneous confidence bounds can be very conservative when
many strata matter. More samples do not automatically establish relative
accuracy for extremely rare failures. Unresolved results are explicitly marked
`budget_exhausted`, and default polynomial export rejects them. Efficient large
QLDPC profiling and tighter rigorously justified aggregate bounds remain work
to do before a general-performance claim or release.

## Reproduction

```text
python -m pytest tests
python -m benchmark.conditional_fix_validation surface_13_z_nonuniform_depolarizing --seconds 300 --output experiment_results/conditional_rerun
python -m benchmark.conditional_fix_validation bicycle_144_z_nonuniform_depolarizing --seconds 300 --skip-mc --output experiment_results/conditional_rerun
python -m benchmark.summarize_conditional_fixes --output experiment_results/conditional_rerun
```

The benchmark runner records one case per invocation; run the remaining names
from the result table before summarizing, since the summary requires all
11 listed cases to have completed. The runner refuses to overwrite an existing
case; select a fresh directory with `--output`. Full test results, module coverage, dependency versions, source
hashes, and additional verification details are in
[verification.json](../experiment_results/conditional_fix_validation/verification.json).
