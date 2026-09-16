# Reusable LER profiles and reliability plan

## Scientific contract

For N independent, uniform single-qubit depolarizing locations and a fixed
decoder D, define q[w] = Pr(D fails | exactly w nonidentity Pauli faults).
Then LER(p) = sum(q[w] * binom(N,w) * p**w * (1-p)**(N-w), w=0..N).
The spectrum q is independent of p; a decoder rebuilt or retuned at each p
need not have the same spectrum. The fitted spectrum is an approximation,
not an exact symbolic polynomial. S-curve extrapolation has systematic error
that more samples, high R-squared, or a narrow sampling interval cannot rule out.

Reference: https://arxiv.org/pdf/2602.04921, equations 2-4, algorithms 1-2,
and section 9. Preserve these restrictions in the public interface.

## Sequence and acceptance criteria

1. Establish a local build and test baseline; retain pre-existing user edits.
2. Before implementation, enumerate a tiny SID circuit and check its complete
   conditional spectrum against an analytic answer and seeded Stim trials at
   several p values, using one fixed decoder.
3. Add an independently evaluable, versioned JSON profile, scalar/vector p
   evaluation, plotting, provenance, and measured/model contribution diagnostics.
   Evaluate the full binomial support with bounded temporary memory: no silent
   Gaussian tail truncation, no refitting or resampling during evaluation.
4. Add profile-once entry points while retaining the single-p API. Fail clearly
   for unsupported circuit/noise/observable combinations. Record decoder
   reference p and the circuit-level distance assumption.
5. Audit important Python and C++ paths. Prioritize reproducible correctness,
   memory safety, parser consistency, numerical reliability, and avoidable
   allocations. Add regression tests for each fixed defect, exact small-circuit
   cross-checks, and sampler distribution/boundary tests.
6. Run the complete existing suite, focused new-module coverage, native checks,
   an end-to-end surface-code sweep versus Stim, and a sweep performance check.
7. Build source and wheel artifacts, verify metadata and clean installation,
   update examples/release notes and enforce tests in wheel CI. Publish to PyPI
   using available authorized credentials or the repository's trusted publisher.
   Report an explicit external blocker if publishing is unavailable.

## Limits of this release

Do not claim that arbitrary Stim instructions, noise models, decoder retuning,
or multi-observable block error rates are supported by the current SID backend.
The S-curve's 0.5 asymptote is not a universal high-p guarantee. General nonuniform
and correlated noise requires a different mathematical weighting scheme.
No finite audit can certify that every module is bug-free or fully optimized;
record concrete validation and remaining scope instead.
