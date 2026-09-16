# Changelog

## 1.1.0

- Profile a circuit and fixed decoder once, then evaluate or plot LER over an
  arbitrary p grid without resampling. Save and reload versioned JSON profiles
  with raw counts, provenance, and model-contribution diagnostics.
- Integrate the full binomial support in bounded memory; remove the primary
  Scaler's integer-rounded five-sigma truncation.
- Normalize Stim repeats and grouped operations in Scaler. Reject unsupported
  noise and observable configurations instead of silently changing their meaning.
- Make plots optional, clear state between runs, retain the selected fitted
  model, and reject absent fits instead of exporting default parameters.
- Correct SID injection for coalesced operations and use a decomposed detector
  error model for the default PyMatching decoder.
- Correct repeated-measurement parity propagation, Y-axis decompositions,
  measurement poststates, and composition of mixed Pauli channels.
- Replace Poisson approximations in native Monte Carlo with exact binomial
  sampling and geometric thinning. Correct probability-one behavior and validate
  native batch sizes, probabilities, strided arrays, and record references.
- Write multi-weight samples directly into NumPy buffers. Use portable SIMD
  defaults, with optional local AVX2 via SCALERQEC_NATIVE=1.
- Bundle required C++ and OpenMP runtime libraries in Windows release wheels.
- Add exact small-circuit, sampler, serialization, and native regression tests;
  enforce cross-platform tests and focused profile coverage in release CI.

### Compatibility and interpretation

The single-p API remains available and accepts omitted plot names. Scaler now
requires noiseless SID input and exactly one observable. It does not reuse a
profile for a decoder retuned at each p, and its extrapolation uncertainty is
not a statistical confidence interval. Correctness fixes can change numerical
results compared with 1.0.0. Historical estimates should be revalidated.
