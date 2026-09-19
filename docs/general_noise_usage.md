See [QStabIR, configurable noise, and all four schemes](qstabir_noise_and_schemes.md)
for the `NoiseModel` interface and tested code-definition examples.

# Experimental nonuniform-noise profiles

For a complete gate-dependent example using two-qubit depolarization p,
single-qubit noise p/5, and measurement noise 5p, see
[gate-scaled noise with confidence bars](gate_scaled_noise_example.md).

For automatic sample allocation and a stopping rule that remains valid as
sampling continues, use `model.sample_until_accuracy(decoder, probabilities,
relative_error=0.1, confidence=0.99)`. Check `result.converged` before exporting
its polynomial. This covers unseen failures, unsampled weights, and the entire
requested finite p grid. A resource-limited run reports unresolved accuracy.
See [the API, assumptions, and proof](accuracy_control.md).

For a conservative pointwise uncertainty interval, use
`profile.confidence_bounds(p, confidence=0.95)`. It accounts for unobserved
failures and omitted weights under fixed-budget sampling. The interval may be
wide even when the estimated standard error is small. High ESS alone cannot
detect missed rare failures. These are fixed-p bounds, not an automatically
simultaneous confidence band or an optional-stopping guarantee. See the
[broader validation](general_noise_broad_validation.md) for an observed example.

The implementation is in `scalerqec.Stratified.general_noise`. It is separate
from the released uniform-SID `Scaler` interface. See [the derivation](general_noise_math.md)
and [numerical validation](general_noise_validation.md).

## Export the polynomial itself

The exported object represents a fixed polynomial with sampled coefficients:

```python
from scalerqec.Stratified import GeneralNoiseProfile, LERPolynomial

profile = GeneralNoiseProfile.load("surface_profile.npz")
polynomial = profile.to_polynomial()
print(polynomial.degree, polynomial.num_terms)
print(polynomial.to_sympy())  # Symbolic factored polynomial in p.

polynomial.save("surface_polynomial.npz")
restored = LERPolynomial.load("surface_polynomial.npz")
values = restored([0.001, 0.002, 0.005])  # No circuit, decoder, or sampling.

# For small examples: coefficients of 1, p, p**2, ... as Decimal values.
# Default max_degree=100 guards expensive and numerically delicate expansion.
if polynomial.degree <= 100:
    coefficients = polynomial.power_coefficients(precision=50)
    expression = polynomial.to_sympy(expanded=True)

# Sampling SE, omitted-weight bounds, and ESS remain available on the profile.
estimates = profile.curve([0.001, 0.002, 0.005])
```

The existing uniform-SID weighted profiling workflow also supports
`polynomial = scaler.get_profile().to_polynomial()`. Its measured and
extrapolated spectrum entries are preserved. Exporting does not remove S-curve
model bias or turn that spectrum into a nonuniform-noise profile.

No p grid is used to infer or fit the coefficients. The export groups histories
with identical likelihood factors, preserving their contributions exactly up
to floating-point arithmetic. Curve evaluation also groups these records and
retains failure/nonfailure counts to preserve the sampling uncertainty.

The positive factored form is recommended for numerical evaluation. Expanding
high-degree polynomials into ordinary powers can introduce severe cancellation.
More decimal digits in exported coefficients do not imply more statistical
accuracy. A profile that omits weights exports their sampled contribution;
the polynomial object does not independently carry SE or certify a small tail.

For the previous nonuniform DEPOLARIZE2 counterexample, the rejected weight-only
formula's 0.00219230 is not the corrected estimator's output. Both
`profile.evaluate(0.001).ler` and `profile.to_polynomial()(0.001)` give about
0.00253659 from the same saved sampling experiment, versus exact 0.00253120.
The remaining difference is sampling error. Scalar weight-only profiles from
the old uniform interface cannot be losslessly converted into general-noise
polynomials: they lack the required likelihood records.

## Sample once, evaluate several noise strengths

Give the circuit at an interior reference p. Every Stim noise argument becomes
its value divided by that reference, multiplied by the requested p. For example,
`DEPOLARIZE2(0.02)` at reference 0.01 represents `DEPOLARIZE2(2*p)`.
The original noise locations and mutually exclusive outcomes are preserved.

```python
import numpy as np
import pymatching
import stim
from scalerqec.Stratified.general_noise import LinearNoiseModel, GeneralNoiseProfile

p0 = 0.02
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=3,
    after_clifford_depolarization=p0,
    before_round_data_depolarization=0.2*p0,
    before_measure_flip_probability=2*p0,
    after_reset_flip_probability=0.5*p0,
)
model = LinearNoiseModel(circuit, reference_p=p0)

# Fix the decoder for the entire experiment. The DEM is only used to build
# this decoder; the sampler uses the original circuit noise distribution.
decoder = pymatching.Matching.from_detector_error_model(
    model.circuit_at(0.005).detector_error_model(decompose_errors=True)
)
profile = model.sample_profile(
    decoder, shots_per_weight=8000, max_weight=24, seed=260204921,
    metadata={"decoder": "PyMatching", "decoder_reference_p": 0.005},
)
profile.save("surface_profile.npz")

# This can run in a separate process. No sampling or decoding is performed.
loaded = GeneralNoiseProfile.load("surface_profile.npz")
for estimate in loaded.curve(np.geomspace(0.001, 0.02, 30)):
    print(estimate.p, estimate.ler, estimate.standard_error,
          estimate.missing_probability_mass, estimate.minimum_ess)
```

`ler` estimates the contribution from sampled weights. The omitted contribution
is nonnegative and at most `missing_probability_mass`; that bound is separate
from the reported sampling standard error. The implementation accumulates small
tail probabilities directly instead of subtracting them from one.

`minimum_ess` reports the smallest importance-weight effective sample size
across sampled strata that have positive target probability. It can be dominated
by a stratum with negligible contribution. It does not detect every missed rare
failure. A zero measured failure count is not a proof that a stratum is safe.

The proposal must have support over the target family: the reference must be
strictly inside the valid probability interval. Evaluation at interval endpoints
is supported. Very distant target p values may need more samples or a different
proposal. No S-curve approximation is applied.

## Use StabIR

```python
from scalerqec.QEC.surface import SurfaceCode
from scalerqec.QEC.noisemodel import SI1000NoiseModel

code = SurfaceCode(distance=3, rounds=2)
code.scheme = "Standard"
code.noisemodel = SI1000NoiseModel(p0)
model = LinearNoiseModel.from_stabcode(code, reference_p=p0)
```

This runs the existing StabIR compiler if needed, then consumes its noisy Stim
circuit. It does not add noise a second time. If you change a previously compiled
code or its noise model, rebuild its circuit before constructing the profile.

## Noise and syntax coverage

For the measured limits at larger sizes, see
[distance-13 and QLDPC validation](large_code_validation.md). In particular,
budget exhaustion and underflow are unresolved results, not valid accuracy
certificates. The legacy native/Python QEPG backends accept only observable
index 0; use `LinearNoiseModel` for multiple logical outputs.

The experimental parser handles DEPOLARIZE1/2, PAULI_CHANNEL_1/2, X/Y/Z_ERROR,
E/ELSE_CORRELATED_ERROR chains, HERALDED_ERASE, HERALDED_PAULI_CHANNEL_1,
measurement error arguments (including MPP, pair measurements, and MPAD), and
identity-noise annotations. It delegates Clifford gates, aliases, annotations,
record-controlled Paulis, and expanded REPEAT blocks to Stim. Tests cover the
complete installed unitary gate catalog under Stim 1.15 and 1.16.

The full state space of possible Stim programs has not been exhaustively
verified. Ideal detectors and observables must be deterministic. Sweep bits
currently have their default zero assignment. Repeats are expanded, so the
prototype is not appropriate for enormous repeated circuits. Unknown noise
instructions raise instead of being discarded. This is a classical Pauli-noise
family, not a simulator for arbitrary coherent or non-Markovian noise.

Weight counts inserted Pauli factors. XI has weight one and XX has weight two;
faults at distinct times count separately. A classical measurement-record flip
and a heralded identity outcome have weight zero. An explicit X_ERROR before a
Z measurement instead has weight one. The representation is part of the profile.

Run the reproducible comparison with:

```text
python benchmark/general_noise_validation.py
```

It produces saved profiles, numerical results, and a comparison figure in
`experiment_results/general_noise_validation/`.

## Alternative without conditional-weight tables

`model.sample_bernstein_profile(decoder, probabilities, ...)` uses an auxiliary
count of uniformized trials together with the original Pauli weight. It produces
a fixed Bernstein-form polynomial for the supported linear noise family, and
uses global sequential confidence bounds instead of adding a separate bound
for every physical weight. This avoids the suffix-table memory limit; dense
response compilation and slow decoders remain possible bottlenecks.

See [the derivation, interface, and limits](uniformized_profiling.md). The new
method is experimental. Use its `converged`/`accuracy_met` results; successful
completion alone is not evidence that rare-event accuracy was established.
