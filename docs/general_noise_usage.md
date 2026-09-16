# Experimental nonuniform-noise profiles

The implementation is in `scalerqec.Stratified.general_noise`. It is separate
from the released uniform-SID `Scaler` interface. See [the derivation](general_noise_math.md)
and [numerical validation](general_noise_validation.md).

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
