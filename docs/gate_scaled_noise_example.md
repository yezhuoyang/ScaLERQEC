# Gate-dependent depolarization and measurement noise

The experimental `LinearNoiseModel` API supports the requested family:
single-qubit gate depolarization p/5, two-qubit gate depolarization p, and
measurement error 5p. The existing `SI1000NoiseModel` builder accepts these
overrides; their values define a custom model, rather than the default SI1000
rates. The implementation is in this checkout and has not been published to
PyPI yet.

## Explicit Stim syntax

Stim files contain numeric probabilities. Supply their common reference p to
ScaLER. At `p_ref=0.01`, the following expressions produce a valid Stim fragment:

```python
p_ref = 0.01
fragment = f"""
H 0
DEPOLARIZE1({p_ref / 5}) 0
CX 0 1
DEPOLARIZE2({p_ref}) 0 1
M({5 * p_ref}) 0 1
"""
```

This becomes `DEPOLARIZE1(0.002)`, `DEPOLARIZE2(0.01)`, and `M(0.05)`.
Place such noise at the intended locations in your full circuit, including its
detectors and logical observables, then construct:

```python
import stim
from scalerqec.Stratified import LinearNoiseModel

circuit = stim.Circuit.from_file("my_noisy_circuit_at_p_ref.stim")
model = LinearNoiseModel(circuit, reference_p=0.01)
```

Each numeric noise argument q becomes `(q/p_ref)*p`. This permits different
coefficients by gate type, qubit, round, or individual location. It does not
infer missing noise locations or their rates from noiseless gate names.

`DEPOLARIZE2(p)` assigns p/15 to each of the 15 nonidentity two-qubit Paulis.
There are nine weight-two outcomes, giving total weight-two probability 3p/5.
For example, XX occurs with probability p/15, of order p. The channel's outcomes
are mutually exclusive, and the sampler preserves that structure. See the
[Stim channel definition](https://github.com/quantumlib/Stim/blob/main/doc/gates.md#the-depolarize2-instruction).

## Build through StabIR, profile once, and draw error bars

This uses the existing surface-code compiler and a fixed PyMatching decoder.
Preparation and idle noise are explicitly zero here; choose their coefficients
as part of your own hardware model.

```python
import matplotlib.pyplot as plt
import numpy as np
import pymatching

from scalerqec.QEC.noisemodel import SI1000NoiseModel
from scalerqec.QEC.surface import SurfaceCode
from scalerqec.Stratified import LinearNoiseModel

p_ref = 0.01
code = SurfaceCode(distance=3, rounds=3)
code.scheme = "Standard"
code.noisemodel = SI1000NoiseModel(
    p_ref,
    p_1q=p_ref / 5,
    p_2q=p_ref,
    p_meas=5 * p_ref,
    p_reset=0,
    p_idle=0,
)
model = LinearNoiseModel.from_stabcode(code, reference_p=p_ref)
decoder = pymatching.Matching.from_detector_error_model(
    model.circuit_at(p_ref).detector_error_model(decompose_errors=True)
)

ps = np.array([0.002, 0.005, 0.01, 0.02])
result = model.sample_bernstein_profile(
    decoder, ps,
    relative_error=0.2, confidence=0.99,
    max_shots=2_000_000, max_seconds=60, seed=260924,
)
print(result.status, result.shots)
for e in result.estimates:
    print(e.p, e.ler, (e.lower, e.upper), e.accuracy_met)

estimates = result.estimates
plt.vlines(ps, [e.lower for e in estimates], [e.upper for e in estimates],
           label="99% simultaneous confidence intervals")
plt.plot(ps, [e.ler for e in estimates], "o", label="LER estimates")
plt.xscale("log")
plt.xlabel("Base physical noise parameter p")
plt.ylabel("Logical error rate")
plt.title(result.status)
plt.legend()
plt.show()

if result.converged:
    polynomial = result.to_polynomial()
    polynomial.save("ler_polynomial.npz")
    print(polynomial([0.004, 0.007, 0.015]))  # No new simulation or decoding.
```

The sample count is automatic. `relative_error` specifies the requested
precision; `max_shots` and `max_seconds` cap resources. If the budget is
insufficient, the status is `budget_exhausted`, with valid uncertainty bounds
and unresolved accuracy. Polynomial export requires convergence by default.
The intervals are simultaneous over the four specified p values and sampling
checkpoints, under the model and IID assumptions. They are not a confidence
band for every p on the interpolated-looking smooth line; that line is the
estimated polynomial itself, not interpolation.

The DEM constructs the decoder only. Profiling samples the original channels,
not an independent-error approximation to their DEM. Keep this decoder fixed
throughout the curve. Retuning the decoder with p generally changes the
function being estimated and requires a different treatment.

## Why the same interface applies more broadly

The parser consumes a complete noisy Stim circuit and retains channel outcomes
and their positions. It does not hard-code the three gate classes used above.
It supports DEPOLARIZE1/2, biased PAULI_CHANNEL_1/2, X/Y/Z errors, explicit
correlated/ELSE chains, heralded errors, and measurement error arguments in
the existing parser. StabIR feeds its compiled Stim circuit into the same path.
Automatic gate-based noise insertion is limited to the builder's supported
operations. For other Stim operations, specify their noise explicitly in the
input circuit.

For each fixed circuit, fixed decoder, and fixed set of coefficients c_i,
uniformization gives a polynomial in the common parameter p with coefficients
independent of p. The saved polynomial can be evaluated throughout its physical
domain. The auxiliary trial count is retained jointly with the original Pauli
weight, so a two-qubit XX error remains weight two. See the
[derivation](uniformized_profiling.md).

One representation detail matters for weight-resolved comparisons: native
`M(5*p)` flips the classical measurement record and has Pauli weight zero in
this convention. The existing StabIR noise builder inserts a basis-appropriate
Pauli immediately before readout, which has weight one. For these simple
readouts the measurement statistics agree, but their weight profiles use
different physical fault representations. The sampler preserves whichever
representation was supplied.

This is general within the supported linear Pauli-noise family. It does not
model coherent overrotation from a hardware control error automatically,
arbitrary nonlinear p dependence, or fixed background noise independent of p.
All modeled probabilities must remain valid; this example has `0 <= p <= 0.2`
because the measurement channel has probability 5p. Changing the coefficients,
circuit, or decoder requires a new profile. Reusing a scalar SID weight spectrum
does not supply the missing general-noise information.

## Runnable examples and checks

From the source checkout, after installing it with `python -m pip install -e .`:

```text
python -m examples.gate_scaled_noise --output myresults_gate_scaled_noise
python -m examples.gate_scaled_noise --surface-distance 3 --output myresults_surface_noise
```

The default is a small three-qubit repetition Z-memory example with noisy
encoding, not a code protecting against phase errors. The second command uses
the StabIR surface example above. Both save the concrete Stim circuit, numerical
intervals, a plot, and (when converged) the reusable polynomial. Use an empty
output directory so previous results are preserved.

In the recorded surface run, all four points met the 20% relative-accuracy
target at 99% confidence after 267,775 samples. The profiling step took about
3.47 seconds on this machine, excluding imports and circuit compilation. This
is a demonstration, not a claim of performance on every code or at rare LERs.

Fresh, independently generated Stim circuits at each p gave the following
comparison, with 100,000 Monte Carlo shots per point and the same fixed decoder:

| p | Reusable polynomial | Independent Stim Monte Carlo |
|---:|---:|---:|
| 0.002 | 0.01638705 | 0.01630 |
| 0.005 | 0.05062056 | 0.05141 |
| 0.010 | 0.12228280 | 0.12238 |
| 0.020 | 0.27085710 | 0.27011 |

All four profile/reference confidence intervals overlap. The saved
[plot](../example_results/gate_scaled_surface/ler_vs_p.png),
[profile results](../example_results/gate_scaled_surface/results.json), and
[independent Monte Carlo intervals](../example_results/gate_scaled_surface/monte_carlo.json)
retain the uncertainty and accuracy status.

The focused tests check the p/5, p, and 5p factors, the valid p domain,
weight-two probabilities, and the complete small-circuit LER against an
independent bit-propagation calculation and direct Stim Monte Carlo. They
also verify the StabIR rate overrides.
