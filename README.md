# ScaLERQEC

<p align="center">
  <img src="Figures/logo.png" alt="ScaLERQEC logo" width="350"/>
</p>

ScaLERQEC estimates logical error rates (LERs) for quantum error-correction
circuits. It provides weighted fault sampling, reusable LER profiles and
polynomials, Stim and stabilizer-code interfaces, and a C++ error-propagation
backend (QEPG).

**The development code supports one-parameter noise families with unequal gate
rates and correlated Pauli faults**, including standard two-qubit depolarization.
You can use measurement noise `5*p`, one-qubit noise `p/5`, and two-qubit noise
`p`, then estimate and reuse a curve of logical error probability `p_L` versus
`p` with statistical error bars.

**This is experimental research software.** The original SID S-curve estimator
still has systematic fitting/extrapolation error. The newer general-noise
polynomial method avoids that fit, but its coefficients are sampled and its
accuracy targets can remain unresolved, especially at low LER. Exporting a
polynomial does not make the answer exact. See [accuracy and limits](#accuracy-and-current-limits).

## Choose an interface

| Goal | Interface | Output and scope |
|---|---|---|
| Gate-dependent Pauli noise; reusable curves with error bars | `LinearNoiseModel.sample_bernstein_profile()` | Joint trial-count/Pauli-weight profile and estimated `LERPolynomial`, with automatic precision checks |
| Explicit allocation by physical Pauli weight | `LinearNoiseModel.sample_profile()` | Saved `GeneralNoiseProfile`, likelihood reweighting, sampling diagnostics and omitted-weight bounds |
| Original uniform independent single-qubit depolarizing (SID) method | `Scaler.profile_from_file()` | Saved `LERProfile` with measured and fitted weight spectrum; systematic error is not bounded |

These profiles belong to a **fixed circuit, noise family and decoder**. Changing
gate-rate ratios, noise locations, circuit structure or decoder requires a new
profile. A single profile varies only the common parameter `p`.

## Installation

Use a source checkout for the interfaces described here; do not assume a PyPI
release contains the current development APIs. Run the examples from the
repository root in a Python environment with Python **3.10 or later**:

```console
git clone https://github.com/yezhuoyang/ScaLERQEC.git
cd ScaLERQEC
python -m pip install -e .
python -c "import scalerqec; import scalerqec.qepg; print('OK')"
```

Source installation builds the C++20 backend and installs the Python
dependencies, including Stim, PyMatching, NumPy, SciPy, SymPy and Matplotlib.
A C++20 compiler is required: Visual Studio Build Tools on Windows, Xcode
command-line tools on macOS, or GCC/Clang on Linux. OpenMP is enabled by default
on Windows/Linux; local macOS builds use Homebrew `libomp` when detected.
Set the environment variable `SCALERQEC_NO_OPENMP=1` to build without OpenMP.
No Boost installation is required.

For BP+OSD decoding of LDPC codes or detector hypergraphs:

```console
python -m pip install -e ".[ldpc]"
```

The published package remains available with `python -m pip install scalerqec`.
The `examples/`, `benchmark/` and `stimprograms/` paths below refer to the source
checkout.

## Quick start: gate-dependent noise and error bars

This example builds a distance-3 surface-code memory circuit, adds the requested
noise rates, and holds a PyMatching decoder fixed at `p_ref`. Reset and idle
noise are explicitly disabled for this example; choose their rates for your
experiment. The reported LER is the failure probability of the **whole circuit**,
not a per-round rate.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

from scalerqec import NoiseModel, LinearNoiseModel, LERPolynomial

p_ref = 0.001
ideal = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=3,
)
noise = NoiseModel(
    p_ref,
    p_1q=p_ref / 5,
    p_2q=p_ref,
    p_meas=5 * p_ref,
    p_reset=0,
    p_idle=0,
)
noisy = noise.apply(ideal)
model = LinearNoiseModel(noisy, reference_p=p_ref)

# The detector error model configures the decoder. Profiling samples the
# original circuit channels, preserving their mutually exclusive outcomes.
decoder = pymatching.Matching.from_detector_error_model(
    noisy.detector_error_model(decompose_errors=True),
)
ps = np.array([0.0001, 0.0003, 0.001])
result = model.sample_bernstein_profile(
    decoder, ps,
    relative_error=0.2, confidence=0.99,
    max_shots=1_000_000, max_seconds=60, seed=123,
)
print(result.status, result.reason)
for e in result.estimates:
    print(e.p, e.ler, (e.lower, e.upper), e.accuracy_met)

output = Path("scaler_example_results")
output.mkdir(parents=True, exist_ok=True)
noisy.to_file(output / "circuit_at_reference.stim")

# Draw intervals directly; a stopped point estimate can lie outside the
# intersection of earlier intervals. Keep a linear y-axis if a bound is zero.
fig, ax = plt.subplots()
lower = np.array([e.lower for e in result.estimates])
upper = np.array([e.upper for e in result.estimates])
ax.vlines(ps, lower, upper, label="99% simultaneous confidence intervals")
ax.plot(ps, [e.ler for e in result.estimates], "o", label="Estimated LER")
ax.set_xscale("log")
if np.all(lower > 0):
    ax.set_yscale("log")
else:
    ax.set_ylim(bottom=0)
ax.set(xlabel="Base noise parameter p", ylabel="Logical error probability p_L",
       title=result.status)
ax.legend()
fig.savefig(output / "ler_vs_p.png", dpi=160, bbox_inches="tight")
plt.close(fig)
```

A run may finish with `budget_exhausted`. Its intervals remain useful, but the
requested precision was not reached at every target. This small low-p example
is **not guaranteed to converge within the example budget**. Check
`result.converged` and each `e.accuracy_met`; do not interpret zero observed
failures as zero LER. `absolute_error` can also be specified when absolute
precision is appropriate.

### Export the weighted profile

Continue from the example above:

```python
# Rows are (auxiliary trial count T, physical Pauli weight W, shots, failures).
np.savetxt(
    output / "weighted_profile.csv",
    np.asarray(result.joint_counts, dtype=np.int64).reshape(-1, 4),
    delimiter=",", fmt="%d", comments="",
    header="trial_count,pauli_weight,shots,failures",
)

# Estimated contribution P(failure AND W=2), not P(failure | W=2).
# Explicitly allow exploratory export even if total-LER precision is unresolved.
weight_two = result.weight_polynomial(2, allow_unconverged=True)
weight_two.save(output / "weight_two_contribution.npz")
print("Weight-two contribution:", weight_two(ps))
```

`W` counts inserted nonidentity Pauli factors at their original locations:
`XI` has weight 1 and `XX` has weight 2. Faults at distinct times count separately,
even if their propagated effects cancel. Native measurement-record flips and
heralded identity outcomes have weight 0. The `NoiseModel` builder instead
represents single-qubit readout errors as Paulis before measurement, with weight
1. Thus the noise representation is part of a weighted profile.

`T` is an auxiliary sampling count, not the number of physical faults. The
joint counts preserve both coordinates; use `weight_polynomial(w)` for a
reweighted contribution. Raw failure fractions grouped only by `W` are not a
reusable general-noise spectrum. **Per-weight polynomials do not inherit the
total-LER confidence bounds.** Keep the total polynomial and its uncertainty
alongside these diagnostic counts.

For explicit sampling by `W`, `model.sample_profile(decoder,
shots_per_weight=1000, max_weight=6, seed=123)` returns a `GeneralNoiseProfile`
with `save()`/`load()`, `evaluate(p)` and `to_polynomial()`. Its `ler` estimates
only the sampled-weight contribution. Inspect `missing_probability_mass`
separately from `standard_error`, and use `confidence_bounds(p)` for conservative
fixed-budget, pointwise intervals. Standard errors and effective sample size
alone do not detect all missed rare failures. See the
[conditional-weight guide](docs/general_noise_usage.md) and
[adaptive allocation alternative](docs/accuracy_control.md).

### Export and reuse the polynomial

Continue from the same `result`:

```python
if result.converged:
    polynomial = result.to_polynomial()
    path = output / "ler_polynomial.npz"
else:
    # Explicit diagnostic export: retains broad bounds and
    # metadata["accuracy_certified"] == False.
    polynomial = result.to_polynomial(allow_unconverged=True)
    path = output / "ler_polynomial_unresolved.npz"
polynomial.save(path)

# Reuse without a circuit, decoder, fitting, or additional sampling.
restored = LERPolynomial.load(path)
print(restored(ps))
print(restored.confidence_interval(ps))
print("Precision target met:", restored.metadata["accuracy_certified"])

# A p chosen after sampling: always inspect value AND uncertainty/status.
assessment = restored.estimate(0.0005)
print(assessment.ler, assessment.lower, assessment.upper, assessment.status)

print("Degree upper bound:", restored.degree)
if restored.degree <= 100:
    print(restored.to_sympy())                 # Factored expression in p.
    print(restored.to_sympy(expanded=True))    # Ordinary powers of p.
    print(restored.power_coefficients())      # Coefficients of 1, p, p**2, ...
```

Ordinary `to_polynomial()` refuses an unconverged result unless you explicitly
set `allow_unconverged=True`. The saved object contains a fixed estimated
polynomial, not an interpolation of the plotted points. Keep its positive
factored form for numerical evaluation: expanding a high-degree polynomial
can cause severe cancellation. Extra digits do not improve sampling accuracy.

Current Bernstein exports retain simultaneous uncertainty over the physical
p domain. `estimate(p)` reports `accuracy_met` or `insufficient_precision` at
new points; bounds may be very wide. Include every point requiring automatic
precision control in the original `ps` grid. Plain evaluation outside that grid
requires `restored(new_ps, allow_uncertified=True)` and supplies **no precision
claim by itself**. Older exports without domain statistics reject off-grid
error-bar queries. Polynomials exported from other profile types may not carry
confidence information.

For a small polynomial that can be printed and compared with an exact answer:

```console
python -m examples.polynomial_with_error_bars --output myresults_polynomial
```

This example uses two-qubit and conditional correlated noise; its exact
fixed-decoder LER is `(31/30)*p - (23/15)*p**2 + (16/15)*p**3`. The exported
coefficients are sampled estimates of that answer. Choose a new output
directory when rerunning it.

## Noise model contract

Stim accepts **numeric probabilities**, not symbolic expressions. Construct a
noisy circuit at an interior `p_ref`; `LinearNoiseModel` interprets each noise
argument `q` as `(q / p_ref) * p`. Coefficients may differ by gate type, qubit,
round or individual location. Zero rates stay zero. All noise arguments scale
together; arbitrary functions of p, constant background rates mixed with
variable rates, and several independently varying parameters are outside this
interface.

For example, `DEPOLARIZE2(p)` chooses one of 15 nonidentity two-qubit Paulis,
each with probability `p/15`. Nine outcomes have weight 2, with combined
probability `3*p/5`. A correlated `XX` fault at one location is therefore of
order p, not p squared. Channel outcomes remain mutually exclusive.

Supported channels include `DEPOLARIZE1/2`, `PAULI_CHANNEL_1/2`, `X/Y/Z_ERROR`,
`E`/`ELSE_CORRELATED_ERROR` chains, `HERALDED_ERASE`,
`HERALDED_PAULI_CHANNEL_1`, measurement-error arguments (including parity
measurements), and identity-noise annotations. ELSE arguments retain their
conditional meaning; the resulting branch probabilities can be higher-degree
polynomials even though each supplied argument is linear in p. Unsupported
noise instructions raise an error. This covers supported stochastic Pauli and
classical-record noise, not arbitrary physical noise such as coherent errors,
leakage or general non-Markovian dynamics.

The model exposes `max_p`; evaluate only within `0 <= p <= model.max_p`, with
`0 < p_ref < model.max_p`. The quick-start family has `max_p=0.2` because its
largest rate is `5*p`. All channels, including the sum of probabilities within
a Pauli channel, must remain valid. `model.circuit_at(p)` reconstructs the
corresponding noisy Stim circuit for an independent comparison.

For your own circuit, replace the builder with:

```python
noisy = stim.Circuit.from_file("my_noisy_circuit_at_p_ref.stim")
model = LinearNoiseModel(noisy, reference_p=0.001)
```

Supply detectors and logical observables defining the experiment. Ideal
(no-noise) detectors and logical parities must be deterministic. The profiler
does not insert missing noise; `NoiseModel.apply()` preserves existing noise,
so applying it twice adds noise twice. The builder's unspecified gate, reset
and readout rates default to its first argument; idle noise defaults to zero.
Idle intervals are defined by completed `TICK` layers, not inferred durations.

A decoder can implement `decode_batch(detectors)` or be a callable returning
one prediction per shot and logical observable. Keep it deterministic and
fixed across p. The general-noise interface supports multiple observables and
counts a shot as a failure if **any** logical prediction is wrong. PyMatching
requires a compatible graphlike detector model; use a suitable custom decoder
or the optional BP+OSD decoder for hypergraphs. Decoder compatibility is
separate from the profiler's noise support.

## Stabilizer-code input

You can define a code using `StabCode` and attach the same `NoiseModel`, or use
a built-in surface code:

```python
from scalerqec.QEC.surface import SurfaceCode

code = SurfaceCode(distance=3, rounds=3)
code.scheme = "Standard"
code.noisemodel = NoiseModel(
    p_ref, p_1q=p_ref / 5, p_2q=p_ref,
    p_meas=5 * p_ref, p_reset=0, p_idle=0,
)
stab_model = LinearNoiseModel.from_stabcode(code, reference_p=p_ref)
print(code.stimcirc)
# Build a decoder for stab_model.circuit, then use the same profiling workflow.
```

The memory compiler implements Standard, Flag, Shor (decoded-cat) and Knill
extraction circuits. These are specific circuit constructions, not a universal
fault-tolerance or circuit-distance guarantee. See the
[custom StabCode example, scheme definitions and limits](docs/qstabir_noise_and_schemes.md)
and [executable example](examples/qstabir_noise.py). General adaptive LogiQ or
MagicQ protocols, including postselection/retry, are not a supported end-to-end
LER workflow here.

## What the polynomial represents

For uniform SID noise at N locations and a fixed decoder, the original method
uses a weight spectrum `s_w = P(failure | W=w)`:

```text
p_L(p) = sum_w s_w * binom(N,w) * p^w * (1-p)^(N-w).
```

For nonuniform or correlated noise, `P(failure | W=w)` generally depends on p.
Substituting a different weight distribution into the original scalar spectrum
is therefore insufficient. The general method retains additional likelihood
information, or the joint `(T,W)` profile. Its Bernstein representation is:

```text
p_L(p) = sum_t b_t * binom(M,t) * (C*p)^t * (1-C*p)^(M-t),
b_t = P(failure | T=t).
```

Here M is the number of auxiliary trials and C their common rate coefficient;
`result.num_trials` and `result.uniform_rate` expose them. The identity is exact
under the supported model, while the exported coefficients are estimates.
Physical-weight contributions sum to the total polynomial. See the
[derivation and confidence assumptions](docs/uniformized_profiling.md).

## Accuracy and current limits

- **Systematic error remains in the original S-curve approach.** Its fitted or
  extrapolated weights can bias LER, particularly in rare-event regimes.
  A good fit score, a smooth curve, sampling bars, or polynomial export does
  not bound that bias. `LERProfile.plot()` does not provide total-error bars.
- **The general-noise method has statistical uncertainty.** It avoids S-curve
  extrapolation, but a finite sample may miss important failure histories.
  Its bounds assume the supported noise model, correct circuit responses,
  independent draws and a fixed deterministic decoder. They do not cover
  hardware-model mismatch or software/numerical errors.
- **Confidence is different from precision.** `confidence=0.99` specifies
  simultaneous statistical coverage, including automatic stopping; it does
  not mean 1% relative error. Precision is checked against `relative_error`
  and `absolute_error`. An error bar at a few points alone does not certify
  the rest of a curve; use the saved domain bounds and new-point assessments.
- **Low-LER and large-code efficiency is unresolved.** In the recorded
  [September 17 low-p baseline](docs/low_p_validation.md), none of 15 full grids
  met the requested precision within their budgets at p = 0.0001, 0.0003 and
  0.001. Later conditional-sampling prototypes improved selected cases; they
  are benchmark prototypes, not the default API. No universal speedup over
  direct Stim Monte Carlo is established.
- **Resources and syntax have limits.** Repeats are expanded; response
  compilation, conditional-weight tables and slow decoders can dominate
  memory/runtime. The Bernstein path avoids conditional-weight suffix tables,
  but is not cost-free. Time limits are checked between batches; setup and a
  long decoder call can exceed the requested time. Sweep bits use their
  default zero assignment.

Validate a new circuit/noise/decoder combination against exact small cases or
independent Stim sampling before relying on an extrapolated rare-event result.
See [polynomial soundness and observed failures](docs/polynomial_soundness_validation.md)
and [large-code validation](docs/large_code_validation.md) for the measured
scope, including unresolved cases.

## Original SID workflow

The original C++-backed ScaLER method remains available for a **noiseless**
Stim circuit with one logical observable (index 0). It inserts uniform SID
noise before primitive gates; it does not accept the explicitly noisy circuit
from the general-noise example.

```python
from scalerqec import Scaler, LERProfile

sid_profile = Scaler(time_budget=60).profile_from_file(
    "stimprograms/surface/surface3", codedistance=3,
    decoder_reference_p=0.001,
)
sid_profile.save("surface3_sid_profile.json")
sid_profile = LERProfile.load("surface3_sid_profile.json")
print(sid_profile.evaluate([0.0001, 0.001]))
sid_profile.to_polynomial().save("surface3_sid_polynomial.npz")
```

`codedistance` must be the circuit-level distance for the chosen circuit and
decoder, not merely the nominal code distance. Inspect `conditional_ler`,
`sample_counts`, `failure_counts`, `modeled_weights` and `curve(ps)` to see
which contributions were measured or fitted. These are diagnostics, not
confidence intervals. Use the general-noise workflow above, also applicable
to explicitly specified SID channels, when statistical error bars are needed.
See the [SID profile guide](docs/source/profiles.rst).

## Documentation and development

- [Polynomial interface](docs/source/polynomials.rst),
  [general-noise guide](docs/general_noise_usage.md), and
  [gate-rate conventions](docs/gate_scaled_noise_example.md).
- [Tutorial notebook](Tutorial.ipynb) and [examples](examples/).
- [Benchmark instructions](benchmark/README.md) and [migration notes](docs/legacy_cleanup.md).
- [Generated API documentation](https://yezhuoyang.github.io/ScaLERQEC/).
  For development behavior, use this checkout's code and documentation; older
  examples may predate the off-grid evaluation guard and domain bounds.

The source lives in `src/scalerqec/`: `Stratified/` contains the profilers and
polynomials, `QEC/` the circuit/noise builders, `Monte/` direct Monte Carlo,
`Symbolic/` small-circuit exact SID analysis, and `Analysis/` hotspot attribution.
Native QEPG code is in `QEPG/`. Exact symbolic analysis has its own small-circuit
and noise restrictions; it is distinct from sampled general-noise polynomial
export. Bundled fixtures in `stimprograms/` include surface, repetition, color,
toric, hexagonal, square and bicycle LDPC circuits, plus small test cases.

```console
python -m pip install -e ".[dev]"
python -m pytest tests
python -m pytest tests/test_stratified/test_gate_scaled_example.py tests/test_stratified/test_noise_polynomial.py tests/test_stratified/test_uniformized.py
```

After changing C++ sources, rebuild with `python -m pip install -e .`.
To build the API documentation:

```console
python -m pip install sphinx furo
python -m sphinx.cmd.build -b html docs/source docs
```

## Citation

If you use ScaLERQEC in research, please cite:

```bibtex
@misc{ye2026scalabletestingquantumerror,
      title={Scalable testing of quantum error correction},
      author={John Zhuoyang Ye and Jens Palsberg},
      year={2026},
      eprint={2602.04921},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2602.04921},
}
```

[MIT license](LICENSE).
