# QStabIR, configurable noise, and reusable LER curves

`NoiseModel` is the generic operation-dependent interface. `SI1000NoiseModel`
remains a compatible preset, but is not required for custom rates. This API is
in the development checkout; it has not yet been published to PyPI.

## Define a code, choose an extraction circuit, and attach noise

This example defines a Steane [[7,1,3]] code through the public `StabCode` API.
The compiler constructs its stabilizer memory IR, then lowers it to Stim.
Install the optional general decoder with `pip install 'scalerqec[ldpc]'`.
When working from this checkout, install with `pip install -e '.[ldpc]'`.

```python
from scalerqec.QEC import StabCode, NoiseModel
from scalerqec.Stratified import LinearNoiseModel
from stimbposd import BPOSD

p_ref = 0.01
noise = NoiseModel(
    p_ref,
    p_1q=p_ref / 5,
    p_2q=p_ref,
    p_meas=5 * p_ref,
    p_reset=0,
    p_idle=0,
)

code = StabCode(n=7, k=1, d=3)
for stabilizer in [
    "IIIXXXX", "IXXIIXX", "XIXIXIX",
    "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ",
]:
    code.add_stab(stabilizer)
code.set_logical_Z(0, "ZZZZZZZ")
code.rounds = 2
code.scheme = "Flag"   # "Standard", "Flag", "Shor", or "Knill"
code.noisemodel = noise

model = LinearNoiseModel.from_stabcode(code, reference_p=p_ref)
code.show_IR()
print(code.stimcirc)

# Fixed across the entire p curve. Accepts detector hyperedges and flag bits.
decoder = BPOSD(
    model.circuit.detector_error_model(),
    max_bp_iters=30, bp_method="min_sum", osd_order=0,
)
profile = model.sample_bernstein_profile(
    decoder, [0.002, 0.005, 0.01],
    relative_error=0.25, confidence=0.99,
    max_shots=500_000, max_seconds=60, seed=260927,
)
for result in profile.estimates:
    print(result.p, result.ler, result.lower, result.upper)
print(profile.status, profile.reason)
if profile.converged:
    polynomial = profile.to_polynomial()
    polynomial.save("ler_polynomial.npz")
    print(polynomial(0.007))  # Reuses the same samples; no new decoding.
```

For Standard, Flag and Shor, the IR contains entries such as:

```text
c0 = Prop[r=0, s=0] IIIXXXX
c1 = Prop[r=0, s=1] IXXIIXX
...
DataMeasure Z
...
o0 = Parity m0 m1 m2 m3 m4 m5 m6
```

The selected scheme determines how each `Prop` is implemented. Extra flag or
decoded-cat records become Stim detectors. Knill instead uses block-level
`Teleport[r=...]` instructions: it does not independently measure each `Prop`.
Its compiler includes Bell preparation, physical Bell measurements, syndrome
parities, and the logical Pauli frame in the observable.

This example uses the **StabCode memory IR**. It does not claim that arbitrary
`LogicQCompiler` programs containing adaptive `Decode`/`Correct` instructions
have a complete physical backend. The four memory schemes are fixed circuits.

## Apply the same object directly to Stim

```python
import stim

ideal = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=3, rounds=3,
)
noisy = noise.apply(ideal)
model = LinearNoiseModel(noisy, reference_p=p_ref)
# Use the same decoder/profile steps as above.
```

`apply` returns a new circuit. `inject_noise` remains an equivalent legacy
method name. Do not call both: existing noise is preserved and additional
noise would be added twice. The QStabIR path calls this same operation. Changing
`rounds`, `scheme`, a stabilizer, a logical operator, or the attached noise model
invalidates compiled state; repeated compilation does not duplicate noise.

The input must be a `stim.Circuit`, including circuits loaded using
`stim.Circuit.from_file`. Numeric probabilities are evaluated in Python;
`p_ref` is not symbolic Stim syntax. The family is formed by scaling every
noise argument by `p/p_ref`, with circuit structure and decoder fixed.

## Exact meanings of the four choices

| Scheme | Circuit implemented | Limits of the claim |
|---|---|---|
| Standard | Bare-ancilla stabilizer extraction; basis rotations for mixed Pauli checks | Hook errors can reduce circuit distance. |
| Flag | One syndrome ancilla plus one flag, with flag CNOTs bracketing data interactions | Flag outcomes go to the decoder; no universal distance-d fault-tolerance guarantee. |
| Shor | Cat state on the check's support, transversal controlled-Pauli coupling, inverse cat preparation, and readout | Specifically the **decoded-cat variant**; no verified-cat rejection/retry protocol. |
| Knill | Synthesized encoded Bell pair, transversal physical Bell measurement, output-block rotation and deferred logical frame | Bell preparation is noisy and unverified; no ancilla distillation or rejection/retry. |

These variants have different qubit counts, operation counts and schedules, and
therefore different LERs. They are real circuit implementations, not aliases
for Standard. Functionally correct extraction is a weaker property than a
complete fault-tolerant protocol with a decoder proved to correct all relevant
fault patterns. Published flag protocols can require additional checks,
conditions or flags; see [Chamberland–Beverland](https://arxiv.org/abs/1708.02246)
and [Chao–Reichardt](https://arxiv.org/abs/1912.09549). Ancilla decoding is
discussed by [DiVincenzo–Aliferis](https://arxiv.org/abs/quant-ph/0607047).

The memory compiler initializes data in Z and reads it out in Z. It therefore
requires commuting stabilizers of rank n-k and independent **Z-type logical Z
operators**. Mixed X/Y/Z stabilizers are supported. For other preparation and
readout bases, supply an explicit Stim circuit to the same estimator. The
declared distance `d` is metadata, not a verified circuit-distance certificate.
Multiple logical observables work in the general estimator. Their LER is the
probability that **any** logical prediction is wrong. The older QEPG object
supports only one observable; for multiple logicals `code.circuit` is `None`
and `code.legacy_backend_error` explains why. `code.stimcirc` remains available.

## Noise and time conventions

* Unspecified 1Q, 2Q, reset and measurement rates default to the first argument.
  Idle defaults to zero. Explicit zero disables the corresponding channel.
* All current Stim one- and two-qubit Clifford gate types are recognized through
  Stim metadata. A coalesced instruction containing overlapping gate pairs is
  split so that noise occurs immediately after each individual gate.
* `DEPOLARIZE2(p_2q)` samples one of 15 nonidentity two-qubit Paulis uniformly.
  Nine are weight two, so their combined probability is `3*p_2q/5`.
* Single-qubit reset errors are anticommuting Paulis after reset; single-qubit
  readout errors are anticommuting Paulis before measurement. For X basis the
  Pauli is Z; for Z or Y basis it is X. Measure-reset receives both channels.
* `MPP`, `MXX`, `MYY`, `MZZ` use native record-flip probabilities. Existing
  nonzero readout probabilities on those instructions must not be combined
  implicitly, since their effective probability would be nonlinear in p.
* Record/sweep-controlled Paulis are treated as ideal frame updates. A physical
  implementation requiring noisy feedback must include that noise explicitly.
* `SPP`/`SPP_DAG` are decomposed using Stim and their elementary gates receive
  noise. This is a specified implementation policy, not a native many-body
  noise model. Metadata, existing channels, and measurement record references
  are retained. When idle tracking is enabled, repeats are flattened.
* Idle noise applies to circuit qubits unused during each completed TICK layer.
  A layer may contain sequential operations: `TICK` boundaries, rather than an
  inferred hardware duration, define this model. No idle interval is added
  after a final layer without a closing TICK.

The inserted Pauli-factor weight convention is unchanged: XI has weight 1, XX
has weight 2, faults at different locations count separately, and native record
flips have weight 0. Moving a fault across a gate would change this model.

## Accuracy and validation

The polynomial estimates the failure function of the **fixed decoder**. The
99% bounds are simultaneous over the declared p grid and sequential stopping
checks. A newly evaluated p has no fresh confidence certificate. If a budget
is exhausted before accuracy is reached, the status is explicit and normal
polynomial export refuses to imply convergence. The time budget is cooperative
between batches; a slow decode call can overrun it. No universal speedup over
direct Monte Carlo is asserted.

`examples/qstabir_noise.py` runs every choice with the same API and can save an
independent Stim Monte Carlo comparison:

```console
python -m examples.qstabir_noise --scheme Flag --output myresults_flag --mc-shots 50000
```

The tests include signed stabilizer flows for both logical axes of Knill
teleportation, parity measurement and preservation of the Pauli commutant for
Flag/Shor, odd-Y stabilizers, multiple logicals, complete single-fault replay on
small circuits, nonuniform-noise comparisons with direct Stim sampling, and
distance-13 surface detector checks for every scheme. Signed flow checks use
Stim's algorithm with at most 2^-256 false-positive probability. These tests
provide reproducible evidence, not a proof that all software is bug-free.
