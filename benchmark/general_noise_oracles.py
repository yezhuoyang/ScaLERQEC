"""Independent syndrome-distribution oracles for seven Pauli-noise families.

The exact calculation uses Pauli commutation and probability convolution. It
does not call the production noise parser, fault responses, or enumerator.
These are code-capacity experiments: ideal encoding and syndrome measurement,
except for the explicitly named readout-noise experiment.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import stim

from scalerqec.Stratified import LinearNoiseModel

OUT = Path("experiment_results/general_noise_oracles")
FAMILIES = [
    "uniform_single",
    "nonuniform_single",
    "mixed_depolarizing",
    "biased_two_qubit",
    "correlated_else",
    "heralded_pauli",
    "readout",
]
SPECS = {
    "five_qubit": ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
    "steane": ["IIIXXXX", "IXXIIXX", "XIXIXIX", "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"],
    "shor": [
        "ZZIIIIIII",
        "IZZIIIIII",
        "IIIZZIIII",
        "IIIIZZIII",
        "IIIIIIZZI",
        "IIIIIIIZZ",
        "XXXXXXIII",
        "XXXIIIXXX",
    ],
}


def code_spec(name):
    if name.startswith("repetition_"):
        n = int(name.split("_")[1])
        stabs = ["I" * j + "ZZ" + "I" * (n - j - 2) for j in range(n - 1)]
        logical = "Z" + "I" * (n - 1)
    else:
        stabs = SPECS[name]
        logical = "Z" * len(stabs[0])
    return stabs, logical


def signature(pauli, checks):
    result = 0
    for j, check in enumerate(checks):
        parity = sum(a != "I" and b != "I" and a != b for a, b in zip(pauli, check)) % 2
        result |= parity << j
    return result


def targets(pauli, qubits):
    factory = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
    return [factory[a](q) for a, q in zip(pauli, qubits) if a != "I"]


@dataclass
class Channel:
    gate: str
    qubits: list
    rates: list
    paulis: list

    def probabilities(self, p):
        if self.gate == "chain":
            surviving = 1.0
            probabilities = []
            for rate in self.rates:
                probabilities.append(surviving * rate * p)
                surviving *= 1 - rate * p
            return [surviving] + probabilities
        probabilities = [rate * p for rate in self.rates]
        return [1 - math.fsum(probabilities)] + probabilities

    def append(self, circuit, p):
        if self.gate == "chain":
            for j, (rate, pauli) in enumerate(zip(self.rates, self.paulis)):
                circuit.append(
                    "E" if j == 0 else "ELSE_CORRELATED_ERROR",
                    targets(pauli, self.qubits),
                    rate * p,
                )
        elif self.gate in {"DEPOLARIZE1", "DEPOLARIZE2"}:
            circuit.append(self.gate, self.qubits, math.fsum(self.rates) * p)
        else:
            circuit.append(self.gate, self.qubits, np.asarray(self.rates) * p)


def channels(n, family):
    result = []
    if family == "readout":
        return result
    for q in range(n):
        scale = [0.25, 1.0, 2.0][q % 3]
        if family == "uniform_single":
            result.append(Channel("DEPOLARIZE1", [q], [1 / 3] * 3, list("XYZ")))
        elif family == "nonuniform_single":
            result.append(
                Channel(
                    "PAULI_CHANNEL_1",
                    [q],
                    [scale * x for x in [0.1, 0.2, 0.7]],
                    list("XYZ"),
                )
            )
        elif family == "heralded_pauli":
            result.append(
                Channel(
                    "HERALDED_PAULI_CHANNEL_1",
                    [q],
                    [scale * x for x in [0.3, 0.1, 0.2, 0.4]],
                    list("IXYZ"),
                )
            )
        else:
            result.append(Channel("DEPOLARIZE1", [q], [0.2 / 3] * 3, list("XYZ")))
    pairs = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]
    for q in range(0, n - 1, 2):
        scale = 0.5 if q % 4 else 1.5
        if family == "mixed_depolarizing":
            result.append(Channel("DEPOLARIZE2", [q, q + 1], [scale / 15] * 15, pairs))
        elif family == "biased_two_qubit":
            rates = np.full(15, 0.01)
            rates[4] += 0.2
            rates[14] += 0.65
            result.append(
                Channel("PAULI_CHANNEL_2", [q, q + 1], (scale * rates).tolist(), pairs)
            )
        elif family == "correlated_else":
            result.append(
                Channel(
                    "chain",
                    [q, q + 1],
                    [0.8 * scale, 0.5 * scale, 0.3 * scale],
                    ["XX", "ZY", "IZ"],
                )
            )
    return result


def exact_joint(name, family, p, *, second_moment_reference=None):
    """Independent probability DP, or sum P_p(h)^2/P_reference(h) by state."""
    stabs, logical = code_spec(name)
    checks = stabs + [logical]
    n = len(logical)
    factors = []
    for channel in channels(n, family):
        outcomes = [(0, 0)]
        for pauli in channel.paulis:
            full = ["I"] * n
            for q, a in zip(channel.qubits, pauli):
                full[q] = a
            outcomes.append((sum(a != "I" for a in full), signature(full, checks)))
        probabilities = channel.probabilities(p)
        if second_moment_reference is not None:
            probabilities = [
                a * a / b if b else 0.0
                for a, b in zip(
                    probabilities, channel.probabilities(second_moment_reference)
                )
            ]
        factors.append((probabilities, outcomes))
    if family == "readout":
        for j in range(len(checks)):
            probability = [0.5, 1.0, 1.5][j % 3] * p
            probabilities = [1 - probability, probability]
            if second_moment_reference is not None:
                reference = [0.5, 1.0, 1.5][j % 3] * second_moment_reference
                probabilities = [
                    (1 - probability) ** 2 / (1 - reference),
                    probability**2 / reference,
                ]
            factors.append((probabilities, [(0, 0), (0, 1 << j)]))
    size = 1 << len(checks)
    indices = np.arange(size)
    mass = np.zeros((1, size))
    mass[0, 0] = 1
    for probabilities, outcomes in factors:
        next_mass = np.zeros((len(mass) + max(w for w, _ in outcomes), size))
        for probability, (weight, effect) in zip(probabilities, outcomes):
            next_mass[weight : weight + len(mass)] += (
                probability * mass[:, indices ^ effect]
            )
        mass = next_mass
    return mass


class LookupDecoder:
    def __init__(self, joint):
        distribution = joint.sum(axis=0)
        half = len(distribution) // 2
        self.predictions = distribution[half:] > distribution[:half]
        self.powers = 1 << np.arange(int(math.log2(half)))

    def decode_batch(self, det):
        return self.predictions[det.astype(np.int64) @ self.powers, None]

    def failure_mass(self, joint):
        half = joint.shape[1] // 2
        failure_indices = np.arange(half) + (~self.predictions) * half
        return joint[:, failure_indices].sum(axis=1)


def exact_profile_variance(name, family, p, reference_p, shots_per_weight):
    """True sampling variance, including failure histories absent from a profile."""
    reference = exact_joint(name, family, reference_p)
    decoder = LookupDecoder(reference)
    target_failures = decoder.failure_mass(exact_joint(name, family, p))
    second = decoder.failure_mass(
        exact_joint(name, family, p, second_moment_reference=reference_p)
    )
    variance = reference.sum(axis=1) * second - target_failures**2
    return float(np.maximum(0.0, variance).sum() / shots_per_weight)


def circuit_at(name, family, p):
    stabs, logical = code_spec(name)
    checks = stabs + [logical]
    n = len(logical)
    encoder = stim.Tableau.from_stabilizers(
        [stim.PauliString(s) for s in checks]
    ).to_circuit()
    circuit = stim.Circuit()
    circuit.append("R", range(n))
    circuit += encoder
    for channel in channels(n, family):
        channel.append(circuit, p)
    for j, check in enumerate(checks):
        measured = targets(check, range(n))
        product = []
        for t in measured:
            if product:
                product.append(stim.target_combiner())
            product.append(t)
        args = [[0.5, 1.0, 1.5][j % 3] * p] if family == "readout" else []
        circuit.append("MPP", product, args)
    for j in range(len(stabs)):
        circuit.append("DETECTOR", [stim.target_rec(j - len(checks))])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit


def run_case(name, family, *, shots_per_weight=5000, mc_shots=300_000):
    p0 = 0.05
    ps = [0.001, 0.01, 0.05, 0.15]
    start = perf_counter()
    decoder = LookupDecoder(exact_joint(name, family, p0))
    model = LinearNoiseModel(circuit_at(name, family, p0), p0)
    setup_seconds = perf_counter() - start
    start = perf_counter()
    profile = model.sample_profile(
        decoder, shots_per_weight=shots_per_weight, seed=130927
    )
    sampling_seconds = perf_counter() - start
    rows = []
    for j, p in enumerate(ps):
        exact = exact_joint(name, family, p)
        weight_mass = exact.sum(axis=1)
        failure_mass = decoder.failure_mass(exact)
        np.testing.assert_allclose(
            model.weight_distribution(p), weight_mass, rtol=3e-13, atol=1e-15
        )
        np.testing.assert_allclose(exact.sum(), 1, atol=3e-14)
        truth = float(failure_mass.sum())
        estimate = profile.evaluate(p)
        true_se = math.sqrt(
            exact_profile_variance(name, family, p, p0, shots_per_weight)
        )
        # The MC circuit is generated independently at p; do not call model.circuit_at.
        sampler = circuit_at(name, family, p).compile_detector_sampler(seed=92812 + j)
        failures = 0
        start = perf_counter()
        for first in range(0, mc_shots, 20_000):
            det, obs = sampler.sample(
                min(20_000, mc_shots - first), separate_observables=True
            )
            failures += int(np.any(decoder.decode_batch(det) != obs, axis=1).sum())
        mc_seconds = perf_counter() - start
        sigma = math.sqrt(truth * (1 - truth) / mc_shots)
        z = (
            (estimate.ler - truth) / estimate.standard_error
            if estimate.standard_error
            else (0 if abs(estimate.ler - truth) < 1e-15 else None)
        )
        rows.append(
            {
                "p": p,
                "exact_ler": truth,
                "profile_ler": estimate.ler,
                "profile_se": estimate.standard_error,
                "profile_z": z,
                "exact_profile_se": true_se,
                "exact_profile_z": (estimate.ler - truth) / true_se if true_se else 0.0,
                "estimated_se_over_exact_se": estimate.standard_error / true_se
                if true_se
                else None,
                "profile_ess": estimate.minimum_ess,
                "mc_ler": failures / mc_shots,
                "mc_failures": failures,
                "mc_shots": mc_shots,
                "mc_z": (failures / mc_shots - truth) / sigma if sigma else 0,
                "mc_seconds": mc_seconds,
                "projected_mc_shots_to_match_true_variance": truth
                * (1 - truth)
                / true_se**2
                if true_se
                else None,
                "projected_sampling_time_ratio_true_variance": truth
                * (1 - truth)
                / true_se**2
                / mc_shots
                * mc_seconds
                / (setup_seconds + sampling_seconds)
                if true_se
                else None,
            }
        )
    return {
        "name": name,
        "family": family,
        "reference_p": p0,
        "shots_per_weight": shots_per_weight,
        "total_profile_shots": len(profile.weights),
        "setup_seconds": setup_seconds,
        "sampling_seconds": sampling_seconds,
        "rows": rows,
    }


def calibration(repeats=30):
    rows = []
    for family in FAMILIES:
        name, p0, target = "five_qubit", 0.05, 0.01
        decoder = LookupDecoder(exact_joint(name, family, p0))
        model = LinearNoiseModel(circuit_at(name, family, p0), p0)
        truth = float(decoder.failure_mass(exact_joint(name, family, target)).sum())
        for j in range(repeats):
            profile = model.sample_profile(
                decoder, shots_per_weight=1000, seed=8800 + j
            )
            estimate = profile.evaluate(target)
            rows.append(
                {
                    "family": family,
                    "seed": 8800 + j,
                    "exact_ler": truth,
                    "estimate": estimate.ler,
                    "se": estimate.standard_error,
                    "z": (estimate.ler - truth) / estimate.standard_error
                    if estimate.standard_error
                    else None,
                }
            )
    return rows


def missed_failure_study():
    rows = []
    for family in ["nonuniform_single", "heralded_pauli"]:
        name, p0, p = "repetition_7", 0.05, 0.001
        reference = exact_joint(name, family, p0)
        decoder = LookupDecoder(reference)
        model = LinearNoiseModel(circuit_at(name, family, p0), p0)
        profile = model.sample_profile(decoder, shots_per_weight=5000, seed=130927)
        q3 = float(decoder.failure_mass(reference)[3] / reference[3].sum())
        rows.append(
            {
                "family": family,
                "p": p,
                "exact_ler": float(
                    decoder.failure_mass(exact_joint(name, family, p)).sum()
                ),
                "estimate": asdict(profile.evaluate(p)),
                "bounds": asdict(profile.confidence_bounds(p)),
                "reference_q3": q3,
                "expected_weight3_failures": 5000 * q3,
                "probability_zero_weight3_failures": (1 - q3) ** 5000,
            }
        )
    return rows


def matched_precision_study():
    """Actually run the MC shots needed to match analytically known variance."""
    results = []
    for name, family in [
        ("steane", "nonuniform_single"),
        ("repetition_5", "mixed_depolarizing"),
    ]:
        p0, p, shots = 0.05, 0.001, 5000
        # Oracle work determines a fixed MC budget before either experiment.
        reference = exact_joint(name, family, p0)
        start = perf_counter()
        decoder = LookupDecoder(reference)
        decoder_seconds = perf_counter() - start
        truth = float(decoder.failure_mass(exact_joint(name, family, p)).sum())
        variance = exact_profile_variance(name, family, p, p0, shots)
        mc_shots = math.ceil(truth * (1 - truth) / variance)
        start = perf_counter()
        model = LinearNoiseModel(circuit_at(name, family, p0), p0)
        profile = model.sample_profile(decoder, shots_per_weight=shots, seed=130927)
        estimate = profile.evaluate(p)
        profile_seconds = perf_counter() - start + decoder_seconds
        start = perf_counter()
        sampler = circuit_at(name, family, p).compile_detector_sampler(seed=292123)
        failures = 0
        for first in range(0, mc_shots, 20_000):
            det, obs = sampler.sample(
                min(20_000, mc_shots - first), separate_observables=True
            )
            failures += int(np.any(decoder.decode_batch(det) != obs, axis=1).sum())
        mc_seconds = perf_counter() - start + decoder_seconds
        row = {
            "name": name,
            "family": family,
            "p": p,
            "reference_p": p0,
            "exact_ler": truth,
            "profile_shots": len(profile.weights),
            "profile_ler": estimate.ler,
            "profile_estimated_se": estimate.standard_error,
            "profile_true_se": math.sqrt(variance),
            "profile_seconds": profile_seconds,
            "mc_shots": mc_shots,
            "mc_failures": failures,
            "mc_ler": failures / mc_shots,
            "mc_true_se": math.sqrt(truth * (1 - truth) / mc_shots),
            "mc_seconds": mc_seconds,
            "observed_time_ratio": mc_seconds / profile_seconds,
        }
        results.append(row)
        print("MATCHED PRECISION", json.dumps(row), flush=True)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in list(SPECS) + ["repetition_3", "repetition_5", "repetition_7"]:
        for family in FAMILIES:
            result = run_case(name, family)
            results.append(result)
            (OUT / f"{name}_{family}.json").write_text(json.dumps(result, indent=2))
            print(
                name,
                family,
                "max |z|",
                max(
                    abs(r["profile_z"])
                    for r in result["rows"]
                    if r["profile_z"] is not None
                ),
                flush=True,
            )
    report = {"stim": stim.__version__, "cases": results, "calibration": calibration()}
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    (OUT / "missed_failures.json").write_text(
        json.dumps(missed_failure_study(), indent=2)
    )
    (OUT / "matched_precision.json").write_text(
        json.dumps(matched_precision_study(), indent=2)
    )
    print("DONE", len(results), "cases", flush=True)


if __name__ == "__main__":
    main()
