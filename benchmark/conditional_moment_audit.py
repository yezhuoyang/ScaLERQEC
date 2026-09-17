"""Check the actual conditional sampling law using independent exact moments."""

from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark.fault_replay_oracle import ReplayOracle
from benchmark.large_code_cases import P0, cases, make_circuit
from benchmark.large_code_validation import write_json
from scalerqec.Stratified import LinearNoiseModel

NAMES = [
    "active_locations",
    "early_half_active",
    "two_pauli_outcomes",
    "single_X_outcomes",
]


def labels(event, j, n):
    active = np.arange(len(event.weights)) > 0
    return np.array(
        [
            active,
            active & (j < n // 2),
            np.array(event.weights) == 2,
            [len(p) == 1 and p[0][1] == "X" for p in event.paulis],
        ],
        dtype=float,
    )


def conditional_moments(events, p, p0, weight):
    mass = np.zeros(weight + 1)
    mass[0] = 1
    first = np.zeros((4, weight + 1))
    second = first.copy()
    for j, event in enumerate(events):
        probs = event.probabilities(p, p0)
        g = np.bincount(event.weights, weights=probs)
        new_first = np.zeros_like(first)
        new_second = np.zeros_like(second)
        for k, flag in enumerate(labels(event, j, len(events))):
            a = np.bincount(event.weights, weights=probs * flag)
            inc = np.convolve(mass, a)[: weight + 1]
            new_first[k] = np.convolve(first[k], g)[: weight + 1] + inc
            new_second[k] = (
                np.convolve(second[k], g)[: weight + 1]
                + 2 * np.convolve(first[k], a)[: weight + 1]
                + inc
            )
        mass = np.convolve(mass, g)[: weight + 1]
        first, second = new_first, new_second
    means = first[:, weight] / mass[weight]
    variances = np.maximum(0, second[:, weight] / mass[weight] - means**2)
    return means, variances


def run(spec, shots=2048):
    started = perf_counter()
    circuit = make_circuit(spec, P0)
    model = LinearNoiseModel(circuit, P0)
    oracle = ReplayOracle(circuit, P0)
    mean = sum(np.dot(e.weights, e.probabilities(P0, P0)) for e in oracle.events)
    weight = max(1, round(mean))
    expected, variance = conditional_moments(oracle.events, P0, P0, weight)
    histories = np.empty((shots, len(oracle.events)), dtype=np.int64)
    model._sample_stratum(
        weight,
        shots,
        np.random.default_rng(20260918),
        model._suffix_table(weight),
        model._sampling_plan(),
        outcome_buffer=histories,
    )
    stats = np.zeros((shots, 4))
    for j, event in enumerate(oracle.events):
        stats += labels(event, j, len(oracle.events))[:, histories[:, j]].T
    actual = stats.mean(axis=0)
    se = np.sqrt(variance / shots)
    score = np.abs(actual - expected) / np.maximum(se, 1e-9)
    row = {
        "spec": spec,
        "shots": shots,
        "pauli_weight": weight,
        "seconds": perf_counter() - started,
        "statistics": [
            {
                "name": name,
                "observed_mean": float(a),
                "exact_conditional_mean": float(e),
                "true_standard_error": float(s),
                "standardized_difference": float(z),
            }
            for name, a, e, s, z in zip(NAMES, actual, expected, se, score)
        ],
        "within_7_true_standard_errors": bool(np.all(score < 7)),
    }
    return row


def main():
    wanted = {
        "surface_13_z_nonuniform_depolarizing",
        "surface_13_x_biased_pauli",
        "stabir_13_z_SI1000",
        "bicycle_72_z_nonuniform_depolarizing",
        "bicycle_144_z_biased_pauli",
        "hgp_58_z_correlated_else",
        "hgp_245_z_biased_pauli",
        "color_7_xyz_nonuniform_depolarizing",
    }
    out = Path("experiment_results/large_code_validation/moments")
    out.mkdir(parents=True, exist_ok=True)
    for spec in cases():
        if spec["name"] not in wanted:
            continue
        row = run(spec)
        write_json(out / (spec["name"] + ".json"), row)
        print(
            spec["name"],
            row["within_7_true_standard_errors"],
            max(s["standardized_difference"] for s in row["statistics"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
