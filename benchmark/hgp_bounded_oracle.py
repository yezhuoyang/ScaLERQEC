"""Independent exact Pauli-weight <=2 oracle with rigorous omitted-mass bounds.

Unlike a Monte Carlo comparison, these small-p HGP references enclose the full
block LER deterministically (up to floating point). The fixed decoder is the
same as in the production estimate. No production parser/sampler is used to
form the oracle's syndrome signatures or history probabilities.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark.accuracy_control_validation import serialize
from benchmark.fault_replay_oracle import ReplayOracle
from benchmark.large_code_cases import P0, hgp_matrices, make_circuit
from benchmark.large_code_validation import decoder_for, write_json
from scalerqec.Stratified import LinearNoiseModel

OUT = Path("experiment_results/large_code_validation/hgp_oracles")
PS = [0.00005, 0.0001, 0.0002]


class SingleFaultRepair:
    """A fixed BP+OSD4 decoder with an independent exact single-Pauli lookup.

    The lookup uses CSS algebra, not training samples or production fault
    columns. It is part of the declared decoder, shared by oracle and profiler.
    """

    def __init__(self, spec):
        from stimbposd import BPOSD

        _, hx, hz, lz = hgp_matrices(spec["n"])
        self.num_observables = len(lz)
        self.lookup = {}
        for q in range(spec["n"]):
            for axis in "IXYZ":
                det = np.r_[
                    hx[:, q] if axis in "YZ" else np.zeros(len(hx)),
                    hz[:, q] if axis in "XY" else np.zeros(len(hz)),
                ].astype(bool)
                obs = (
                    lz[:, q].astype(bool)
                    if axis in "XY"
                    else np.zeros(len(lz), dtype=bool)
                )
                key = np.packbits(det).tobytes()
                if key in self.lookup and not np.array_equal(self.lookup[key], obs):
                    raise AssertionError("Conflicting single-Pauli logical predictions")
                self.lookup[key] = obs
        self.fallback = BPOSD(
            make_circuit(spec, 0.003).detector_error_model(
                approximate_disjoint_errors=True
            ),
            max_bp_iters=30,
            bp_method="product_sum",
            osd_order=4,
            osd_method="osd_cs",
        )

    def decode_batch(self, det):
        result = np.zeros((len(det), self.num_observables), dtype=bool)
        unknown = []
        for j, row in enumerate(np.packbits(det, axis=1)):
            predicted = self.lookup.get(row.tobytes())
            if predicted is None:
                unknown.append(j)
            else:
                result[j] = predicted
        if unknown:
            result[unknown] = self.fallback.decode_batch(det[unknown])
        return result


def bounded_oracle(spec, decoder, probabilities):
    circuit = make_circuit(spec, P0)
    replay = ReplayOracle(circuit, P0)
    _, hx, hz, lz = hgp_matrices(spec["n"])
    xs = np.vstack([hx, np.zeros_like(hz), np.zeros_like(lz)]).astype(bool)
    zs = np.vstack([np.zeros_like(hx), hz, lz]).astype(bool)
    width = len(xs)
    nd = len(hx) + len(hz)
    base = np.ones(len(probabilities))
    tails = np.zeros(len(probabilities))
    pref = np.zeros((len(probabilities), 3))
    pref[:, 0] = 1
    signatures = []
    ratios = []
    weights = []
    locations = []
    for j, event in enumerate(replay.events):
        probs = np.array([event.probabilities(p, P0) for p in probabilities])
        base *= probs[:, 0]
        for k in range(len(probabilities)):
            pgf = np.bincount(event.weights, weights=probs[k])
            conv = np.convolve(pref[k], pgf)
            tails[k] += conv[3:].sum()
            pref[k] = conv[:3]
        for a, paulis in enumerate(event.paulis[1:], 1):
            w = event.weights[a]
            if not 1 <= w <= 2:
                raise ValueError(
                    "Oracle requires strictly positive active Pauli weight <=2"
                )
            signature = np.zeros(width, dtype=bool)
            for q, axis in paulis:
                if axis in "XY":
                    signature ^= zs[:, q]
                if axis in "YZ":
                    signature ^= xs[:, q]
            signatures.append(signature)
            ratios.append(probs[:, a] / probs[:, 0])
            weights.append(w)
            locations.append(j)
    signatures = np.array(signatures)
    ratios = np.array(ratios)
    locations = np.array(locations)
    lower = np.zeros(len(probabilities))
    histories = 0
    failing = 0

    def include(bits, multipliers):
        nonlocal lower, histories, failing
        fail = np.any(decoder.decode_batch(bits[:, :nd]) != bits[:, nd:], axis=1)
        lower += multipliers[fail].sum(axis=0) * base
        histories += len(bits)
        failing += int(fail.sum())

    include(np.zeros((1, width), dtype=bool), np.ones((1, len(probabilities))))
    include(signatures, ratios)
    ones = np.flatnonzero(np.array(weights) == 1)
    a, b = np.triu_indices(len(ones), 1)
    a = ones[a]
    b = ones[b]
    valid = locations[a] != locations[b]
    a = a[valid]
    b = b[valid]
    for start in range(0, len(a), 2048):
        i = a[start : start + 2048]
        j = b[start : start + 2048]
        include(signatures[i] ^ signatures[j], ratios[i] * ratios[j])
    return {
        "histories": histories,
        "failing_histories": failing,
        "points": [
            {
                "p": p,
                "lower": float(lo),
                "upper": float(lo + tail),
                "omitted_mass": float(tail),
            }
            for p, lo, tail in zip(probabilities, lower, tails)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong", action="store_true")
    args = parser.parse_args()
    output = OUT.with_name("hgp_oracles_strong") if args.strong else OUT
    output.mkdir(parents=True, exist_ok=True)
    for noise in [
        "uniform_single",
        "nonuniform_depolarizing",
        "biased_pauli",
        "correlated_else",
    ]:
        spec = {"family": "hgp", "n": 58, "noise": noise, "rounds": 1}
        decoder = (
            SingleFaultRepair(spec)
            if args.strong
            else decoder_for(spec, make_circuit(spec, 0.003))[0]
        )
        started = perf_counter()
        oracle = bounded_oracle(spec, decoder, PS)
        oracle["seconds"] = perf_counter() - started
        print(noise, "oracle", oracle["histories"], oracle["seconds"], flush=True)
        model = LinearNoiseModel(make_circuit(spec, P0), P0)
        profile = model.sample_until_accuracy(
            decoder,
            PS,
            relative_error=0.1,
            confidence=0.99,
            max_shots=500_000,
            max_seconds=120,
            exact_budget=100_000,
            seed=20260917,
        )
        row = {
            "spec": spec,
            "oracle": oracle,
            "adaptive": serialize(profile),
            "decoder": "BP+OSD4 with exact single-Pauli lookup"
            if args.strong
            else "BP+OSD0",
        }
        for point, estimate in zip(oracle["points"], profile.estimates):
            point["accuracy_verified_by_oracle"] = bool(
                estimate.ler / 1.1 <= point["lower"]
                and point["upper"] <= estimate.ler / 0.9
            )
            point["profile_interval_overlaps_oracle"] = bool(
                estimate.lower <= point["upper"] and point["lower"] <= estimate.upper
            )
        write_json(output / (noise + ".json"), row)
        print(
            noise,
            profile.status,
            [p["accuracy_verified_by_oracle"] for p in oracle["points"]],
            flush=True,
        )
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
