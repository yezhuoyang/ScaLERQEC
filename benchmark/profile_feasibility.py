"""Exact pre-implementation check of one spectrum reused at multiple p values."""

import itertools
import json
import math

import numpy as np
import pymatching
import stim
from scipy.stats import binom


def run():
    circuit = stim.Circuit("""R 0 1 2
M 0 1 2
DETECTOR rec[-3] rec[-2]
DETECTOR rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-3]
""")

    def noisy(p):
        return stim.Circuit(f"R 0 1 2\nDEPOLARIZE1({p}) 0 1 2\n") + circuit[1:]

    decoder = pymatching.Matching.from_detector_error_model(
        noisy(0.01).detector_error_model()
    )
    failures = np.zeros(4, dtype=int)
    totals = np.zeros(4, dtype=int)
    for pattern in itertools.product("IXYZ", repeat=3):
        injected = stim.Circuit("R 0 1 2")
        for qubit, pauli in enumerate(pattern):
            if pauli != "I":
                injected.append(pauli, [qubit])
        injected += circuit[1:]
        # Detector sampling reports flips relative to the noiseless reference;
        # explicitly inserted Pauli gates are part of that reference. Use the
        # measurement-to-detection converter with the original reference instead.
        measurements = injected.compile_sampler(seed=11).sample(1)
        det, obs = circuit.compile_m2d_converter().convert(
            measurements=measurements, separate_observables=True
        )
        weight = sum(p != "I" for p in pattern)
        totals[weight] += 1
        failures[weight] += int(np.any(decoder.decode_batch(det) != obs))
    spectrum = failures / totals
    rows = []
    for p in [0.0, 0.02, 0.05, 0.1, 0.2, 1.0]:
        predicted = float(binom.pmf(np.arange(4), 3, p) @ spectrum)
        q = 2 * p / 3
        analytic = 3 * q**2 - 2 * q**3
        assert math.isclose(predicted, analytic, rel_tol=1e-13, abs_tol=1e-15)
        shots = 200_000
        det, obs = (
            noisy(p)
            .compile_detector_sampler(seed=2026)
            .sample(shots, separate_observables=True)
        )
        mc = float(np.mean(np.any(decoder.decode_batch(det) != obs, axis=1)))
        sigma = math.sqrt(analytic * (1 - analytic) / shots)
        assert abs(mc - analytic) <= 6 * sigma + 1 / shots
        rows.append(dict(p=p, reweighted=predicted, analytic=analytic, stim=mc))
    return dict(
        conditional_rates=spectrum.tolist(),
        failures=failures.tolist(),
        configurations=totals.tolist(),
        comparisons=rows,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
