"""Reproducible independent audits of the uniformized Bernstein profiler.

Run one case per process. Original benchmark evidence is never overwritten.
Large-code labels/decoder limitations are inherited from large_code_cases.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark.fault_replay_oracle import ReplayOracle
from benchmark.large_code_cases import P0, PS, cases, make_circuit
from benchmark.large_code_validation import decoder_for, monte_carlo, write_json
from scalerqec.Stratified import LERPolynomial, LinearNoiseModel, UniformizedSampler

SOURCES = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (
        Path("src/scalerqec/Stratified") / name
        for name in [
            "uniformized.py",
            "general_noise.py",
            "confidence.py",
            "noise_polynomial.py",
        ]
    )
}


def run(name, output, seconds=30, high_only=False, fresh_shots=256):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"Preserving {path}; use a new output directory.")
    spec = next(s for s in cases() if s["name"] == name)
    ps = [PS[-1]] if high_only else PS
    start = perf_counter()
    c = make_circuit(spec, P0)
    row = {
        "spec": spec,
        "p_values": ps,
        "seed": 260920,
        "status": "running",
        "circuit_sha256": hashlib.sha256(str(c).encode()).hexdigest(),
        "profile_seconds_budget": seconds,
        "source_sha256": SOURCES,
    }

    def checkpoint(phase):
        row["phase"] = phase
        row["seconds_total"] = perf_counter() - start
        write_json(path, row)
        print(name, phase, flush=True)

    checkpoint("compile")
    model = LinearNoiseModel(c, P0)
    row["compile_seconds"] = perf_counter() - start
    row["factors"] = len(model._factors)
    row["dense_response_bytes"] = sum(r.nbytes for r in model._responses)
    sampler = UniformizedSampler(model)
    row["latent_trials"] = sampler.num_trials
    row["uniform_rate"] = sampler.rate
    row["sampler_array_bytes"] = sum(
        a.nbytes for a in vars(sampler).values() if isinstance(a, np.ndarray)
    )
    checkpoint("replay")
    oracle = ReplayOracle(c, P0)
    # Include T=0, T=1, T=2, the typical reference count, and the endpoint T=M.
    # The endpoint detects first-success/deduplication and cancellation errors.
    typical = round(sampler.num_trials * sampler.rate * P0)
    counts = np.array([0, 1, 2, typical, sampler.num_trials] * 2)
    counts = np.minimum(counts, sampler.num_trials)
    histories = np.empty((len(counts), len(model._factors)), dtype=np.int64)
    bits, weights = sampler.sample(
        counts, np.random.default_rng(260920), outcome_buffer=histories
    )
    for h, b, w in zip(histories, bits, weights):
        np.testing.assert_array_equal(b, oracle.bits(h))
        assert w == sum(e.weights[a] for e, a in zip(oracle.events, h))
    row["replay"] = {
        "histories": len(counts),
        "trial_counts": counts.tolist(),
        "pauli_weights": weights.tolist(),
        "passed": True,
    }
    del histories, oracle, sampler
    decoder, row["decoder"] = decoder_for(spec, make_circuit(spec, PS[1]))
    checkpoint("profile")
    profile = model.sample_bernstein_profile(
        decoder,
        ps,
        relative_error=0.1,
        confidence=0.99,
        max_shots=1_000_000,
        max_seconds=seconds,
        batch_size=512 if spec["family"] in {"surface", "stabir"} else 32,
        seed=260920,
    )
    row["profile"] = {
        "status": profile.status,
        "reason": profile.reason,
        "shots": profile.shots,
        "seconds": profile.seconds,
        "estimates": [asdict(e) for e in profile.estimates],
        "joint_histogram_cells": len(profile.joint_counts),
        "joint_histogram_shots": sum(n for t, w, n, f in profile.joint_counts),
    }
    polynomial = profile.to_polynomial(allow_unconverged=True)
    poly_path = output / (name + "_polynomial.npz")
    polynomial.save(poly_path)
    reloaded = LERPolynomial.load(poly_path)
    np.testing.assert_allclose(
        reloaded(ps), [e.ler for e in profile.estimates], rtol=2e-10, atol=1e-15
    )
    row["polynomial_reload_verified"] = True
    row["polynomial_degree"] = polynomial.degree
    historical = Path("experiment_results/large_code_validation") / (name + ".json")
    old = json.loads(historical.read_text())
    assert old["circuit_sha256"] == row["circuit_sha256"]
    if "decoder" in old:
        assert old["decoder"] == row["decoder"]
    row["historical_reference_file"] = str(historical)
    row["historical_reference_sha256"] = hashlib.sha256(
        historical.read_bytes()
    ).hexdigest()
    row["historical_monte_carlo"] = old.get("monte_carlo", [])
    row["fresh_monte_carlo"] = []
    checkpoint("monte_carlo")
    for p in ps if fresh_shots else []:
        row["fresh_monte_carlo"].append(
            {"p": p, **monte_carlo(make_circuit(spec, p), decoder, fresh_shots, 260921)}
        )
        checkpoint("monte_carlo")
    row["status"] = "completed"
    checkpoint("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("experiment_results/uniformized_validation")
    )
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fresh-shots", type=int, default=256)
    parser.add_argument("--high-only", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        run(name, args.output, args.seconds, args.high_only, args.fresh_shots)
