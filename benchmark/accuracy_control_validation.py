"""Validate automatic stopping against exact oracles and actual Stim Monte Carlo.

Run from the repository root: python -m benchmark.accuracy_control_validation
This validates an accuracy contract; its resource-limited runs are not a speedup
benchmark. The JSON keeps unresolved cases instead of silently discarding them.
"""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pymatching
import stim

from benchmark.general_noise_matrix import P0, cases, direct_mc, make_circuit
from benchmark.general_noise_oracles import (
    FAMILIES,
    SPECS,
    LookupDecoder,
    circuit_at,
    exact_joint,
)
from scalerqec.Stratified import LinearNoiseModel

OUT = Path("experiment_results/accuracy_control")
PS = [0.001, 0.01, 0.05]


def serialize(result):
    return {
        "status": result.status,
        "reason": result.reason,
        "confidence": result.confidence,
        "relative_error": result.relative_error,
        "absolute_error": result.absolute_error,
        "shots": result.shots,
        "exact_histories": result.exact_histories,
        "sample_counts": result.sample_counts,
        "exact_weights": result.exact_weights,
        "proposal_probabilities": result.proposal_probabilities,
        "seconds": result.seconds,
        "estimates": [asdict(e) for e in result.estimates],
        "polynomial_terms": result.to_polynomial(allow_unconverged=True).num_terms,
    }


def oracle_cases():
    rows = []
    for name in list(SPECS) + ["repetition_3", "repetition_5", "repetition_7"]:
        for family in FAMILIES:
            start = perf_counter()
            decoder = LookupDecoder(exact_joint(name, family, 0.05))
            model = LinearNoiseModel(circuit_at(name, family, 0.05), 0.05)
            result = model.sample_until_accuracy(
                decoder,
                PS,
                relative_error=0.1,
                confidence=0.99,
                max_shots=200_000,
                max_seconds=8,
                exact_budget=100_000,
                seed=130927,
            )
            row = {"code": name, "noise": family, **serialize(result)}
            row["total_seconds"] = perf_counter() - start
            for e in row["estimates"]:
                truth = float(
                    decoder.failure_mass(exact_joint(name, family, e["p"])).sum()
                )
                e["truth"] = truth
                e["relative_actual_error"] = (
                    abs(e["ler"] - truth) / truth if truth else 0
                )
                e["covers_truth"] = bool(
                    e["lower"] - 1e-14 * truth <= truth <= e["upper"] + 1e-14 * truth
                )
                e["actual_accuracy_met"] = abs(e["ler"] - truth) <= 0.1 * truth
            print(
                name,
                family,
                row["status"],
                result.shots,
                result.exact_histories,
                flush=True,
            )
            rows.append(row)
    return rows


def repeated_sampling():
    rows = []
    # Readout W=0 is ordinary Bernoulli sampling at the reference. Across the
    # grid it exercises importance mixtures without any enumeration shortcut.
    model = LinearNoiseModel("R 0\nM(.2) 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.2)
    decoder = lambda det: np.zeros((len(det), 1), dtype=bool)
    ps = np.linspace(0.02, 0.3, 11)
    for seed in range(100):
        result = model.sample_until_accuracy(
            decoder,
            ps,
            relative_error=0.15,
            absolute_error=0.005,
            confidence=0.99,
            exact_budget=0,
            max_shots=200_000,
            max_seconds=None,
            seed=seed,
        )
        row = {"seed": seed, **serialize(result)}
        row["all_covered"] = all(e.lower <= e.p <= e.upper for e in result.estimates)
        row["all_accurate"] = all(
            abs(e.ler - e.p) <= 0.005 + 0.15 * e.p for e in result.estimates
        )
        rows.append(row)
    print("100 entirely stochastic repeated runs complete", flush=True)
    return rows


def circuit_cases():
    selected = {
        "stim_repetition_d3_nonuniform_depolarizing",
        "stim_repetition_d5_biased_pauli",
        "stim_rotated_z_d3_nonuniform_depolarizing",
        "stim_rotated_z_d5_biased_pauli",
        "stim_rotated_z_d7_nonuniform_depolarizing",
        "stabir_repetition_d3_SI1000",
        "stabir_surface_d3_SD6",
    }
    rows = []
    for spec in cases():
        if spec["name"] not in selected:
            continue
        start = perf_counter()
        model = LinearNoiseModel(make_circuit(spec), P0)
        decoder = pymatching.Matching.from_detector_error_model(
            model.circuit_at(0.003).detector_error_model(
                decompose_errors=True, approximate_disjoint_errors=True
            )
        )
        result = model.sample_until_accuracy(
            decoder,
            [0.001, 0.003, 0.01],
            relative_error=0.1,
            confidence=0.99,
            max_shots=200_000,
            max_seconds=8,
            exact_budget=100_000,
            seed=260216,
        )
        row = {**spec, **serialize(result)}
        row["total_profile_seconds"] = perf_counter() - start
        for e in row["estimates"]:
            e["mc"] = direct_mc(model.circuit_at(e["p"]), decoder, 2_000_000, 260916)
            lo, hi = e["mc"]["interval_95"]
            e["intervals_overlap"] = bool(e["lower"] <= hi and lo <= e["upper"])
            if result.converged and e["p"] == 0.001:
                # A fresh, larger fixed-budget run resolves the rare point more
                # precisely than the initial 2m-shot comparison.
                e["mc_confirmation"] = direct_mc(
                    model.circuit_at(e["p"]), decoder, 20_000_000, 260917
                )
        rows.append(row)
        print(
            spec["name"],
            result.status,
            result.shots,
            result.exact_histories,
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["all", "oracles", "repeated", "circuits"], default="all"
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, run in [
        ("oracles", oracle_cases),
        ("repeated", repeated_sampling),
        ("circuits", circuit_cases),
    ]:
        if args.suite not in (name, "all"):
            continue
        results = run()
        (OUT / f"{name}.json").write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "stim": stim.__version__,
                    "pymatching": pymatching.__version__,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
