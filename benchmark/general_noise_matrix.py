"""Accuracy and cost matrix against actual Stim sampling, with a fixed decoder.

Run: python benchmark/general_noise_matrix.py [--case NAME] [--resume]
Timings include model/decoder setup. Precision-cost ratios are projections from
measured variance and throughput, not additional Monte Carlo experiments.
"""

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pymatching
import stim
from scipy.stats import beta

from scalerqec.QEC.noisemodel import SD6NoiseModel, SI1000NoiseModel
from scalerqec.QEC.surface import RepetitionCode, SurfaceCode
from scalerqec.Stratified import LinearNoiseModel

OUT = Path("experiment_results/general_noise_matrix")
PS = [0.001, 0.003, 0.01]
P0 = 0.01


def cases():
    result = []
    for code, sizes in [("repetition", [3, 5, 9]), ("rotated_z", [3, 5, 7])]:
        for d in sizes:
            for noise in ["uniform_single", "nonuniform_depolarizing", "biased_pauli"]:
                if code == "rotated_z" and d == 7 and noise == "biased_pauli":
                    continue
                result.append(
                    {"frontend": "stim", "code": code, "distance": d, "noise": noise}
                )
    for code, sizes in [("rotated_x", [3, 5]), ("unrotated_z", [3, 5])]:
        for d in sizes:
            result.append(
                {
                    "frontend": "stim",
                    "code": code,
                    "distance": d,
                    "noise": "nonuniform_depolarizing",
                }
            )
    for code, sizes in [("surface", [3, 5]), ("repetition", [3, 7])]:
        for d in sizes:
            for noise in ["SD6", "SI1000"] if code == "surface" else ["SI1000"]:
                result.append(
                    {"frontend": "stabir", "code": code, "distance": d, "noise": noise}
                )
    for spec in result:
        spec["rounds"] = (
            min(spec["distance"], 3)
            if spec["frontend"] == "stabir"
            else spec["distance"]
        )
        spec["name"] = "{frontend}_{code}_d{distance}_{noise}".format(**spec)
    return result


def make_circuit(spec):
    d = spec["distance"]
    if spec["frontend"] == "stabir":
        cls = SurfaceCode if spec["code"] == "surface" else RepetitionCode
        code = cls(distance=d, rounds=spec["rounds"])
        code.scheme = "Standard"
        code.noisemodel = (
            SD6NoiseModel if spec["noise"] == "SD6" else SI1000NoiseModel
        )(P0)
        code.construct_circuit()
        return code.stimcirc
    task = {
        "repetition": "repetition_code:memory",
        "rotated_z": "surface_code:rotated_memory_z",
        "rotated_x": "surface_code:rotated_memory_x",
        "unrotated_z": "surface_code:unrotated_memory_z",
    }[spec["code"]]
    circuit = stim.Circuit.generated(
        task,
        distance=d,
        rounds=spec["rounds"],
        after_clifford_depolarization=P0,
        before_round_data_depolarization=0.2 * P0,
        before_measure_flip_probability=2 * P0,
        after_reset_flip_probability=0.5 * P0,
    )
    if spec["noise"] == "nonuniform_depolarizing":
        return circuit
    transformed = stim.Circuit()
    pair_fractions = np.full(15, 0.01)
    # Stim order IX,IY,IZ,XI,XX,XY,XZ,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ.
    pair_fractions[4] += 0.20
    pair_fractions[14] += 0.65
    for op in circuit.flattened():
        if op.name not in {
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
        }:
            transformed.append(op)
        elif spec["noise"] == "uniform_single":
            transformed.append("DEPOLARIZE1", op.targets_copy(), P0)
        elif op.name == "DEPOLARIZE2":
            transformed.append(
                "PAULI_CHANNEL_2", op.targets_copy(), pair_fractions * P0
            )
        elif op.name == "DEPOLARIZE1":
            transformed.append(
                "PAULI_CHANNEL_1",
                op.targets_copy(),
                np.array([0.025, 0.025, 0.95]) * op.gate_args_copy()[0],
            )
        else:
            transformed.append(op)
    return transformed


def binomial_interval(failures, shots, alpha=0.05):
    return [
        float(beta.ppf(alpha / 2, failures, shots - failures + 1)) if failures else 0.0,
        float(beta.ppf(1 - alpha / 2, failures + 1, shots - failures))
        if failures < shots
        else 1.0,
    ]


def direct_mc(circuit, decoder, shots, seed):
    start = perf_counter()
    sampler = circuit.compile_detector_sampler(seed=seed)
    setup_seconds = perf_counter() - start
    failures = 0
    start = perf_counter()
    for first in range(0, shots, 20_000):
        det, obs = sampler.sample(min(20_000, shots - first), separate_observables=True)
        failures += int(np.any(decoder.decode_batch(det) != obs, axis=1).sum())
    sampling_seconds = perf_counter() - start
    rate = failures / shots
    return {
        "shots": shots,
        "failures": failures,
        "ler": rate,
        "standard_error": math.sqrt(rate * (1 - rate) / shots),
        "interval_95": binomial_interval(failures, shots),
        "setup_seconds": setup_seconds,
        "sampling_seconds": sampling_seconds,
    }


def choose_limit(model, p, tolerance=1e-10):
    # A tail computed by positive accumulation, not 1-cdf cancellation.
    limit = min(16, model.max_weight)
    while True:
        _, tail = model._weight_distribution_with_tail(p, limit)
        if tail <= tolerance or limit == model.max_weight:
            return limit, tail
        limit = min(model.max_weight, max(limit + 1, math.ceil(limit * 1.5)))


def run_case(spec, *, shots_per_weight=2000, mc_shots=1_000_000):
    name = spec["name"]
    print("START", name, flush=True)
    start = perf_counter()
    circuit = make_circuit(spec)
    circuit_seconds = perf_counter() - start
    circuit.to_file(OUT / f"{name}.stim")
    start = perf_counter()
    model = LinearNoiseModel(circuit, P0)
    model_seconds = perf_counter() - start
    start = perf_counter()
    decoder = pymatching.Matching.from_detector_error_model(
        model.circuit_at(0.003).detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
    )
    decoder_seconds = perf_counter() - start
    start = perf_counter()
    limit, _ = choose_limit(model, max(PS))
    limit_seconds = perf_counter() - start
    start = perf_counter()
    profile = model.sample_profile(
        decoder,
        shots_per_weight=shots_per_weight,
        max_weight=limit,
        seed=260216,
        metadata={"case": name, "fixed_decoder_p": 0.003},
    )
    sampling_seconds = perf_counter() - start
    start = perf_counter()
    estimates = profile.curve(PS)
    evaluation_seconds = perf_counter() - start
    polynomial = profile.to_polynomial()
    np.testing.assert_allclose(polynomial(PS), [x.ler for x in estimates], rtol=2e-11)
    polynomial.save(OUT / f"{name}.npz")
    total_seconds = (
        circuit_seconds
        + model_seconds
        + decoder_seconds
        + limit_seconds
        + sampling_seconds
        + evaluation_seconds
    )
    rows = []
    for j, (p, estimate) in enumerate(zip(PS, estimates)):
        start = perf_counter()
        target = model.circuit_at(p)
        target_seconds = perf_counter() - start
        mc = direct_mc(target, decoder, mc_shots, 82931 + j)
        mc["total_seconds"] = (
            circuit_seconds
            + decoder_seconds
            + target_seconds
            + mc["setup_seconds"]
            + mc["sampling_seconds"]
        )
        se = math.hypot(estimate.standard_error, mc["standard_error"])
        discrepancy = max(
            0.0, abs(estimate.ler - mc["ler"]) - estimate.missing_probability_mass
        )
        row = {
            "p": p,
            "profile": asdict(estimate),
            "monte_carlo": mc,
            "difference_in_combined_se": discrepancy / se if se else None,
            "mc_has_100_failures": mc["failures"] >= 100,
            "profile_relative_se": estimate.standard_error / estimate.ler
            if estimate.ler
            else None,
            "tail_relative_to_profile": estimate.missing_probability_mass / estimate.ler
            if estimate.ler
            else None,
        }
        # Only estimate precision-cost when MC has substantial event counts.
        if mc["failures"] >= 100 and estimate.standard_error > 0:
            equivalent_shots = mc["ler"] * (1 - mc["ler"]) / estimate.standard_error**2
            equivalent_time = (
                circuit_seconds
                + decoder_seconds
                + target_seconds
                + mc["setup_seconds"]
                + equivalent_shots / mc_shots * mc["sampling_seconds"]
            )
            row["projected_mc_shots_to_match_se"] = equivalent_shots
            row["projected_mc_seconds_to_match_se"] = equivalent_time
            row["projected_single_point_speedup"] = equivalent_time / total_seconds
        rows.append(row)
        print(
            name,
            p,
            "LER",
            estimate.ler,
            "MC",
            mc["ler"],
            "z",
            row["difference_in_combined_se"],
            flush=True,
        )
    result = {
        **spec,
        "reference_p": P0,
        "rounds": spec["rounds"],
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "noise_locations": len(model._factors),
        "rate_groups": len(model.rates),
        "max_sampled_weight": limit,
        "total_profile_shots": len(profile.weights),
        "shots_per_weight": shots_per_weight,
        "polynomial_terms": polynomial.num_terms,
        "polynomial_degree": polynomial.degree,
        "timing": {
            "circuit_seconds": circuit_seconds,
            "model_seconds": model_seconds,
            "decoder_seconds": decoder_seconds,
            "limit_seconds": limit_seconds,
            "sampling_seconds": sampling_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": total_seconds,
        },
        "rows": rows,
    }
    if all("projected_mc_seconds_to_match_se" in row for row in rows):
        # Shared circuit/decoder setup is paid once by both methods.
        projected_curve = sum(row["projected_mc_seconds_to_match_se"] for row in rows)
        projected_curve -= (len(rows) - 1) * (circuit_seconds + decoder_seconds)
        result["projected_curve_speedup"] = projected_curve / total_seconds
    return result


def confirm_rare_points():
    """Exploratory follow-up: increase MC counts for two sparse comparisons."""
    results = []
    for name in [
        "stim_rotated_z_d5_biased_pauli",
        "stim_rotated_z_d7_nonuniform_depolarizing",
    ]:
        baseline = json.loads((OUT / f"{name}.json").read_text())
        model = LinearNoiseModel(
            stim.Circuit.from_file(OUT / f"{name}.stim"), P0, compile_responses=False
        )
        decoder = pymatching.Matching.from_detector_error_model(
            model.circuit_at(0.003).detector_error_model(
                decompose_errors=True, approximate_disjoint_errors=True
            )
        )
        direct = direct_mc(model.circuit_at(0.001), decoder, 10_000_000, 203971)
        result = {
            "name": name,
            "p": 0.001,
            "profile": baseline["rows"][0]["profile"],
            "new_monte_carlo": direct,
        }
        results.append(result)
        print("RARE POINT FOLLOW-UP", json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shots-per-weight", type=int, default=2000)
    parser.add_argument("--mc-shots", type=int, default=1_000_000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in cases():
        if args.case and spec["name"] not in args.case:
            continue
        path = OUT / f"{spec['name']}.json"
        if args.resume and path.exists():
            result = json.loads(path.read_text())
            if "error" in result:
                raise ValueError(
                    f"Cached case {spec['name']} failed; rerun it without --resume."
                )
            cached_shots = result.get(
                "shots_per_weight",
                result["total_profile_shots"] / (result["max_sampled_weight"] + 1),
            )
            if (
                result["rounds"] != spec["rounds"]
                or result["reference_p"] != P0
                or cached_shots != args.shots_per_weight
                or [row["p"] for row in result["rows"]] != PS
                or any(
                    row["monte_carlo"]["shots"] != args.mc_shots
                    for row in result["rows"]
                )
            ):
                raise ValueError(
                    f"Cached case {spec['name']} has different settings; rerun without --resume."
                )
        else:
            try:
                result = run_case(
                    spec, shots_per_weight=args.shots_per_weight, mc_shots=args.mc_shots
                )
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                AssertionError,
            ) as exc:
                result = {**spec, "error": repr(exc)}
                print("ERROR", spec["name"], repr(exc), flush=True)
            path.write_text(json.dumps(result, indent=2))
        results.append(result)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "stim": stim.__version__,
        "numpy": np.__version__,
        "pymatching": pymatching.__version__,
        "settings": vars(args),
        "cases": results,
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print("DONE", len(results), "cases", flush=True)


if __name__ == "__main__":
    main()
