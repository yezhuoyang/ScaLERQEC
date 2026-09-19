"""Reproducible general-noise study: exact oracles and actual Stim Monte Carlo.

Run from the repository root: python benchmark/general_noise_validation.py
No PyPI publishing, S-curve fitting, or decoder retuning occurs here.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

from scalerqec.QEC.noisemodel import SI1000NoiseModel
from scalerqec.QEC.surface import SurfaceCode
from scalerqec.Stratified.general_noise import GeneralNoiseProfile, LinearNoiseModel

OUT = Path("experiment_results/general_noise_validation")


def majority(det):
    return (det[:, 0] & ~det[:, 1])[:, None]


def repetition():
    return stim.Circuit("""R 0 1 2
DEPOLARIZE2(.016) 0 1
DEPOLARIZE1(.004) 2
M 0 1 2
DETECTOR rec[-3] rec[-2]
DETECTOR rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-3]
""")


def mc(model, decoder, p, shots, seed):
    sampler = model.circuit_at(p).compile_detector_sampler(seed=seed)
    failures = 0
    start = perf_counter()
    for first in range(0, shots, 20_000):
        det, obs = sampler.sample(min(20_000, shots - first), separate_observables=True)
        predictions = (
            decoder.decode_batch(det)
            if hasattr(decoder, "decode_batch")
            else decoder(det)
        )
        failures += int(np.any(predictions != obs, axis=1).sum())
    rate = failures / shots
    return {
        "ler": rate,
        "failures": failures,
        "shots": shots,
        "standard_error": math.sqrt(rate * (1 - rate) / shots),
        "seconds": perf_counter() - start,
    }


def run_case(name, model, decoder, ps, shots_per_weight, max_weight, exact=False):
    start = perf_counter()
    profile = model.sample_profile(
        decoder,
        shots_per_weight=shots_per_weight,
        max_weight=max_weight,
        seed=260204921,
        metadata={"case": name, "fixed_decoder": True},
    )
    sampling_seconds = perf_counter() - start
    path = OUT / f"{name}.npz"
    profile.save(path)
    loaded = GeneralNoiseProfile.load(path)
    start = perf_counter()
    estimates = loaded.curve(ps)
    evaluation_seconds = perf_counter() - start
    rows = []
    for i, (p, estimate) in enumerate(zip(ps, estimates)):
        direct = mc(model, decoder, p, 1_000_000, 20260916 + i)
        sigma = math.hypot(estimate.standard_error, direct["standard_error"])
        row = {
            "p": p,
            "profile": asdict(estimate),
            "monte_carlo": direct,
            "difference_in_combined_se": (estimate.ler - direct["ler"]) / sigma
            if sigma
            else 0,
        }
        if exact:
            row["exact_ler"] = model.enumerate_histories(decoder, p)["ler"]
        rows.append(row)
        print(name, json.dumps(row), flush=True)
    result = {
        "name": name,
        "reference_p": model.reference_p,
        "max_weight": model.max_weight,
        "sampled_weights": profile.sampled_weights.tolist(),
        "total_profile_shots": len(profile.weights),
        "noise_locations": len(model._factors),
        "rate_groups": len(model.rates),
        "sampling_seconds": sampling_seconds,
        "evaluation_seconds": evaluation_seconds,
        "rows": rows,
    }
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Adversarial exact example: weight-only reweighting must demonstrably fail.
    counter = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.2) 0 1\nX_ERROR(.4) 0\nX_ERROR(.6) 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]",
        0.2,
    )
    zero = lambda det: np.zeros((len(det), 1), dtype=np.bool_)
    reference = counter.enumerate_histories(zero, 0.2)
    profile = counter.sample_profile(zero, shots_per_weight=10_000, seed=58)
    counter_rows = []
    for p in [0.001, 0.01, 0.1]:
        exact = counter.enumerate_histories(zero, p)
        naive = float(exact["weight_mass"] @ reference["conditional_ler"])
        counter_rows.append(
            {
                "p": p,
                "exact_ler": exact["ler"],
                "naive_weight_only_ler": naive,
                "profile": asdict(profile.evaluate(p)),
                "monte_carlo": mc(counter, zero, p, 1_000_000, 20260916),
            }
        )
    (OUT / "weight_only_counterexample.json").write_text(
        json.dumps(counter_rows, indent=2)
    )
    results = []
    p0 = 0.02
    results.append(
        run_case(
            "repetition_depolarize2",
            LinearNoiseModel(repetition(), p0),
            majority,
            [0.0001, 0.001, 0.005, 0.01, 0.02],
            12_000,
            3,
            True,
        )
    )
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=p0,
        before_round_data_depolarization=0.2 * p0,
        before_measure_flip_probability=2 * p0,
        after_reset_flip_probability=0.5 * p0,
    )
    start = perf_counter()
    model = LinearNoiseModel(circuit, p0)
    print(
        "Stim surface response compilation seconds", perf_counter() - start, flush=True
    )
    decoder = pymatching.Matching.from_detector_error_model(
        model.circuit_at(0.005).detector_error_model(decompose_errors=True)
    )
    results.append(
        run_case(
            "stim_surface_d3_r3",
            model,
            decoder,
            [0.001, 0.002, 0.005, 0.01, 0.02],
            8000,
            24,
        )
    )
    code = SurfaceCode(distance=3, rounds=2)
    code.scheme = "Standard"
    code.noisemodel = SI1000NoiseModel(p0)
    start = perf_counter()
    model = LinearNoiseModel.from_stabcode(code, p0)
    print("StabIR response compilation seconds", perf_counter() - start, flush=True)
    decoder = pymatching.Matching.from_detector_error_model(
        model.circuit_at(0.005).detector_error_model(decompose_errors=True)
    )
    results.append(
        run_case(
            "stabir_surface_d3_r2_si1000",
            model,
            decoder,
            [0.001, 0.002, 0.005, 0.01, 0.02],
            8000,
            26,
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    for ax, result in zip(axes, results):
        rows = result["rows"]
        ps = [row["p"] for row in rows]
        ax.errorbar(
            ps,
            [row["profile"]["ler"] for row in rows],
            yerr=[2 * row["profile"]["standard_error"] for row in rows],
            fmt="o-",
            label="One profile (±2 SE)",
        )
        ax.errorbar(
            ps,
            [row["monte_carlo"]["ler"] for row in rows],
            yerr=[2 * row["monte_carlo"]["standard_error"] for row in rows],
            fmt="s",
            capsize=3,
            label="Stim MC (±2 SE)",
        )
        if "exact_ler" in rows[0]:
            ax.plot(
                ps, [row["exact_ler"] for row in rows], "k--", label="Exact enumeration"
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Global noise parameter p",
            ylabel="Logical error probability",
            title=result["name"].replace("_", " "),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    figure.savefig(OUT / "comparison.png", dpi=180)
    figure.savefig(OUT / "comparison.pdf")
    report = {
        "stim_version": stim.__version__,
        "numpy_version": np.__version__,
        "weight_only_counterexample": counter_rows,
        "results": results,
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print("Saved", OUT.resolve(), flush=True)


if __name__ == "__main__":
    main()
