"""Profile a surface circuit once and validate a p sweep with a fixed decoder.

Run from the repository root: python benchmark/reusable_profile.py
Artifacts go to ignored experiment_results/profile_validation/.
"""

from contextlib import redirect_stdout
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scalerqec.Stratified import LERProfile, Scaler
from scalerqec.QEC.noisemodel import SIDNoiseModel


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "experiment_results" / "profile_validation"
    output.mkdir(parents=True, exist_ok=True)
    scaler = Scaler(time_budget=15)
    start = perf_counter()
    with (
        (output / "sampling.log").open("w", encoding="utf-8") as log,
        redirect_stdout(log),
    ):
        profile = scaler.profile_from_file(
            root / "stimprograms/surface/surface3", 3, decoder_reference_p=0.001
        )
    sampling_seconds = perf_counter() - start
    profile.save(output / "surface3-profile.json")
    loaded = LERProfile.load(output / "surface3-profile.json")
    p_grid = np.geomspace(0.0001, 0.02, 101)
    start = perf_counter()
    curve = loaded.curve(p_grid)
    evaluation_seconds = perf_counter() - start
    np.testing.assert_array_equal(curve.ler, profile.evaluate(p_grid))
    rows = []
    shots = 500_000
    start = perf_counter()
    for i, p in enumerate([0.001, 0.002, 0.005, 0.01, 0.02]):
        noisy = SIDNoiseModel(p).inject_noise(scaler._cliffordcircuit.stimcircuit)
        sampler = noisy.compile_detector_sampler(seed=2026 + i)
        errors = 0
        # Keep memory independent of the Monte Carlo trial budget.
        for _ in range(shots // 10_000):
            det, obs = sampler.sample(10_000, separate_observables=True)
            pred = scaler._matcher.decode_batch(det)
            errors += int(np.count_nonzero(np.any(pred != obs, axis=1)))
        mc = errors / shots
        estimate = profile.evaluate(p)
        rows.append(
            dict(
                p=p,
                profile_ler=estimate,
                stim_ler=mc,
                stim_errors=errors,
                stim_standard_error=float(np.sqrt(mc * (1 - mc) / shots)),
                relative_difference=(estimate / mc - 1) if mc else None,
            )
        )
    monte_seconds = perf_counter() - start
    report = dict(
        noise_locations=profile.num_noise,
        profile_seconds=sampling_seconds,
        grid_size=len(p_grid),
        evaluation_seconds=evaluation_seconds,
        monte_carlo_seconds=monte_seconds,
        shots_per_stim_point=shots,
        total_profile_samples=int(profile.sample_counts.sum()),
        decoder_reference_p=0.001,
        comparisons=rows,
        caveat="Fixed decoder. Differences include S-curve systematic error and sampling error.",
    )
    (output / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    profile.plot(p_grid, ax=ax, label="One reusable ScaLER profile")
    ax.errorbar(
        [r["p"] for r in rows],
        [r["stim_ler"] for r in rows],
        yerr=[2 * r["stim_standard_error"] for r in rows],
        fmt="o",
        capsize=3,
        label="Stim, same decoder (±2 sampling SE)",
    )
    ax.set_title("Distance-3 surface code · uniform SID · fixed decoder")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.savefig(output / "surface3-sweep.png", dpi=180)
    fig.savefig(output / "surface3-sweep.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
