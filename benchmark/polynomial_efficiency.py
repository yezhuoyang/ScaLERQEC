"""Measure polynomial reuse and verify the formerly biased counterexample.

Run general_noise_validation.py first to create the saved surface profiles.
This benchmark does not resample those profiles. The raw-history implementation
below is the previous estimator and serves as the speed/correctness baseline.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from scalerqec.Stratified import GeneralNoiseProfile, LERPolynomial, LinearNoiseModel

OUT = Path("experiment_results/polynomial_efficiency")
PROFILES = Path("experiment_results/general_noise_validation")


def raw_evaluate(profile, p):
    logs = profile._log_likelihood(p)
    total = se = 0.0
    ess = math.inf
    mass, tail = profile.model._weight_distribution_with_tail(
        p, int(profile.weights.max())
    )
    for w, n in zip(profile.sampled_weights, profile.counts):
        selected = profile.weights == w
        peak = logs[selected].max()
        scaled = np.exp(logs[selected] - peak) if np.isfinite(peak) else np.zeros(n)
        scale = math.exp(math.log(profile._reference_mass[w]) + peak)
        values = scaled * profile.failures[selected]
        total += scale * values.mean()
        se = math.hypot(se, scale * values.std(ddof=1) / math.sqrt(n))
        if mass[w] > 0:
            ess = min(
                ess,
                float(scaled.sum() ** 2 / (scaled @ scaled)) if np.any(scaled) else 0.0,
            )
    missing = np.ones(len(mass), dtype=bool)
    missing[profile.sampled_weights] = False
    return [total, se, ess, tail + mass[missing].sum()]


def median_time(fn, repetitions=3):
    times = []
    value = None
    for _ in range(repetitions):
        start = perf_counter()
        value = fn()
        times.append(perf_counter() - start)
    return value, float(np.median(times))


def rare_event_comparison():
    """An end-to-end example with known true LER, not just evaluation timing."""
    reference = GeneralNoiseProfile.load(PROFILES / "repetition_depolarize2.npz")
    model = reference.model
    model.compile_responses()
    decoder = lambda det: (det[:, 0] & ~det[:, 1])[:, None]
    profile, sampling_time = median_time(
        lambda: model.sample_profile(
            decoder, shots_per_weight=12_000, max_weight=3, seed=260204921
        )
    )
    p = 0.0001
    exact = 4 * (0.8 * p) / 15 + 8 * (0.8 * p) / 15 * (2 * (0.2 * p) / 3)
    estimated = profile.evaluate(p)
    shots = 10_000_000
    sampler = model.circuit_at(p).compile_detector_sampler(seed=20260916)
    start = perf_counter()
    failures = 0
    for _ in range(shots // 100_000):
        det, obs = sampler.sample(100_000, separate_observables=True)
        failures += int(np.any(decoder(det) != obs, axis=1).sum())
    elapsed = perf_counter() - start
    # Truth is known here, so the MC standard error need not be estimated
    # from a small number of failures.
    mc_se = math.sqrt(exact * (1 - exact) / shots)
    assert abs(failures / shots - exact) < 7 * mc_se
    assert abs(estimated.ler - exact) < 7 * estimated.standard_error
    return {
        "p": p,
        "exact_ler": exact,
        "profile_trials": len(profile.weights),
        "profile_sampling_seconds": sampling_time,
        "profile": asdict(estimated),
        "monte_carlo_shots": shots,
        "monte_carlo_seconds": elapsed,
        "monte_carlo_ler": failures / shots,
        "monte_carlo_failures": failures,
        "monte_carlo_true_standard_error": mc_se,
        "mc_shots_for_profile_se_estimate": exact
        * (1 - exact)
        / estimated.standard_error**2,
        "interpretation": "This small exactly solvable rare-event example is favorable to profiling; the matched-precision shot count is an estimate, not a run of that size.",
    }


def write_figure(report, profile):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = np.geomspace(0.0001, 1 / 3, 401)
    exact = (38 * ps - 32 * ps**2) / 15
    curve = profile.curve(ps)
    corrected = np.array([r.ler for r in curve])
    se = np.array([r.standard_error for r in curve])
    zero = lambda det: np.zeros((len(det), 1), dtype=np.bool_)
    reference = profile.model.enumerate_histories(zero, 0.2)
    naive = np.array(
        [
            profile.model.weight_distribution(p) @ reference["conditional_ler"]
            for p in ps
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    ax = axes[0]
    ax.axhline(0, color="black", linewidth=1, label="Exact polynomial")
    ax.plot(
        ps,
        100 * (naive / exact - 1),
        color="#b44435",
        label="Unsupported weight-only formula",
    )
    ax.plot(
        ps,
        100 * (corrected / exact - 1),
        color="#237c69",
        label="Corrected polynomial estimate",
    )
    ax.fill_between(
        ps,
        100 * ((corrected - 2 * se) / exact - 1),
        100 * ((corrected + 2 * se) / exact - 1),
        color="#237c69",
        alpha=0.18,
        label="Estimated ±2 SE",
    )
    ax.set(
        xscale="log",
        xlabel="Global noise parameter p",
        ylabel="Relative error against exact LER (%)",
        title="Fixing reweighting within Pauli-weight strata",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    rows = report["grid_benchmarks"][1:]
    x = np.arange(len(rows))
    ax = axes[1]
    for offset, key, label, color in [
        (-0.25, "raw_evaluation_seconds", "Raw histories + diagnostics", "#777777"),
        (
            0,
            "compressed_evaluation_with_se_seconds",
            "Compressed profile + diagnostics",
            "#486ea9",
        ),
        (0.25, "polynomial_evaluation_seconds", "Polynomial values only", "#237c69"),
    ]:
        bars = ax.bar(
            x + offset, [r[key] for r in rows], width=0.24, label=label, color=color
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set(
        yscale="log",
        ylabel="Seconds for 1,001 p values",
        title="Reuse of the same saved profiles",
        xticks=x,
        xticklabels=["Stim surface d3", "StabIR surface d3"],
        ylim=(0.04, 300),
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / "polynomial_comparison.png", dpi=180)
    plt.close(fig)


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    reports = []
    ps = np.geomspace(0.001, 0.02, 1001)
    for name in [
        "repetition_depolarize2",
        "stim_surface_d3_r3",
        "stabir_surface_d3_r2_si1000",
    ]:
        profile = GeneralNoiseProfile.load(PROFILES / f"{name}.npz")
        polynomial, construction = median_time(profile.to_polynomial)
        polynomial.save(OUT / f"{name}_polynomial.npz")
        restored = LERPolynomial.load(OUT / f"{name}_polynomial.npz")
        values, polynomial_time = median_time(lambda restored=restored: restored(ps))
        curve, compressed_time = median_time(lambda profile=profile: profile.curve(ps))
        raw, raw_time = median_time(
            lambda profile=profile: np.array([raw_evaluate(profile, p) for p in ps]),
            repetitions=1,
        )
        np.testing.assert_allclose(values, raw[:, 0], rtol=3e-12, atol=1e-15)
        compressed = np.array(
            [
                [r.ler, r.standard_error, r.minimum_ess, r.missing_probability_mass]
                for r in curve
            ]
        )
        np.testing.assert_allclose(compressed, raw, rtol=3e-12, atol=1e-12)
        entry = {
            "case": name,
            "grid_points": len(ps),
            "profile_histories": len(profile.weights),
            "likelihood_records": profile.num_likelihood_records,
            "polynomial_terms": polynomial.num_terms,
            "degree_upper_bound": polynomial.degree,
            "polynomial_construction_seconds": construction,
            "raw_evaluation_seconds": raw_time,
            "compressed_evaluation_with_se_seconds": compressed_time,
            "polynomial_evaluation_seconds": polynomial_time,
            "polynomial_speedup": raw_time / polynomial_time,
            "compressed_with_se_speedup": raw_time / compressed_time,
            "profile_file_bytes": (PROFILES / f"{name}.npz").stat().st_size,
            "polynomial_file_bytes": (OUT / f"{name}_polynomial.npz").stat().st_size,
        }
        reports.append(entry)
        print(json.dumps(entry), flush=True)

    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.2) 0 1\nX_ERROR(.4) 0\nX_ERROR(.6) 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]",
        0.2,
    )
    zero = lambda det: np.zeros((len(det), 1), dtype=bool)
    profile = model.sample_profile(zero, shots_per_weight=10_000, seed=58)
    polynomial = profile.to_polynomial()
    polynomial.save(OUT / "corrected_counterexample_polynomial.npz")
    coefficients = [str(c) for c in polynomial.power_coefficients()]
    reference = model.enumerate_histories(zero, 0.2)
    counter = []
    for p in [0.0001, 0.001, 0.01, 0.1, 0.2, 1 / 3]:
        truth = model.enumerate_histories(zero, p)
        exact = (38 * p - 32 * p * p) / 15
        np.testing.assert_allclose(truth["ler"], exact, rtol=1e-13)
        prediction = profile.evaluate(p)
        assert abs(polynomial(p) - exact) <= 7 * prediction.standard_error + 1e-13
        counter.append(
            {
                "p": p,
                "exact_ler": exact,
                "unsupported_weight_only_baseline": float(
                    truth["weight_mass"] @ reference["conditional_ler"]
                ),
                "corrected_polynomial_ler": polynomial(p),
                "estimate": asdict(prediction),
            }
        )
    report = {
        "grid_benchmarks": reports,
        "counterexample_power_coefficients_constant_first": coefficients,
        "counterexample_exact_power_coefficients_constant_first": [
            "0",
            "38/15",
            "-32/15",
            "0",
        ],
        "counterexample": counter,
        "rare_event_comparison": rare_event_comparison(),
        "interpretation": "Sampling coefficient estimates; polynomial export does not remove sampling uncertainty or omitted-weight contributions.",
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    (OUT / "corrected_counterexample_polynomial.txt").write_text(
        str(polynomial.to_sympy(expanded=True))
    )
    write_figure(report, profile)
    return report


if __name__ == "__main__":
    run()
