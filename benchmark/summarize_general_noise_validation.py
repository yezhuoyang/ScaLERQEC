"""Render the completed general-noise experiments into a report and figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("experiment_results")


def main():
    matrix = json.loads((ROOT / "general_noise_matrix/results.json").read_text())
    oracles = json.loads((ROOT / "general_noise_oracles/results.json").read_text())
    matched = json.loads(
        (ROOT / "general_noise_oracles/matched_precision.json").read_text()
    )
    missed = json.loads(
        (ROOT / "general_noise_oracles/missed_failures.json").read_text()
    )
    followup = json.loads(
        (ROOT / "general_noise_matrix/rare_point_followup.json").read_text()
    )
    valid = [c for c in matrix["cases"] if "error" not in c]
    failures = [c for c in matrix["cases"] if "error" in c]
    rows = [r for c in valid for r in c["rows"]]
    resolved = [r for r in rows if r["mc_has_100_failures"]]
    ratios = [
        r["projected_single_point_speedup"]
        for r in resolved
        if "projected_single_point_speedup" in r
    ]
    oracle_rows = [r for c in oracles["cases"] for r in c["rows"]]
    max_true_z = max(abs(r["exact_profile_z"]) for r in oracle_rows)
    shots = sum(r["monte_carlo"]["shots"] for r in rows)
    shots += sum(r["mc_shots"] for r in oracle_rows)
    shots += sum(r["mc_shots"] for r in matched)
    shots += sum(r["new_monte_carlo"]["shots"] for r in followup)
    summary = {
        "circuit_cases": len(valid),
        "circuit_errors": failures,
        "circuit_points": len(rows),
        "points_with_100_mc_failures": len(resolved),
        "projected_profile_wins": sum(r > 1 for r in ratios),
        "projected_ratio_range": [min(ratios), max(ratios)],
        "oracle_cases": len(oracles["cases"]),
        "oracle_points": len(oracle_rows),
        "max_exact_standardized_discrepancy": max_true_z,
        "actual_mc_shots": shots,
        "matched_precision": matched,
    }
    (ROOT / "general_noise_matrix/summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    x = np.arange(2)
    ax = axes[0]
    for offset, key, color, label in [
        (-0.19, "profile_seconds", "#187d6d", "Profile incl. setup"),
        (0.19, "mc_seconds", "#a44637", "Actual Stim MC"),
    ]:
        values = [r[key] for r in matched]
        bars = ax.bar(x + offset, values, width=0.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.3g}s" for v in values], padding=4, fontsize=9)
    ax.set(
        yscale="log",
        ylim=(0.05, 80),
        xticks=x,
        xticklabels=["Steane\nnonuniform Pauli", "Repetition-5\nmixed DEPOL1/2"],
        ylabel="Measured runtime (seconds)",
        title="Rare events: matched true precision",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax = axes[1]
    seen = set()
    for c in valid:
        row = c["rows"][-1]
        if "projected_single_point_speedup" not in row:
            continue
        category = (
            "StabIR"
            if c["frontend"] == "stabir"
            else ("Stim repetition" if c["code"] == "repetition" else "Stim surface")
        )
        color = {
            "StabIR": "#8960a8",
            "Stim repetition": "#2072a2",
            "Stim surface": "#bc7023",
        }[category]
        offset = {
            "uniform_single": -0.12,
            "nonuniform_depolarizing": 0,
            "biased_pauli": 0.12,
            "SD6": -0.08,
            "SI1000": 0.08,
        }[c["noise"]]
        ax.scatter(
            c["distance"] + offset,
            row["projected_single_point_speedup"],
            s=40,
            color=color,
            alpha=0.75,
            label=category if category not in seen else None,
        )
        seen.add(category)
    ax.axhline(1, color="#555", linestyle="--", linewidth=1)
    ax.text(3.1, 1.17, "Above 1 favors profiling", fontsize=8)
    ax.set(
        yscale="log",
        ylim=(0.001, 2),
        xticks=[3, 5, 7, 9],
        xlabel="Code distance",
        ylabel="MC time / profile time",
        title="Circuit-level: projected matched SE",
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.2)
    ax = axes[2]
    for c in oracles["cases"]:
        for r in c["rows"]:
            value = r["estimated_se_over_exact_se"]
            if value is None:
                continue
            color = "#a44637" if value < 0.1 else "#2072a2"
            ax.scatter(r["exact_ler"], value, s=19, alpha=0.6, color=color)
    ax.axhline(1, color="#555", linestyle="--", linewidth=1)
    ax.text(
        1e-12,
        0.023,
        "Two missed-failure cases:\nSE understated 82x and 100x",
        color="#a44637",
        fontsize=9,
    )
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(0.005, 3),
        xlabel="Exact LER",
        ylabel="Reported SE / true standard deviation",
        title="High ESS can miss rare failures",
    )
    ax.grid(alpha=0.2)
    fig.savefig(ROOT / "general_noise_matrix/validation_summary.png", dpi=180)
    fig.savefig(ROOT / "general_noise_matrix/validation_summary.pdf")

    perf_table = "\n".join(
        f"| {r['name']} / {r['family']} | {r['profile_shots']:,} | {r['profile_seconds']:.3f} s | "
        f"{r['mc_shots']:,} | {r['mc_seconds']:.2f} s | {r['observed_time_ratio']:.1f}x |"
        for r in matched
    )
    case_table = "\n".join(
        f"| {c['name']} | {c['rounds']} | {c['noise_locations']} | {c['total_profile_shots']:,} | "
        f"{c['timing']['total_seconds']:.2f} s | "
        f"{c['rows'][-1].get('projected_single_point_speedup', float('nan')):.4f} |"
        for c in valid
    )
    follow_table = "\n".join(
        f"| {r['name']} | {r['profile']['ler']:.6g} | {r['new_monte_carlo']['ler']:.6g} | "
        f"{r['new_monte_carlo']['failures']} / {r['new_monte_carlo']['shots']:,} |"
        for r in followup
    )
    errors = "None." if not failures else "\n".join(str(c) for c in failures)
    text = f"""# Broader general-noise validation and Monte Carlo comparison

Local research results, September 16, 2026. The method has a demonstrated
advantage on selected rare-event examples. The current general-noise prototype
is slower than direct Stim Monte Carlo on the tested circuit-level workloads.
The tests also expose seriously overconfident sample standard errors when rare
failures inside a weight stratum are unobserved. These results do not justify
a universal speedup, universal accuracy, or a bug-free implementation claim.

![Measured benefits, circuit-level costs, and missed-failure uncertainty](../experiment_results/general_noise_matrix/validation_summary.png)

## Experimental scope

* **{len(oracles["cases"])} exact-oracle configurations, {len(oracle_rows)} p comparisons.** Five-qubit,
  Steane, Shor, and repetition-3/5/7 codes crossed with seven noise families:
  uniform single-qubit depolarization; nonuniform biased single-qubit Paulis;
  mixed DEPOLARIZE1/2; biased PAULI_CHANNEL_2; correlated E/ELSE chains;
  heralded Pauli channels; and measurement-record noise. The checks use
  p=0.001, 0.01, 0.05, 0.15 with reference p0=0.05.
* **{len(valid)} circuit-level configurations, {len(rows)} p comparisons.** Stim repetition
  distances 3/5/9; rotated Z surface distances 3/5/7; rotated X and unrotated Z
  surface distances 3/5; StabIR surface distances 3/5 and repetition 3/7.
  Stim circuits use rounds=distance; StabIR circuits use three rounds. Noise
  includes uniform single-qubit channels, nonuniform circuit-level depolarization,
  biased single-/two-qubit channels, SD6, and SI1000. This is a selected matrix,
  not the complete Cartesian product: distance-7 surface experiments use
  uniform-single and nonuniform-depolarizing noise in the Z memory basis.
* **210 additional replicate profiles** for SE calibration: 30 seeds per noise
  family on the five-qubit example at p=0.01. Seeds are reused across families,
  and p points from one profile share samples; do not treat all reported points
  as independent experiments.
* **{shots:,} actual Stim Monte Carlo shots**, including the matched-precision
  experiments and the two rare-point follow-ups. The original matrix uses one
  million MC shots per circuit point and 300,000 per exact-oracle point.

The exact oracle uses independent Pauli commutation and syndrome-state
convolution. It never calls the production noise parser, fault responses, or
enumerator. Unit tests compare every channel outcome's fault response and
Pauli weight directly against that independent calculation. The exact MC
circuits are also generated directly at each p instead of using the production
`circuit_at` scaling function. Code-capacity examples have ideal encoding and
syndrome extraction except in the named readout experiment. Their fixed lookup
decoder uses the exact distribution at p0; the heralded experiment ignores the
flags in decoding. Circuit-level experiments use a fixed PyMatching decoder
constructed at p=0.003. Approximate disjoint DEM conversion is used only to
define that decoder, never to sample noise in either method.
Weight remains the number of inserted single-qubit Pauli factors at their
original locations; XX has weight two. Record flips and heralded identity
outcomes have weight zero under this convention.

## Actual matched-precision advantage

At p=0.001, exact variance determines the ordinary MC sample count before the
experiment. We actually ran those shots; these two rows are not extrapolations.
Both methods use the same fixed decoder. Profile time includes circuit/model
construction, fault-response compilation, sampling, decoding, and evaluation;
MC time includes target-circuit/sampler construction, sampling, and decoding.
The common lookup decoder is used by both. Oracle calculations used solely for
validation and setting the MC budget are outside both timings.

| Code / noise | Profile shots | Profile time | Actual MC shots | MC time | Measured speedup |
|---|---:|---:|---:|---:|---:|
{perf_table}

The Steane example has exact LER {matched[0]["exact_ler"]:.8g} and matched true SE
{matched[0]["profile_true_se"]:.8g}; the repetition example has exact LER
{matched[1]["exact_ler"]:.8g} and matched true SE {matched[1]["profile_true_se"]:.8g}.
The profiles simultaneously retain their entire estimated polynomial. These
are code-capacity examples, not evidence of the same speedup at circuit level.

## Circuit-level cost and accuracy limits

Of {len(rows)} matrix points, only {len(resolved)} had at least 100 MC failures. The
remaining {len(rows) - len(resolved)} provide weaker or inconclusive rare-event comparisons.
Across the {len(ratios)} points where a precision-cost projection was usable,
profiling won {sum(r > 1 for r in ratios)} times. Projected MC/profile runtime ratios
ranged from {min(ratios):.4g} to {max(ratios):.4g}. Values below 1 favor direct MC.
These projections use the reported profile SE and measured MC throughput;
they are not experiments at every projected budget. An underestimated profile
SE can make profiling look better in this comparison, so these numbers do not
certify its uncertainty. A three-point polynomial sweep does not erase the
large setup/sampling cost on these tested circuits.

Follow-up MC at p=0.001, selected after observing sparse matrix discrepancies:

| Circuit | Profile estimate | Independent MC estimate | MC failures / shots |
|---|---:|---:|---:|
{follow_table}

These follow-ups are exploratory diagnostics, not a pre-registered multiple-test
significance analysis. They test the same fixed decoder. They must be retained
alongside favorable results rather than hidden by aggregate agreement counts.
Their minimum likelihood ESS values were approximately 3.90 and 1.04 out of
2000 per weight, warning of poor overlap. The exact examples below show a
different limitation: high ESS also cannot certify adequate failure sampling.

## A failure of the estimated SE, not of the expectation identity

For repetition-7 with nonuniform single-qubit noise at p=0.001, the exact LER
is 5.14473535e-13. The profile gives 1.07230110e-13 with reported SE 1.83216444e-14,
despite minimum ESS 4989.6 out of 5000. It observed no failures at weight 3.
The expected count of weight-3 failures in this budget is only 0.0774; observing
none has probability 92.6%. Those histories dominate the low-p answer.

The independent second-moment calculation finds true standard deviation
1.50724901e-12, about 82x the reported SE. The heralded variant understates SE
by about 100x. The corresponding point estimates are about 79% low in this
realization. The estimator is unbiased over repeated experiments: rare large
contributions restore its expectation. A typical small sample can still miss
those contributions and report a misleadingly small SE.

Across all {len(oracle_rows)} oracle comparisons, the largest discrepancy is {max_true_z:.3f}
true standard deviations. This supports the mathematical estimator and fault
semantics, while the failed reported-SE checks expose inadequate uncertainty
diagnostics. The 210 replicate five-qubit profiles had 200/210 nominal 95%
normal intervals cover truth; that aggregate result does not protect the
repetition-7 rare-failure examples.

`profile.confidence_bounds(p)` now adds a conservative fixed-budget pointwise
interval using an exact maximum-likelihood-ratio DP and a two-sided empirical
Bernstein bound. It covers both counterexamples but is extremely loose: the
upper endpoints are approximately {missed[0]["bounds"]["upper"]:.5g} and
{missed[1]["bounds"]["upper"]:.5g}, versus true LER near 5e-13. It exposes insufficient
information; it does not repair the estimate or certify useful rare-event
precision. See [the derivation](general_noise_math.md#finite-sample-confidence-and-missed-failures).

## Reproduction and verification

```
python benchmark/general_noise_oracles.py
python benchmark/general_noise_matrix.py
python -c "import json; from benchmark.general_noise_matrix import OUT, confirm_rare_points; (OUT/'rare_point_followup.json').write_text(json.dumps(confirm_rare_points(), indent=2))"
python benchmark/summarize_general_noise_validation.py
```

The complete suite passed **810 tests, two existing skips**. All **80 new
oracle/uncertainty tests** also passed under Stim 1.16. Coverage is 100% for the
new confidence module, 100% for polynomial export, and 97% for general-noise
profiling. Tests were run locally on Windows; remote CI has not been run.
The local Python 3.14 coverage run needed numerical dependencies imported before
starting coverage because otherwise NumPy raised a duplicate extension-load
error; tests without coverage and the separate Stim 1.16 environment passed.

Timings are observations from a shared development machine with other processes
running, not isolated hardware benchmarks. They establish the observed regimes,
not portable speedup constants. The new experiments exercise the Python+Stim
general-noise path. Existing native propagation/sampling audits and the full
suite also pass, but this is not a new exhaustive audit of every C++ path.
Recorded matrix execution errors: {errors}

## What the evidence supports, and what needs work

The finite-history reweighting identity is exact for the documented supported
noise family and a fixed decoder. Finite tests cannot prove that all code is
bug-free. Generality here means supported classical Pauli/record noise with
probabilities proportional to one p (including conditional ELSE probabilities),
not arbitrary coherent, non-Pauli, or unrestricted non-Markovian noise. This
study does not validate every Stim program or large LDPC code family.

The next statistical improvement should allocate samples to rare likelihood
groups within each Pauli weight, using additional activation/rate information.
Post-sampling compression already records that information but does not ensure
those groups were sampled. Independent pilot and production budgets can support
allocation without silently invoking invalid optional-stopping guarantees.
The next performance improvement should target conditional sampling and bulk
fault-response compilation, which dominate the larger circuit runs. Faster
polynomial evaluation alone does not solve either bottleneck. The extension
remains experimental and has not been published as a validated general solver.

## Complete circuit matrix

The final column projects MC/profile runtime at p=0.01 for the same reported SE.
NaN means too few MC failures for that comparison, not zero cost or zero LER.

| Configuration | Rounds | Noise locations | Profile trials | Profile total | MC/profile ratio |
|---|---:|---:|---:|---:|---:|
{case_table}
"""
    Path("docs/general_noise_broad_validation.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
