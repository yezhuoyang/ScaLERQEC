"""Regenerate the report from all declared cases, including failures/timeouts."""

import csv
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.large_code_cases import cases
from benchmark.large_code_validation import OUT, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = [
        read(OUT / (spec["name"] + ".json"))
        if (OUT / (spec["name"] + ".json")).exists()
        else {"spec": spec, "status": "not_started"}
        for spec in cases()
    ]
    groups = {}
    for family in ["surface", "stabir", "color", "hgp", "bicycle"]:
        subset = [r for r in rows if r["spec"]["family"] == family]
        groups[family] = {
            "declared_cases": len(subset),
            "statuses": dict(Counter(r["status"] for r in subset)),
            "fault_audits_passed": sum(
                r.get("audit", {}).get("status") == "passed" for r in subset
            ),
            "mc_points": sum(len(r.get("monte_carlo", [])) for r in subset),
            "accuracy_met_profiles": sum(
                r.get("adaptive", {}).get("status") == "accuracy_met" for r in subset
            ),
        }
    mc = [p for r in rows for p in r.get("monte_carlo", [])]
    oracles = [read(p) for p in sorted(OUT.glob("hgp_oracles*/*.json"))]
    for row in oracles:
        row.setdefault("decoder", "BP+OSD0")
    moments = [read(p) for p in sorted((OUT / "moments").glob("*.json"))]
    native = [read(p) for p in sorted((OUT / "native").glob("*.json"))]
    longer = [read(p) for p in sorted((OUT / "longer_budget").glob("*.json"))]
    accepted = [read(p) for p in sorted((OUT / "accepted_checks").glob("*.json"))]
    summary = {
        "declared_cases": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "families": groups,
        "profile_statuses": dict(
            Counter(r.get("adaptive", {}).get("status", "not_reached") for r in rows)
        ),
        "fault_histories_replayed": sum(
            r.get("audit", {}).get("histories_replayed", 0) for r in rows
        ),
        "independent_hgp_commutation_columns": sum(
            r.get("independent_commutation_columns", 0) for r in rows
        ),
        "mc_shots": sum(p["shots"] for p in mc),
        "mc_points": len(mc),
        "mc_interval_disagreements": sum(not p["profile_interval_overlap"] for p in mc),
        "accepted_accuracy_contradictions": sum(
            p["accepted_accuracy_contradiction"] for p in mc
        ),
        "mc_references_resolve_10_percent": sum(
            p["reference_resolves_10_percent"] for p in mc
        ),
        "bounded_hgp_runs": len(oracles),
        "bounded_hgp_profiles_accuracy_met": sum(
            r["adaptive"]["status"] == "accuracy_met" for r in oracles
        ),
        "bounded_hgp_points_verified": sum(
            p["accuracy_verified_by_oracle"]
            for r in oracles
            for p in r["oracle"]["points"]
        ),
        "bounded_hgp_points": sum(len(r["oracle"]["points"]) for r in oracles),
        "moment_cases": len(moments),
        "moment_draws": sum(r["shots"] for r in moments),
        "largest_moment_standardized_difference": max(
            (s["standardized_difference"] for r in moments for s in r["statistics"]),
            default=None,
        ),
        "native_cases": len(native),
        "native_histories_replayed": sum(r["histories_replayed"] for r in native),
        "longer_budget_followups": [
            {
                "name": r["spec"]["name"],
                "status": r["status"],
                "profile_status": r.get("adaptive", {}).get("status"),
                "shots": r.get("adaptive", {}).get("shots"),
                "max_seconds": r["max_seconds"],
            }
            for r in longer
        ],
        "accepted_profile_oracle_points": sum(
            len(r["oracle"]["points"]) for r in accepted
        ),
        "accepted_profile_oracle_points_verified": sum(
            p["accuracy_verified_by_oracle"]
            for r in accepted
            for p in r["oracle"]["points"]
        ),
        "accepted_profile_oracle_contradictions": sum(
            p["contradiction"] for r in accepted for p in r["oracle"]["points"]
        ),
    }
    write_json(OUT / "summary.json", summary)
    largest_moment_difference = summary["largest_moment_standardized_difference"]
    moment_description = (
        "unavailable"
        if largest_moment_difference is None
        else f"{largest_moment_difference:.3f} true standard errors"
    )
    with (OUT / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "run_status",
                "p",
                "profile_status",
                "profile_estimate",
                "profile_lower",
                "profile_upper",
                "mc_shots",
                "mc_failures",
                "mc_estimate",
                "mc_lower",
                "mc_upper",
                "reference_resolves_10_percent",
            ]
        )
        for row in rows:
            estimates = row.get("adaptive", {}).get("estimates", [])
            for p, estimate in zip(row.get("monte_carlo", []), estimates):
                writer.writerow(
                    [
                        row["spec"]["name"],
                        row["status"],
                        p["p"],
                        row["adaptive"]["status"],
                        estimate["ler"],
                        estimate["lower"],
                        estimate["upper"],
                        p["shots"],
                        p["failures"],
                        p["ler"],
                        p["lower"],
                        p["upper"],
                        p["reference_resolves_10_percent"],
                    ]
                )
    lines = [
        "# Large-code validation: recorded results",
        "",
        (
            "Generated from the predeclared matrix and supplementary oracle runs. "
            "Completion is not an accuracy pass. Bicycle case IDs retain the historical "
            "`z` token, but the stored circuits are X memory with only logical k−1 defined."
        ),
        "",
        "| Family | Cases | Completed | Failed / timed out | Audits passed | MC points | Accuracy met |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, g in groups.items():
        s = g["statuses"]
        lines.append(
            f"| {name} | {g['declared_cases']} | {s.get('completed', 0)} | "
            f"{s.get('failed', 0)} / {s.get('process_timeout', 0)} | "
            f"{g['fault_audits_passed']} | {g['mc_points']} | {g['accuracy_met_profiles']} |"
        )
    lines += [
        "",
        (
            f"Monte Carlo: {summary['mc_shots']:,} shots across {len(mc)} points. "
            f"Accepted accuracy contradictions: {summary['accepted_accuracy_contradictions']}. "
            "Interval overlap with an unresolved [0,1]-like profile interval is uninformative."
        ),
        "",
        "## Distance 13, 13 rounds",
        "",
        "| Case | Run / profile status | MC failures / shots at p=0.001, 0.003, 0.01 |",
        "|---|---|---|",
    ]
    for r in rows:
        if r["spec"].get("distance") != 13:
            continue
        evidence = "; ".join(
            f"{p['failures']:,} / {p['shots']:,}" for p in r.get("monte_carlo", [])
        )
        lines.append(
            f"| {r['spec']['name']} | {r['status']} / "
            f"{r.get('adaptive', {}).get('status', 'not reached')} | {evidence or 'unavailable'} |"
        )
    if longer:
        lines += ["", "### Longer distance-13 budget", ""]
    for r in longer:
        a = r.get("adaptive", {})
        lines.append(
            f"{r['spec']['name']}, {r['max_seconds']}-second profiling budget: "
            f"{r['status']}, profile {a.get('status', 'running')}, "
            f"{a.get('shots', 0):,} random samples. "
            "This is a separate follow-up; the original 30-second run remains recorded."
        )
        if a.get("estimates"):
            lines += [
                "",
                "| p | Estimate | Lower | Upper | Accuracy met |",
                "|---:|---:|---:|---:|---|",
            ]
            for e in a["estimates"]:
                lines.append(
                    f"| {e['p']:g} | {e['ler']:.6g} | {e['lower']:.6g} | "
                    f"{e['upper']:.6g} | {e['accuracy_met']} |"
                )
    lines += [
        "",
        "## Independent bounded HGP58 reference",
        "",
        "| Decoder | Noise | Profile status | p points verified to ±10% | Random / exact histories |",
        "|---|---|---|---:|---:|",
    ]
    for r in oracles:
        n = sum(p["accuracy_verified_by_oracle"] for p in r["oracle"]["points"])
        a = r["adaptive"]
        lines.append(
            f"| {r['decoder']} | {r['spec']['noise']} | {a['status']} | {n}/3 | "
            f"{a['shots']:,} / {a['exact_histories']:,} |"
        )
    lines += [
        "",
        "### Additional checks on accepted main-matrix profiles",
        "",
        (
            "Each accepted main-matrix profile was checked against the independent "
            "weight-at-most-two oracle at its actual p grid. "
            f"{summary['accepted_profile_oracle_points_verified']} of "
            f"{summary['accepted_profile_oracle_points']} points were verified to 10%; "
            f"there were {summary['accepted_profile_oracle_contradictions']} interval contradictions. "
            "A wide oracle enclosure is inconclusive, not a failure or a pass."
        ),
    ]
    lines += ["", "## Cases that did not complete", ""]
    for r in rows:
        if r["status"] != "completed":
            lines.append(
                f"- **{r['spec']['name']}**: {r['status']}, phase `{r.get('phase', 'not started')}`. "
                f"{r.get('error', '')}"
            )
    lines += [
        "",
        "## Component checks",
        "",
        (
            f"Independent general-noise replay: {summary['fault_histories_replayed']} histories. "
            f"Independent HGP CSS commutation: {summary['independent_hgp_commutation_columns']:,} outcome columns. "
            f"Native replay: {summary['native_histories_replayed']} histories in {len(native)} cases."
        ),
        "",
        (
            f"Conditional moments: {len(moments)} large-code cases, "
            f"{summary['moment_draws']:,} draws; largest difference "
            f"{moment_description}. "
            "This checks selected moments, not the entire joint distribution."
        ),
        "",
        "See [method, limits, and reproduction](../../docs/large_code_validation.md).",
        "",
    ]
    (OUT / "results.md").write_text(
        "\n".join(line.rstrip() for line in lines), encoding="utf-8"
    )

    selected = [
        r for r in rows if r["spec"].get("distance") == 13 and r.get("monte_carlo")
    ]
    if selected:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
        colors = ["#116466", "#d48b13", "#824ca0"]
        for i, r in enumerate(selected):
            for j, p in enumerate(r["monte_carlo"]):
                x = i + (j - 1) * 0.2
                if p["failures"]:
                    axes[0].errorbar(
                        x,
                        p["ler"],
                        yerr=[[p["ler"] - p["lower"]], [p["upper"] - p["ler"]]],
                        fmt="o",
                        color=colors[j],
                        markersize=4,
                        capsize=2,
                    )
                else:
                    axes[0].plot(x, p["upper"], "v", color=colors[j], markersize=5)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Direct Stim Monte Carlo LER / upper bound")
        axes[0].set_title("Distance 13, 13 rounds\nProfile accuracy remains unresolved")
        axes[0].set_xticks(
            range(len(selected)),
            [
                r["spec"]["name"]
                .replace("surface_13_", "")
                .replace("stabir_13_z_", "StabIR ")
                for r in selected
            ],
            rotation=55,
            ha="right",
            fontsize=8,
        )
        for j, p in enumerate([0.001, 0.003, 0.01]):
            axes[0].plot([], [], "o", color=colors[j], label=f"p={p:g}")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.2)
        labels = []
        for i, r in enumerate(moments):
            labels.append(
                r["spec"]["name"]
                .replace("nonuniform_depolarizing", "mixed")
                .replace("biased_pauli", "biased")
                .replace("bicycle_72_z_", "BB72 X, logical 11: ")
                .replace("bicycle_144_z_", "BB144 X, logical 11: ")
            )
            for s in r["statistics"]:
                axes[1].plot(
                    s["standardized_difference"], i, "o", color="#116466", alpha=0.6
                )
        axes[1].axvline(7, color="#b34735", linestyle="--", label="declared 7-SE check")
        axes[1].set_yticks(range(len(labels)), labels, fontsize=8)
        axes[1].set_xlabel("Absolute discrepancy / exact standard error")
        axes[1].set_title(
            "Conditional sampler moment checks\n2,048 draws per circuit; four statistics"
        )
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="x", alpha=0.2)
        fig.savefig(OUT / "validation_summary.png", dpi=170)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
