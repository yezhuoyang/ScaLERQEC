"""Summarize the completed reruns, retaining unresolved outcomes explicitly."""

import argparse
import json
from pathlib import Path

OUT = Path("experiment_results/conditional_fix_validation")
NAMES = [
    "surface_13_z_nonuniform_depolarizing",
    "surface_13_x_biased_pauli",
    "stabir_13_z_SD6",
    "stabir_13_z_SI1000",
    "hgp_58_z_correlated_else",
    "hgp_180_z_nonuniform_depolarizing",
    "hgp_245_z_biased_pauli",
    "color_7_xyz_nonuniform_depolarizing",
    "bicycle_72_z_nonuniform_depolarizing",
    "bicycle_144_z_nonuniform_depolarizing",
    "bicycle_144_z_biased_pauli",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(output=OUT):
    output = Path(output)
    rows = [read(output / (name + ".json")) for name in NAMES]
    if any(r["status"] != "completed" for r in rows):
        raise RuntimeError("Not every requested rerun completed; inspect raw records.")
    baseline = read(
        Path(
            "experiment_results/large_code_validation/longer_budget/surface_13_z_nonuniform_depolarizing.json"
        )
    )
    d13 = rows[0]["adaptive"]
    moments = [r["moments"] for r in rows if "moments" in r]
    moments += [read(p) for p in (output / "moments").glob("*.json")]
    summary = {
        "completed_configurations": len(rows),
        "accuracy_met": sum(r["adaptive"]["status"] == "accuracy_met" for r in rows),
        "histories_replayed": sum(r["audit"]["histories_replayed"] for r in rows),
        "moment_draws": sum(r["shots"] for r in moments),
        "moment_checks_passed": all(
            r["within_7_true_standard_errors"] for r in moments
        ),
        "max_moment_standardized_difference": max(
            s["standardized_difference"] for r in moments for s in r["statistics"]
        ),
        "baseline_d13_shots": baseline["adaptive"]["shots"],
        "fixed_d13_shots": d13["shots"],
        "d13_shot_count_ratio": d13["shots"] / baseline["adaptive"]["shots"],
        "d13_high_p_estimate": d13["estimates"][-1]["ler"],
        "d13_mc_high_p_estimate": rows[0]["historical_monte_carlo"][-1]["ler"],
        "scope": "Representative reruns and sampler checks; no claim of universal accuracy or speedup.",
    }
    summary["d13_high_p_point_difference_fraction"] = abs(
        summary["d13_high_p_estimate"] / summary["d13_mc_high_p_estimate"] - 1
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Conditional sampler fix reruns",
        "",
        "Every listed run used the same fixed decoder and circuit as its historical reference (circuit hash checked). Most comparisons reuse those independent MC samples; fresh d13 MC is also saved. Unresolved intervals are not accuracy passes.",
        "",
        "| Case | Random shots | Profile seconds | Status |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        a = row["adaptive"]
        lines.append(
            f"| {row['spec']['name']} | {a['shots']:,} | {a['seconds']:.1f} | {a['status']}: {a['reason']} |"
        )
    lines += [
        "",
        f"{summary['histories_replayed']} actual histories were replayed independently in Stim. {summary['moment_draws']:,} conditional draws passed the predeclared seven-standard-error moment checks; largest standardized difference {summary['max_moment_standardized_difference']:.3f}.",
        "",
        "Both BB144 cases must explicitly include low weight 2 in their replay records. The intermediate run in before_log_audit_fix checked only typical weight and stopped during its third MC point after more than 15 minutes; it is not an underflow regression pass.",
        "",
        "BB fixtures expose one defined logical observable, not full-block LER. HGP tests use ideal syndrome measurements and do not validate circuit-level QLDPC noise. Wall times include different concurrent local workloads; these are resource-capped validation runs, not a controlled throughput benchmark.",
    ]
    for row in rows:
        if row["spec"].get("n") == 144:
            assert row["audit"]["histories_per_weight"].get("2") == 4
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
