"""Summarize all outcomes, including interrupted and uncertified runs."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scalerqec.Stratified import LERPolynomial

ROOT = Path("experiment_results")
FOLDERS = [
    "uniformized_validation",
    "uniformized_final_validation",
    "uniformized_precision_validation",
]


def references(name, circuit_hash, decoder):
    folders = [
        "large_code_validation",
        "conditional_fix_validation",
        "conditional_fix_validation/before_log_audit_fix",
        *FOLDERS,
        "uniformized_references",
        "uniformized_lowp_reference",
    ]
    best = {}
    for folder in folders:
        path = ROOT / folder / (name + ".json")
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row.get("circuit_sha256") != circuit_hash or row.get("decoder") != decoder:
            continue
        for ref in row.get("monte_carlo", []) + row.get("fresh_monte_carlo", []):
            if "lower" not in ref or "upper" not in ref:
                continue
            if ref["shots"] > best.get(ref["p"], {}).get("shots", 0):
                best[ref["p"]] = {
                    **ref,
                    "source": str(path),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    return best


def run():
    runs, chosen = [], {}
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).glob("*.json")):
            row = json.loads(path.read_text())
            if "spec" not in row:
                continue
            name = row["spec"]["name"]
            assert row["status"] in {"completed", "interrupted"}, (path, row["status"])
            runs.append(
                {
                    "file": str(path),
                    "name": name,
                    "status": row["status"],
                    "profile_status": row.get("profile", {}).get("status"),
                }
            )
            if row["status"] != "completed":
                continue
            assert row["replay"]["passed"] and row["replay"]["histories"] == 10
            assert row["profile"]["joint_histogram_shots"] == row["profile"]["shots"]
            polynomial = LERPolynomial.load(path.with_name(name + "_polynomial.npz"))
            np.testing.assert_allclose(
                polynomial(row["p_values"]),
                [e["ler"] for e in row["profile"]["estimates"]],
                rtol=2e-10,
                atol=1e-15,
            )
            assert polynomial.metadata["accuracy_certified"] == (
                row["profile"]["status"] == "accuracy_met"
            )
            chosen[name] = (path, row)
    assert len(chosen) == 16, f"Expected 16 distinct cases, found {len(chosen)}"
    details = []
    for name, (path, row) in sorted(chosen.items()):
        refs = references(name, row["circuit_sha256"], row["decoder"])
        comparisons = []
        for e in row["profile"]["estimates"]:
            ref = refs.get(e["p"])
            comparison = {**e, "reference": ref}
            if ref:
                comparison["intervals_overlap"] = max(e["lower"], ref["lower"]) <= min(
                    e["upper"], ref["upper"]
                )
                if ref["lower"] > 0:
                    relative_bound = max(
                        abs(e["ler"] - ref["lower"]) / ref["lower"],
                        abs(e["ler"] - ref["upper"]) / ref["upper"],
                    )
                else:
                    relative_bound = None
                comparison["reference_relative_error_bound"] = relative_bound
                comparison["reference_verifies_10_percent"] = (
                    relative_bound is not None and relative_bound <= 0.1
                )
            comparisons.append(comparison)
        details.append(
            {
                "name": name,
                "selected_run": str(path),
                "spec": row["spec"],
                "status": row["profile"]["status"],
                "shots": row["profile"]["shots"],
                "seconds": row["profile"]["seconds"],
                "sampler_array_bytes": row["sampler_array_bytes"],
                "dense_response_bytes": row["dense_response_bytes"],
                "comparisons": comparisons,
            }
        )
    all_points = [e for row in details for e in row["comparisons"]]
    summary = {
        "scope": "16 fixed circuit/decoder/noise cases; no universal efficiency claim",
        "confidence_scope": "99% simultaneous over the three p points and sampling checkpoints within each run; not a familywise claim across all runs",
        "distinct_cases": len(details),
        "completed_runs": sum(r["status"] == "completed" for r in runs),
        "interrupted_runs": sum(r["status"] == "interrupted" for r in runs),
        "whole_grid_certified_cases": sum(
            r["status"] == "accuracy_met" for r in details
        ),
        "certified_points": sum(e["accuracy_met"] for e in all_points),
        "total_points": len(all_points),
        "reference_points": sum(e["reference"] is not None for e in all_points),
        "nonoverlapping_intervals": sum(
            e.get("intervals_overlap") is False for e in all_points
        ),
        "independently_verified_10_percent_points": sum(
            e.get("reference_verifies_10_percent", False) for e in all_points
        ),
        "certified_and_independently_verified_points": sum(
            e["accuracy_met"] and e.get("reference_verifies_10_percent", False)
            for e in all_points
        ),
        "runs": runs,
        "cases": details,
    }
    target = ROOT / "uniformized_final_validation"
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Uniformized polynomial validation",
        "",
        (
            f"{len(details)} distinct cases; {summary['whole_grid_certified_cases']} certify the whole requested p grid; "
            f"{summary['certified_points']}/{len(all_points)} individual points meet the statistical 10% accuracy target."
        ),
        "",
        (
            f"Independent references exist for {summary['reference_points']}/{len(all_points)} points. "
            f"{summary['nonoverlapping_intervals']} profile/reference interval pairs do not overlap. "
            f"The reference intervals independently bound relative error by 10% at {summary['independently_verified_10_percent_points']} points. "
            "Wide overlapping intervals are not evidence of 10% accuracy."
        ),
        "",
        (
            "Intervals are 99% simultaneous within one declared p grid and over its sequential checks. "
            "They are not a simultaneous 99% assertion across all benchmark cases. "
            "The selected run is the final implementation where available, and the declared longer precision run for two HGP cases. "
            "Four additional cases retain the initial KL-only run. Every run, including the interrupted first bicycle attempt, is listed in summary.json."
        ),
        "",
        "| Case | Shots | Profile seconds | Points certified | Whole grid |",
        "|---|---:|---:|---:|---|",
    ]
    for row in details:
        lines.append(
            f"| {row['name']} | {row['shots']:,} | {row['seconds']:.2f} | {sum(e['accuracy_met'] for e in row['comparisons'])}/3 | {row['status']} |"
        )
    lines += [
        "",
        "## One polynomial across three p values",
        "",
        (
            "The HGP-58 correlated-noise run stopped automatically after 479,743 samples. "
            "Each independent Stim comparison below uses 250,000 preselected shots. "
            "Both experiments use the same fixed decoder."
        ),
        "",
        "| p | Polynomial LER | Profile interval | Independent Stim LER |",
        "|---:|---:|---:|---:|",
    ]
    correlated = next(r for r in details if r["name"] == "hgp_58_z_correlated_else")
    for e in correlated["comparisons"]:
        lines.append(
            f"| {e['p']} | {e['ler']:.8f} | [{e['lower']:.8f}, {e['upper']:.8f}] | {e['reference']['ler']:.8f} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        (
            "Surface distances 3, 7, and 13 are represented. Distance 13 includes both memory bases plus StabIR SD6 and SI1000. "
            "QLDPC cases include HGP 58/180/245 and bicycle 72/144; a color-code case is also included. "
            "The tests additionally compare seven noise families on small codes to independent exact Pauli calculations."
        ),
        "",
        (
            "HGP examples here use ideal encoding/syndrome measurement, with all defined logical outputs. "
            "Bicycle fixtures define one logical observable, despite their multiple output slots. "
            "The historical bicycle filenames containing z label X-memory fixtures. "
            "These are the same fixed decoders used by the references; the baseline HGP decoder is not a claim of optimal code performance."
        ),
        "",
        (
            "Dense response compilation is still separate from the new sparse sampler. "
            "The sampler avoids conditional-weight suffix tables, but large BP+OSD decoding remains slow. "
            "Profile seconds include sampler setup and decoding, while circuit response compilation and the independent replay audit are recorded separately. "
            "Budgets are cooperative between batches, so one call may overrun. Concurrent jobs make timings unsuitable for a controlled speedup claim."
        ),
        "",
        (
            "Low-p distance-13 values remain uncertified. Extremely small importance point estimates can simply reflect unobserved relevant failures. "
            "Uncertified curves must not be used as accuracy-validated predictions. The public exporter rejects them by default."
        ),
        "",
        "## Reproduce",
        "",
        "```text",
        "python -m benchmark.uniformized_validation CASE --seconds 60 --fresh-shots 4096 --output NEW_DIRECTORY",
        "python -m benchmark.uniformized_references hgp_58_z_correlated_else hgp_58_z_uniform_single --shots 250000 --output NEW_REFERENCE_DIRECTORY",
        "python -m benchmark.summarize_uniformized",
        "```",
        "",
        (
            "Original result directories are never overwritten by the experiment runner. "
            "Each run records source and circuit hashes, decoder description, independent replay outcomes, actual budgets, and a reload-verified polynomial."
        ),
    ]
    (target / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({k: v for k, v in summary.items() if k not in {"cases", "runs"}})


if __name__ == "__main__":
    run()
