"""Check and plot the saved QStabIR example runs without resampling."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim

from examples.qstabir_noise import define_code, make_noise
from scalerqec.Stratified import LERPolynomial


def report(directory):
    directory = Path(directory)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    rows = []
    for scheme, ax in zip(["Standard", "Flag", "Shor", "Knill"], axes.flat):
        folder = directory / scheme.lower()
        result = json.loads((folder / "results.json").read_text())
        fresh = define_code(scheme)
        fresh.noisemodel = make_noise(result["p_ref"])
        fresh.construct_circuit()
        assert fresh.stimcirc.flattened() == stim.Circuit.from_file(
            str(folder / "circuit.stim")
        )
        polynomial = LERPolynomial.load(folder / "ler_polynomial.npz")
        assert result["status"] == "accuracy_met"
        ps = np.array(result["probabilities"])
        dense = np.geomspace(ps.min(), ps.max(), 200)
        ax.plot(dense, polynomial(dense), color="#146a89", label="Reused polynomial")
        for e, mc in zip(result["estimates"], result["independent_stim_mc"]):
            assert e["p"] == mc["p"]
            assert e["accuracy_met"]
            np.testing.assert_allclose(polynomial(e["p"]), e["ler"], rtol=1e-12)
            overlap = max(e["lower"], mc["lower"]) <= min(e["upper"], mc["upper"])
            assert overlap
            rows.append(
                {
                    "scheme": scheme,
                    "p": e["p"],
                    "profile_ler": e["ler"],
                    "mc_ler": mc["ler"],
                    "intervals_overlap": overlap,
                    "profile_shots": result["shots"],
                }
            )
        es = result["estimates"]
        ms = result["independent_stim_mc"]
        ax.vlines(
            ps,
            [e["lower"] for e in es],
            [e["upper"] for e in es],
            color="#146a89",
            linewidth=2,
        )
        ax.scatter(
            ps,
            [e["ler"] for e in es],
            color="#146a89",
            s=24,
            label="Profile: 99% bounds",
        )
        ax.vlines(
            ps * 1.035,
            [m["lower"] for m in ms],
            [m["upper"] for m in ms],
            color="#c65a27",
            linewidth=2,
        )
        ax.scatter(
            ps * 1.035,
            [m["ler"] for m in ms],
            color="#c65a27",
            marker="x",
            s=32,
            label="Independent Stim MC",
        )
        ax.set(
            xscale="log",
            yscale="log",
            title=f"{scheme} | {result['shots']:,} profile shots",
            xlabel="Physical parameter p",
            ylabel="Logical error rate",
        )
        ax.set_xticks(ps, [f"{p:g}" for p in ps])
        ax.set_ylim(0.008, 0.55)
        ax.set_yticks([0.01, 0.03, 0.1, 0.3], ["0.01", "0.03", "0.1", "0.3"])
        ax.minorticks_off()
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "QStabIR Steane [[7,1,3]], two rounds: gate-dependent noise", fontsize=13
    )
    fig.text(
        0.5,
        0.012,
        "1Q: p/5   |   2Q: p   |   readout: 5p   |   fixed BP+OSD0 decoder per scheme\n"
        "Bounds simultaneous over each run's three p points; MC markers shifted right for visibility.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)
    sources = [
        "src/scalerqec/QEC/noisemodel.py",
        "src/scalerqec/QEC/extraction.py",
        "src/scalerqec/QEC/qeccircuit.py",
        "src/scalerqec/Stratified/general_noise.py",
        "src/scalerqec/Stratified/uniformized.py",
        "examples/qstabir_noise.py",
    ]
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    result = {
        "comparisons": rows,
        "all_intervals_overlap": True,
        "versions": {p: version(p) for p in ["stim", "numpy", "scipy", "stimbposd"]},
        "source_sha256_at_verification": hashes,
        "limitations": "Per-run 99% simultaneous grid coverage; not 99% familywise over four runs. "
        "Functional circuits; no general fault-tolerance or speedup certificate.",
    }
    (directory / "verification.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# QStabIR example verification",
        "",
        "All four runs met 25% relative error at 99% per-run simultaneous confidence.",
        "Each was compared to 50,000 independent Stim shots per p (600,000 total).",
        "All 12 pairs of confidence intervals overlap. This is not a proof of universal accuracy.",
        "",
        "| Scheme | p | Profile LER | Stim MC LER | Profile shots (whole curve) |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['scheme']} | {r['p']} | {r['profile_ler']:.7f} | {r['mc_ler']:.7f} | {r['profile_shots']:,} |"
        for r in rows
    ]
    lines += [
        "",
        "Exact intervals, decoder settings, seeds and status are in each scheme's results.json.",
        "Saved NPZ polynomials reproduce the profile point estimates. Timing is not a controlled speed comparison.",
        "",
        "![Comparison](comparison.png)",
    ]
    (directory / "verification.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    report(parser.parse_args().directory)
