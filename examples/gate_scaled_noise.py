"""Sample one gate-scaled noise family and plot its LER with confidence bars.

From an installed source checkout:
    python -m examples.gate_scaled_noise --output myresults_gate_scaled_noise
    python -m examples.gate_scaled_noise --surface-distance 3 --output myresults_surface_noise

This small three-qubit repetition Z-memory experiment is for demonstrating
the API. It includes noisy encoding and does not protect against phase errors.
Replace memory_circuit(p_ref) with your actual noisy Stim circuit for research.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

from scalerqec.Stratified import LinearNoiseModel


def memory_circuit(p):
    """S gate: p/5; each CX: DEPOLARIZE2(p); each readout: 5*p.

    Preparation is ideal in this example. Noise follows each individual gate;
    the two CX locations are kept separate because their targets overlap.
    Stim receives numeric probabilities; Python evaluates the expressions.
    """
    return stim.Circuit(f"""
        R 0 1 2
        S 0
        DEPOLARIZE1({p / 5}) 0
        CX 0 1
        DEPOLARIZE2({p}) 0 1
        CX 0 2
        DEPOLARIZE2({p}) 0 2
        M({5 * p}) 0 1 2
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """)


def surface_model(p_ref=0.01, *, distance=3, rounds=3):
    """The same coefficients through the existing StabIR builder.

    This builder uses anticommuting Paulis immediately before measurement.
    Those inserted readout faults have weight one; native M(5*p) record flips
    in memory_circuit have Pauli weight zero. Neither representation changes
    the convention of counting Pauli factors at their original locations.
    """
    from scalerqec.QEC.noisemodel import NoiseModel
    from scalerqec.QEC.surface import SurfaceCode

    code = SurfaceCode(distance=distance, rounds=rounds)
    code.scheme = "Standard"
    code.noisemodel = NoiseModel(
        p_ref,
        p_1q=p_ref / 5,
        p_2q=p_ref,
        p_meas=5 * p_ref,
        p_reset=0,
        p_idle=0,
    )
    return LinearNoiseModel.from_stabcode(code, reference_p=p_ref)


def run(output, *, max_seconds=60, max_shots=2_000_000, surface_distance=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving {output}; choose an empty output directory.")
    p_ref = 0.01
    ps = np.array([0.002, 0.005, 0.01, 0.02])
    model = (
        LinearNoiseModel(memory_circuit(p_ref), reference_p=p_ref)
        if surface_distance is None
        else surface_model(p_ref, distance=surface_distance, rounds=surface_distance)
    )

    # Fix the decoder once. The DEM configures the decoder, while profiling
    # samples the original circuit channels, including their correlations.
    decoder = pymatching.Matching.from_detector_error_model(
        model.circuit_at(p_ref).detector_error_model(decompose_errors=True)
    )
    result = model.sample_bernstein_profile(
        decoder,
        ps,
        relative_error=0.2,
        confidence=0.99,
        max_shots=max_shots,
        max_seconds=max_seconds,
        seed=260924,
    )

    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(e) for e in result.estimates]
    payload = {
        "status": result.status,
        "reason": result.reason,
        "shots": result.shots,
        "seconds": result.seconds,
        "confidence": result.confidence,
        "relative_error_requested": 0.2,
        "reference_p": p_ref,
        "seed": 260924,
        "circuit": "repetition Z memory"
        if surface_distance is None
        else f"StabIR surface d={surface_distance}, rounds={surface_distance}",
        "valid_p_domain": [0, model.max_p],
        "interval_scope": "simultaneous over the four requested p values and sequential sampling checkpoints",
        "noise": {
            "single_qubit": "p/5",
            "two_qubit": "p",
            "measurement": "5*p",
            "reset": "ideal",
        },
        "estimates": rows,
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    model.circuit.to_file(output / "circuit_at_reference.stim")

    values = np.array([e.ler for e in result.estimates])
    lower = np.array([e.lower for e in result.estimates])
    upper = np.array([e.upper for e in result.estimates])
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    # Draw intervals directly: a stopped estimate can occasionally fall outside
    # the intersection of earlier intervals, so yerr = value-lower is not safe.
    ax.vlines(
        ps,
        lower,
        upper,
        color="tab:blue",
        linewidth=1.6,
        label="99% simultaneous intervals",
    )
    ax.plot(ps, values, "o", color="tab:blue", label="LER estimates")
    if result.converged:
        polynomial = result.to_polynomial()
        polynomial.save(output / "ler_polynomial.npz")
        dense_ps = np.geomspace(ps.min(), ps.max(), 200)
        ax.plot(
            dense_ps,
            polynomial(dense_ps),
            color="tab:blue",
            alpha=0.6,
            label="Same estimated polynomial",
        )
        # The smooth line does not acquire a continuum confidence band.
    ax.set_xscale("log")
    ax.set_xticks(ps, [f"{p:g}" for p in ps])
    if np.all(lower > 0):
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set(
        xlabel="Base noise parameter p",
        ylabel="Logical error rate",
        title="Gate-scaled noise: 1q = p/5, 2q = p, readout = 5p\n"
        + (
            "20% accuracy target met at all four points"
            if result.converged
            else "Accuracy target unresolved; inspect the intervals"
        ),
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "ler_vs_p.png", dpi=160)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    return model, decoder, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("myresults_gate_scaled_noise")
    )
    parser.add_argument("--surface-distance", type=int)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--max-shots", type=int, default=2_000_000)
    args = parser.parse_args()
    run(
        args.output,
        max_seconds=args.max_seconds,
        max_shots=args.max_shots,
        surface_distance=args.surface_distance,
    )
