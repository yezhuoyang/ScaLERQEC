"""Define a Steane code with QStabIR and estimate a reusable LER polynomial.

Install the optional decoder: python -m pip install 'scalerqec[ldpc]'
From this checkout: python -m examples.qstabir_noise --scheme Flag --output myresults_flag
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scalerqec.QEC import NoiseModel, StabCode
from scalerqec.Stratified import LinearNoiseModel


def make_noise(p_ref):
    return NoiseModel(
        p_ref,
        p_1q=p_ref / 5,
        p_2q=p_ref,
        p_meas=5 * p_ref,
        p_reset=0,
        p_idle=0,
    )


def define_code(scheme="Standard", rounds=2):
    """Define [[7,1,3]] using stabilizers; StabCode constructs the memory IR."""
    code = StabCode(n=7, k=1, d=3)
    for stabilizer in [
        "IIIXXXX",
        "IXXIIXX",
        "XIXIXIX",
        "IIIZZZZ",
        "IZZIIZZ",
        "ZIZIZIZ",
    ]:
        code.add_stab(stabilizer)
    code.set_logical_Z(0, "ZZZZZZZ")
    code.rounds = rounds
    code.scheme = scheme
    return code


def run(scheme, output, *, max_shots=500_000, max_seconds=60, mc_shots=0):
    from stimbposd import BPOSD

    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Preserving existing results in {directory}.")
    directory.mkdir(parents=True, exist_ok=True)
    p_ref = 0.01
    code = define_code(scheme)
    code.noisemodel = make_noise(p_ref)
    model = LinearNoiseModel.from_stabcode(code, reference_p=p_ref)
    # This decoder accepts general detector hyperedges, including flag records.
    # Keep it fixed across p so the whole curve represents ONE failure function.
    decoder = BPOSD(
        model.circuit.detector_error_model(),
        max_bp_iters=30,
        bp_method="min_sum",
        osd_order=0,
    )
    ps = [0.002, 0.005, 0.01]
    profile = model.sample_bernstein_profile(
        decoder,
        ps,
        relative_error=0.25,
        confidence=0.99,
        max_shots=max_shots,
        max_seconds=max_seconds,
        seed=260927,
    )
    result = {
        "scheme": scheme,
        "code": "[[7,1,3]] Steane, two rounds, Z memory",
        "seed": 260927,
        "p_ref": p_ref,
        "probabilities": ps,
        "noise": {
            "p_1q/p": 0.2,
            "p_2q/p": 1,
            "p_meas/p": 5,
            "p_reset/p": 0,
            "p_idle/p": 0,
        },
        "decoder": "fixed BPOSD: min_sum, 30 iterations, OSD0",
        "relative_error": 0.25,
        "confidence": 0.99,
        "status": profile.status,
        "reason": profile.reason,
        "shots": profile.shots,
        "seconds": profile.seconds,
        "estimates": [asdict(e) for e in profile.estimates],
    }
    if profile.converged:
        profile.to_polynomial().save(directory / "ler_polynomial.npz")
    if mc_shots:
        from scipy.stats import beta

        reference = []
        for j, p in enumerate(ps):
            # Recompile and inject at the actual p, independently of circuit_at.
            fresh = define_code(scheme)
            fresh.noisemodel = make_noise(p)
            fresh.construct_circuit()
            det, obs = fresh.stimcirc.compile_detector_sampler(seed=260928 + j).sample(
                mc_shots, separate_observables=True
            )
            failed = int(np.any(decoder.decode_batch(det) != obs, axis=1).sum())
            alpha = 0.01 / len(ps)
            lo = (
                0
                if not failed
                else float(beta.ppf(alpha / 2, failed, mc_shots - failed + 1))
            )
            hi = (
                1
                if failed == mc_shots
                else float(beta.ppf(1 - alpha / 2, failed + 1, mc_shots - failed))
            )
            reference.append(
                {
                    "p": p,
                    "shots": mc_shots,
                    "failures": failed,
                    "ler": failed / mc_shots,
                    "lower": lo,
                    "upper": hi,
                    "seed": 260928 + j,
                }
            )
        result["independent_stim_mc"] = reference
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    model.circuit.to_file(str(directory / "circuit.stim"))
    (directory / "memory.qstabir").write_text("\n".join(map(str, code._IRList)))
    print(json.dumps(result, indent=2))
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheme", choices=["Standard", "Flag", "Shor", "Knill"], default="Standard"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-shots", type=int, default=500_000)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--mc-shots", type=int, default=0)
    args = parser.parse_args()
    run(
        args.scheme,
        args.output,
        max_shots=args.max_shots,
        max_seconds=args.max_seconds,
        mc_shots=args.mc_shots,
    )
