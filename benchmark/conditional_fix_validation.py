"""Re-run historical failures without overwriting their original evidence.

Each invocation handles one case. Reference MC is reused only after checking
the original circuit hash and fixed-decoder description. New small MC batches
and independent Stim replays additionally exercise the changed sampler.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from benchmark.accuracy_control_validation import serialize
from benchmark.conditional_moment_audit import run as audit_moments
from benchmark.fault_replay_oracle import ReplayOracle, audit_sampled_histories
from benchmark.large_code_cases import P0, PS, cases, make_circuit
from benchmark.large_code_validation import decoder_for, monte_carlo, write_json
from scalerqec.Stratified import LinearNoiseModel

OUT = Path("experiment_results/conditional_fix_validation")


def run(name, seconds, skip_mc=False, output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = next(s for s in cases() if s["name"] == name)
    path = output / (name + ".json")
    if path.exists():
        raise FileExistsError(
            f"Preserving {path}; select a new directory with --output."
        )
    started = perf_counter()
    circuit = make_circuit(spec, P0)
    row = {
        "spec": spec,
        "seed": 260919,
        "max_seconds": seconds,
        "p_values": PS,
        "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
        "status": "running",
    }
    row["source_sha256"] = {
        name: hashlib.sha256(
            (Path("src/scalerqec/Stratified") / name).read_bytes()
        ).hexdigest()
        for name in [
            "general_noise.py",
            "adaptive.py",
            "conditional.py",
            "confidence.py",
            "noise_polynomial.py",
        ]
    }

    def checkpoint(phase):
        row["phase"] = phase
        row["elapsed_seconds"] = perf_counter() - started
        write_json(path, row)
        print(name, phase, flush=True)

    checkpoint("compile")
    model = LinearNoiseModel(circuit, P0)
    row["compile_seconds"] = perf_counter() - started
    checkpoint("audit")
    oracle = ReplayOracle(circuit, P0)
    row["audit"] = audit_sampled_histories(model, oracle)
    row["moments"] = audit_moments(spec, model=model, oracle=oracle)
    if not row["moments"]["within_7_true_standard_errors"]:
        raise AssertionError("Conditional moment audit failed")
    decoder, row["decoder"] = decoder_for(spec, make_circuit(spec, PS[1]))
    checkpoint("profile")
    result = model.sample_until_accuracy(
        decoder,
        PS,
        relative_error=0.1,
        confidence=0.99,
        max_shots=1_000_000,
        max_seconds=seconds,
        exact_budget=100_000,
        seed=row["seed"],
    )
    row["adaptive"] = serialize(result)
    result.to_polynomial(allow_unconverged=True).save(
        output / (name + "_polynomial.npz")
    )
    checkpoint("compare")
    previous_path = Path("experiment_results/large_code_validation") / (name + ".json")
    previous = json.loads(previous_path.read_text())
    assert previous["circuit_sha256"] == row["circuit_sha256"]
    if "decoder" in previous:
        assert previous["decoder"] == row["decoder"]
    row["historical_reference_file"] = str(previous_path)
    row["historical_monte_carlo"] = previous.get("monte_carlo", [])
    # Independent fresh MC; the old study remains the higher-statistics reference.
    shots = 8192 if spec["family"] in {"surface", "stabir"} else 256
    row["fresh_monte_carlo"] = []
    for p in [] if skip_mc else PS:
        row["fresh_monte_carlo"].append(
            {"p": p, **monte_carlo(make_circuit(spec, p), decoder, shots, 290916)}
        )
        checkpoint("compare")
    row["status"] = "completed"
    checkpoint("completed")
    print(name, result.status, result.shots, result.seconds, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--skip-mc",
        action="store_true",
        help="Run profile and replay checks only; retain historical MC separately.",
    )
    args = parser.parse_args()
    run(args.name, args.seconds, args.skip_mc, args.output)
