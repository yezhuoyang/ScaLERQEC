"""Independent, preselected fixed-shot Stim references for polynomial audits."""

import argparse
import hashlib
from pathlib import Path

from benchmark.large_code_cases import P0, PS, cases, make_circuit
from benchmark.large_code_validation import decoder_for, monte_carlo, write_json


def run(name, shots, output, probabilities=PS, seed=260922):
    output.mkdir(parents=True, exist_ok=True)
    path = output / (name + ".json")
    if path.exists():
        raise FileExistsError(f"Preserving {path}")
    spec = next(s for s in cases() if s["name"] == name)
    circuit = make_circuit(spec, P0)
    decoder, description = decoder_for(spec, make_circuit(spec, PS[1]))
    row = {
        "spec": spec,
        "status": "running",
        "decoder": description,
        "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
        "seed": seed,
        "fixed_shots_per_point": shots,
        "monte_carlo": [],
    }
    write_json(path, row)
    for p in probabilities:
        row["monte_carlo"].append(
            {"p": p, **monte_carlo(make_circuit(spec, p), decoder, shots, seed)}
        )
        write_json(path, row)
        print(name, p, row["monte_carlo"][-1]["ler"], flush=True)
    row["status"] = "completed"
    write_json(path, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--shots", type=int, default=250000)
    parser.add_argument("--probabilities", type=float, nargs="+", default=PS)
    parser.add_argument("--seed", type=int, default=260922)
    parser.add_argument(
        "--output", type=Path, default=Path("experiment_results/uniformized_references")
    )
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("shots must be positive")
    for name in args.names:
        run(name, args.shots, args.output, args.probabilities, args.seed)
