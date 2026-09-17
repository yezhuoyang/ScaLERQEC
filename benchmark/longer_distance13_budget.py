"""Predeclared five-minute follow-up to the d13 mixed-depolarization case."""

import hashlib
from time import perf_counter

from benchmark.accuracy_control_validation import serialize
from benchmark.large_code_cases import P0, PS, cases, make_circuit
from benchmark.large_code_validation import OUT, decoder_for, write_json
from scalerqec.Stratified import LinearNoiseModel


def main():
    spec = next(
        s for s in cases() if s["name"] == "surface_13_z_nonuniform_depolarizing"
    )
    output = OUT / "longer_budget"
    output.mkdir(parents=True, exist_ok=True)
    row = {
        "spec": spec,
        "seed": 260919,
        "p_values": PS,
        "max_seconds": 300,
        "max_shots": 1_000_000,
        "status": "running",
    }
    path = output / (spec["name"] + ".json")
    write_json(path, row)
    started = perf_counter()
    circuit = make_circuit(spec, P0)
    row["circuit_sha256"] = hashlib.sha256(str(circuit).encode()).hexdigest()
    model = LinearNoiseModel(circuit, P0)
    row["compile_seconds"] = perf_counter() - started
    decoder, row["decoder"] = decoder_for(spec, make_circuit(spec, PS[1]))
    write_json(path, row)
    print("Distance-13 five-minute profile started", flush=True)
    result = model.sample_until_accuracy(
        decoder,
        PS,
        relative_error=0.1,
        confidence=0.99,
        max_shots=1_000_000,
        max_seconds=300,
        exact_budget=100_000,
        seed=row["seed"],
    )
    row["adaptive"] = serialize(result)
    row["status"] = "completed"
    row["seconds"] = perf_counter() - started
    write_json(path, row)
    print(result.status, result.shots, result.seconds, flush=True)


if __name__ == "__main__":
    main()
