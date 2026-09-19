"""Replay native fixed-weight SID samples on large circuits in independent Stim.

The legacy native backend has one logical output. QLDPC examples explicitly
project onto one logical observable at a time; these are not block-LER tests.
Actual returned fault vectors are retained because native sampling has no seed
argument. This does not change the general-noise matrix's noise definitions.
"""

import hashlib
from time import perf_counter

import numpy as np
import stim

from benchmark.fault_replay_oracle import ReplayOracle
from benchmark.large_code_cases import P0, ROOT, cases, make_circuit
from benchmark.large_code_validation import write_json
from scalerqec import qepg
from scalerqec.Clifford.stimparser import rewrite_stim_code


def single_observable(circuit, index):
    result = stim.Circuit()
    found = False
    for op in circuit.without_noise().flattened():
        if op.name == "OBSERVABLE_INCLUDE":
            if int(op.gate_args_copy()[0]) == index:
                found = True
                result.append("OBSERVABLE_INCLUDE", op.targets_copy(), 0)
        else:
            result.append(op)
    if not found:
        raise ValueError(f"Circuit does not define logical observable {index}")
    return result


def sid_oracle(program):
    """Independent implementation of native's documented SID site convention."""
    noisy = stim.Circuit()
    for line in program.splitlines():
        op = stim.Circuit(line)[0]
        if op.name in {"H", "S", "X", "Y", "Z", "CX", "M"}:
            noisy.append("DEPOLARIZE1", op.targets_copy(), P0)
        noisy.append(op)
    return ReplayOracle(noisy, P0)


def run(spec, index, output):
    started = perf_counter()
    circuit = make_circuit(spec, P0)
    program = rewrite_stim_code(str(single_observable(circuit, index)))
    oracle = sid_oracle(program)
    row = {
        "spec": spec,
        "original_observables": circuit.num_observables,
        "logical_projection": index,
        "noise": "uniform single-qubit depolarization before each native primitive",
        "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
        "noise_locations": len(oracle.events),
        "detectors": circuit.num_detectors,
        "extension": qepg.__file__,
        "samples": [],
    }
    for weight in [0, 1, 2, 13]:
        vectors, actual = qepg.return_samples_with_noise_vector(program, weight, 4)
        for faults, bits in zip(vectors, actual):
            assert len(faults) == weight
            assert len({site for site, _ in faults}) == weight
            history = np.zeros(len(oracle.events), dtype=np.int64)
            for site, pauli in faults:
                assert 0 <= site < len(history) and 1 <= pauli <= 3
                history[site] = pauli
            expected = oracle.bits(history)
            np.testing.assert_array_equal(bits, expected)
            row["samples"].append(
                {
                    "weight": weight,
                    "faults": [[int(j), int(a)] for j, a in faults],
                    "packed_bits_sha256": hashlib.sha256(
                        np.packbits(expected).tobytes()
                    ).hexdigest(),
                }
            )
    row["status"] = "passed"
    row["histories_replayed"] = len(row["samples"])
    row["seconds"] = perf_counter() - started
    write_json(output, row)
    print(spec["name"], index, row["status"], row["seconds"], flush=True)


def main():
    out = ROOT / "experiment_results" / "large_code_validation" / "native"
    out.mkdir(parents=True, exist_ok=True)
    specs = {s["name"]: s for s in cases()}
    selected = [
        ("surface_13_x_nonuniform_depolarizing", 0),
        ("surface_13_z_nonuniform_depolarizing", 0),
        ("bicycle_72_z_nonuniform_depolarizing", 11),
        ("bicycle_144_z_nonuniform_depolarizing", 11),
    ]
    for name, index in selected:
        output = out / f"{name}_logical{index}.json"
        if not output.exists():
            run(specs[name], index, output)


if __name__ == "__main__":
    main()
