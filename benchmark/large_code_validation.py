"""Resumable, isolated large-code validation. No unresolved case counts as a pass.

Run python -m benchmark.large_code_validation --workers 2.
Each case is a separate bounded subprocess. Partial phases and failures are
saved immediately; --resume never silently reruns or discards a failed case.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import pymatching
from scipy.stats import beta

from benchmark.accuracy_control_validation import serialize
from benchmark.fault_replay_oracle import (
    ReplayOracle,
    audit_sampled_histories,
    hgp_all_columns,
)
from benchmark.large_code_cases import P0, PS, ROOT, cases, hgp_matrices, make_circuit
from scalerqec.Stratified import LinearNoiseModel

OUT = ROOT / "experiment_results" / "large_code_validation"


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def decoder_for(spec, circuit):
    if (
        spec["family"] in {"hgp", "bicycle", "color"}
        or spec["noise"] == "correlated_else"
    ):
        from stimbposd import BPOSD

        decoder = BPOSD(
            circuit.detector_error_model(approximate_disjoint_errors=True),
            max_bp_iters=30,
            bp_method="min_sum",
            osd_order=0,
            osd_method="osd_0",
        )
        return decoder, "BP+OSD0, min_sum, 30 iterations"
    return pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
    ), "PyMatching"


def monte_carlo(circuit, decoder, shots, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    failures = 0
    per_observable = np.zeros(circuit.num_observables, dtype=np.int64)
    started = perf_counter()
    for first in range(0, shots, 256):
        det, obs = sampler.sample(min(256, shots - first), separate_observables=True)
        failed = decoder.decode_batch(det) != obs
        failures += int(failed.any(axis=1).sum())
        per_observable += failed.sum(axis=0)
    alpha = 0.01 / (len(cases()) * len(PS))
    lower = (
        float(beta.ppf(alpha / 2, failures, shots - failures + 1)) if failures else 0.0
    )
    upper = (
        float(beta.ppf(1 - alpha / 2, failures + 1, shots - failures))
        if failures < shots
        else 1.0
    )
    return {
        "shots": shots,
        "failures": failures,
        "ler": failures / shots,
        "per_observable_failures": per_observable.tolist(),
        "lower": lower,
        "upper": upper,
        "pointwise_confidence": 1 - alpha,
        "seconds": perf_counter() - started,
    }


def run_case(spec):
    path = OUT / (spec["name"] + ".json")
    row = {
        "spec": spec,
        "phase": "starting",
        "status": "running",
        "p_values": PS,
        "seed": 260916,
    }

    def checkpoint(phase):
        row["phase"] = phase
        write_json(path, row)
        print(spec["name"], phase, flush=True)

    started = perf_counter()
    try:
        circuit = make_circuit(spec, P0)
        row["circuit_sha256"] = hashlib.sha256(str(circuit).encode()).hexdigest()
        row["qubits"] = circuit.num_qubits
        row["detectors"] = circuit.num_detectors
        row["observables"] = circuit.num_observables
        row["defined_observables"] = sorted(
            {
                int(op.gate_args_copy()[0])
                for op in circuit.flattened()
                if op.name == "OBSERVABLE_INCLUDE"
            }
        )
        circuit.to_file(OUT / (spec["name"] + ".stim"))
        checkpoint("circuit_built")
        t = perf_counter()
        model = LinearNoiseModel(circuit, P0)
        row["compile_seconds"] = perf_counter() - t
        row["factors"] = len(model._factors)
        row["response_columns"] = sum(len(x) for x in model._responses)
        row["response_bytes"] = sum(x.nbytes for x in model._responses)
        checkpoint("responses_compiled")
        oracle = ReplayOracle(circuit, P0)
        row["audit"] = audit_sampled_histories(model, oracle)
        if spec["family"] == "hgp":
            h, hx, hz, lz = hgp_matrices(spec["n"])
            row["classical_matrix"] = h.tolist()
            row["independent_commutation_columns"] = hgp_all_columns(
                model, oracle, hx, hz, lz
            )
        checkpoint("histories_audited")
        targets = [make_circuit(spec, p) for p in PS]
        row["independent_scaling_matches"] = all(
            model.circuit_at(p).approx_equals(c.flattened(), atol=1e-13)
            for p, c in zip(PS, targets)
        )
        if not row["independent_scaling_matches"]:
            raise AssertionError(
                "Independent circuit generator disagrees with scaled circuit"
            )
        decoder, row["decoder"] = decoder_for(spec, targets[1])
        det = targets[1].compile_detector_sampler(seed=101).sample(16)
        first = decoder.decode_batch(det)
        if not np.array_equal(first, decoder.decode_batch(det[::-1])[::-1]):
            raise AssertionError("Decoder depends on sample order")
        row["decoder_order_check"] = True
        checkpoint("decoder_ready")
        t = perf_counter()
        result = model.sample_until_accuracy(
            decoder,
            PS,
            relative_error=0.1,
            confidence=0.99,
            max_shots=300_000,
            max_seconds=30,
            exact_budget=100_000,
            seed=260916,
        )
        row["adaptive"] = serialize(result)
        row["adaptive_wall_seconds"] = perf_counter() - t
        checkpoint("profile_complete")
        shots = 100_000
        if spec["family"] in {"hgp", "color"}:
            shots = 4096
        if spec["family"] == "bicycle":
            shots = 1024 if spec["n"] in {72, 288} else 256
        row["monte_carlo"] = []
        for j, (p, c) in enumerate(zip(PS, targets)):
            mc = monte_carlo(c, decoder, shots, 260917 + j)
            mc["p"] = p
            estimate = result.estimates[j]
            mc["profile_interval_overlap"] = (
                estimate.lower <= mc["upper"] and mc["lower"] <= estimate.upper
            )
            mc["accepted_accuracy_contradiction"] = bool(
                result.converged
                and (
                    estimate.ler / 0.9 < mc["lower"] or estimate.ler / 1.1 > mc["upper"]
                )
            )
            mc["reference_resolves_10_percent"] = bool(
                mc["lower"] >= estimate.ler / 1.1 and mc["upper"] <= estimate.ler / 0.9
            )
            row["monte_carlo"].append(mc)
            checkpoint(f"mc_{j + 1}_complete")
        row["status"] = "completed"
        row["seconds"] = perf_counter() - started
        checkpoint("complete")
    except Exception as exc:
        row["status"] = "failed"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc()
        row["seconds"] = perf_counter() - started
        write_json(path, row)
        print(spec["name"], "FAILED", str(exc)[:200], flush=True)
        raise


def launch(spec, timeout):
    log = OUT / (spec["name"] + ".log")
    env = dict(
        os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg"
    )
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with log.open("w", encoding="utf-8") as stream:
        try:
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmark.large_code_validation",
                    "--case",
                    spec["name"],
                ],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                creationflags=flags,
                check=False,
            )
            return spec["name"], child.returncode
        except subprocess.TimeoutExpired:
            path = OUT / (spec["name"] + ".json")
            row = json.loads(path.read_text()) if path.exists() else {"spec": spec}
            row["status"] = "process_timeout"
            row["timeout_seconds"] = timeout
            write_json(path, row)
            return spec["name"], "timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--family")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cases": cases(),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["stim", "numpy", "pymatching", "ldpc", "stimbposd"]
        },
        "p_values": PS,
        "relative_error": 0.1,
        "confidence": 0.99,
        "max_seconds": 30,
        "max_shots": 300_000,
        "exact_budget": 100_000,
        "mc_matrix_confidence": 0.99,
    }
    if args.case:
        run_case(next(s for s in cases() if s["name"] == args.case))
        return
    write_json(OUT / "manifest.json", manifest)
    selected = [s for s in cases() if args.family is None or s["family"] == args.family]
    if args.resume:
        selected = [s for s in selected if not (OUT / (s["name"] + ".json")).exists()]
    # Largest required distances start early, so they cannot disappear behind
    # short-case successes. Workers isolate memory and decoder failures.
    selected.sort(
        key=lambda s: (s.get("distance", 0) < 13, -s.get("distance", 0), s["name"])
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(launch, s, args.timeout): s for s in selected}
        for future in as_completed(futures):
            print("FINISHED", *future.result(), flush=True)


if __name__ == "__main__":
    main()
