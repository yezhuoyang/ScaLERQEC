"""Predeclared large-code validation cases and independently generated noise.

HGP examples use the standard CSS hypergraph product. BB files are repository
fixtures; their n,k,d labels are not new distance proofs. Those fixtures define
only logical observable k-1, in X memory. Their other output slots are empty,
so their LER is a single-logical error rate. HGP includes all k logical Z bits.
"""

from functools import cache
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parents[1]
P0 = 0.01
PS = [0.001, 0.003, 0.01]


def cases():
    rows = []
    for d in [3, 5, 7, 9, 11, 13]:
        for noise in ["uniform_single", "nonuniform_depolarizing", "biased_pauli"]:
            rows.append(
                {
                    "family": "surface",
                    "distance": d,
                    "basis": "z",
                    "noise": noise,
                    "rounds": d,
                }
            )
    for d in [3, 7, 13]:
        for noise in ["nonuniform_depolarizing", "biased_pauli"]:
            rows.append(
                {
                    "family": "surface",
                    "distance": d,
                    "basis": "x",
                    "noise": noise,
                    "rounds": d,
                }
            )
    for noise in ["correlated_else", "phenomenological"]:
        rows.append(
            {
                "family": "surface",
                "distance": 13,
                "basis": "z",
                "noise": noise,
                "rounds": 13,
            }
        )
    for d in [3, 7, 13]:
        for noise in ["SD6", "SI1000"]:
            rows.append(
                {
                    "family": "stabir",
                    "distance": d,
                    "basis": "z",
                    "noise": noise,
                    "rounds": d,
                }
            )
    for d in [3, 5, 7]:
        rows.append(
            {
                "family": "color",
                "distance": d,
                "basis": "xyz",
                "noise": "nonuniform_depolarizing",
                "rounds": d,
            }
        )
    for n in [58, 180, 245]:
        for noise in [
            "uniform_single",
            "nonuniform_depolarizing",
            "biased_pauli",
            "correlated_else",
        ]:
            rows.append({"family": "hgp", "n": n, "noise": noise, "rounds": 1})
    for n, k, d, r in [
        (72, 12, 6, 18),
        (90, 8, 10, 30),
        (108, 8, 8, 24),
        (144, 12, 12, 36),
        (288, 16, 18, 54),
    ]:
        for noise in (
            ["biased_pauli"] if n == 90 else ["nonuniform_depolarizing", "biased_pauli"]
        ):
            rows.append(
                {
                    "family": "bicycle",
                    "n": n,
                    "k": k,
                    "distance": d,
                    "rounds": r,
                    "noise": noise,
                    "data_only": n == 288,
                    "memory_basis": "x",
                    "defined_logical_ids": [k - 1],
                }
            )
    for row in rows:
        identity = str(row.get("n", row.get("distance")))
        row["name"] = (
            f"{row['family']}_{identity}_{row.get('basis', 'z')}_{row['noise']}"
        )
        if row.get("data_only"):
            row["name"] += "_data_only"
    return rows


def channel(circuit, targets, p, family, index, *, pair=False):
    """Append the actual requested noise directly at p; no production scaler."""
    scale = [0.25, 1.0, 2.0][index % 3]
    if family == "uniform_single":
        circuit.append("DEPOLARIZE1", targets, p)
    elif family == "nonuniform_depolarizing":
        circuit.append("DEPOLARIZE2" if pair else "DEPOLARIZE1", targets, scale * p)
    elif family == "biased_pauli":
        if pair:
            fractions = np.full(15, 0.01)
            fractions[4] += 0.20
            fractions[14] += 0.65
            circuit.append("PAULI_CHANNEL_2", targets, fractions * scale * p)
        else:
            circuit.append(
                "PAULI_CHANNEL_1", targets, np.array([0.025, 0.025, 0.95]) * scale * p
            )
    elif family == "correlated_else":
        if pair:
            circuit.append("E", [stim.target_x(q) for q in targets], 0.8 * scale * p)
            circuit.append(
                "ELSE_CORRELATED_ERROR",
                [stim.target_z(targets[0]), stim.target_y(targets[1])],
                0.5 * scale * p,
            )
        else:
            circuit.append("DEPOLARIZE1", targets, 0.2 * scale * p)
    else:
        raise ValueError(f"Unknown channel family {family}")


def circuit_noise(ideal, p, family):
    """Gate, preparation and readout noise, preserving the supplied schedule."""
    out = stim.Circuit()
    index = 0
    for op in ideal.flattened():
        ts = op.targets_copy()
        op.gate_args_copy()
        # Error before measurement, after reset, and after each Clifford gate.
        if op.name in {"M", "MX", "MY", "MR", "MRX", "MRY"}:
            out.append(op.name, ts, 2 * p)
        else:
            out.append(op)
        if op.name in {"R", "RX", "RY", "MR", "MRX", "MRY"}:
            out.append("Z_ERROR" if op.name.endswith("X") else "X_ERROR", ts, 0.5 * p)
        elif op.name in {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "C_XYZ", "C_ZYX"}:
            if family != "phenomenological":
                for t in ts:
                    channel(out, [t.value], p, family, index)
                    index += 1
        elif op.name in {"CX", "CY", "CZ", "SWAP"} and family != "phenomenological":
            for first in range(0, len(ts), 2):
                if not all(t.is_qubit_target for t in ts[first : first + 2]):
                    continue
                channel(
                    out,
                    [t.value for t in ts[first : first + 2]],
                    p,
                    family,
                    index,
                    pair=True,
                )
                index += 1
        elif op.name == "TICK" and family == "phenomenological":
            # Explicitly a noisy idle at every tick, not a gate-noise model.
            out.append("DEPOLARIZE1", range(ideal.num_qubits), 0.05 * p)
    return out


def gf2_rref(matrix):
    a = np.asarray(matrix, dtype=np.uint8).copy()
    pivots = []
    for col in range(a.shape[1]):
        choices = np.flatnonzero(a[len(pivots) :, col])
        if not len(choices):
            continue
        row = len(pivots)
        found = row + int(choices[0])
        a[[row, found]] = a[[found, row]]
        others = np.flatnonzero(a[:, col])
        others = others[others != row]
        a[others] ^= a[row]
        pivots.append(col)
        if len(pivots) == len(a):
            break
    return a[: len(pivots)], pivots


def kernel(matrix):
    a, pivots = gf2_rref(matrix)
    result = []
    for col in sorted(set(range(a.shape[1])) - set(pivots)):
        v = np.zeros(a.shape[1], dtype=np.uint8)
        v[col] = 1
        v[pivots] = a[:, col]
        result.append(v)
    return np.asarray(result, dtype=np.uint8)


@cache
def hgp_matrices(n):
    if n == 58:
        h = np.array(
            [[(j >> i) & 1 for j in range(1, 8)] for i in range(3)], dtype=np.uint8
        )
    else:
        m = {180: 6, 245: 7}[n]
        rng = np.random.default_rng(260916 + m)
        # Sparse regular bipartite graph, rejecting parallel edges and repeated
        # columns. The resulting fixed matrix is recorded in the case artifacts.
        for _ in range(200_000):
            h = np.zeros((m, 2 * m), dtype=np.uint8)
            slots = rng.permutation(np.repeat(np.arange(m), 6)).reshape(2 * m, 3)
            if any(len(set(row)) != 3 for row in slots):
                continue
            for col, rs in enumerate(slots):
                h[rs, col] = 1
            if len(np.unique(h.T, axis=0)) == 2 * m and len(gf2_rref(h)[1]) == m:
                break
        else:
            raise RuntimeError(
                "Could not generate the declared regular classical code."
            )
    m, q = h.shape
    hx = np.column_stack(
        [np.kron(h, np.eye(q, dtype=np.uint8)), np.kron(np.eye(m, dtype=np.uint8), h.T)]
    )
    hz = np.column_stack(
        [np.kron(np.eye(q, dtype=np.uint8), h), np.kron(h.T, np.eye(m, dtype=np.uint8))]
    )
    assert hx.shape[1] == n and not (hx @ hz.T % 2).any()
    # Choose independent logical Z representatives modulo Z stabilizers.
    basis = {}

    def add(row):
        v = sum(int(x) << j for j, x in enumerate(row))
        while v:
            pivot = v.bit_length() - 1
            if pivot in basis:
                v ^= basis[pivot]
            else:
                basis[pivot] = v
                return True
        return False

    for row in hz:
        add(row)
    lz = np.array([row for row in kernel(hx) if add(row)], dtype=np.uint8)
    return h, hx, hz, lz


@cache
def hgp_ideal(n):
    _, hx, hz, lz = hgp_matrices(n)
    checks = [stim.PauliString("".join("X" if v else "I" for v in row)) for row in hx]
    checks += [
        stim.PauliString("".join("Z" if v else "I" for v in row))
        for row in np.vstack([hz, lz])
    ]
    c = stim.Circuit()
    c.append("R", range(n))
    c += stim.Tableau.from_stabilizers(checks, allow_redundant=True).to_circuit()
    measurement = stim.Circuit()
    for check in checks:
        ts = []
        for q, pauli in enumerate(check):
            if pauli:
                if ts:
                    ts.append(stim.target_combiner())
                ts.append((stim.target_x if pauli == 1 else stim.target_z)(q))
        measurement.append("MPP", ts)
    for j in range(len(hx) + len(hz)):
        measurement.append("DETECTOR", [stim.target_rec(j - len(checks))])
    for j in range(len(lz)):
        measurement.append("OBSERVABLE_INCLUDE", [stim.target_rec(j - len(lz))], j)
    return c, measurement


def make_circuit(spec, p):
    family = spec["family"]
    if family == "hgp":
        before, after = hgp_ideal(spec["n"])
        c = before.copy()
        for q in range(spec["n"]):
            channel(c, [q], p, spec["noise"], q)
        if spec["noise"] != "uniform_single":
            for q in range(0, spec["n"] - 1, 2):
                channel(c, [q, q + 1], p, spec["noise"], q, pair=True)
        return c + after
    if family == "bicycle":
        filename = "bbcode_{n}_{k}_{distance}_rounds{rounds}".format(**spec)
        ideal = stim.Circuit.from_file(ROOT / "stimprograms" / "ldpc" / filename)
        if not spec["data_only"]:
            return circuit_noise(ideal, p, spec["noise"])
        # Full stored schedule is retained. Noise is restricted to data qubits
        # immediately before M in their final H+M X readout; explicitly labelled.
        flat = list(ideal.flattened())
        last_measure = {}
        for j, op in enumerate(flat):
            if op.name == "M":
                for t in op.targets_copy():
                    last_measure[t.value] = j
        c = stim.Circuit()
        data_readouts = {}
        for q in range(spec["n"]):
            if q in last_measure:
                data_readouts.setdefault(last_measure[q], []).append(q)
        for j, op in enumerate(flat):
            for q in data_readouts.get(j, []):
                channel(c, [q], p, spec["noise"], q)
            c.append(op)
        return c
    if family == "stabir":
        from scalerqec.QEC.noisemodel import SD6NoiseModel, SI1000NoiseModel
        from scalerqec.QEC.surface import SurfaceCode

        code = SurfaceCode(distance=spec["distance"], rounds=spec["rounds"])
        code.scheme = "Standard"
        code.noisemodel = (
            SD6NoiseModel if spec["noise"] == "SD6" else SI1000NoiseModel
        )(p)
        code.construct_circuit()
        return code.stimcirc
    task = (
        "color_code:memory_xyz"
        if family == "color"
        else f"surface_code:rotated_memory_{spec['basis']}"
    )
    ideal = stim.Circuit.generated(
        task, distance=spec["distance"], rounds=spec["rounds"]
    )
    return circuit_noise(ideal, p, spec["noise"])
