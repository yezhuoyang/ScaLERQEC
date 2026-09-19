"""Explicit extraction gadgets and encoded Bell teleportation.

These are fixed circuits, not adaptive fault-tolerant protocols. Flag records
and decoded-cat records are supplied to the decoder, without postselection.
Knill resources are synthesized using noisy Clifford gates, without ancilla
verification. Code distance alone does not certify their circuit distance.
"""

from __future__ import annotations

import stim


def controlled_pauli(circuit, control, target, pauli):
    """Append a controlled Pauli using H, S and CX only, with correct Y phase."""
    if pauli == "Z":
        circuit.append("H", [target])
    elif pauli == "Y":
        circuit.append("S_DAG", [target])
    circuit.append("CX", [control, target])
    if pauli == "Z":
        circuit.append("H", [target])
    elif pauli == "Y":
        circuit.append("S", [target])


def append_parity_gadget(circuit, pauli, data, ancilla_start, scheme):
    """Measure a Pauli product; return absolute syndrome and flag record indices.

    Flag: one syndrome ancilla, bracketed by two CNOTs to one flag.
    Shor: one cat qubit per support qubit, transversal coupling, inverse cat
    preparation, then readout. Non-root cat outcomes are additional detectors.
    """
    support = [(q, p) for q, p in zip(data, pauli) if p != "I"]
    if not support:
        raise ValueError("A parity gadget requires nonidentity support.")
    root = ancilla_start
    if scheme == "Flag":
        flag = root + 1
        circuit.append("R", [root, flag])
        circuit.append("H", [root])
        circuit.append("TICK")
        circuit.append("CX", [root, flag])
        for q, p in support:
            controlled_pauli(circuit, root, q, p)
            circuit.append("TICK")
        circuit.append("CX", [root, flag])
        circuit.append("H", [root])
        first = circuit.num_measurements
        circuit.append("M", [root, flag])
        return [first], [first + 1]
    if scheme != "Shor":
        raise ValueError(f"Unknown parity gadget: {scheme}")
    cat = list(range(root, root + len(support)))
    circuit.append("R", cat)
    circuit.append("H", [root])
    for q in cat[1:]:
        circuit.append("CX", [root, q])
    circuit.append("TICK")
    for a, (q, p) in zip(cat, support):
        controlled_pauli(circuit, a, q, p)
    circuit.append("TICK")
    for q in reversed(cat[1:]):
        circuit.append("CX", [root, q])
    circuit.append("H", [root])
    first = circuit.num_measurements
    circuit.append("M", cat)
    return [first], list(range(first + 1, first + len(cat)))


def code_tableau(stabilizers, logical_z, n):
    """Validate a complete stabilizer code and obtain canonical logical Xs."""
    stabs = [stim.PauliString(s) for s in stabilizers]
    # The first independent z outputs are the code stabilizers. Count rank
    # directly; redundant checks are allowed, dependent logicals are not.
    pivots = {}
    for s in stabs:
        xs, zs = s.to_numpy()
        row = sum(int(b) << i for i, b in enumerate(list(xs) + list(zs)))
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    rank = len(pivots)
    if rank != n - len(logical_z):
        raise ValueError(
            "Stabilizer rank must equal n-k; redundant checks are allowed."
        )
    tableau = stim.Tableau.from_stabilizers(
        stabs + [stim.PauliString(z) for z in logical_z], allow_redundant=True
    )
    # A dependent logical could otherwise be silently skipped by Stim.
    if any(
        tableau.z_output(rank + j) != stim.PauliString(z)
        for j, z in enumerate(logical_z)
    ):
        raise ValueError(
            "Logical Z operators must be independent modulo the stabilizers."
        )
    return tableau, rank


def encoded_bell_circuit(tableau, rank):
    """Prepare conjugate(code) on A entangled with code on B.

    Conjugation on A is essential for codes whose generators contain an odd
    number of Ys. It makes physical Bell measurement implement teleportation.
    """
    n = len(tableau)
    constraints = []
    for j in range(rank):
        s = tableau.z_output(j)
        constraints += [
            conjugate_pauli(s) + stim.PauliString(n),
            stim.PauliString(n) + s,
        ]
    for j in range(rank, n):
        for s in (tableau.x_output(j), tableau.z_output(j)):
            constraints.append(conjugate_pauli(s) + s)
    return stim.Tableau.from_stabilizers(constraints).to_circuit()


def conjugate_pauli(pauli):
    result = pauli.copy()
    result.sign = complex(pauli.sign).conjugate() * (-1) ** sum(p == 2 for p in pauli)
    return result


def bell_record_parity(pauli, x_records, z_records):
    """Record parity representing P(data) P*(resource) in Bell readout."""
    return [x_records[i] for i, p in enumerate(pauli) if p in "XY"] + [
        z_records[i] for i, p in enumerate(pauli) if p in "ZY"
    ]


def append_knill_round(circuit, data, resource_a, resource_b, bell_prep):
    """Teleport a block into B; return physical Bell X/Z record indices.

    Logical byproducts are tracked in observable parities by the caller.
    No physical, noisy correction gates are inserted for a Pauli frame update.
    """
    targets = resource_a + resource_b
    circuit.append("R", targets)
    for op in bell_prep:
        circuit.append(
            op.name, [targets[t.value] for t in op.targets_copy()], op.gate_args_copy()
        )
    circuit.append("TICK")
    for d, a in zip(data, resource_a):
        circuit.append("CX", [d, a])
    circuit.append("H", data)
    circuit.append("TICK")
    first = circuit.num_measurements
    circuit.append("M", data + resource_a)
    circuit.append("TICK")
    n = len(data)
    return list(range(first, first + n)), list(range(first + n, first + 2 * n))


def append_detector(circuit, records):
    count = circuit.num_measurements
    circuit.append("DETECTOR", [stim.target_rec(i - count) for i in records])


def compile_memory_gadgets(code):
    """Lower StabCode's memory IR with Flag or decoded-cat Shor gadgets."""
    from .qeccircuit import (
        DataMeasureInstruction,
        DetectorInstruction,
        ObservableInstruction,
        StabPropInstruction,
    )

    circuit = stim.Circuit()
    records = {}
    data = list(range(code.n))
    circuit.append("R", data)
    circuit.append("TICK")
    for instruction in code._IRList:
        if isinstance(instruction, StabPropInstruction):
            result, flags = append_parity_gadget(
                circuit, instruction.stab, data, code.n, code.scheme.name.title()
            )
            records[instruction.dest] = result
            for flag in flags:
                append_detector(circuit, [flag])
            circuit.append("TICK")
        elif isinstance(instruction, DataMeasureInstruction):
            first = circuit.num_measurements
            circuit.append("M", data)
            records.update({f"m{q}": [first + q] for q in data})
        elif isinstance(instruction, (DetectorInstruction, ObservableInstruction)):
            refs = [i for arg in instruction.args for i in records[arg]]
            if isinstance(instruction, DetectorInstruction):
                append_detector(circuit, refs)
            else:
                circuit.append(
                    "OBSERVABLE_INCLUDE",
                    [stim.target_rec(i - circuit.num_measurements) for i in refs],
                    int(instruction.dest[1:]),
                )
        else:
            raise NotImplementedError(
                f"Unsupported memory IR instruction: {instruction}"
            )
    return circuit


def compile_knill_memory(code):
    """Encoded Bell extraction with deferred logical Pauli frame tracking."""
    logical_z = [code._logicalZ[j] for j in range(code.k)]
    tableau, rank = code_tableau(code._stabs, logical_z, code.n)
    prep = encoded_bell_circuit(tableau, rank)
    circuit = stim.Circuit()
    blocks = [list(range(j * code.n, (j + 1) * code.n)) for j in range(3)]
    current = 0
    circuit.append("R", blocks[current])
    circuit.append("TICK")
    frames = [[] for _ in logical_z]
    for r in range(code.rounds):
        a, b = [j for j in range(3) if j != current]
        xs, zs = append_knill_round(
            circuit, blocks[current], blocks[a], blocks[b], prep
        )
        for s in code._stabs:
            if r or all(p in "IZ" for p in s):
                append_detector(circuit, bell_record_parity(s, xs, zs))
        for j, z in enumerate(logical_z):
            frames[j] += bell_record_parity(z, xs, zs)
        current = b
    first = circuit.num_measurements
    circuit.append("M", blocks[current])
    for s in code._stabs:
        if all(p in "IZ" for p in s):
            append_detector(circuit, [first + i for i, p in enumerate(s) if p == "Z"])
    for j, z in enumerate(logical_z):
        refs = frames[j] + [first + i for i, p in enumerate(z) if p == "Z"]
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(i - circuit.num_measurements) for i in refs],
            j,
        )
    return circuit
