"""STIM circuit program normalization.

This module provides utilities to preprocess raw STIM circuit programs into a
canonical one-operation-per-line format. The normalized output can then be
parsed by :meth:`~scalerqec.Clifford.clifford.CliffordCircuit.compile_from_stim_circuit_str`.

Both a :class:`stimparser` class and a standalone :func:`rewrite_stim_code`
function are provided; they perform the same transformation.

Gate support is table-driven: to add a new gate, add an entry to
``_1Q_PASSTHROUGH``, ``_1Q_DECOMPOSITIONS``, ``_2Q_PASSTHROUGH``, or
``_2Q_DECOMPOSITIONS`` below.
"""

# ---------------------------------------------------------------------------
# Gate dispatch tables
# ---------------------------------------------------------------------------

# Single-qubit gates that pass through as-is (one line per qubit)
_1Q_PASSTHROUGH = {"H", "S", "M", "R", "X", "Y", "Z"}

# Single-qubit gates decomposed to primitive sequences.
# Each value is a list of primitive gate names to emit per qubit.
_1Q_DECOMPOSITIONS = {
    "S_DAG": ["S", "S", "S"],
    "MX": ["H", "M", "H"],
    "MY": ["S", "S", "S", "H", "M", "H", "S"],
    "MR": ["M", "R"],
    "MRX": ["H", "M", "R", "H"],
    "MRY": ["S", "S", "S", "H", "M", "R", "H", "S"],
    "RX": ["R", "H"],
    "RY": ["R", "H", "S"],
    "SQRT_X": ["H", "S", "H"],
    "SQRT_X_DAG": ["H", "S", "S", "S", "H"],
    "SQRT_Y": ["H", "X"],
    "SQRT_Y_DAG": ["H", "Z"],
}

# Two-qubit gates that pass through as-is (pairwise split)
_2Q_PASSTHROUGH = {"CX"}

# Two-qubit gates decomposed to primitive sequences per pair.
# Each value is a list of (gate_name, "c"|"t") tuples indicating which qubit.
_2Q_DECOMPOSITIONS = {
    "CZ": [("H", "t"), ("CX", "ct"), ("H", "t")],
}

# Annotation keywords — matched by first token (with any trailing '(' stripped).
_ANNOTATION_KEYWORDS = {"TICK", "DETECTOR", "QUBIT_COORDS", "OBSERVABLE_INCLUDE"}

# Noise/metadata directives stripped by the normalizer.
# Matched by first token (with any trailing '(' stripped).
_NOISE_KEYWORDS = {
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "SHIFT_COORDS",
}

# Single-qubit noise instructions (one qubit per line when splitting)
_1Q_NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1"}

# Two-qubit noise instructions (qubit pair per line when splitting)
_2Q_NOISE = {"DEPOLARIZE2", "PAULI_CHANNEL_2"}


# ---------------------------------------------------------------------------
# Core rewrite logic
# ---------------------------------------------------------------------------


def _rewrite(code: str, *, keep_noise: bool = False) -> str:
    """Normalize a STIM program so each line has at most one gate operation.

    Transformations:
    1. Splits multi-target instructions (one op per qubit or qubit pair).
    2. Decomposes composite gates to primitives via the tables above.
    3. Strips noise directives unless *keep_noise* is ``True``.

    Args:
        code: The raw STIM circuit program as a multi-line string.
        keep_noise: If ``True``, noise directives are preserved and split
            per qubit (or qubit pair) instead of being stripped.

    Returns:
        A normalized STIM program string with one operation per line.
    """
    lines = code.splitlines()
    output_lines: list[str] = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        tokens = stripped_line.split()
        # Extract keyword: strip trailing '(' for instructions like "DETECTOR(0,1)"
        keyword = tokens[0].split("(")[0]

        # Preserve annotations
        if keyword in _ANNOTATION_KEYWORDS:
            output_lines.append(stripped_line)
            continue

        # Handle noise directives
        if keyword in _NOISE_KEYWORDS:
            if not keep_noise:
                continue
            # Preserve noise: split per qubit or qubit pair
            gate_with_args = tokens[0]  # e.g. "DEPOLARIZE1(0.001)"
            qubits = tokens[1:]
            if keyword in _1Q_NOISE:
                for q in qubits:
                    output_lines.append(f"{gate_with_args} {q}")
            elif keyword in _2Q_NOISE:
                for i in range(0, len(qubits), 2):
                    q1, q2 = qubits[i], qubits[i + 1]
                    output_lines.append(f"{gate_with_args} {q1} {q2}")
            else:
                # SHIFT_COORDS or other metadata — preserve as-is
                output_lines.append(stripped_line)
            continue

        gate = tokens[0]
        qubits = tokens[1:]

        # --- Single-qubit passthrough ---
        if gate in _1Q_PASSTHROUGH:
            for q in qubits:
                output_lines.append(f"{gate} {q}")

        # --- Single-qubit decomposition ---
        elif gate in _1Q_DECOMPOSITIONS:
            seq = _1Q_DECOMPOSITIONS[gate]
            for q in qubits:
                for prim in seq:
                    output_lines.append(f"{prim} {q}")

        # --- Two-qubit passthrough ---
        elif gate in _2Q_PASSTHROUGH:
            for i in range(0, len(qubits), 2):
                q1, q2 = qubits[i], qubits[i + 1]
                output_lines.append(f"{gate} {q1} {q2}")

        # --- Two-qubit decomposition ---
        elif gate in _2Q_DECOMPOSITIONS:
            seq = _2Q_DECOMPOSITIONS[gate]
            for i in range(0, len(qubits), 2):
                q1, q2 = qubits[i], qubits[i + 1]
                for prim_name, which in seq:
                    if which == "c":
                        output_lines.append(f"{prim_name} {q1}")
                    elif which == "t":
                        output_lines.append(f"{prim_name} {q2}")
                    elif which == "ct":
                        output_lines.append(f"{prim_name} {q1} {q2}")

        else:
            # Unknown gate: preserve as-is
            output_lines.append(stripped_line)

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class stimparser:
    """Utility class for normalizing STIM circuit programs.

    Provides :meth:`rewrite_stim_code` to transform a raw STIM program into
    a canonical form where each line contains exactly one gate or measurement
    operation, suitable for consumption by
    :meth:`~scalerqec.Clifford.clifford.CliffordCircuit.compile_from_stim_circuit_str`.
    """

    def __init__(self):
        pass

    def rewrite_stim_code(self, code: str, *, keep_noise: bool = False) -> str:
        """Normalize a STIM program so each line has at most one gate operation.

        See :func:`rewrite_stim_code` for full details.

        Args:
            code: The raw STIM circuit program as a multi-line string.
            keep_noise: If ``True``, noise directives are preserved and
                split per qubit instead of being stripped.

        Returns:
            A normalized STIM program string with one operation per line.
        """
        return _rewrite(code, keep_noise=keep_noise)


def rewrite_stim_code(code: str, *, keep_noise: bool = False) -> str:
    """Normalize a STIM program so each line has at most one gate operation.

    This is a module-level convenience function equivalent to
    ``stimparser().rewrite_stim_code(code)``. See the module-level dispatch
    tables for the full list of supported gates and their decompositions.

    Args:
        code: The raw STIM circuit program as a multi-line string.
        keep_noise: If ``True``, noise directives are preserved and split
            per qubit instead of being stripped.

    Returns:
        A normalized STIM program string with one operation per line.
    """
    return _rewrite(code, keep_noise=keep_noise)
