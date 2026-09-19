"""Parse Stim circuits to extract per-noise-source probability information.

This module bridges the gap between Stim's rich noise model (non-uniform
error rates, axis-specific channels, two-qubit depolarization) and the
QEPG framework's binary propagation matrix. It extracts per-noise-source
error probabilities ``(px, py, pz)`` that can be used for non-uniform
Monte Carlo sampling.

The key insight is that the QEPG propagation matrix is noise-rate-agnostic
(binary GF(2)), so we only need probability information at sampling time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

from ..Clifford.stimparser import (
    _1Q_DECOMPOSITIONS,
    _1Q_PASSTHROUGH,
    _2Q_DECOMPOSITIONS,
    _2Q_PASSTHROUGH,
    rewrite_stim_code,
)

# The 15 non-identity two-qubit Paulis for DEPOLARIZE2.
# Each tuple is (pauli_on_qubit_a, pauli_on_qubit_b) where 0=I, 1=X, 2=Y, 3=Z.
TWO_QUBIT_PAULIS = [
    (1, 0),
    (2, 0),
    (3, 0),  # XI, YI, ZI
    (0, 1),
    (0, 2),
    (0, 3),  # IX, IY, IZ
    (1, 1),
    (1, 2),
    (1, 3),  # XX, XY, XZ
    (2, 1),
    (2, 2),
    (2, 3),  # YX, YY, YZ
    (3, 1),
    (3, 2),
    (3, 3),  # ZX, ZY, ZZ
]


@dataclass
class CorrelatedNoisePair:
    """A DEPOLARIZE2 event linking two noise source indices.

    Attributes:
        source_a: QEPG noise source index for the first qubit.
        source_b: QEPG noise source index for the second qubit.
        prob: Total DEPOLARIZE2 probability (each of 15 Paulis has prob/15).
    """

    source_a: int
    source_b: int
    prob: float


@dataclass
class NonuniformNoiseModel:
    """Per-noise-source probability model extracted from a Stim circuit.

    Attributes:
        noise_probs: Array of shape ``(num_noise, 3)`` with columns
            ``[px, py, pz]`` for each independent noise source.
        correlated_pairs: List of DEPOLARIZE2 correlation events.
        num_noise: Total number of noise sources.
        correlated_source_indices: Set of noise source indices involved
            in DEPOLARIZE2 pairs.
    """

    noise_probs: np.ndarray
    correlated_pairs: list[CorrelatedNoisePair] = field(default_factory=list)
    num_noise: int = 0
    correlated_source_indices: set[int] = field(default_factory=set)


# Number of QEPG noise sources created by each gate after normalization.
# Auto-derived from stimparser decomposition tables to stay in sync.
# Convention: 1 depolarize per non-Reset primitive gate.
def _build_gate_noise_count() -> dict[str, int]:
    """Derive noise source counts from stimparser gate tables."""
    counts: dict[str, int] = {}
    # Single-qubit passthrough: 1 noise source each (except R which has 0)
    for gate in _1Q_PASSTHROUGH:
        counts[gate] = 0 if gate == "R" else 1
    # Single-qubit decompositions: count non-R primitives
    for gate, seq in _1Q_DECOMPOSITIONS.items():
        counts[gate] = sum(1 for prim in seq if prim != "R")
    # Two-qubit passthrough: 1 per qubit
    for gate in _2Q_PASSTHROUGH:
        counts[gate] = 2
    # Two-qubit decompositions: count primitives (each "ct" is 2 sources)
    for gate, seq in _2Q_DECOMPOSITIONS.items():
        counts[gate] = sum(2 if which == "ct" else 1 for _, which in seq)
    return counts


_GATE_NOISE_COUNT: dict[str, int] = _build_gate_noise_count()

# Noise directives recognized from Stim
_NOISE_CHANNELS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}

# Annotation/metadata instructions (no noise sources)
_ANNOTATIONS = {
    "TICK",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
}


def _flatten_stim_circuit(circuit: stim.Circuit):
    """Yield all instructions from a Stim circuit, expanding REPEAT blocks."""
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            body = inst.body_copy()
            for _ in range(inst.repeat_count):
                yield from _flatten_stim_circuit(body)
        else:
            yield inst


def extract_noise_model(original_circuit_str: str) -> NonuniformNoiseModel:
    """Map supported noise to the next primitive source on each operand.

    Each DEPOLARIZE2 instruction remains an independent categorical event.
    Its two operands can map to different subsequent operations. A reset or
    the end of a qubit's lifetime marginalizes that operand. Unsupported
    syntax raises instead of silently omitting noise; the experimental
    LinearNoiseModel supports the wider Stim instruction set.
    """
    circuit = stim.Circuit(original_circuit_str).flattened()
    events = []
    total_noise = 0
    supported_gates = set(_GATE_NOISE_COUNT)
    channels = _NOISE_CHANNELS | {"PAULI_CHANNEL_1"}
    for op in circuit:
        if op.name in channels:
            events.append((op, None))
            continue
        if op.name in _ANNOTATIONS:
            continue
        if (
            op.name not in supported_gates
            or op.gate_args_copy()
            or any(
                not t.is_qubit_target or t.is_inverted_result_target
                for t in op.targets_copy()
            )
        ):
            raise NotImplementedError(
                f"Legacy QEPG noise mapping does not support {op.name}; "
                "use Stim or Stratified.general_noise.LinearNoiseModel."
            )
        # Normalize one original instruction so the next source is BEFORE
        # the first primitive of a decomposition, including CZ's leading H.
        untagged = stim.CircuitInstruction(op.name, op.targets_copy())
        normalized = stim.Circuit(rewrite_stim_code(str(untagged)))
        for primitive in normalized:
            targets = primitive.targets_copy()
            stride = 2 if primitive.name == "CX" else 1
            for offset in range(0, len(targets), stride):
                group = targets[offset : offset + stride]
                indices = (
                    None
                    if primitive.name == "R"
                    else list(range(total_noise, total_noise + len(group)))
                )
                if indices is not None:
                    total_noise += len(group)
                events.append((stim.CircuitInstruction(primitive.name, group), indices))

    pending = {}
    noise_probs = np.zeros((total_noise, 3), dtype=np.float64)
    pairs = []
    next_source = {}
    for op, indices in reversed(events):
        targets = [t.value for t in op.targets_copy()]
        if op.name not in channels:
            for i, q in enumerate(targets):
                next_source[q] = None if indices is None else indices[i]
            continue
        args = op.gate_args_copy()
        if op.name == "DEPOLARIZE2":
            for a, b in zip(targets[::2], targets[1::2]):
                sa, sb = next_source.get(a), next_source.get(b)
                if sa is not None and sb is not None:
                    pairs.append(CorrelatedNoisePair(sa, sb, args[0]))
                elif sa is not None or sb is not None:
                    # Tracing out one operand leaves X/Y/Z each at 4p/15.
                    _accumulate_noise(
                        pending, sa if sa is not None else sb, *([4 * args[0] / 15] * 3)
                    )
        else:
            if op.name == "DEPOLARIZE1":
                probabilities = [args[0] / 3] * 3
            elif op.name == "PAULI_CHANNEL_1":
                probabilities = args
            else:
                probabilities = [args[0] if axis == op.name[0] else 0 for axis in "XYZ"]
            for q in targets:
                source = next_source.get(q)
                if source is not None:
                    _accumulate_noise(pending, source, *probabilities)
    for source, probabilities in pending.items():
        noise_probs[source] = probabilities
    return NonuniformNoiseModel(
        noise_probs=noise_probs,
        correlated_pairs=list(reversed(pairs)),
        num_noise=total_noise,
        correlated_source_indices={
            s for pair in pairs for s in (pair.source_a, pair.source_b)
        },
    )


def _compose_prob(p1: float, p2: float) -> float:
    """Compose two independent error probabilities: p1 + p2 - 2*p1*p2."""
    return p1 + p2 - 2.0 * p1 * p2


def _accumulate_noise(
    pending: dict[int, tuple[float, float, float]],
    qubit: int,
    px: float,
    py: float,
    pz: float,
) -> None:
    """Accumulate noise probabilities for a qubit.

    Compose categorical I/X/Y/Z distributions by Pauli multiplication.
    Cross-axis products matter: X followed by Y is Z, ignoring global phase.
    """
    if qubit in pending:
        ox, oy, oz = pending[qubit]
        oi = 1.0 - ox - oy - oz
        pi = 1.0 - px - py - pz
        pending[qubit] = (
            oi * px + ox * pi + oy * pz + oz * py,
            oi * py + oy * pi + ox * pz + oz * px,
            oi * pz + oz * pi + ox * py + oy * px,
        )
    else:
        pending[qubit] = (px, py, pz)
