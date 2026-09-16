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
    rewrite_stim_code,
    _1Q_PASSTHROUGH,
    _1Q_DECOMPOSITIONS,
    _2Q_PASSTHROUGH,
    _2Q_DECOMPOSITIONS,
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
    """Extract a non-uniform noise model from a Stim circuit.

    Walks the original Stim circuit instruction by instruction. For each
    noise directive, records per-qubit ``(px, py, pz)`` as "pending". When
    a gate is encountered, consumes the pending noise and assigns it to
    the QEPG noise sources created by that gate.

    The noise source ordering matches the QEPG compiler's convention:
    one depolarize per non-Reset primitive gate in the normalized circuit.

    Args:
        original_circuit_str: Raw Stim circuit string with noise directives.

    Returns:
        A ``NonuniformNoiseModel`` with per-source probabilities.
    """
    circuit = stim.Circuit(original_circuit_str)

    # First pass: count total noise sources
    total_noise = 0
    for inst in _flatten_stim_circuit(circuit):
        name = inst.name
        if name in _NOISE_CHANNELS or name in _ANNOTATIONS:
            continue
        targets = [t.value for t in inst.targets_copy() if not t.is_combiner]
        count = _GATE_NOISE_COUNT.get(name, 0)
        if name in ("CX", "CZ"):
            # Pairwise: count per pair
            num_pairs = len(targets) // 2
            total_noise += count * num_pairs
        else:
            total_noise += count * len(targets)

    if total_noise == 0:
        return NonuniformNoiseModel(
            noise_probs=np.zeros((0, 3), dtype=np.float64),
            num_noise=0,
        )

    noise_probs = np.zeros((total_noise, 3), dtype=np.float64)
    correlated_pairs: list[CorrelatedNoisePair] = []
    correlated_indices: set[int] = set()

    # Per-qubit pending noise: qubit → (px, py, pz)
    pending_1q: dict[int, tuple[float, float, float]] = {}
    # Per-qubit-pair pending DEPOLARIZE2: (q1, q2) → prob
    pending_2q: dict[tuple[int, int], float] = {}

    noise_idx = 0

    for inst in _flatten_stim_circuit(circuit):
        name = inst.name
        args = inst.gate_args_copy()
        targets = [t.value for t in inst.targets_copy() if not t.is_combiner]

        # --- Noise directives: store as pending ---
        if name == "DEPOLARIZE1":
            p = args[0] if args else 0.0
            for q in targets:
                _accumulate_noise(pending_1q, q, p / 3, p / 3, p / 3)
            continue

        if name == "X_ERROR":
            p = args[0] if args else 0.0
            for q in targets:
                _accumulate_noise(pending_1q, q, p, 0.0, 0.0)
            continue

        if name == "Y_ERROR":
            p = args[0] if args else 0.0
            for q in targets:
                _accumulate_noise(pending_1q, q, 0.0, p, 0.0)
            continue

        if name == "Z_ERROR":
            p = args[0] if args else 0.0
            for q in targets:
                _accumulate_noise(pending_1q, q, 0.0, 0.0, p)
            continue

        if name == "DEPOLARIZE2":
            p = args[0] if args else 0.0
            for i in range(0, len(targets), 2):
                q1, q2 = targets[i], targets[i + 1]
                pending_2q[(q1, q2)] = pending_2q.get((q1, q2), 0.0) + p
            continue

        # --- Annotations: skip ---
        if name in _ANNOTATIONS:
            continue

        # --- Gates: consume pending noise ---
        if name == "CX":
            for i in range(0, len(targets), 2):
                q1, q2 = targets[i], targets[i + 1]
                # Control qubit noise source
                ctrl_idx = noise_idx
                pn = pending_1q.pop(q1, None)
                if pn and ctrl_idx < total_noise:
                    noise_probs[ctrl_idx] = pn
                noise_idx += 1
                # Target qubit noise source
                tgt_idx = noise_idx
                pn = pending_1q.pop(q2, None)
                if pn and tgt_idx < total_noise:
                    noise_probs[tgt_idx] = pn
                noise_idx += 1
                # DEPOLARIZE2 correlation
                dep2_key = (q1, q2)
                if dep2_key in pending_2q:
                    p = pending_2q.pop(dep2_key)
                    correlated_pairs.append(
                        CorrelatedNoisePair(source_a=ctrl_idx, source_b=tgt_idx, prob=p)
                    )
                    correlated_indices.add(ctrl_idx)
                    correlated_indices.add(tgt_idx)

        elif name == "CZ":
            # Decomposes to H(t) + CX(c,t) + H(t)
            for i in range(0, len(targets), 2):
                q1, q2 = targets[i], targets[i + 1]
                # H on target: 1 source (no pending noise assigned here)
                noise_idx += 1
                # CX: 2 sources
                ctrl_idx = noise_idx
                pn = pending_1q.pop(q1, None)
                if pn and ctrl_idx < total_noise:
                    noise_probs[ctrl_idx] = pn
                noise_idx += 1
                tgt_idx = noise_idx
                pn = pending_1q.pop(q2, None)
                if pn and tgt_idx < total_noise:
                    noise_probs[tgt_idx] = pn
                noise_idx += 1
                # H on target: 1 source
                noise_idx += 1
                # DEPOLARIZE2 correlation
                dep2_key = (q1, q2)
                if dep2_key in pending_2q:
                    p = pending_2q.pop(dep2_key)
                    correlated_pairs.append(
                        CorrelatedNoisePair(source_a=ctrl_idx, source_b=tgt_idx, prob=p)
                    )
                    correlated_indices.add(ctrl_idx)
                    correlated_indices.add(tgt_idx)

        elif name == "R":
            # R creates 0 noise sources — discard pending noise
            for q in targets:
                pending_1q.pop(q, None)

        elif name in ("H", "S", "X", "Y", "Z", "M"):
            # Single primitive gate: 1 noise source per qubit
            for q in targets:
                pn = pending_1q.pop(q, None)
                if pn and noise_idx < total_noise:
                    noise_probs[noise_idx] = pn
                noise_idx += 1

        elif name in ("MR",):
            # M(1)+R(0) per qubit
            for q in targets:
                pn = pending_1q.pop(q, None)
                if pn and noise_idx < total_noise:
                    noise_probs[noise_idx] = pn
                noise_idx += 1  # M

        elif name in ("MX",):
            # H(1)+M(1) per qubit
            for q in targets:
                pn = pending_1q.pop(q, None)
                if pn and noise_idx < total_noise:
                    noise_probs[noise_idx] = pn
                noise_idx += _GATE_NOISE_COUNT[name]

        elif name in ("MY",):
            # S+S+S+H+M per qubit: 5 sources
            for q in targets:
                pn = pending_1q.pop(q, None)
                if pn and noise_idx < total_noise:
                    noise_probs[noise_idx] = pn
                noise_idx += _GATE_NOISE_COUNT[name]

        elif name == "RX":
            for q in targets:
                pending_1q.pop(q, None)  # R discards noise
                noise_idx += 1  # H only

        elif name == "S_DAG":
            for q in targets:
                pn = pending_1q.pop(q, None)
                if pn and noise_idx < total_noise:
                    noise_probs[noise_idx] = pn
                noise_idx += 3

        else:
            # Other composite gates: consume pending noise, advance by table count
            count = _GATE_NOISE_COUNT.get(name, 0)
            if name in ("CX", "CZ"):
                pass  # handled above
            else:
                for q in targets:
                    pn = pending_1q.pop(q, None)
                    if pn and noise_idx < total_noise:
                        noise_probs[noise_idx] = pn
                    noise_idx += count

    return NonuniformNoiseModel(
        noise_probs=noise_probs,
        correlated_pairs=correlated_pairs,
        num_noise=total_noise,
        correlated_source_indices=correlated_indices,
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
