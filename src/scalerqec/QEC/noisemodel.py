"""Noise models for quantum error correction circuits.

This module provides noise model classes that inject noise into
``stim.Circuit`` objects. The base :class:`NoiseModel` injects uniform
depolarizing noise. Subclasses :class:`SD6NoiseModel`,
:class:`SI1000NoiseModel`, and :class:`SIDNoiseModel` implement
standard circuit-level noise models.

Individual error channels can be selectively disabled via
:meth:`NoiseModel.disable_error`.
"""

from __future__ import annotations

from enum import Enum

import stim

from ..Clifford.clifford import (
    CliffordCircuit,
    Measurement,
    Reset,
    SingleQGate,
    TwoQGate,
)


class ErrorType(Enum):
    """Enumeration of gate/operation types that can carry noise.

    Each member corresponds to a physical operation in the circuit.
    Used with :meth:`NoiseModel.disable_error` to selectively silence
    individual noise channels.

    Attributes:
        MEASUREMENT: Measurement operation.
        RESET: Qubit reset operation.
        CNOT: Two-qubit CNOT gate.
        HADAMARD: Hadamard gate.
        PHASE: Phase (S) gate.
        PAULIX: Pauli-X gate.
        PAULIY: Pauli-Y gate.
        PAULIZ: Pauli-Z gate.
        CZ: Two-qubit controlled-Z gate.
    """

    MEASUREMENT = 0
    RESET = 1
    CNOT = 2
    HADAMARD = 3
    PHASE = 4
    PAULIX = 5
    PAULIY = 6
    PAULIZ = 7
    CZ = 8


# Gate names that are single-qubit Clifford gates
_1Q_GATES = {
    "H",
    "S",
    "S_DAG",
    "X",
    "Y",
    "Z",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
}
# Gate names that are two-qubit gates
_2Q_GATES = {"CX", "CZ"}


class NoiseModel:
    """Configurable depolarizing noise model.

    Injects uniform depolarizing noise into a ``stim.Circuit``:

    * ``DEPOLARIZE1(p)`` after each single-qubit gate
    * ``DEPOLARIZE2(p)`` after each two-qubit gate
    * An anticommuting Pauli error after each reset
    * An anticommuting Pauli error before each measurement

    Args:
        error_rate: Probability of depolarizing error per gate.
    """

    def __init__(self, error_rate: float) -> None:
        self._error_rate = error_rate

        self._has_MEASUREMENT_error = True
        self._has_RESET_error = True
        self._has_CNOT_error = True
        self._has_CZ_error = True
        self._has_HADAMARD_error = True
        self._has_PHASE_error = True
        self._has_PAULIX_error = True
        self._has_PAULIY_error = True
        self._has_PAULIZ_error = True

    @property
    def error_rate(self) -> float:
        return self._error_rate

    @error_rate.setter
    def error_rate(self, value: float) -> None:
        self._error_rate = value

    def disable_error(self, error_type: str) -> None:
        """Disable noise injection for a specific gate type.

        Args:
            error_type: String identifier of the gate type to silence.
                Accepted values: ``"MEASUREMENT"``, ``"RESET"``,
                ``"CNOT"``, ``"CZ"``, ``"H"``, ``"P"``, ``"X"``,
                ``"Y"``, ``"Z"``.
        """
        if error_type == "MEASUREMENT":
            self._has_MEASUREMENT_error = False
        elif error_type == "RESET":
            self._has_RESET_error = False
        elif error_type == "CNOT":
            self._has_CNOT_error = False
        elif error_type == "CZ":
            self._has_CZ_error = False
        elif error_type == "H":
            self._has_HADAMARD_error = False
        elif error_type == "P":
            self._has_PHASE_error = False
        elif error_type == "X":
            self._has_PAULIX_error = False
        elif error_type == "Y":
            self._has_PAULIY_error = False
        elif error_type == "Z":
            self._has_PAULIZ_error = False

    def inject_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """Inject uniform depolarizing noise into a stim.Circuit.

        Walks through the circuit instruction by instruction and inserts
        noise channels after each gate:

        * Single-qubit gates: ``DEPOLARIZE1(p)``
        * Two-qubit gates: ``DEPOLARIZE2(p)``
        * Reset: an anticommuting Pauli after reset (Z for X basis; X otherwise)
        * Measurement: the same basis-aware Pauli before measurement
        * Measure-reset: both the measurement fault and the reset fault

        Args:
            circuit: A noiseless ``stim.Circuit``.

        Returns:
            A new ``stim.Circuit`` with noise injected.
        """
        return self._inject_operation_noise(
            circuit,
            self._error_rate,
            self._error_rate,
            self._error_rate,
            self._error_rate,
        )

    def _inject_operation_noise(self, circuit, p_reset, p_meas, p_1q, p_2q, p_idle=0):
        # Split coalesced operations: overlapping pairs are sequential gates.
        # Noise must occur between them, not after the whole instruction.
        noisy = stim.Circuit()
        single_flags = {
            "H": self._has_HADAMARD_error,
            "S": self._has_PHASE_error,
            "S_DAG": self._has_PHASE_error,
            "X": self._has_PAULIX_error,
            "Y": self._has_PAULIY_error,
            "Z": self._has_PAULIZ_error,
        }
        if p_idle:
            circuit = circuit.flattened()
        all_qubits = (
            {
                t.value
                for op in circuit.flattened()
                if op.name
                not in {
                    "DETECTOR",
                    "OBSERVABLE_INCLUDE",
                    "QUBIT_COORDS",
                    "SHIFT_COORDS",
                }
                for t in op.targets_copy()
                if t.is_qubit_target
            }
            if p_idle
            else set()
        )
        active = set()
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                body = self._inject_operation_noise(
                    op.body_copy(), p_reset, p_meas, p_1q, p_2q
                )
                noisy.append(stim.CircuitRepeatBlock(op.repeat_count, body))
                continue
            name, targets, args = op.name, op.targets_copy(), op.gate_args_copy()
            if name == "TICK" and p_idle:
                idle = sorted(all_qubits - active)
                if idle:
                    noisy.append("DEPOLARIZE1", idle, p_idle)
                active.clear()
            elif name not in {
                "DETECTOR",
                "OBSERVABLE_INCLUDE",
                "QUBIT_COORDS",
                "SHIFT_COORDS",
            }:
                active.update(t.value for t in targets if t.is_qubit_target)
            stride = 2 if name in _2Q_GATES else 1
            if name not in _1Q_GATES | _2Q_GATES | {
                "R",
                "RX",
                "RY",
                "M",
                "MX",
                "MY",
                "MR",
                "MRX",
                "MRY",
            }:
                noisy.append(op)
                continue
            for offset in range(0, len(targets), stride):
                group = targets[offset : offset + stride]
                qubits = [t.value for t in group if t.is_qubit_target]
                is_measurement = name in {"M", "MX", "MY", "MR", "MRX", "MRY"}
                is_reset = name in {"R", "RX", "RY", "MR", "MRX", "MRY"}
                # Z anticommutes with X-basis preparation/readout. X handles
                # both Z and Y bases. X_ERROR on |+> would have no effect.
                flip = "Z_ERROR" if name.endswith("X") else "X_ERROR"
                if is_measurement and self._has_MEASUREMENT_error and p_meas:
                    noisy.append(flip, qubits, p_meas)
                noisy.append(stim.CircuitInstruction(name, group, args, tag=op.tag))
                if is_reset and self._has_RESET_error and p_reset:
                    noisy.append(flip, qubits, p_reset)
                elif name in _2Q_GATES:
                    enabled = (
                        self._has_CNOT_error if name == "CX" else self._has_CZ_error
                    )
                    # Record-controlled Paulis have one physical operand.
                    if enabled and p_2q and len(qubits) == 2:
                        noisy.append("DEPOLARIZE2", qubits, p_2q)
                elif name in _1Q_GATES and single_flags.get(name, True) and p_1q:
                    noisy.append("DEPOLARIZE1", qubits, p_1q)
        return noisy

    def reconstruct_clifford_circuit(
        self, clifford_circuit: CliffordCircuit
    ) -> CliffordCircuit:
        """Rebuild a Clifford circuit with depolarizing noise injected.

        .. deprecated:: Use :meth:`inject_noise` on a ``stim.Circuit`` instead.

        Creates a new :class:`~scalerqec.Clifford.clifford.CliffordCircuit`
        by iterating over every gate in *clifford_circuit*.  For each
        gate whose type is enabled, a single-qubit depolarizing channel
        (at the stored :attr:`error_rate`) is inserted on the relevant
        qubit(s) immediately before the gate.

        Args:
            clifford_circuit: The noise-free Clifford circuit to augment.

        Returns:
            A new CliffordCircuit with depolarizing noise inserted.
        """
        num_qubits = clifford_circuit.qubitnum
        new_circuit = CliffordCircuit(num_qubits)

        new_circuit.error_rate = self._error_rate
        gate_list = clifford_circuit.gatelists

        for gate in gate_list:
            if isinstance(gate, TwoQGate):
                if gate.name == "CNOT":
                    if self._has_CNOT_error:
                        new_circuit.add_depolarize(gate.control)
                        new_circuit.add_depolarize(gate.target)
                    new_circuit.add_cnot(gate.control, gate.target)
                elif gate.name == "CZ":
                    if self._has_CZ_error:
                        new_circuit.add_depolarize(gate.control)
                        new_circuit.add_depolarize(gate.target)
                    new_circuit.add_cz(gate.control, gate.target)

            elif isinstance(gate, SingleQGate):
                if gate.name == "H":
                    if self._has_HADAMARD_error:
                        new_circuit.add_depolarize(gate.qubitindex)
                    new_circuit.add_hadamard(gate.qubitindex)
                elif gate.name == "P":
                    if self._has_PHASE_error:
                        new_circuit.add_depolarize(gate.qubitindex)
                    new_circuit.add_phase(gate.qubitindex)
                elif gate.name == "X":
                    if self._has_PAULIX_error:
                        new_circuit.add_depolarize(gate.qubitindex)
                    new_circuit.add_paulix(gate.qubitindex)
                elif gate.name == "Y":
                    if self._has_PAULIY_error:
                        new_circuit.add_depolarize(gate.qubitindex)
                    new_circuit.add_pauliy(gate.qubitindex)
                elif gate.name == "Z":
                    if self._has_PAULIZ_error:
                        new_circuit.add_depolarize(gate.qubitindex)
                    new_circuit.add_pauliz(gate.qubitindex)

            elif isinstance(gate, Measurement):
                if self._has_MEASUREMENT_error:
                    new_circuit.add_depolarize(gate.qubitindex)
                new_circuit.add_measurement(gate.qubitindex)

            elif isinstance(gate, Reset):
                if self._has_RESET_error:
                    new_circuit.add_depolarize(gate.qubitindex)
                new_circuit.add_reset(gate.qubitindex)

        new_circuit.parityMatchGroup = clifford_circuit.parityMatchGroup
        new_circuit.observable = clifford_circuit.observable
        new_circuit.compile_detector_and_observable()

        return new_circuit


class SD6NoiseModel(NoiseModel):
    """Standard depolarizing noise model with 6 noise locations (SD6).

    This is the standard circuit-level noise model used in many QEC
    papers. All noise channels use the same physical error rate *p*:

    1. ``X_ERROR(p)`` after each reset
    2. ``DEPOLARIZE1(p)`` after each single-qubit gate
    3. ``DEPOLARIZE2(p)`` after each two-qubit gate
    4. ``X_ERROR(p)`` before each measurement
    5. ``DEPOLARIZE1(p)`` on idle data qubits each tick (not yet implemented)
    6. ``X_ERROR(p)`` after final data measurements (handled by rule 4)

    This is functionally identical to the base :class:`NoiseModel` for
    circuits without idle qubit tracking. The main distinction is
    semantic: SD6 is the standard reference model for benchmarking.

    Args:
        p: Physical error rate.
    """

    def __init__(self, p: float) -> None:
        super().__init__(error_rate=p)


class SI1000NoiseModel(NoiseModel):
    """Superconducting-inspired noise model (SI1000).

    Models a superconducting quantum processor with separate error
    rates for different operation types, reflecting the asymmetry
    between single-qubit, two-qubit, measurement, and idle errors
    in real hardware.

    Default rates follow the SI1000 convention at a given base
    physical error rate *p*:

    * Reset error: ``p``
    * Measurement error: ``5p``
    * Single-qubit gate: ``p / 10``
    * Two-qubit gate: ``p``
    * Idle (per completed TICK-delimited layer): ``p / 10``

    Idle noise affects circuit qubits unused in that layer. A final layer
    without a closing TICK has no additional idle interval.

    Users can override any rate individually.

    Args:
        p: Base physical error rate.
        p_reset: Reset error rate. Defaults to ``p``.
        p_meas: Measurement error rate. Defaults to ``5p``.
        p_1q: Single-qubit gate error rate. Defaults to ``p / 10``.
        p_2q: Two-qubit gate error rate. Defaults to ``p``.
        p_idle: Idle error rate per tick. Defaults to ``p / 10``.
    """

    def __init__(
        self,
        p: float,
        p_reset: float | None = None,
        p_meas: float | None = None,
        p_1q: float | None = None,
        p_2q: float | None = None,
        p_idle: float | None = None,
    ) -> None:
        super().__init__(error_rate=p)
        self._p_reset = p_reset if p_reset is not None else p
        self._p_meas = p_meas if p_meas is not None else 5 * p
        self._p_1q = p_1q if p_1q is not None else p / 10
        self._p_2q = p_2q if p_2q is not None else p
        self._p_idle = p_idle if p_idle is not None else p / 10

    def inject_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """Inject SI1000-style noise with per-operation error rates."""
        return self._inject_operation_noise(
            circuit,
            self._p_reset,
            self._p_meas,
            self._p_1q,
            self._p_2q,
            self._p_idle,
        )


class SIDNoiseModel(NoiseModel):
    """Single-qubit Independent Depolarizing noise model (SID).

    Injects uniform single-qubit depolarizing noise (``DEPOLARIZE1(p)``)
    before every operation on each involved qubit. This is a simple,
    gate-independent noise model where every qubit experiences the same
    depolarizing channel at every time step regardless of the operation
    being performed.

    * Before each single-qubit gate: ``DEPOLARIZE1(p)`` on the qubit
    * Before each two-qubit gate: ``DEPOLARIZE1(p)`` on both qubits
    * Before each measurement: ``DEPOLARIZE1(p)`` on the qubit
    * Before each reset: ``DEPOLARIZE1(p)`` on the qubit

    Args:
        p: Single-qubit depolarizing error rate.
    """

    # All gate types that carry qubit operands
    _NOISY_OPS = (
        _1Q_GATES
        | _2Q_GATES
        | {
            "R",
            "RX",
            "RY",
            "M",
            "MX",
            "MY",
            "MR",
            "MRX",
            "MRY",
        }
    )

    def __init__(self, p: float) -> None:
        super().__init__(error_rate=p)

    def inject_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """Inject uniform DEPOLARIZE1 before all operations."""
        p = self._error_rate
        noisy = stim.Circuit()

        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                inner_noisy = self.inject_noise(instruction.body_copy())
                noisy.append(
                    stim.CircuitRepeatBlock(instruction.repeat_count, inner_noisy)
                )
                continue

            name = instruction.name
            targets = instruction.targets_copy()
            gate_args = instruction.gate_args_copy()

            if name in self._NOISY_OPS and p > 0:
                # Coalesced CX pairs can share qubits. Insert noise before
                # each operation, not before the whole coalesced instruction.
                stride = 2 if name in {"CX", "CZ"} else 1
                for offset in range(0, len(targets), stride):
                    group = targets[offset : offset + stride]
                    noisy.append("DEPOLARIZE1", group, [p])
                    noisy.append(name, group, gate_args)
            else:
                noisy.append(name, targets, gate_args)

        return noisy
