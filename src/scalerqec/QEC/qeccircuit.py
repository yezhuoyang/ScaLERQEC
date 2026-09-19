"""Quantum error correction circuit construction and compilation.

This module provides the core :class:`StabCode` class for representing
stabilizer-based quantum error-correcting codes and compiling them into
STIM circuits via a two-stage pipeline:

1. **IR construction** -- stabilizer generators and logical operators are
   translated into an intermediate representation (IR) consisting of
   :class:`StabPropInstruction`, :class:`DetectorInstruction`, and
   :class:`ObservableInstruction` objects.
2. **Circuit compilation** -- the IR is lowered into a
   :class:`~scalerqec.Clifford.clifford.CliffordCircuit` and then into a
   ``stim.Circuit``.

The module also defines the :class:`SCHEME` and :class:`IRType` enumerations
and the full IR instruction hierarchy used internally by the compiler.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import stim
from numpy.typing import NDArray

from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.Clifford.noiselabel import NoiseLabelMap
from scalerqec.QEC.noisemodel import NoiseModel
from scalerqec.util import commute


class SCHEME(Enum):
    """Syndrome-extraction schemes for stabilizer codes.

    Each member selects a different strategy for measuring stabilizer
    generators and propagating parity information.

    Attributes:
        STANDARD: Bare ancilla syndrome extraction (one ancilla per
            stabilizer generator, no verification).
        SHOR: Cat-state extraction with ancilla decoding (no postselection).
        KNILL: Knill-style extraction via teleportation.
        FLAG: Extraction using one flag per measured stabilizer.
    """

    STANDARD = 0
    SHOR = 1
    KNILL = 2
    FLAG = 3


class IRType(Enum):
    """Types of intermediate-representation (IR) instructions.

    The IR captures the logical structure of a syndrome-extraction circuit
    before it is lowered to physical gates.

    Attributes:
        PROP: Stabilizer propagation (ancilla preparation, entanglement,
            and measurement).
        DETECTOR: Detector declaration comparing measurement outcomes
            across rounds.
        OBSERVABLE: Logical observable declaration.
        IF_THEN: Conditional branch (not yet implemented).
        WHILE: While-loop control flow (not yet implemented).
        REPEAT_UNTIL: Repeat-until control flow (not yet implemented).
        REPEAT: Fixed-count repetition (not yet implemented).
        DATA_MEASURE: Measure all data qubits in a given basis.
    """

    PROP = 0
    DETECTOR = 1
    OBSERVABLE = 2
    IF_THEN = 3
    WHILE = 4
    REPEAT_UNTIL = 5
    REPEAT = 6
    DATA_MEASURE = 7
    TELEPORT = 8


class IRInstruction:
    """Base class for all intermediate-representation (IR) instructions.

    Every IR instruction carries a type tag (:class:`IRType`) that
    identifies the kind of operation it represents.  Concrete subclasses
    add the operand-specific fields.

    Args:
        instr_type: The IR instruction type tag.
    """

    def __init__(self, instr_type: IRType) -> None:
        self._instr_type = instr_type


class StabPropInstruction(IRInstruction):
    """IR instruction for stabilizer propagation (syndrome measurement).

    Represents the preparation of an ancilla qubit, entanglement with data
    qubits according to a Pauli stabilizer string, and subsequent
    measurement.  The result is stored in a symbolic destination register.

    Args:
        round: The syndrome-extraction round index (0-based).
        stabindex: Index of the stabilizer generator within the code's
            stabilizer list.
        dest: Symbolic name for the measurement result (e.g., ``"c0"``).
        stab: Pauli string describing the stabilizer generator, composed
            of characters ``I``, ``X``, ``Y``, ``Z``.
        is_observable: If ``True``, this instruction propagates a logical
            observable rather than a stabilizer generator.
        observable_index: Index of the logical observable when
            *is_observable* is ``True``; ``-1`` otherwise.
    """

    def __init__(
        self,
        round: int,
        stabindex: int,
        dest: str,
        stab: str,
        is_observable: bool = False,
        observable_index: int = -1,
    ) -> None:
        super().__init__(IRType.PROP)
        self._round = round
        self._stabindex = stabindex
        self._dest = dest
        self._stab = stab
        self._is_observable = is_observable
        self._observable_index = observable_index

    @property
    def round(self) -> int:
        """
        Get the round number of the stabilizer propagation.

        Returns:
            int: The round number.
        """
        return self._round

    def is_observable(self) -> bool:
        """
        Check if the stabilizer propagation is for an observable.

        Returns:
            bool: True if it is an observable, False otherwise.
        """
        return self._is_observable

    def get_observable_index(self) -> int:
        """
        Get the index of the observable if applicable.

        Returns:
            int: The observable index.
        """
        return self._observable_index

    def get_stabindex(self) -> int:
        """
        Get the stabilizer index of the stabilizer propagation.

        Returns:
            int: The stabilizer index.
        """
        return self._stabindex

    def __str__(self) -> str:
        if self._is_observable:
            return f"{self._dest} = Prop {self._stab}"
        else:
            return f"{self._dest} = Prop[r={self._round}, s={self._stabindex}] {self._stab}"

    @property
    def dest(self) -> str:
        """
        Get the destination qubit/observable/detector from the instruction.

        Returns:
            str: The destination string.
        """
        return self._dest

    @property
    def stab(self) -> str:
        """
        Get the stabilizer from the instruction.

        Returns:
            str: The stabilizer string.
        """
        return self._stab


class ParityInstruction(IRInstruction):
    """Base class for parity-check IR instructions.

    A parity instruction computes the XOR (parity) of a set of
    measurement results identified by their symbolic names.  It is the
    common base for :class:`DetectorInstruction` and
    :class:`ObservableInstruction`.

    Args:
        ir_type: The IR instruction type (``DETECTOR`` or ``OBSERVABLE``).
        dest: Symbolic name for the parity result (e.g., ``"d0"`` or
            ``"o0"``).
        args: List of symbolic measurement-result names whose parity is
            computed.
    """

    def __init__(self, ir_type: IRType, dest: str, args: list[str]) -> None:
        super().__init__(ir_type)
        self._dest = dest
        self._args = args

    def __str__(self) -> str:
        return f"{self._dest} = Parity {' '.join(self._args)}"

    @property
    def dest(self) -> str:
        return self._dest

    @property
    def args(self) -> list[str]:
        return self._args


class DetectorInstruction(ParityInstruction):
    """IR instruction declaring a detector.

    A detector compares stabilizer measurement results across consecutive
    syndrome-extraction rounds.  In the absence of errors the parity of
    the referenced measurements is deterministic (always 0).

    Args:
        dest: Symbolic detector name (e.g., ``"d0"``).
        args: Symbolic names of the measurement results whose parity
            defines this detector.
    """

    def __init__(self, dest: str, args: list[str]) -> None:
        super().__init__(IRType.DETECTOR, dest, args)


class ObservableInstruction(ParityInstruction):
    """IR instruction declaring a logical observable.

    A logical observable ties measurement results to the logical-level
    outcome that the decoder must predict.  Typically it references the
    measurement of a logical Z (or X) operator.

    Args:
        dest: Symbolic observable name (e.g., ``"o0"``).
        args: Symbolic names of the measurement results whose parity
            defines this observable.
    """

    def __init__(self, dest: str, args: list[str]) -> None:
        super().__init__(IRType.OBSERVABLE, dest, args)


class IF_THENInstruction(IRInstruction):
    """IR instruction for conditional (if-then) control flow.

    .. note:: Not yet implemented; reserved for future use.

    Args:
        condition: Boolean expression string evaluated at runtime.
        then_branch: List of IR instructions to execute when
            *condition* is true.
    """

    def __init__(self, condition: str, then_branch: list[IRInstruction]) -> None:
        super().__init__(IRType.IF_THEN)
        self._condition = condition
        self._then_branch = then_branch


class WHILEInstruction(IRInstruction):
    """IR instruction for while-loop control flow.

    .. note:: Not yet implemented; reserved for future use.

    Args:
        condition: Boolean expression string re-evaluated each iteration.
        body: List of IR instructions forming the loop body.
    """

    def __init__(self, condition: str, body: list[IRInstruction]) -> None:
        super().__init__(IRType.WHILE)
        self._condition = condition
        self._body = body


class REPEAT_UNTILInstruction(IRInstruction):
    """IR instruction for repeat-until control flow.

    .. note:: Not yet implemented; reserved for future use.

    Args:
        body: List of IR instructions executed at least once per
            iteration.
        until_condition: Boolean expression string; the loop exits when
            this evaluates to true.
    """

    def __init__(self, body: list[IRInstruction], until_condition: str) -> None:
        super().__init__(IRType.REPEAT_UNTIL)
        self._body = body
        self._until_condition = until_condition


class REPEATInstruction(IRInstruction):
    """IR instruction for fixed-count repetition.

    .. note:: Not yet implemented; reserved for future use.

    Args:
        body: List of IR instructions forming the loop body.
        times: Number of times the body is repeated.
    """

    def __init__(self, body: list[IRInstruction], times: int) -> None:
        super().__init__(IRType.REPEAT)
        self._body = body
        self._times = times


class DataMeasureInstruction(IRInstruction):
    """IR instruction for measuring all data qubits.

    After the final syndrome extraction round, all data qubits are measured
    in the specified basis. The measurement results are used to construct
    final-round detectors and the logical observable.

    Args:
        basis: Measurement basis (``"Z"``, ``"X"``, or ``"Y"``).
            Defaults to ``"Z"`` for memory-Z experiments.
    """

    def __init__(self, basis: str = "Z") -> None:
        super().__init__(IRType.DATA_MEASURE)
        self._basis = basis

    @property
    def basis(self) -> str:
        return self._basis

    def __str__(self) -> str:
        return f"DataMeasure {self._basis}"


class KnillTeleportInstruction(IRInstruction):
    """A block teleportation round, including encoded Bell preparation."""

    def __init__(self, round: int):
        super().__init__(IRType.TELEPORT)
        self.round = round

    def __str__(self):
        return f"Teleport[r={self.round}] encoded Bell; track logical Pauli frame"


class StabCode:
    """Stabilizer quantum error-correcting code.

    ``StabCode`` is the primary entry point for defining a stabilizer code,
    attaching a noise model, and compiling the resulting syndrome-extraction
    circuit into a ``stim.Circuit``.

    The compilation is a two-stage pipeline:

    1. **IR construction** -- call :meth:`construct_IR_standard_scheme` (or
       the scheme-specific variant) to build an intermediate representation
       from the stabilizer generators and logical operators.
    2. **Circuit compilation** -- call
       :meth:`compile_stim_circuit_from_IR_standard` to lower the IR into a
       :class:`~scalerqec.Clifford.clifford.CliffordCircuit` and a
       ``stim.Circuit``.

    Both stages are executed automatically by :meth:`construct_circuit`.

    Example::

        code = StabCode(n=5, k=1, d=3)
        code.add_stab("XZZXI")
        code.add_stab("IXZZX")
        code.add_stab("XIXZZ")
        code.add_stab("ZXIXZ")
        code.set_logical_Z(0, "ZZZZZ")
        code.noisemodel = NoiseModel(error_rate=0.001)
        code.scheme = "Standard"
        code.rounds = 3
        code.construct_circuit()
        print(code.stimcirc)

    Args:
        n: Number of physical (data) qubits.
        k: Number of logical qubits encoded by the code.
        d: Code distance (minimum weight of a non-trivial logical
            operator).
    """

    def __init__(self, n: int, k: int, d: int) -> None:
        self._n: int = n
        self._k: int = k
        self._d: int = d
        self._stabs: list[str] = []
        self._scheme: SCHEME = SCHEME.STANDARD
        self._circuit: CliffordCircuit | None = None
        self._stimcirc: stim.Circuit | None = None
        self._IRList: list[IRInstruction] = []
        self._rounds: int = 3 * d
        # Define the k different logical Z operators
        self._logicalZ: dict[int, str] = {}
        self._paritymatrix: NDArray[np.int_] | None = None
        self._noisemodel: NoiseModel | None = None
        self._IR_compiled = False
        self._circuit_compiled = False
        self._ideal_stimcirc: stim.Circuit | None = None

    def _invalidate(self) -> None:
        """Discard derived state after a code, schedule or noise change."""
        self._IRList = []
        self._IR_compiled = False
        self._circuit_compiled = False
        self._ideal_stimcirc = None
        self._stimcirc = None
        self._circuit = None

    def is_IR_compiled(self) -> bool:
        """
        Check if the intermediate representation (IR) has been compiled.

        Returns:
            bool: True if the IR is compiled, False otherwise.
        """
        return self._IR_compiled

    def is_circuit_compiled(self) -> bool:
        """
        Check if the quantum error-correcting circuit has been compiled.

        Returns:
            bool: True if the circuit is compiled, False otherwise.
        """
        return self._circuit_compiled

    @property
    def n(self) -> int:
        """
        Get the number of physical qubits in the QECC.

        Returns:
            int: The number of physical qubits.
        """
        return self._n

    @property
    def k(self) -> int:
        """
        Get the number of logical qubits in the QECC.

        Returns:
            int: The number of logical qubits.
        """
        return self._k

    @property
    def d(self) -> int:
        """
        Get the distance of the QECC.

        Returns:
            int: The distance.
        """
        return self._d

    @property
    def noisemodel(self) -> NoiseModel | None:
        """
        Get the noise model associated with the QECC.

        Returns:
            NoiseModel: The noise model.
        """
        return self._noisemodel

    @noisemodel.setter
    def noisemodel(self, noisemodel: NoiseModel) -> None:
        """
        Set the noise model associated with the QECC.

        Args:
            noisemodel (NoiseModel): The noise model to set.
        """
        self._noisemodel = noisemodel
        self._invalidate()

    def init_by_parity_check_matrix(self, paritymatrix: NDArray[np.int_]) -> None:
        """Initialize the code from a binary parity-check matrix.

        Sets the number of physical qubits ``n`` and logical qubits ``k``
        from the matrix dimensions and replaces the current stabilizer
        list.  The stabilizer generators themselves are not yet extracted
        from the matrix (to be implemented).

        Args:
            paritymatrix: Binary parity-check matrix of shape
                ``(n - k, n)`` where rows correspond to stabilizer
                generators and columns to physical qubits.
        """
        self._paritymatrix = paritymatrix
        self._n = paritymatrix.shape[1]
        self._k = self._n - paritymatrix.shape[0]
        self._stabs = []

    def construct_parity_check_matrix(self) -> None:
        """Construct the binary symplectic parity-check matrix from stabilizers.

        Converts the stored Pauli-string stabilizer generators into a
        binary matrix in the standard ``[X | Z]`` symplectic format and
        stores it internally.

        .. note:: Not yet implemented.
        """

    def get_parity_check_matrix(self) -> NDArray[np.int_] | None:
        """Return the binary symplectic parity-check matrix, if available.

        Returns:
            The parity-check matrix as a NumPy integer array of shape
            ``(n - k, n)``, or ``None`` if it has not been set or
            constructed.
        """
        return self._paritymatrix

    @property
    def circuit(self) -> CliffordCircuit | None:
        """
        Get the Clifford circuit for the quantum error-correcting code.

        Returns:
            The Clifford circuit, or None if not yet compiled.
        """
        return self._circuit

    @property
    def stimcirc(self):
        """
        Get the STIM circuit for the quantum error-correcting code.

        Returns:
            The STIM circuit, or None if not yet compiled.
        """
        return self._stimcirc

    def set_logical_Z(self, index: int, logicalZ: str) -> None:
        """Set the logical Z operator for a logical qubit.

        The logical Z operator must be specified for every logical qubit
        (indices ``0`` through ``k - 1``) before calling
        :meth:`construct_circuit`.

        Args:
            index: Zero-based index of the logical qubit.
            logicalZ: Pauli string of length ``n`` (characters from
                ``I``, ``X``, ``Y``, ``Z``) representing the logical Z
                operator.

        Raises:
            AssertionError: If *logicalZ* has incorrect length or contains
                invalid characters.
        """
        assert len(logicalZ) == self._n, "Logical Z length must match number of qubits."
        assert all(c in "IXYZ" for c in logicalZ), (
            "Logical Z must only contain I, X, Y, and Z."
        )

        self._logicalZ[index] = logicalZ
        self._invalidate()

    @property
    def rounds(self) -> int:
        """
        Get the number of error correction rounds.

        Returns:
            int: The number of rounds.
        """
        return self._rounds

    @rounds.setter
    def rounds(self, rounds: int) -> None:
        """
        Set the number of error correction rounds.

        Args:
            rounds (int): The number of rounds to set.
        """
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be a positive integer.")
        self._rounds = rounds
        self._invalidate()

    def add_stab(self, stab: str) -> None:
        """Add a stabilizer generator to the code.

        Args:
            stab: Pauli string of length ``n`` (characters from ``I``,
                ``X``, ``Y``, ``Z``) representing the stabilizer
                generator.

        Raises:
            AssertionError: If *stab* has incorrect length or contains
                invalid characters.
        """
        assert len(stab) == self._n, "Stabilizer length must match number of qubits."
        assert all(c in "IXYZ" for c in stab), (
            "Stabilizer must only contain I, X, Y, Z."
        )

        self._stabs.append(stab)
        self._invalidate()

    @property
    def scheme(self) -> SCHEME:
        """
        Get the error correction scheme for the code.

        Returns:
            SCHEME: The error correction scheme.
        """
        return self._scheme

    @property
    def stabilizers(self) -> list[str]:
        """
        Get the list of stabilizer generators for this code.

        Returns:
            list[str]: The stabilizer generators as Pauli strings.
        """
        return self._stabs

    @scheme.setter  # type: ignore[no-redef, attr-defined]
    def scheme(self, scheme: str) -> None:
        """
        Set the error correction scheme for the code.

        Args:
            scheme (SCHEME): The error correction scheme to use.
        """
        if scheme == "Standard":
            self._scheme = SCHEME.STANDARD
        elif scheme == "Shor":
            self._scheme = SCHEME.SHOR
        elif scheme == "Knill":
            self._scheme = SCHEME.KNILL
        elif scheme == "Flag":
            self._scheme = SCHEME.FLAG
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        self._invalidate()

    def construct_circuit(self) -> None:
        """Run the full compilation pipeline for the selected scheme.

        Performs two stages:

        1. **IR construction** -- translates stabilizer generators and
           logical operators into an intermediate representation.
        2. **Circuit compilation** -- lowers the IR into a
           :class:`~scalerqec.Clifford.clifford.CliffordCircuit` and a
           ``stim.Circuit``.

        If a :attr:`noisemodel` has been attached, noise is injected into
        the compiled circuit after stage 2, and the CliffordCircuit is
        rebuilt from the noisy circuit so that the QEPG gate list reflects
        the actual noise locations.

        The IR uses a simple text-like notation internally::

            c0 = Prop XYZIX
            c1 = Prop IXYZI
            d0 = Parity c0 c1
            o0 = Parity c0

        All four schemes compile fixed circuits. Flag and decoded-cat records
        are detectors. Knill prepares an encoded Bell resource and tracks its
        logical Pauli frame in observable parities. These circuits do not
        include adaptive ancilla rejection or a fault-tolerance certificate.

        Raises:
            ValueError: If the stabilizer/logical definitions are invalid or
                the logical operators are incompatible with Z-memory readout.
        """
        self._validate_memory_code()
        if self._scheme == SCHEME.STANDARD:
            self.construct_IR_standard_scheme()
            self.compile_stim_circuit_from_IR_standard()
        elif self._scheme in {SCHEME.FLAG, SCHEME.SHOR}:
            self.construct_IR_standard_scheme()
            if not self._circuit_compiled:
                from .extraction import compile_memory_gadgets

                self._store_compiled_circuit(compile_memory_gadgets(self))
        else:
            self.construct_IR_knill_scheme()
            self.compile_stim_circuit_from_knill()
        if self._noisemodel is not None:
            self._stimcirc = self._noisemodel.apply(self._ideal_stimcirc)
            self._build_legacy_circuit(noisy=True)

    def _validate_memory_code(self):
        from .extraction import code_tableau

        logicals = []
        for j in range(self.k):
            if j not in self._logicalZ:
                raise ValueError(
                    f"Logical Z operator for logical qubit {j} not defined."
                )
            z = self._logicalZ[j]
            if any(p not in "IZ" for p in z):
                raise ValueError(
                    "The memory compiler uses Z preparation/readout and requires "
                    "Z-type logical Z operators. Use an explicit Stim circuit "
                    "for other preparation/readout bases."
                )
            logicals.append(z)
        code_tableau(self._stabs, logicals, self.n)

    def _store_compiled_circuit(self, circuit):
        self._ideal_stimcirc = circuit.copy()
        self._stimcirc = circuit
        self._circuit_compiled = True
        self._build_legacy_circuit(noisy=False)

    def _build_legacy_circuit(self, *, noisy):
        self.legacy_backend_error = None
        if self._stimcirc.num_observables > 1:
            self._circuit = None
            self.legacy_backend_error = (
                "Multiple observables require LinearNoiseModel, not legacy QEPG."
            )
            return
        self._circuit = CliffordCircuit(self._stimcirc.num_qubits)
        try:
            if noisy:
                self._circuit.compile_from_noisy_stim_circuit_str(str(self._stimcirc))
            else:
                from scalerqec.Clifford.stimparser import rewrite_stim_code

                self._circuit.compile_from_stim_circuit_str(
                    rewrite_stim_code(str(self._stimcirc))
                )
        except NotImplementedError as exc:
            # The general Stim estimator supports more syntax/observables than
            # the legacy QEPG backend. Never expose a partially parsed graph.
            self._circuit = None
            self.legacy_backend_error = str(exc)

    def label_noise(self) -> NoiseLabelMap:
        """Auto-label all noise sources in the compiled circuit.

        Must be called after :meth:`construct_circuit`.  Returns a
        :class:`~scalerqec.Clifford.noiselabel.NoiseLabelMap` with each
        noise source labeled by category (hook_error, measurement_error,
        etc.) and enriched with round/stabilizer context.

        Returns:
            A populated NoiseLabelMap.

        Raises:
            AssertionError: If the circuit has not been compiled.
        """
        from .autolabel import auto_label_from_stabcode

        return auto_label_from_stabcode(self)

    def construct_IR_shor_scheme(self) -> None:
        """Build memory IR to be lowered with decoded-cat extraction."""
        self.scheme = "Shor"
        self.construct_IR_standard_scheme()

    def compile_stim_circuit_from_shor_standard(self) -> str | None:
        """Compile decoded-cat memory IR (historical method name)."""
        from .extraction import compile_memory_gadgets

        if not self._IR_compiled:
            raise RuntimeError("IR not compiled yet.")
        self._store_compiled_circuit(compile_memory_gadgets(self))
        return str(self._stimcirc)

    compile_stim_circuit_from_shor = compile_stim_circuit_from_shor_standard

    def construct_IR_flag_scheme(self) -> None:
        """Build memory IR to be lowered with flagged parity gadgets."""
        self.scheme = "Flag"
        self.construct_IR_standard_scheme()

    def compile_stim_circuit_from_flag(self) -> str:
        """Lower flagged memory IR to Stim."""
        return self.compile_stim_circuit_from_shor_standard()

    def construct_IR_knill_scheme(self) -> None:
        """Build block teleportation IR, including logical frame tracking."""
        if self.scheme != SCHEME.KNILL:
            self.scheme = "Knill"
        # A Knill round is block teleportation, not a list of individual Prop
        # gadgets. Keep this distinction visible in the exported IR.
        if not self._IR_compiled:
            self._IRList = [KnillTeleportInstruction(r) for r in range(self.rounds)]
            self._IR_compiled = True

    def compile_stim_circuit_from_knill(self) -> str | None:
        """Compile an encoded-Bell teleportation memory circuit."""
        from .extraction import compile_knill_memory

        if not self._IR_compiled:
            raise RuntimeError("IR not compiled yet.")
        if not self._circuit_compiled:
            self._validate_memory_code()
            self._store_compiled_circuit(compile_knill_memory(self))
        return str(self._stimcirc)

    def _is_z_type_stabilizer(self, stab: str) -> bool:
        """Check if a stabilizer is pure Z-type (only I and Z entries)."""
        return all(c in "IZ" for c in stab)

    def _is_x_type_stabilizer(self, stab: str) -> bool:
        """Check if a stabilizer is pure X-type (only I and X entries)."""
        return all(c in "IX" for c in stab)

    def construct_IR_standard_scheme(self) -> None:
        """Build the IR for standard (bare-ancilla) syndrome extraction.

        Produces a complete IR with:

        1. **Round-0 detectors**: Z-type stabilizers are deterministic with
           all-zero initialization, so their outcomes become detectors.
        2. **Inter-round detectors**: From round 1 onward, detectors compare
           the same stabilizer across consecutive rounds.
        3. **Final data qubit measurements**: All data qubits are measured
           in the Z basis (memory-Z experiment).
        4. **Final-round detectors**: For each Z-type stabilizer, a detector
           XORs the last syndrome measurement with the data qubit
           measurements at the stabilizer's Z support.
        5. **Observable**: References data qubit measurements at positions
           where the Z-type logical operator has Z support.

        This method is idempotent until a code or schedule property changes.

        Raises:
            ValueError: If a logical Z operator has not been set for
                every logical qubit index ``0 .. k-1``.
        """
        self._validate_memory_code()
        if self._IR_compiled:
            return

        for logical_idx in range(self._k):
            if logical_idx not in self._logicalZ:
                raise ValueError(
                    f"Logical Z operator for logical qubit {logical_idx} not defined. "
                    f"Use set_logical_Z({logical_idx}, '<pauli_string>') before constructing the circuit."
                )

        current_measurement_idx = 0
        current_detector_idx = 0
        prev_stab_meas_addr: dict[int, str] = {}  # stabindex -> dest

        for r in range(self._rounds):
            for stabidx, stab in enumerate(self._stabs):
                dest = f"c{current_measurement_idx}"
                instr = StabPropInstruction(r, stabidx, dest, stab)
                self._IRList.append(instr)
                current_measurement_idx += 1

                if r == 0:
                    # Round-0 detector: single measurement, deterministic
                    # Only Z-type stabilizers are deterministic with |0⟩ init
                    # (X-type stabilizers have random outcomes in Z-basis init)
                    if self._is_z_type_stabilizer(stab):
                        detector_dest = f"d{current_detector_idx}"
                        detector_instr = DetectorInstruction(detector_dest, [dest])
                        self._IRList.append(detector_instr)
                        current_detector_idx += 1
                else:
                    # Inter-round detector: compare with previous round
                    prev_dest = prev_stab_meas_addr[stabidx]
                    detector_dest = f"d{current_detector_idx}"
                    detector_instr = DetectorInstruction(
                        detector_dest, [prev_dest, dest]
                    )
                    self._IRList.append(detector_instr)
                    current_detector_idx += 1

                prev_stab_meas_addr[stabidx] = dest

        # Final: measure all data qubits in Z basis
        data_meas_instr = DataMeasureInstruction(basis="Z")
        self._IRList.append(data_meas_instr)

        # Data qubit measurement destinations
        data_meas_dests = []
        for q in range(self._n):
            dest = f"m{q}"  # data qubit measurement
            data_meas_dests.append(dest)

        # Final-round detectors: for Z-type stabilizers, XOR last syndrome
        # measurement with data qubit measurements at the stabilizer's support
        for stabidx, stab in enumerate(self._stabs):
            if self._is_z_type_stabilizer(stab):
                last_syndrome_dest = prev_stab_meas_addr[stabidx]
                # Data qubit indices where this stabilizer has Z
                data_dests = [
                    data_meas_dests[q] for q, pauli in enumerate(stab) if pauli == "Z"
                ]
                detector_dest = f"d{current_detector_idx}"
                detector_instr = DetectorInstruction(
                    detector_dest, [last_syndrome_dest] + data_dests
                )
                self._IRList.append(detector_instr)
                current_detector_idx += 1

        # Logical observables from data qubit measurements
        for logical_idx in range(self._k):
            logicalZ = self._logicalZ[logical_idx]
            # Observable references data qubit measurements where logicalZ has Z or Y
            obs_dests = [
                data_meas_dests[q] for q, pauli in enumerate(logicalZ) if pauli in "ZY"
            ]
            observable_dest = f"o{logical_idx}"
            observable_instr = ObservableInstruction(observable_dest, obs_dests)
            self._IRList.append(observable_instr)

        self._IR_compiled = True

    def show_IR(self) -> None:
        """Print the intermediate representation to stdout.

        Each IR instruction is printed on its own line using its
        ``__str__`` representation.  Useful for debugging the compilation
        pipeline.
        """
        for irinst in self._IRList:
            print(irinst)

    def compile_stim_circuit_from_IR_standard(self) -> str | None:
        """Lower the standard-scheme IR into a STIM circuit.

        Builds a ``stim.Circuit`` directly from the IR, batching
        operations by round with proper TICK instructions:

        1. Reset all ancillas for the round → TICK
        2. Entangling gates for all stabilizers → TICK
        3. Measure all ancillas → emit detectors → TICK

        This round structure enables noise models to inject per-time-step
        noise (idle depolarization, etc.) between TICK boundaries.

        Qubit layout convention:

        * Data qubits: indices ``0 .. n-1``.
        * Syndrome ancillas: indices ``n .. n + num_stabilizers - 1``.

        Returns:
            The compiled STIM circuit as a string if the circuit was
            already compiled, or ``None`` on first compilation (the
            result is stored in :attr:`stimcirc`).

        Raises:
            RuntimeError: If the IR has not been compiled yet.
        """
        if not self._IR_compiled:
            raise RuntimeError("IR not compiled yet.")
        if self._circuit_compiled:
            return str(self._stimcirc)

        # ------------------------------------------------------------------
        # Phase 1: Group IR instructions by round
        # ------------------------------------------------------------------
        rounds_data: list[
            tuple[list[StabPropInstruction], list[DetectorInstruction]]
        ] = []
        current_round_stabs: list[StabPropInstruction] = []
        current_round_detectors: list[DetectorInstruction] = []
        data_measure: DataMeasureInstruction | None = None
        final_detectors: list[DetectorInstruction] = []
        observables: list[ObservableInstruction] = []

        for irinst in self._IRList:
            if isinstance(irinst, StabPropInstruction):
                # New round? flush the previous one
                if current_round_stabs and irinst.round != current_round_stabs[0].round:
                    rounds_data.append((current_round_stabs, current_round_detectors))
                    current_round_stabs = []
                    current_round_detectors = []
                current_round_stabs.append(irinst)
            elif isinstance(irinst, DetectorInstruction):
                if data_measure is None:
                    current_round_detectors.append(irinst)
                else:
                    final_detectors.append(irinst)
            elif isinstance(irinst, DataMeasureInstruction):
                # Flush the last syndrome round
                if current_round_stabs:
                    rounds_data.append((current_round_stabs, current_round_detectors))
                    current_round_stabs = []
                    current_round_detectors = []
                data_measure = irinst
            elif isinstance(irinst, ObservableInstruction):
                observables.append(irinst)

        # Flush any remaining round
        if current_round_stabs:
            rounds_data.append((current_round_stabs, current_round_detectors))

        # ------------------------------------------------------------------
        # Phase 2: Build stim.Circuit with batched operations and TICKs
        # ------------------------------------------------------------------
        circuit = stim.Circuit()
        circuit.append("R", range(self.n))
        circuit.append("TICK")

        # Emit qubit coordinates if available
        if hasattr(self, "_qubit_coords"):
            for q, (x, y) in self._qubit_coords.items():
                circuit.append("QUBIT_COORDS", [q], [x, y])

        total_meas = 0
        dest_to_meas_idx: dict[str, int] = {}

        for round_stabs, round_detectors in rounds_data:
            # --- Reset all ancillas for this round ---
            ancillas = [self._n + sp.get_stabindex() for sp in round_stabs]
            circuit.append("R", ancillas)
            circuit.append("TICK")

            # --- Entangling gates for all stabilizers ---
            for sp in round_stabs:
                stab = sp.stab
                ancilla = self._n + sp.get_stabindex()

                is_pure_x = self._is_x_type_stabilizer(stab)
                is_pure_z = self._is_z_type_stabilizer(stab)

                if is_pure_x:
                    circuit.append("H", [ancilla])
                    for qubit_index, pauli in enumerate(stab):
                        if pauli == "X":
                            circuit.append("CX", [ancilla, qubit_index])
                    circuit.append("H", [ancilla])
                elif is_pure_z:
                    for qubit_index, pauli in enumerate(stab):
                        if pauli == "Z":
                            circuit.append("CX", [qubit_index, ancilla])
                else:
                    for qubit_index, pauli in enumerate(stab):
                        if pauli == "X":
                            circuit.append("H", [qubit_index])
                            circuit.append("CX", [qubit_index, ancilla])
                            circuit.append("H", [qubit_index])
                        elif pauli == "Z":
                            circuit.append("CX", [qubit_index, ancilla])
                        elif pauli == "Y":
                            circuit.append("S_DAG", [qubit_index])
                            circuit.append("H", [qubit_index])
                            circuit.append("CX", [qubit_index, ancilla])
                            circuit.append("H", [qubit_index])
                            circuit.append("S", [qubit_index])

            circuit.append("TICK")

            # --- Measure all ancillas ---
            for sp in round_stabs:
                ancilla = self._n + sp.get_stabindex()
                circuit.append("M", [ancilla])
                dest_to_meas_idx[sp.dest] = total_meas
                total_meas += 1

            # --- Emit detectors for this round ---
            for det in round_detectors:
                rec_targets = []
                for arg in det.args:
                    abs_idx = dest_to_meas_idx[arg]
                    rec_targets.append(stim.target_rec(abs_idx - total_meas))
                circuit.append("DETECTOR", rec_targets)

            circuit.append("TICK")

        # ------------------------------------------------------------------
        # Final: data qubit measurements
        # ------------------------------------------------------------------
        if data_measure is not None:
            data_qubits = list(range(self._n))
            circuit.append("M", data_qubits)
            for q in range(self._n):
                dest_to_meas_idx[f"m{q}"] = total_meas
                total_meas += 1

        # Final-round detectors
        for det in final_detectors:
            rec_targets = []
            for arg in det.args:
                abs_idx = dest_to_meas_idx[arg]
                rec_targets.append(stim.target_rec(abs_idx - total_meas))
            circuit.append("DETECTOR", rec_targets)

        # Observables
        for obs in observables:
            rec_targets = []
            for arg in obs.args:
                abs_idx = dest_to_meas_idx[arg]
                rec_targets.append(stim.target_rec(abs_idx - total_meas))
            obs_idx = int(obs.dest[1:])
            circuit.append("OBSERVABLE_INCLUDE", rec_targets, obs_idx)

        self._store_compiled_circuit(circuit)


def test_commute():
    assert commute("IXYZ", "IYZX") == False
    assert commute("XZZI", "ZXXI") == False
    assert commute("IIII", "ZZZZ") == True
    assert commute("XIZY", "YZXI") == True


if __name__ == "__main__":
    noise_model = NoiseModel(error_rate=0.001)
    qeccirc = StabCode(n=5, k=1, d=3)
    qeccirc.noisemodel = noise_model
    # Specify your stabilizers
    # Stabilizer generators
    qeccirc.add_stab("XZZXI")
    qeccirc.add_stab("IXZZX")
    qeccirc.add_stab("XIXZZ")
    qeccirc.add_stab("ZXIXZ")
    qeccirc.set_logical_Z(0, "ZZZZZ")
    # Set stabilizer parity measurement scheme, round of repetition
    qeccirc.scheme = "Standard"  # type: ignore[misc, assignment]
    qeccirc.rounds = 3
    qeccirc.construct_circuit()
    stim_circuit = qeccirc.stimcirc
    print(stim_circuit)
