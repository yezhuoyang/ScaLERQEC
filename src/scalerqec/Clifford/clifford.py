"""Clifford circuit representation with gates, noise, measurements, and resets.

This module defines the data structures for representing a Clifford quantum
circuit and provides methods for constructing circuits either programmatically
or by parsing STIM-format circuit strings. Each circuit tracks its gate list,
noise sources, measurements, detector parity groups, and an observable, and
maintains a parallel ``stim.Circuit`` object for simulation.

Module-level constants:

- ``oneQGate_`` / ``oneQGateindices``: names and index mapping for single-qubit
  Clifford gates (H, P, X, Y, Z).
- ``twoQGate_`` / ``twoQGateindices``: names and index mapping for two-qubit
  gates (CNOT, CZ).
- ``pauliNoise_`` / ``pauliNoiseindices``: names and index mapping for the
  Pauli noise channel outcomes (I, X, Y, Z).
"""

from __future__ import annotations

import stim


oneQGate_ = ["H", "P", "X", "Y", "Z"]
oneQGateindices = {"H": 0, "P": 1, "X": 2, "Y": 3, "Z": 4}


twoQGate_ = ["CNOT", "CZ"]
twoQGateindices = {"CNOT": 0, "CZ": 1}

pauliNoise_ = ["I", "X", "Y", "Z"]
pauliNoiseindices = {"I": 0, "X": 1, "Y": 2, "Z": 3}


class SingleQGate:
    """A single-qubit Clifford gate operation.

    Represents one of {H, P, X, Y, Z} applied to a specific qubit.

    Args:
        gateindex: Index into ``oneQGate_`` identifying the gate type
            (0=H, 1=P, 2=X, 3=Y, 4=Z).
        qubitindex: The qubit this gate acts on.

    Attributes:
        name (str): Human-readable gate name (e.g. ``"H"``).
        qubitindex (int): Target qubit index.
    """

    def __init__(self, gateindex: int, qubitindex: int):
        self._name = oneQGate_[gateindex]
        self._qubitindex = qubitindex

    @property
    def qubitindex(self) -> int:
        return self._qubitindex

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name + "[" + str(self._qubitindex) + "]"


class TwoQGate:
    """A two-qubit gate operation (CNOT or CZ).

    Args:
        gateindex: Index into ``twoQGate_`` identifying the gate type
            (0=CNOT, 1=CZ).
        control: The control qubit index.
        target: The target qubit index.

    Attributes:
        name (str): Gate name (``"CNOT"`` or ``"CZ"``).
        control (int): Control qubit index.
        target (int): Target qubit index.
    """

    def __init__(self, gateindex: int, control: int, target: int):
        self._name = twoQGate_[gateindex]
        self._control = control
        self._target = target

    @property
    def control(self) -> int:
        return self._control

    @property
    def target(self) -> int:
        return self._target

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return self._name + "[" + str(self._control) + "," + str(self._target) + "]"


class pauliNoise:
    """A Pauli noise channel acting on a single qubit.

    Each noise instance represents a potential error location in the circuit.
    The actual error type (I, X, Y, or Z) is controlled by :attr:`noisetype`
    and can be set after construction.

    Args:
        noiseindex: A unique integer identifier for this noise source, used
            to index into the error propagation matrix.
        qubitindex: The qubit this noise channel acts on.

    Attributes:
        noisetype (int): The Pauli error type applied at this location.
            0=I (no error), 1=X, 2=Y, 3=Z. Defaults to 0.
    """

    def __init__(self, noiseindex: int, qubitindex: int):
        self._name = "n" + str(noiseindex)
        self._noiseindex: int = noiseindex
        self._qubitindex: int = qubitindex
        self._noisetype: int = 0

    @property
    def noisetype(self) -> int:
        return self._noisetype

    @noisetype.setter
    def noisetype(self, noisetype: int):
        self._noisetype = noisetype

    def __str__(self) -> str:
        return (
            self._name
            + "("
            + pauliNoise_[self._noisetype]
            + ")"
            + "["
            + str(self._qubitindex)
            + "]"
        )


class Measurement:
    """A Z-basis measurement operation on a single qubit.

    Args:
        measureindex: A sequential integer identifier for this measurement,
            used to reference it in detector parity groups and the observable.
        qubitindex: The qubit being measured.

    Attributes:
        qubitindex (int): The measured qubit index.
    """

    def __init__(self, measureindex: int, qubitindex: int):
        self._name = "M" + str(measureindex)
        self._qubitindex = qubitindex
        self._measureindex = measureindex

    @property
    def qubitindex(self) -> int:
        return self._qubitindex

    def __str__(self) -> str:
        return self._name + "[" + str(self._qubitindex) + "]"


class Reset:
    """A qubit reset operation that projects the qubit to the |0> state.

    A reset clears any accumulated error propagation on the qubit, since
    the qubit state is discarded and re-initialized.

    Args:
        qubitindex: The qubit being reset.

    Attributes:
        qubitindex (int): The reset qubit index.
    """

    def __init__(self, qubitindex: int):
        self._name = "R"
        self._qubitindex = qubitindex

    @property
    def qubitindex(self) -> int:
        return self._qubitindex

    def __str__(self) -> str:
        return self._name + "[" + str(self._qubitindex) + "]"


class CliffordCircuit:
    """A Clifford quantum circuit with noise, measurements, and syndrome extraction.

    This is the central circuit representation used throughout ScaLERQEC. It
    maintains an ordered list of gate operations (including noise channels) and
    tracks detector parity groups and observables needed for error analysis.

    A parallel ``stim.Circuit`` is built incrementally as gates are added, so
    the circuit can also be used directly with the STIM simulator.

    Args:
        qubit_num: The number of qubits in the circuit. This may be updated
            automatically when compiling from a STIM string.

    Attributes:
        gatelists (list): Ordered list of circuit operations
            (:class:`SingleQGate`, :class:`TwoQGate`, :class:`pauliNoise`,
            :class:`Measurement`, or :class:`Reset`).
        qubitnum (int): Number of qubits.
        totalnoise (int): Total number of noise sources inserted.
        totalMeas (int): Total number of measurements.
        error_rate (float): Physical error rate used for depolarizing noise
            channels injected by :meth:`add_depolarize`.
        parityMatchGroup (list[list[int]]): Detector definitions. Each inner
            list contains measurement indices whose parity should be constant
            in the absence of errors.
        observable (list[int]): Measurement indices whose parity defines the
            logical observable.
        stimcircuit (stim.Circuit): The equivalent STIM circuit, built in
            parallel as gates are added.
        stim_str (str | None): The raw STIM string if the circuit was compiled
            from one, otherwise ``None``.
    """

    def __init__(self, qubit_num: int):
        self._qubit_num: int = qubit_num
        self._totalnoise: int = 0
        self._totalMeas: int = 0
        self._totalgates: int = 0
        self._gatelists: list[
            SingleQGate | TwoQGate | pauliNoise | Measurement | Reset
        ] = []
        self._error_rate: float = 0
        self._index_to_noise: dict[int, pauliNoise] = {}
        self._index_to_measurement: dict[int, Measurement] = {}

        # self._index_to_measurement={}

        self._shownoise: bool = False
        self._syndromeErrorTable: dict[str, int] = {}
        # Store the repeat match group
        # For example, if we require M0=M1, M2=M3, then the match group is [[0,1],[2,3]]
        self._parityMatchGroup: list[list[int]] = []
        self._observable: list[int] = []

        self._measIdx_to_parityIdx: dict[int, list[int]] = {}

        self._stim_str: str | None = None
        self._stimcircuit: stim.Circuit = stim.Circuit()

        # self._error_channel

    @property
    def gatelists(self) -> list:
        return self._gatelists

    @property
    def qubitnum(self) -> int:
        return self._qubit_num

    @qubitnum.setter
    def qubitnum(self, qubit_num: int):
        self._qubit_num = qubit_num

    def get_measIdx_to_parityIdx(self, measIdx: int) -> list[int]:
        """Return the detector indices that include a given measurement.

        Args:
            measIdx: The measurement index to look up.

        Returns:
            A list of detector (parity group) indices that reference this
            measurement.
        """
        return self._measIdx_to_parityIdx[measIdx]

    @property
    def stim_str(self) -> str | None:
        return self._stim_str

    @stim_str.setter
    def stim_str(self, stim_str: str):
        self._stim_str = stim_str

    @property
    def error_rate(self) -> float:
        return self._error_rate

    @error_rate.setter
    def error_rate(self, error_rate: float):
        self._error_rate = error_rate

    @property
    def stimcircuit(self):
        return self._stimcircuit

    @stimcircuit.setter
    def stimcircuit(self, stim_circuit):
        if isinstance(stim_circuit, str):
            self._stim_str = stim_circuit
            self._stimcircuit = stim.Circuit(stim_circuit)
        elif isinstance(stim_circuit, stim.Circuit):
            self._stimcircuit = stim_circuit
            self._stim_str = str(stim_circuit)

    @property
    def observable(self):
        return self._observable

    @observable.setter
    def observable(self, observablemeasurements):
        self._observable = observablemeasurements

    @property
    def parityMatchGroup(self):
        return self._parityMatchGroup

    @parityMatchGroup.setter
    def parityMatchGroup(self, parityMatchGroup):
        self._parityMatchGroup = parityMatchGroup

    @property
    def qubit_num(self):
        return self._qubit_num

    @property
    def totalnoise(self):
        return self._totalnoise

    @property
    def totalMeas(self):
        return self._totalMeas

    def read_circuit_from_file(self, filename: str) -> None:
        """Read a circuit from a plain-text gate file.

        The file format uses one operation per line. The first line must
        declare the qubit count with ``NumberOfQubit <n>``. Subsequent
        lines specify gates: ``cnot``, ``CZ``, ``H``, ``P``, ``X``,
        ``Y``, ``Z``, ``M`` (measurement), or ``R`` (reset), each
        followed by qubit indices.

        Example file::

            NumberOfQubit 6
            cnot 1 2
            H 0
            M 0
            R 0

        Args:
            filename: Path to the circuit description file.

        Raises:
            ValueError: If an unrecognized gate type is encountered.
        """
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                if line.startswith("NumberOfQubit"):
                    # Extract the number of qubits
                    self._qubit_num = int(line.split()[1])
                else:
                    # Parse the gate operation
                    parts = line.split()
                    gate_type = parts[0]
                    qubits = list(map(int, parts[1:]))

                    if gate_type == "cnot":
                        self.add_cnot(qubits[0], qubits[1])
                    elif gate_type == "M":
                        self.add_measurement(qubits[0])
                    elif gate_type == "R":
                        self.add_reset(qubits[0])
                    elif gate_type == "H":
                        self.add_hadamard(qubits[0])
                    elif gate_type == "P":
                        self.add_phase(qubits[0])
                    elif gate_type == "CZ":
                        self.add_cz(qubits[0], qubits[1])
                    elif gate_type == "X":
                        self.add_paulix(qubits[0])
                    elif gate_type == "Y":
                        self.add_pauliy(qubits[0])
                    elif gate_type == "Z":
                        self.add_pauliz(qubits[0])
                    else:
                        raise ValueError(f"Unknown gate type: {gate_type}")

    def compile_from_stim_circuit_str(self, stim_str: str) -> None:
        """Parse a STIM circuit string and build a **noiseless** internal circuit.

        This method expects a STIM string that has already been preprocessed by
        :func:`~scalerqec.Clifford.stimparser.rewrite_stim_code` so that each
        line contains exactly one gate operation. It performs three passes:

        1. **Index measurements** -- assigns sequential measurement indices and
           records which line each measurement appears on.
        2. **Extract detectors and observable** -- resolves ``DETECTOR`` and
           ``OBSERVABLE_INCLUDE`` record references into measurement indices,
           building :attr:`parityMatchGroup` and :attr:`observable`.
        3. **Insert gates** -- adds each gate to the internal gate list,
           registering noise locations (for QEPG) before every gate and
           measurement (except resets).  The internal ``stim.Circuit`` is
           built **noiseless**; noise must be injected afterwards via a
           :class:`~scalerqec.QEC.noisemodel.NoiseModel`.

        After all gates are inserted, :meth:`compile_detector_and_observable`
        is called to append detector and observable instructions to the
        internal ``stim.Circuit``.

        Args:
            stim_str: A normalized STIM circuit string (one gate per line).

        Side Effects:
            Resets and repopulates ``_gatelists``, ``_totalnoise``,
            ``_totalMeas``, ``_parityMatchGroup``, ``_observable``,
            ``_qubit_num``, and the internal ``stim.Circuit``.
        """
        self._totalnoise = 0
        self._totalMeas = 0
        self._totalgates = 0

        lines = stim_str.splitlines()
        self._gatelists = []
        self._index_to_noise = {}
        self._index_to_measurement = {}
        self._measIdx_to_parityIdx = {}
        self._stimcircuit = stim.Circuit()
        self._stim_str = stim_str
        output_lines = []
        maxum_q_index = 0
        """
        First, read and compute the parity match group and the observable
        """
        parityMatchGroup = []
        observable = []

        measure_index_to_line = {}
        measure_line_to_measure_index = {}
        current_line_index = 0
        current_measure_index = 0
        _SKIP_KEYWORDS_PASS1 = {
            "TICK",
            "DETECTOR",
            "QUBIT_COORDS",
            "OBSERVABLE_INCLUDE",
        }
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                current_line_index += 1
                continue

            keyword = stripped_line.split()[0].split("(")[0]
            if keyword in _SKIP_KEYWORDS_PASS1:
                current_line_index += 1
                continue

            tokens = stripped_line.split()
            gate = tokens[0]

            if gate == "M":
                measure_index_to_line[current_measure_index] = current_line_index
                measure_line_to_measure_index[current_line_index] = (
                    current_measure_index
                )
                current_measure_index += 1

            current_line_index += 1

        current_line_index = 0
        measure_stack: list[int] = []
        for line in lines:
            stripped_line = line.strip()
            keyword = stripped_line.split()[0].split("(")[0] if stripped_line else ""
            if keyword == "DETECTOR":
                meas_index_str = [
                    token.strip()
                    for token in stripped_line.split()
                    if token.strip().startswith("rec")
                ]
                meas_index = [int(x[4:-1]) for x in meas_index_str]
                parityMatchGroup.append(
                    [
                        measure_line_to_measure_index[measure_stack[idx]]
                        for idx in meas_index
                    ]
                )
                current_line_index += 1
                continue
            elif keyword == "OBSERVABLE_INCLUDE":
                meas_index_str = [
                    token.strip()
                    for token in stripped_line.split()
                    if token.strip().startswith("rec")
                ]
                meas_index = [int(x[4:-1]) for x in meas_index_str]
                observable = [
                    measure_line_to_measure_index[measure_stack[idx]]
                    for idx in meas_index
                ]
                current_line_index += 1
                continue

            gate = keyword
            if gate == "M":
                measure_stack.append(current_line_index)
            current_line_index += 1

        """
        Insert gates (noiseless stim circuit, noise locations tracked for QEPG)
        """
        _1Q_GATE_MAP = {
            "H": self.add_hadamard,
            "S": self.add_phase,
            "X": self.add_paulix,
            "Y": self.add_pauliy,
            "Z": self.add_pauliz,
            "M": self.add_measurement,
        }
        _SKIP_KEYWORDS_PASS3 = {
            "TICK",
            "DETECTOR",
            "QUBIT_COORDS",
            "OBSERVABLE_INCLUDE",
        }
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            keyword = stripped_line.split()[0].split("(")[0]
            if keyword in _SKIP_KEYWORDS_PASS3:
                output_lines.append(stripped_line)
                continue

            tokens = stripped_line.split()
            gate = tokens[0]

            if gate == "CX":
                control = int(tokens[1])
                target = int(tokens[2])
                maxum_q_index = max(maxum_q_index, control, target)
                # Track noise locations for QEPG (no noise in stim circuit)
                self._track_noise_location(control)
                self._track_noise_location(target)
                self.add_cnot(control, target)

            elif gate in _1Q_GATE_MAP:
                qubit = int(tokens[1])
                maxum_q_index = max(maxum_q_index, qubit)
                self._track_noise_location(qubit)
                _1Q_GATE_MAP[gate](qubit)

            elif gate == "R":
                qubit = int(tokens[1])
                maxum_q_index = max(maxum_q_index, qubit)
                # No noise location before reset
                self.add_reset(qubit)

        """
        Finally, compile detector and observable
        """
        self._parityMatchGroup = parityMatchGroup
        self._observable = observable
        self._qubit_num = maxum_q_index + 1
        self.compile_detector_and_observable()

    def compile_from_noisy_stim_circuit_str(self, stim_str: str) -> None:
        """Parse a **noisy** STIM circuit string, preserving its noise as-is.

        Unlike :meth:`compile_from_stim_circuit_str` which strips noise and
        produces a noiseless circuit, this method keeps the noise instructions
        already present in the STIM string. The internal ``stim.Circuit`` is
        set to the full noisy circuit parsed directly from *stim_str*.

        The gate list is built for QEPG compatibility: noise instructions in
        the STIM string are registered as noise locations in the order they
        appear.

        This is **approach 1**: the user provides a noisy STIM circuit (e.g.
        from ``stim.Circuit.generated(...)`` or hand-written) and the system
        uses it directly without further noise injection.

        Args:
            stim_str: A STIM circuit string that already contains noise
                instructions (DEPOLARIZE1, DEPOLARIZE2, X_ERROR, etc.)
                as well as DETECTOR and OBSERVABLE_INCLUDE definitions.

        Side Effects:
            Resets and repopulates ``_gatelists``, ``_totalnoise``,
            ``_totalMeas``, ``_parityMatchGroup``, ``_observable``,
            ``_qubit_num``, and the internal ``stim.Circuit``.
        """
        from .stimparser import rewrite_stim_code

        self._totalnoise = 0
        self._totalMeas = 0
        self._totalgates = 0
        self._gatelists = []
        self._index_to_noise = {}
        self._index_to_measurement = {}
        self._measIdx_to_parityIdx = {}
        self._stim_str = stim_str

        # --- Build stim circuit directly from the noisy string ---
        self._stimcircuit = stim.Circuit(stim_str)

        # --- Normalize with noise preserved for gate-list iteration ---
        normalized = rewrite_stim_code(stim_str, keep_noise=True)
        lines = normalized.splitlines()

        # ------------------------------------------------------------------
        # Pass 1: index measurements (assign sequential measurement indices)
        # ------------------------------------------------------------------
        measure_index_to_line: dict[int, int] = {}
        measure_line_to_measure_index: dict[int, int] = {}
        _SKIP_KEYWORDS = {
            "TICK",
            "DETECTOR",
            "QUBIT_COORDS",
            "OBSERVABLE_INCLUDE",
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
            "PAULI_CHANNEL_1",
            "PAULI_CHANNEL_2",
            "SHIFT_COORDS",
        }
        current_line_index = 0
        current_measure_index = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_line_index += 1
                continue
            keyword = stripped.split()[0].split("(")[0]
            if keyword in _SKIP_KEYWORDS:
                current_line_index += 1
                continue
            tokens = stripped.split()
            gate = tokens[0]
            if gate == "M":
                measure_index_to_line[current_measure_index] = current_line_index
                measure_line_to_measure_index[current_line_index] = (
                    current_measure_index
                )
                current_measure_index += 1
            current_line_index += 1

        # ------------------------------------------------------------------
        # Pass 2: extract detectors and observable
        # ------------------------------------------------------------------
        parityMatchGroup: list[list[int]] = []
        observable: list[int] = []
        current_line_index = 0
        measure_stack: list[int] = []
        for line in lines:
            stripped = line.strip()
            keyword = stripped.split()[0].split("(")[0] if stripped else ""
            if keyword == "DETECTOR":
                meas_index_str = [
                    token.strip()
                    for token in stripped.split()
                    if token.strip().startswith("rec")
                ]
                meas_index = [int(x[4:-1]) for x in meas_index_str]
                parityMatchGroup.append(
                    [
                        measure_line_to_measure_index[measure_stack[idx]]
                        for idx in meas_index
                    ]
                )
                current_line_index += 1
                continue
            elif keyword == "OBSERVABLE_INCLUDE":
                meas_index_str = [
                    token.strip()
                    for token in stripped.split()
                    if token.strip().startswith("rec")
                ]
                meas_index = [int(x[4:-1]) for x in meas_index_str]
                observable = [
                    measure_line_to_measure_index[measure_stack[idx]]
                    for idx in meas_index
                ]
                current_line_index += 1
                continue

            gate = keyword
            if gate == "M":
                measure_stack.append(current_line_index)
            current_line_index += 1

        # ------------------------------------------------------------------
        # Pass 3: build gate list (no stim circuit writes)
        # ------------------------------------------------------------------
        _1Q_GATE_INDEX = {
            "H": oneQGateindices["H"],
            "S": oneQGateindices["P"],
            "X": oneQGateindices["X"],
            "Y": oneQGateindices["Y"],
            "Z": oneQGateindices["Z"],
        }
        _1Q_NOISE_KEYWORDS = {
            "DEPOLARIZE1",
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
            "PAULI_CHANNEL_1",
        }
        _2Q_NOISE_KEYWORDS = {"DEPOLARIZE2", "PAULI_CHANNEL_2"}

        maxum_q_index = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            keyword = stripped.split()[0].split("(")[0]
            if keyword in (
                "TICK",
                "DETECTOR",
                "QUBIT_COORDS",
                "OBSERVABLE_INCLUDE",
                "SHIFT_COORDS",
            ):
                continue

            tokens = stripped.split()
            gate = keyword

            if gate == "CX":
                control = int(tokens[1])
                target = int(tokens[2])
                maxum_q_index = max(maxum_q_index, control, target)
                self._gatelists.append(
                    TwoQGate(twoQGateindices["CNOT"], control, target)
                )

            elif gate in _1Q_GATE_INDEX:
                qubit = int(tokens[1])
                maxum_q_index = max(maxum_q_index, qubit)
                self._gatelists.append(SingleQGate(_1Q_GATE_INDEX[gate], qubit))

            elif gate == "M":
                qubit = int(tokens[1])
                maxum_q_index = max(maxum_q_index, qubit)
                meas = Measurement(self._totalMeas, qubit)
                self._gatelists.append(meas)
                self._index_to_measurement[self._totalMeas] = meas
                self._measIdx_to_parityIdx[self._totalMeas] = []
                self._totalMeas += 1

            elif gate == "R":
                qubit = int(tokens[1])
                maxum_q_index = max(maxum_q_index, qubit)
                self._gatelists.append(Reset(qubit))

            elif gate in _1Q_NOISE_KEYWORDS:
                qubit = int(tokens[1])
                maxum_q_index = max(maxum_q_index, qubit)
                # Track noise location for QEPG (no stim circuit write)
                self._track_noise_location(qubit)

            elif gate in _2Q_NOISE_KEYWORDS:
                q1 = int(tokens[1])
                q2 = int(tokens[2])
                maxum_q_index = max(maxum_q_index, q1, q2)
                self._track_noise_location(q1)
                self._track_noise_location(q2)

        # ------------------------------------------------------------------
        # Finalize: set parity groups and observable
        # ------------------------------------------------------------------
        self._parityMatchGroup = parityMatchGroup
        self._observable = observable
        self._qubit_num = maxum_q_index + 1

        # Build the measIdx → parityIdx reverse mapping
        for det_idx, paritygroup in enumerate(self._parityMatchGroup):
            for k in paritygroup:
                self._measIdx_to_parityIdx[k].append(det_idx)

    def save_circuit_to_file(self, filename):
        pass

    def set_noise_type(self, noiseindex: int, noisetype: int) -> None:
        """Set the Pauli error type for a specific noise source.

        Args:
            noiseindex: The noise source index to modify.
            noisetype: The Pauli type to assign (0=I, 1=X, 2=Y, 3=Z).
        """
        self._index_to_noise[noiseindex].noisetype = noisetype

    def reset_noise_type(self) -> None:
        """Reset all noise sources to identity (no error)."""
        for i in range(self._totalnoise):
            self._index_to_noise[i].noisetype = 0

    def show_all_noise(self) -> None:
        """Print all noise sources and their current error types to stdout."""
        for i in range(self._totalnoise):
            print(self._index_to_noise[i])

    def get_noise_context(self, noise_index: int) -> dict:
        """Return contextual info about a noise source.

        Looks at the gate list entry immediately after the noise source
        to determine what gate the noise precedes.

        Args:
            noise_index: Index of the noise source.

        Returns:
            A dict with keys ``qubit``, ``following_gate``, and
            ``following_gate_type``.
        """
        noise = self._index_to_noise.get(noise_index)
        if noise is None:
            return {}

        # Find position of this noise in gate list
        result = {"qubit": noise._qubitindex}
        for i, gate in enumerate(self._gatelists):
            if isinstance(gate, pauliNoise) and gate._noiseindex == noise_index:
                # Look at the next non-noise gate
                for j in range(i + 1, len(self._gatelists)):
                    next_gate = self._gatelists[j]
                    if not isinstance(next_gate, pauliNoise):
                        result["following_gate"] = str(next_gate)
                        result["following_gate_type"] = type(next_gate).__name__
                        break
                break

        return result

    def add_xflip_noise(self, qubit: int) -> None:
        """Add an X-flip noise channel on a qubit.

        Appends an ``X_ERROR`` instruction to the STIM circuit and a
        :class:`pauliNoise` entry to the gate list.

        Args:
            qubit: The qubit to apply the noise channel to.
        """
        self._stimcircuit.append("X_ERROR", [qubit], self._error_rate)
        noise = pauliNoise(self._totalnoise, qubit)
        self._gatelists.append(noise)
        self._index_to_noise[self._totalnoise] = noise
        self._totalnoise += 1

    def _track_noise_location(self, qubit: int) -> None:
        """Register a noise location without emitting to the stim circuit.

        Records a :class:`pauliNoise` entry in the gate list so that
        QEPG and weight-stratified sampling know about this noise site,
        but does **not** append any noise instruction to the internal
        ``stim.Circuit``.  The stim circuit remains noiseless; noise
        must be injected separately via a :class:`NoiseModel`.

        Args:
            qubit: The qubit where a noise channel could act.
        """
        noise = pauliNoise(self._totalnoise, qubit)
        self._gatelists.append(noise)
        self._index_to_noise[self._totalnoise] = noise
        self._totalnoise += 1

    def add_depolarize(self, qubit: int) -> None:
        """Add a single-qubit depolarizing noise channel.

        Appends a ``DEPOLARIZE1`` instruction to the STIM circuit and a
        :class:`pauliNoise` entry to the gate list. The depolarizing
        probability is taken from :attr:`error_rate`.

        .. deprecated::
            This method bakes noise into the circuit during compilation.
            Prefer compiling a noiseless circuit and injecting noise
            afterwards via :class:`~scalerqec.QEC.noisemodel.SIDNoiseModel`.

        Args:
            qubit: The qubit to apply the depolarizing channel to.
        """
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)
        noise = pauliNoise(self._totalnoise, qubit)
        self._gatelists.append(noise)
        self._index_to_noise[self._totalnoise] = noise
        self._totalnoise += 1

    def add_cnot(self, control: int, target: int) -> None:
        """Add a CNOT (controlled-X) gate.

        Args:
            control: The control qubit index.
            target: The target qubit index.
        """
        self._gatelists.append(TwoQGate(twoQGateindices["CNOT"], control, target))
        self._stimcircuit.append("CNOT", [control, target])

    def add_hadamard(self, qubit: int) -> None:
        """Add a Hadamard gate.

        Args:
            qubit: The qubit to apply H to.
        """
        self._gatelists.append(SingleQGate(oneQGateindices["H"], qubit))
        self._stimcircuit.append("H", [qubit])

    def add_phase(self, qubit: int) -> None:
        """Add a phase (S) gate.

        Args:
            qubit: The qubit to apply S to.
        """
        self._gatelists.append(SingleQGate(oneQGateindices["P"], qubit))
        self._stimcircuit.append("S", [qubit])

    def add_cz(self, qubit1: int, qubit2: int) -> None:
        """Add a controlled-Z gate.

        Args:
            qubit1: The first qubit (control).
            qubit2: The second qubit (target).
        """
        self._gatelists.append(TwoQGate(twoQGateindices["CZ"], qubit1, qubit2))

    def add_paulix(self, qubit: int) -> None:
        """Add a Pauli X gate.

        Args:
            qubit: The qubit to apply X to.
        """
        self._gatelists.append(SingleQGate(oneQGateindices["X"], qubit))
        self._stimcircuit.append("X", [qubit])

    def add_pauliy(self, qubit: int) -> None:
        """Add a Pauli Y gate.

        Args:
            qubit: The qubit to apply Y to.
        """
        self._gatelists.append(SingleQGate(oneQGateindices["Y"], qubit))
        self._stimcircuit.append("Y", [qubit])

    def add_pauliz(self, qubit: int) -> None:
        """Add a Pauli Z gate.

        Args:
            qubit: The qubit to apply Z to.
        """
        self._gatelists.append(SingleQGate(oneQGateindices["Z"], qubit))
        self._stimcircuit.append("Z", [qubit])

    def add_measurement(self, qubit: int) -> None:
        """Add a Z-basis measurement on a qubit.

        Assigns a sequential measurement index and registers the measurement
        in the internal lookup tables.

        Args:
            qubit: The qubit to measure.
        """
        meas = Measurement(self._totalMeas, qubit)
        self._gatelists.append(meas)
        self._stimcircuit.append("M", [qubit])
        self._index_to_measurement[self._totalMeas] = meas
        self._measIdx_to_parityIdx[self._totalMeas] = []
        self._totalMeas += 1

    def compile_detector_and_observable(self) -> None:
        """Append DETECTOR and OBSERVABLE_INCLUDE instructions to the STIM circuit.

        Uses :attr:`parityMatchGroup` and :attr:`observable` to emit the
        corresponding STIM instructions with relative measurement record
        references. Also populates the ``_measIdx_to_parityIdx`` reverse
        mapping so each measurement knows which detectors it belongs to.

        This method is called automatically at the end of
        :meth:`compile_from_stim_circuit_str`.
        """
        totalMeas = self._totalMeas
        # print(totalMeas)
        detectorIdx = 0
        for paritygroup in self._parityMatchGroup:
            # print(paritygroup)
            # print([k-totalMeas for k in paritygroup])
            self._stimcircuit.append(
                "DETECTOR", [stim.target_rec(k - totalMeas) for k in paritygroup]
            )
            for k in paritygroup:
                self._measIdx_to_parityIdx[k].append(detectorIdx)
            detectorIdx += 1
        self._stimcircuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(k - totalMeas) for k in self._observable],
            0,
        )

    def add_reset(self, qubit: int) -> None:
        """Add a qubit reset (project to |0>).

        Args:
            qubit: The qubit to reset.
        """
        self._gatelists.append(Reset(qubit))
        self._stimcircuit.append("R", [qubit])

    def setShowNoise(self, show: bool) -> None:
        """Control whether noise channels are included in string output.

        Args:
            show: If ``True``, noise channels appear in :meth:`__str__`
                and LaTeX output.
        """
        self._shownoise = show

    def __str__(self):
        str = ""
        for gate in self._gatelists:
            if isinstance(gate, pauliNoise) and not self._shownoise:
                continue
            str += gate.__str__() + "\n"
        return str

    def get_yquant_latex(self):
        """
        Convert the circuit (stored in self._gatelists) into a yquant LaTeX string.
        This version simply prints each gate (or noise box) in the order they appear,
        without grouping or any fancy logic.
        """
        lines = []
        # Begin the yquant environment
        lines.append("\\begin{yquant}")
        lines.append("")

        # Declare qubits and classical bits.
        # Note: Literal braces in the LaTeX code are escaped by doubling them.
        lines.append("% -- Qubits and classical bits --")
        lines.append("qubit {{$\\ket{{q_{{\\idx}}}}$}} q[{}];".format(self._qubit_num))
        lines.append("cbit {{$c_{{\\idx}} = 0$}} c[{}];".format(self._totalMeas))
        lines.append("")
        lines.append("% -- Circuit Operations --")

        # Process each gate in the order they were added.
        for gate in self._gatelists:
            if isinstance(gate, pauliNoise):
                # Print the noise box only if noise output is enabled.
                if self._shownoise:
                    lines.append("[fill=red!80]")
                    # The following format string produces, e.g.,:
                    # "box {$n_{8}$} q[2];"
                    lines.append(
                        "box {{$n_{{{}}}$}} q[{}];".format(
                            gate._noiseindex, gate._qubitindex
                        )
                    )
            elif isinstance(gate, TwoQGate):
                # Two-qubit gate (e.g., CNOT or CZ).
                if gate._name == "CNOT":
                    # Note: yquant syntax for a CNOT is: cnot q[target] | q[control];
                    line = "cnot q[{}] | q[{}];".format(gate._target, gate._control)
                elif gate._name == "CZ":
                    line = "cz q[{}] | q[{}];".format(gate._target, gate._control)
                lines.append(line)
            elif isinstance(gate, SingleQGate):
                # Single-qubit gate.
                if gate._name == "H":
                    line = "h q[{}];".format(gate._qubitindex)

                lines.append(line)
            elif isinstance(gate, Measurement):
                # Measurement is output as three separate lines.
                lines.append("measure q[{}];".format(gate._qubitindex))
                lines.append(
                    "cnot c[{}] | q[{}];".format(gate._measureindex, gate._qubitindex)
                )
                lines.append("discard q[{}];".format(gate._qubitindex))
            elif isinstance(gate, Reset):
                # Reset is output as an initialization command.
                lines.append("init {{$\\ket0$}} q[{}];".format(gate._qubitindex))
            else:
                continue

        lines.append("")
        lines.append("\\end{yquant}")

        return "\n".join(lines)


def example_basic():
    """Basic example without noise."""
    circ = CliffordCircuit(3)
    circ.error_rate = 0.1
    circ.add_hadamard(0)
    circ.add_cnot(0, 1)
    circ.add_cnot(0, 2)
    circ.add_measurement(1)
    circ.add_measurement(2)
    # Convert scaler circuit to stim circuit
    stimcirc = circ.stimcircuit
    print(stimcirc)


def example_with_noise():
    """Example with depolarizing noise."""
    circ = CliffordCircuit(3)
    circ.error_rate = 0.1
    circ.add_depolarize(0)
    circ.add_hadamard(0)
    circ.add_depolarize(0)
    circ.add_depolarize(1)
    circ.add_cnot(0, 1)
    circ.add_cnot(0, 2)
    circ.add_depolarize(1)
    circ.add_measurement(1)
    circ.add_measurement(2)
    # Convert scaler circuit to stim circuit
    stimcirc = circ.stimcircuit
    print(stimcirc)


if __name__ == "__main__":
    example_basic()
    example_with_noise()
