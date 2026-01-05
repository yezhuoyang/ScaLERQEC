"""
Tests for scalerqec.Clifford.clifford module.

Tests the CliffordCircuit class and related gate operations.
"""
import pytest
from scalerqec.Clifford.clifford import (
    CliffordCircuit,
    SingleQGate,
    TwoQGate,
    Measurement,
    Reset,
    pauliNoise,
)


class TestCliffordCircuitInitialization:
    """Tests for CliffordCircuit initialization."""

    def test_basic_initialization(self):
        """Test basic circuit creation with qubit count."""
        circ = CliffordCircuit(5)
        assert circ.qubit_num == 5
        assert circ.totalnoise == 0
        assert circ.totalMeas == 0
        assert circ.error_rate == 0
        assert len(circ.gatelists) == 0

    def test_initialization_single_qubit(self):
        """Test circuit with single qubit."""
        circ = CliffordCircuit(1)
        assert circ.qubit_num == 1

    def test_initialization_large_circuit(self):
        """Test circuit with many qubits."""
        circ = CliffordCircuit(100)
        assert circ.qubit_num == 100


class TestErrorRateProperty:
    """Tests for error rate property."""

    def test_set_error_rate(self):
        """Test setting error rate."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        assert circ.error_rate == 0.1

    def test_error_rate_zero(self):
        """Test zero error rate."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.0
        assert circ.error_rate == 0.0

    def test_error_rate_small(self):
        """Test very small error rate."""
        circ = CliffordCircuit(3)
        circ.error_rate = 1e-6
        assert circ.error_rate == 1e-6


class TestSingleQubitGates:
    """Tests for single qubit gate operations."""

    def test_add_hadamard(self):
        """Test adding Hadamard gate."""
        circ = CliffordCircuit(3)
        circ.add_hadamard(0)
        assert len(circ.gatelists) == 1
        assert isinstance(circ.gatelists[0], SingleQGate)
        assert circ.gatelists[0].name == "H"
        assert circ.gatelists[0].qubitindex == 0

    def test_add_phase(self):
        """Test adding Phase (S) gate."""
        circ = CliffordCircuit(3)
        circ.add_phase(1)
        assert len(circ.gatelists) == 1
        assert circ.gatelists[0].name == "P"
        assert circ.gatelists[0].qubitindex == 1

    def test_add_paulix(self):
        """Test adding Pauli X gate."""
        circ = CliffordCircuit(3)
        circ.add_paulix(2)
        assert len(circ.gatelists) == 1
        assert circ.gatelists[0].name == "X"
        assert circ.gatelists[0].qubitindex == 2

    def test_add_pauliy(self):
        """Test adding Pauli Y gate."""
        circ = CliffordCircuit(3)
        circ.add_pauliy(0)
        assert len(circ.gatelists) == 1
        assert circ.gatelists[0].name == "Y"

    def test_add_pauliz(self):
        """Test adding Pauli Z gate."""
        circ = CliffordCircuit(3)
        circ.add_pauliz(1)
        assert len(circ.gatelists) == 1
        assert circ.gatelists[0].name == "Z"

    def test_add_multiple_single_qubit_gates(self):
        """Test adding multiple single qubit gates."""
        circ = CliffordCircuit(3)
        circ.add_hadamard(0)
        circ.add_phase(0)
        circ.add_hadamard(0)
        assert len(circ.gatelists) == 3


class TestTwoQubitGates:
    """Tests for two qubit gate operations."""

    def test_add_cnot(self):
        """Test adding CNOT gate."""
        circ = CliffordCircuit(3)
        circ.add_cnot(0, 1)
        assert len(circ.gatelists) == 1
        assert isinstance(circ.gatelists[0], TwoQGate)
        assert circ.gatelists[0].name == "CNOT"
        assert circ.gatelists[0].control == 0
        assert circ.gatelists[0].target == 1

    def test_add_cz(self):
        """Test adding CZ gate."""
        circ = CliffordCircuit(3)
        circ.add_cz(1, 2)
        assert len(circ.gatelists) == 1
        assert circ.gatelists[0].name == "CZ"

    def test_add_multiple_cnots(self):
        """Test adding multiple CNOT gates."""
        circ = CliffordCircuit(4)
        circ.add_cnot(0, 1)
        circ.add_cnot(1, 2)
        circ.add_cnot(2, 3)
        assert len(circ.gatelists) == 3


class TestMeasurementOperations:
    """Tests for measurement operations."""

    def test_add_measurement(self):
        """Test adding measurement."""
        circ = CliffordCircuit(3)
        circ.add_measurement(0)
        assert circ.totalMeas == 1
        assert len(circ.gatelists) == 1
        assert isinstance(circ.gatelists[0], Measurement)
        assert circ.gatelists[0].qubitindex == 0

    def test_add_multiple_measurements(self):
        """Test adding multiple measurements."""
        circ = CliffordCircuit(3)
        circ.add_measurement(0)
        circ.add_measurement(1)
        circ.add_measurement(2)
        assert circ.totalMeas == 3

    def test_measurement_index_tracking(self):
        """Test that measurement indices are tracked correctly."""
        circ = CliffordCircuit(3)
        circ.add_measurement(0)
        circ.add_measurement(1)
        # Measurements should be indexed 0, 1
        assert circ.gatelists[0]._measureindex == 0
        assert circ.gatelists[1]._measureindex == 1


class TestResetOperations:
    """Tests for reset operations."""

    def test_add_reset(self):
        """Test adding reset."""
        circ = CliffordCircuit(3)
        circ.add_reset(0)
        assert len(circ.gatelists) == 1
        assert isinstance(circ.gatelists[0], Reset)
        assert circ.gatelists[0].qubitindex == 0

    def test_add_multiple_resets(self):
        """Test adding multiple resets."""
        circ = CliffordCircuit(3)
        circ.add_reset(0)
        circ.add_reset(1)
        assert len(circ.gatelists) == 2


class TestNoiseOperations:
    """Tests for noise operations."""

    def test_add_depolarize(self):
        """Test adding depolarizing noise."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        circ.add_depolarize(0)
        assert circ.totalnoise == 1
        assert len(circ.gatelists) == 1
        assert isinstance(circ.gatelists[0], pauliNoise)

    def test_add_xflip_noise(self):
        """Test adding X-flip noise."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        circ.add_xflip_noise(0)
        assert circ.totalnoise == 1

    def test_add_multiple_noise(self):
        """Test adding multiple noise operations."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        circ.add_depolarize(0)
        circ.add_depolarize(1)
        circ.add_depolarize(2)
        assert circ.totalnoise == 3

    def test_noise_index_tracking(self):
        """Test that noise indices are tracked correctly."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        circ.add_depolarize(0)
        circ.add_depolarize(1)
        # Noise operations should be indexed 0, 1
        assert circ.gatelists[0]._noiseindex == 0
        assert circ.gatelists[1]._noiseindex == 1


class TestStimCircuitGeneration:
    """Tests for STIM circuit generation."""

    def test_stimcircuit_property_exists(self):
        """Test that stimcircuit property exists."""
        circ = CliffordCircuit(3)
        assert circ.stimcircuit is not None

    def test_stimcircuit_after_hadamard(self):
        """Test STIM circuit after adding Hadamard."""
        circ = CliffordCircuit(3)
        circ.add_hadamard(0)
        stim_str = str(circ.stimcircuit)
        assert "H" in stim_str

    def test_stimcircuit_after_cnot(self):
        """Test STIM circuit after adding CNOT."""
        circ = CliffordCircuit(3)
        circ.add_cnot(0, 1)
        stim_str = str(circ.stimcircuit)
        assert "CNOT" in stim_str or "CX" in stim_str

    def test_stimcircuit_after_measurement(self):
        """Test STIM circuit after adding measurement."""
        circ = CliffordCircuit(3)
        circ.add_measurement(0)
        stim_str = str(circ.stimcircuit)
        assert "M" in stim_str

    def test_stimcircuit_with_noise(self):
        """Test STIM circuit with depolarizing noise."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        circ.add_depolarize(0)
        circ.add_hadamard(0)
        stim_str = str(circ.stimcircuit)
        assert "DEPOLARIZE1" in stim_str

    def test_stimcircuit_setter_from_string(self):
        """Test setting stimcircuit from string."""
        circ = CliffordCircuit(3)
        stim_str = "H 0\nCNOT 0 1\nM 0 1"
        circ.stimcircuit = stim_str
        assert circ.stim_str == stim_str


class TestParityMatchGroup:
    """Tests for parity match group (detector) operations."""

    def test_parity_match_group_empty(self):
        """Test empty parity match group."""
        circ = CliffordCircuit(3)
        assert circ.parityMatchGroup == []

    def test_parity_match_group_setter(self):
        """Test setting parity match group."""
        circ = CliffordCircuit(3)
        circ.parityMatchGroup = [[0, 1], [2, 3]]
        assert circ.parityMatchGroup == [[0, 1], [2, 3]]

    def test_observable_property(self):
        """Test observable property."""
        circ = CliffordCircuit(3)
        assert circ.observable == []
        circ.observable = [0, 1]
        assert circ.observable == [0, 1]


class TestCircuitStringRepresentation:
    """Tests for circuit string representation."""

    def test_str_empty_circuit(self):
        """Test string representation of empty circuit."""
        circ = CliffordCircuit(3)
        assert str(circ) == ""

    def test_str_with_gates(self):
        """Test string representation with gates."""
        circ = CliffordCircuit(3)
        circ.add_hadamard(0)
        circ.add_cnot(0, 1)
        result = str(circ)
        assert "H" in result
        assert "CNOT" in result

    def test_str_hides_noise_by_default(self):
        """Test that noise is hidden by default in string representation."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        circ.add_depolarize(0)
        circ.add_hadamard(0)
        result = str(circ)
        # Noise should be hidden
        assert "n0" not in result
        assert "H" in result

    def test_str_shows_noise_when_enabled(self):
        """Test that noise is shown when enabled."""
        circ = CliffordCircuit(3)
        circ.error_rate = 0.1
        circ.add_depolarize(0)
        circ.add_hadamard(0)
        circ.setShowNoise(True)
        result = str(circ)
        assert "n0" in result


class TestCompileFromStimString:
    """Tests for compiling circuit from STIM string."""

    def test_compile_simple_circuit(self):
        """Test compiling a simple STIM circuit."""
        stim_str = """CX 0 1
M 0
M 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circ = CliffordCircuit(2)
        circ.error_rate = 0.01
        circ.compile_from_stim_circuit_str(stim_str)

        # Should have noise + gates
        assert circ.totalnoise > 0
        assert circ.totalMeas == 2

    def test_compile_with_hadamard(self):
        """Test compiling circuit with Hadamard gates."""
        stim_str = """H 0
CX 0 1
M 0
M 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circ = CliffordCircuit(2)
        circ.error_rate = 0.01
        circ.compile_from_stim_circuit_str(stim_str)

        assert circ.totalMeas == 2

    def test_compile_with_reset(self):
        """Test compiling circuit with reset operations."""
        stim_str = """R 0
H 0
CX 0 1
M 0
M 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circ = CliffordCircuit(2)
        circ.error_rate = 0.01
        circ.compile_from_stim_circuit_str(stim_str)

        assert circ.totalMeas == 2

    def test_compile_extracts_detectors(self):
        """Test that detectors are correctly extracted.

        Note: The parser expects DETECTOR(coords) format, not bare DETECTOR.
        """
        stim_str = """CX 0 1
M 0
M 1
DETECTOR(0,0) rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circ = CliffordCircuit(2)
        circ.error_rate = 0.01
        circ.compile_from_stim_circuit_str(stim_str)

        # Should have one detector group
        assert len(circ.parityMatchGroup) == 1

    def test_compile_extracts_observable(self):
        """Test that observable is correctly extracted.

        Note: The parser expects DETECTOR(coords) format, not bare DETECTOR.
        """
        stim_str = """CX 0 1
M 0
M 1
DETECTOR(0,0) rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circ = CliffordCircuit(2)
        circ.error_rate = 0.01
        circ.compile_from_stim_circuit_str(stim_str)

        # Should have observable
        assert len(circ.observable) > 0


class TestGateClasses:
    """Tests for individual gate classes."""

    def test_single_q_gate_str(self):
        """Test SingleQGate string representation."""
        gate = SingleQGate(0, 2)  # H gate on qubit 2
        assert str(gate) == "H[2]"

    def test_two_q_gate_str(self):
        """Test TwoQGate string representation."""
        gate = TwoQGate(0, 1, 2)  # CNOT from 1 to 2
        assert str(gate) == "CNOT[1,2]"

    def test_measurement_str(self):
        """Test Measurement string representation."""
        meas = Measurement(0, 1)  # Measurement 0 on qubit 1
        assert str(meas) == "M0[1]"

    def test_reset_str(self):
        """Test Reset string representation."""
        reset = Reset(2)
        assert str(reset) == "R[2]"

    def test_pauli_noise_str(self):
        """Test pauliNoise string representation."""
        noise = pauliNoise(0, 1)  # Noise 0 on qubit 1
        # Default noise type is 0 (I)
        assert "n0" in str(noise)


class TestComplexCircuits:
    """Tests for complex circuit constructions."""

    def test_bell_state_preparation(self):
        """Test Bell state preparation circuit."""
        circ = CliffordCircuit(2)
        circ.add_hadamard(0)
        circ.add_cnot(0, 1)
        circ.add_measurement(0)
        circ.add_measurement(1)

        assert len(circ.gatelists) == 4
        assert circ.totalMeas == 2

    def test_ghz_state_preparation(self):
        """Test GHZ state preparation circuit."""
        circ = CliffordCircuit(3)
        circ.add_hadamard(0)
        circ.add_cnot(0, 1)
        circ.add_cnot(1, 2)

        assert len(circ.gatelists) == 3

    def test_circuit_with_noise_and_gates(self):
        """Test circuit with interleaved noise and gates."""
        circ = CliffordCircuit(2)
        circ.error_rate = 0.01

        circ.add_depolarize(0)
        circ.add_hadamard(0)
        circ.add_depolarize(0)
        circ.add_depolarize(1)
        circ.add_cnot(0, 1)
        circ.add_depolarize(0)
        circ.add_depolarize(1)
        circ.add_measurement(0)
        circ.add_measurement(1)

        assert circ.totalnoise == 5
        assert circ.totalMeas == 2

    def test_repetition_code_style_circuit(self):
        """Test a repetition-code style circuit."""
        circ = CliffordCircuit(5)  # 3 data + 2 ancilla

        # Syndrome extraction pattern
        circ.add_cnot(0, 3)
        circ.add_cnot(1, 3)
        circ.add_measurement(3)
        circ.add_reset(3)

        circ.add_cnot(1, 4)
        circ.add_cnot(2, 4)
        circ.add_measurement(4)
        circ.add_reset(4)

        assert circ.totalMeas == 2


class TestYquantLatex:
    """Tests for yquant LaTeX generation."""

    def test_yquant_basic_circuit(self):
        """Test yquant LaTeX for basic circuit."""
        circ = CliffordCircuit(2)
        circ.add_hadamard(0)
        circ.add_cnot(0, 1)

        latex = circ.get_yquant_latex()
        assert "\\begin{yquant}" in latex
        assert "\\end{yquant}" in latex
        assert "h q[0]" in latex
        assert "cnot" in latex

    def test_yquant_with_measurement(self):
        """Test yquant LaTeX with measurement."""
        circ = CliffordCircuit(2)
        circ.add_hadamard(0)
        circ.add_measurement(0)

        latex = circ.get_yquant_latex()
        assert "measure" in latex
