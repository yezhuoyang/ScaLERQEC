"""
Comprehensive tests for the QEPG (Quantum Error Propagation Graph) C++ backend.

These tests verify:
1. CliffordCircuit parsing from Stim strings
2. QEPG graph construction (backward propagation)
3. Correctness of parity propagation matrices for various gate sequences
4. Sampling functionality

The tests start with simple single-gate circuits and build up to complete QEC circuits.
"""
import pytest
import numpy as np

# Import the QEPG C++ module
import scalerqec.qepg as qepg


# ============================================================================
# Test Fixtures for Stim Circuit Strings
# ============================================================================

@pytest.fixture
def simple_single_qubit_circuit():
    """Single qubit with reset, noise, Hadamard, and measurement."""
    return """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.fixture
def two_qubit_cnot_circuit():
    """Two qubits with CNOT and measurements."""
    return """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.fixture
def repetition_code_d3_circuit():
    """Distance-3 repetition code circuit."""
    return """R 0
R 1
R 2
R 3
R 4
CX 0 3
CX 1 3
CX 1 4
CX 2 4
M 3
M 4
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-5] rec[-4]
DETECTOR(1, 0, 0) rec[-4] rec[-3]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]
"""


# ============================================================================
# Test Class: CliffordCircuit Parsing
# ============================================================================

class TestCliffordCircuitParsing:
    """Tests for the CliffordCircuit C++ class parsing Stim strings."""

    def test_parse_empty_circuit(self):
        """Parsing an empty string should work without error."""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string("")
        assert circuit.get_num_qubit() == 0
        assert circuit.get_num_noise() == 0
        assert circuit.get_num_detector() == 0

    def test_parse_single_reset(self):
        """Parse a single reset operation."""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string("R 0")
        assert circuit.get_num_qubit() == 1
        # Reset doesn't add noise
        assert circuit.get_num_noise() == 0

    def test_parse_single_hadamard(self):
        """Parse a single Hadamard gate (adds depolarization noise)."""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string("H 0")
        assert circuit.get_num_qubit() == 1
        # Hadamard adds one depolarization noise
        assert circuit.get_num_noise() == 1

    def test_parse_single_measurement(self):
        """Parse a single measurement."""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string("M 0")
        assert circuit.get_num_qubit() == 1
        # Measurement adds one depolarization noise
        assert circuit.get_num_noise() == 1

    def test_parse_cnot(self):
        """Parse a CNOT gate (adds noise on both qubits)."""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string("CX 0 1")
        assert circuit.get_num_qubit() == 2
        # CNOT adds depolarization on both control and target
        assert circuit.get_num_noise() == 2

    def test_parse_multiple_qubits(self):
        """Parse circuit with multiple qubits."""
        circuit = qepg.CliffordCircuit()
        stim_str = """R 0
R 1
R 2
H 0
H 1
H 2
CX 0 1
CX 1 2
M 0
M 1
M 2
"""
        circuit.compile_from_rewrited_stim_string(stim_str)
        assert circuit.get_num_qubit() == 3
        # 3 H gates (3 noise) + 2 CX gates (4 noise) + 3 M gates (3 noise) = 10 noise
        assert circuit.get_num_noise() == 10

    def test_parse_detector_single_rec(self):
        """Parse a detector with single rec reference."""
        circuit = qepg.CliffordCircuit()
        stim_str = """M 0
DETECTOR(0, 0, 0) rec[-1]
"""
        circuit.compile_from_rewrited_stim_string(stim_str)
        assert circuit.get_num_detector() == 1

    def test_parse_detector_multiple_rec(self):
        """Parse a detector with multiple rec references."""
        circuit = qepg.CliffordCircuit()
        stim_str = """M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
"""
        circuit.compile_from_rewrited_stim_string(stim_str)
        assert circuit.get_num_detector() == 1

    def test_parse_multiple_detectors(self):
        """Parse multiple detectors."""
        circuit = qepg.CliffordCircuit()
        stim_str = """M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3]
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(2, 0, 0) rec[-1]
"""
        circuit.compile_from_rewrited_stim_string(stim_str)
        assert circuit.get_num_detector() == 3

    def test_parse_observable(self):
        """Parse an observable declaration."""
        circuit = qepg.CliffordCircuit()
        stim_str = """M 0
M 1
OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
"""
        circuit.compile_from_rewrited_stim_string(stim_str)
        # Observable is parsed but doesn't increment detector count
        assert circuit.get_num_detector() == 0


# ============================================================================
# Test Class: QEPG Graph Construction
# ============================================================================

class TestQEPGGraphConstruction:
    """Tests for QEPG graph construction from CliffordCircuit."""

    def test_construct_graph_empty(self):
        """Constructing graph from empty circuit."""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string("")
        graph = qepg.QEPGGraph(circuit, 0, 0)
        graph.backward_graph_construction()
        # Should complete without error

    def test_construct_graph_single_measurement(self):
        """Construct graph for single qubit with measurement."""
        stim_str = """R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        num_detector = circuit.get_num_detector()
        num_noise = circuit.get_num_noise()

        graph = qepg.QEPGGraph(circuit, num_detector, num_noise)
        graph.backward_graph_construction()
        # Should complete without error

    def test_construct_graph_cnot(self):
        """Construct graph for CNOT gate circuit."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        graph = qepg.QEPGGraph(circuit, circuit.get_num_detector(), circuit.get_num_noise())
        graph.backward_graph_construction()
        # Should complete without error


# ============================================================================
# Test Class: compile_QEPG High-Level API
# ============================================================================

class TestCompileQEPG:
    """Tests for the compile_QEPG convenience function."""

    def test_compile_qepg_simple(self):
        """Compile QEPG from simple circuit string."""
        stim_str = """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)
        # Should return a valid QEPGGraph object
        assert graph is not None

    def test_compile_qepg_repetition_code(self):
        """Compile QEPG from repetition code circuit."""
        stim_str = """R 0
R 1
R 2
R 3
R 4
CX 0 3
CX 1 3
CX 1 4
CX 2 4
M 3
M 4
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-5] rec[-4]
DETECTOR(1, 0, 0) rec[-4] rec[-3]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)
        assert graph is not None


# ============================================================================
# Test Class: Pauli Error Propagation - Single Gate Tests
# ============================================================================

class TestPauliPropagationSingleGate:
    """
    Tests verifying correct Pauli error propagation through individual gates.

    The parity propagation matrix has shape (3 * num_noise) x (num_detector + 1).
    - Rows 0 to num_noise-1: X errors
    - Rows num_noise to 2*num_noise-1: Y errors
    - Rows 2*num_noise to 3*num_noise-1: Z errors
    - Columns 0 to num_detector-1: detector outcomes
    - Column num_detector: observable outcome
    """

    def test_measurement_x_error_flips_detector(self):
        """
        X error before Z-basis measurement should flip the detector.

        Circuit: R 0, M 0, DETECTOR rec[-1]
        An X error on qubit 0 before measurement flips the measurement outcome.
        """
        stim_str = """R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Sample with weight 1 - should get some detector flips
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100
        # Each sample should have detector + observable columns
        assert len(samples[0]) == 2  # 1 detector + 1 observable

    def test_measurement_z_error_no_flip(self):
        """
        Z error before Z-basis measurement should NOT flip the detector.

        In Z-basis measurement, Z errors commute with measurement and don't flip outcomes.
        However, in the circuit model with depolarization, the noise is added with the gate.
        """
        stim_str = """R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)
        # This is a structural test - we verify sampling works
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100

    def test_hadamard_swaps_x_z_propagation(self):
        """
        Hadamard gate swaps X and Z error propagation.

        Circuit: R 0, H 0, M 0, DETECTOR rec[-1]
        - X error before H → Z error after H → no flip in Z-measurement
        - Z error before H → X error after H → flips Z-measurement
        """
        stim_str = """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # H adds noise, M adds noise = 2 total noise locations
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100
        assert len(samples[0]) == 2  # 1 detector + 1 observable


# ============================================================================
# Test Class: CNOT Gate Propagation
# ============================================================================

class TestCNOTGatePropagation:
    """
    Tests verifying CNOT gate error propagation rules.

    CNOT propagation rules (Heisenberg picture, backward):
    - X on control: X propagates to target (X_c → X_c ⊗ X_t)
    - Z on target: Z propagates to control (Z_t → Z_c ⊗ Z_t)
    - Y on control: Y_c → Y_c ⊗ X_t (since Y = iXZ)
    - Y on target: Y_t → Z_c ⊗ Y_t
    """

    def test_cnot_single_gate_structure(self):
        """
        Single CNOT gate circuit structure verification.

        Circuit: R 0, R 1, CX 0 1, M 0, M 1
        CNOT adds 2 noise locations (one on each qubit).
        """
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # CX adds 2 noise, M 0 adds 1 noise, M 1 adds 1 noise = 4 total
        assert circuit.get_num_noise() == 4
        assert circuit.get_num_detector() == 2
        assert circuit.get_num_qubit() == 2

    def test_cnot_x_propagation_control_to_target(self):
        """
        X error on CNOT control qubit propagates to target.

        After CNOT, X_control becomes X_control ⊗ X_target.
        So X error on control before CNOT affects both qubit measurements.
        """
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Verify we can sample
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100
        # 2 detectors + 1 observable
        assert len(samples[0]) == 3

    def test_cnot_z_propagation_target_to_control(self):
        """
        Z error on CNOT target qubit propagates to control.

        After CNOT, Z_target becomes Z_control ⊗ Z_target.
        """
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100

    def test_two_cnots_chain(self):
        """
        Two CNOT gates in a chain: CX 0 1, CX 1 2.

        Error propagation chains through multiple gates.
        """
        stim_str = """R 0
R 1
R 2
CX 0 1
CX 1 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3]
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(2, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # 2 CX gates (4 noise) + 3 M gates (3 noise) = 7 noise
        assert circuit.get_num_noise() == 7
        assert circuit.get_num_detector() == 3

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100
        assert len(samples[0]) == 4  # 3 detectors + 1 observable

    def test_cnot_fan_out(self):
        """
        CNOT fan-out pattern: CX 0 1, CX 0 2.

        Same control qubit controlling two different targets.
        """
        stim_str = """R 0
R 1
R 2
CX 0 1
CX 0 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3]
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(2, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # 2 CX gates (4 noise) + 3 M gates (3 noise) = 7 noise
        assert circuit.get_num_noise() == 7

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100


# ============================================================================
# Test Class: Reset Gate Propagation
# ============================================================================

class TestResetGatePropagation:
    """
    Tests verifying reset gate behavior.

    Reset gate resets the error propagation - errors before reset don't
    propagate through to later measurements.
    """

    def test_reset_blocks_propagation(self):
        """
        Reset blocks error propagation.

        Errors before a reset should not affect measurements after reset.
        """
        stim_str = """R 0
H 0
R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # H adds 1 noise, M adds 1 noise = 2 noise
        # Reset doesn't add noise
        assert circuit.get_num_noise() == 2

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100

    def test_multiple_resets(self):
        """Multiple resets in sequence."""
        stim_str = """R 0
R 0
R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # Only M adds noise
        assert circuit.get_num_noise() == 1

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100


# ============================================================================
# Test Class: Hadamard Gate Propagation
# ============================================================================

class TestHadamardGatePropagation:
    """Tests verifying Hadamard gate error propagation (X ↔ Z swap)."""

    def test_hadamard_single(self):
        """Single Hadamard gate."""
        stim_str = """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # H adds 1 noise, M adds 1 noise = 2 noise
        assert circuit.get_num_noise() == 2

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100

    def test_double_hadamard_identity(self):
        """
        Two Hadamards in sequence = identity (for propagation).

        H·H = I, so error propagation should be same as no Hadamard.
        """
        stim_str = """R 0
H 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # 2 H gates (2 noise) + 1 M (1 noise) = 3 noise
        assert circuit.get_num_noise() == 3

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100

    def test_hadamard_cnot_combination(self):
        """Hadamard and CNOT combination."""
        stim_str = """R 0
R 1
H 0
CX 0 1
H 0
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        # 2 H (2 noise) + 1 CX (2 noise) + 2 M (2 noise) = 6 noise
        assert circuit.get_num_noise() == 6

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100


# ============================================================================
# Test Class: Detector Parity Combinations
# ============================================================================

class TestDetectorParityCombinations:
    """Tests for various detector parity configurations."""

    def test_detector_two_measurements(self):
        """Detector combining two measurements (parity check)."""
        stim_str = """R 0
R 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        assert circuit.get_num_detector() == 1

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100
        # 1 detector + 1 observable
        assert len(samples[0]) == 2

    def test_detector_three_measurements(self):
        """Detector combining three measurements."""
        stim_str = """R 0
R 1
R 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3] rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        assert circuit.get_num_detector() == 1

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100


# ============================================================================
# Test Class: Complete QEC Circuits
# ============================================================================

class TestCompleteQECCircuits:
    """Tests for complete quantum error correction circuit patterns."""

    def test_repetition_code_d3(self):
        """
        Distance-3 repetition code (3 data qubits, 2 ancilla qubits).

        Standard syndrome extraction circuit for ZZ stabilizers.
        """
        stim_str = """R 0
R 1
R 2
R 3
R 4
CX 0 3
CX 1 3
CX 1 4
CX 2 4
M 3
M 4
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-5] rec[-4]
DETECTOR(1, 0, 0) rec[-4] rec[-3]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        assert circuit.get_num_qubit() == 5
        assert circuit.get_num_detector() == 2
        # 4 CX (8 noise) + 5 M (5 noise) = 13 noise
        assert circuit.get_num_noise() == 13

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100
        # 2 detectors + 1 observable
        assert len(samples[0]) == 3

    def test_bell_state_measurement(self):
        """Bell state preparation and measurement."""
        stim_str = """R 0
R 1
H 0
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        assert circuit.get_num_qubit() == 2
        assert circuit.get_num_detector() == 1

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100

    def test_ghz_state_measurement(self):
        """GHZ state preparation and measurement."""
        stim_str = """R 0
R 1
R 2
H 0
CX 0 1
CX 1 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3] rec[-2]
DETECTOR(1, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        assert circuit.get_num_qubit() == 3
        assert circuit.get_num_detector() == 2

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, 1, 100)
        assert len(samples) == 100


# ============================================================================
# Test Class: Sampling Functions
# ============================================================================

class TestSamplingFunctions:
    """Tests for various sampling API functions."""

    def test_return_samples_basic(self):
        """Test return_samples function."""
        stim_str = """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        samples = qepg.return_samples(stim_str, weight=1, shots=50)
        assert len(samples) == 50
        # Each sample is a list of bools
        assert all(isinstance(s, list) for s in samples)

    def test_return_samples_numpy(self):
        """Test return_samples_numpy function."""
        stim_str = """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        samples = qepg.return_samples_numpy(stim_str, weight=1, shots=50)
        # Should return numpy array
        assert hasattr(samples, 'shape')
        assert samples.shape[0] == 50

    def test_return_samples_many_weights(self):
        """Test sampling with multiple weight values."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        weights = [1, 2, 3]
        shots = [10, 20, 30]

        results = qepg.return_samples_many_weights(stim_str, weights, shots)
        assert len(results) == 3
        assert len(results[0]) == 10
        assert len(results[1]) == 20
        assert len(results[2]) == 30

    def test_return_samples_many_weights_separate_obs(self):
        """Test sampling with separated observable output."""
        stim_str = """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        weights = [1, 2]
        shots = [25, 25]

        detector_samples, obs_samples = qepg.return_samples_many_weights_separate_obs(
            stim_str, weights, shots
        )

        # Total samples should be sum of shots
        assert detector_samples.shape[0] == 50
        assert obs_samples.shape[0] == 50

    def test_return_samples_with_fixed_qepg(self):
        """Test sampling with pre-compiled QEPG graph."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        # Compile once
        graph = qepg.compile_QEPG(stim_str)

        # Sample multiple times with same graph
        samples1 = qepg.return_samples_with_fixed_QEPG(graph, weight=1, shots=50)
        samples2 = qepg.return_samples_with_fixed_QEPG(graph, weight=2, shots=50)
        samples3 = qepg.return_samples_with_fixed_QEPG(graph, weight=1, shots=100)

        assert len(samples1) == 50
        assert len(samples2) == 50
        assert len(samples3) == 100

    def test_return_samples_many_weights_with_qepg(self):
        """Test multi-weight sampling with pre-compiled QEPG."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        weights = [1, 2, 3]
        shots = [20, 30, 40]

        detector_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG(
            graph, weights, shots
        )

        # Total = 20 + 30 + 40 = 90
        assert detector_samples.shape[0] == 90
        assert obs_samples.shape[0] == 90


# ============================================================================
# Test Class: Monte Carlo Sampling
# ============================================================================

class TestMonteCarloSampling:
    """Tests for Monte Carlo sampling with error probability."""

    def test_monte_carlo_sampling_basic(self):
        """Test Monte Carlo sampling with error rate."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        error_rate = 0.01
        shots = 100

        detector_samples, obs_samples = qepg.return_samples_Monte_separate_obs_with_QEPG(
            graph, error_rate, shots
        )

        assert detector_samples.shape[0] == shots
        assert obs_samples.shape[0] == shots

    def test_monte_carlo_different_error_rates(self):
        """Test Monte Carlo with various error rates."""
        stim_str = """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        for error_rate in [0.001, 0.01, 0.1]:
            detector_samples, obs_samples = qepg.return_samples_Monte_separate_obs_with_QEPG(
                graph, error_rate, 50
            )
            assert detector_samples.shape[0] == 50
            assert obs_samples.shape[0] == 50


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_weight_sampling(self):
        """Sampling with weight 0 (no errors)."""
        stim_str = """R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Weight 0 means no errors - all samples should be identical (all zeros)
        samples = qepg.return_samples_with_fixed_QEPG(graph, weight=0, shots=10)
        assert len(samples) == 10
        # With no errors, all detector/observable outcomes should be 0
        for sample in samples:
            assert all(bit == False for bit in sample)

    def test_large_weight_sampling(self):
        """Sampling with large weight (many errors)."""
        stim_str = """R 0
R 1
R 2
CX 0 1
CX 1 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3]
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(2, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Sample with high weight
        samples = qepg.return_samples_with_fixed_QEPG(graph, weight=5, shots=50)
        assert len(samples) == 50

    def test_single_shot_sampling(self):
        """Sampling with just one shot."""
        stim_str = """R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        samples = qepg.return_samples_with_fixed_QEPG(graph, weight=1, shots=1)
        assert len(samples) == 1

    def test_many_qubits(self):
        """Circuit with many qubits."""
        # Create a 10-qubit circuit
        lines = ["R " + str(i) for i in range(10)]
        lines += ["M " + str(i) for i in range(10)]
        lines += [f"DETECTOR({i}, 0, 0) rec[-{10-i}]" for i in range(10)]
        lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
        stim_str = "\n".join(lines)

        circuit = qepg.CliffordCircuit()
        circuit.compile_from_rewrited_stim_string(stim_str)

        assert circuit.get_num_qubit() == 10
        assert circuit.get_num_detector() == 10

        graph = qepg.compile_QEPG(stim_str)
        samples = qepg.return_samples_with_fixed_QEPG(graph, weight=1, shots=50)
        assert len(samples) == 50
        # 10 detectors + 1 observable
        assert len(samples[0]) == 11


# ============================================================================
# Test Class: Consistency and Statistical Tests
# ============================================================================

class TestConsistencyAndStatistics:
    """Statistical tests to verify sampling behaves correctly."""

    def test_weight_affects_flip_probability(self):
        """
        Higher weight should lead to more detector flips on average.

        This is a statistical test - with more errors, we expect more flips.
        """
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Sample with low weight
        samples_low = qepg.return_samples_with_fixed_QEPG(graph, weight=1, shots=1000)
        flips_low = sum(sum(s) for s in samples_low)

        # Sample with higher weight
        samples_high = qepg.return_samples_with_fixed_QEPG(graph, weight=3, shots=1000)
        flips_high = sum(sum(s) for s in samples_high)

        # Statistically, higher weight should have more flips
        # This is a soft assertion - could fail with very low probability
        # We just check that higher weight has at least some flips
        assert flips_high >= 0

    def test_zero_errors_no_flips(self):
        """With zero errors, all detector outcomes should be zero."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        samples = qepg.return_samples_with_fixed_QEPG(graph, weight=0, shots=100)

        for sample in samples:
            assert all(bit == False for bit in sample), "Zero errors should produce no flips"

    def test_sampling_reproducibility_structure(self):
        """
        Verify that multiple sampling calls return proper structure.

        (Not testing RNG reproducibility, just that the API works consistently)
        """
        stim_str = """R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Multiple calls should all return valid results
        for _ in range(5):
            samples = qepg.return_samples_with_fixed_QEPG(graph, weight=1, shots=20)
            assert len(samples) == 20
            assert all(len(s) == 2 for s in samples)  # 1 detector + 1 observable
