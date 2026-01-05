"""
Tests for scalerqec.QEC.qeccircuit module.

Tests the StabCode class and circuit compilation, including Y Pauli support.
"""
import pytest
from scalerqec.QEC.qeccircuit import StabCode, SCHEME
from scalerqec.util import commute


class TestStabCodeInitialization:
    """Tests for StabCode initialization."""

    def test_basic_initialization(self):
        """Test basic StabCode creation with n, k, d parameters."""
        code = StabCode(n=5, k=1, d=3)
        assert code.n == 5
        assert code.k == 1
        assert code.d == 3

    def test_initialization_different_parameters(self):
        """Test StabCode with different parameters."""
        code = StabCode(n=7, k=1, d=3)
        assert code.n == 7
        assert code.k == 1
        assert code.d == 3

    def test_initialization_larger_code(self):
        """Test StabCode with larger parameters."""
        code = StabCode(n=9, k=1, d=3)
        assert code.n == 9


class TestAddStabilizer:
    """Tests for adding stabilizers to StabCode."""

    def test_add_single_stabilizer(self):
        """Test adding a single stabilizer."""
        code = StabCode(n=5, k=1, d=3)
        code.add_stab("XZZXI")
        assert len(code.stabilizers) == 1
        assert code.stabilizers[0] == "XZZXI"

    def test_add_multiple_stabilizers(self):
        """Test adding multiple stabilizers."""
        code = StabCode(n=5, k=1, d=3)
        code.add_stab("XZZXI")
        code.add_stab("IXZZX")
        code.add_stab("XIXZZ")
        code.add_stab("ZXIXZ")
        assert len(code.stabilizers) == 4

    def test_add_stabilizers_five_qubit_code(self, five_qubit_stabilizers):
        """Test adding stabilizers for five-qubit code."""
        code = StabCode(n=5, k=1, d=3)
        for stab in five_qubit_stabilizers:
            code.add_stab(stab)
        assert len(code.stabilizers) == 4

    def test_add_stabilizers_steane_code(self, steane_stabilizers):
        """Test adding stabilizers for Steane code."""
        code = StabCode(n=7, k=1, d=3)
        for stab in steane_stabilizers:
            code.add_stab(stab)
        assert len(code.stabilizers) == 6


class TestSetLogicalZ:
    """Tests for setting logical Z operators."""

    def test_set_logical_z(self):
        """Test setting logical Z operator."""
        code = StabCode(n=5, k=1, d=3)
        code.set_logical_Z(0, "ZZZZZ")
        # Access internal state to verify
        assert 0 in code._logicalZ
        assert code._logicalZ[0] == "ZZZZZ"

    def test_set_logical_z_steane(self):
        """Test setting logical Z for Steane code."""
        code = StabCode(n=7, k=1, d=3)
        code.set_logical_Z(0, "ZZZZZZZ")
        assert code._logicalZ[0] == "ZZZZZZZ"


class TestLogicalZValidationError:
    """Tests for logical Z validation error."""

    def test_construct_without_logical_z_raises_error(self):
        """Test that constructing without logical Z raises ValueError."""
        code = StabCode(n=5, k=1, d=3)
        code.add_stab("XZZXI")
        code.add_stab("IXZZX")
        code.add_stab("XIXZZ")
        code.add_stab("ZXIXZ")
        # Don't set logical Z
        code.scheme = "Standard"
        code.rounds = 2

        with pytest.raises(ValueError) as excinfo:
            code.construct_IR_standard_scheme()

        assert "Logical Z operator" in str(excinfo.value)
        assert "not defined" in str(excinfo.value)


class TestConstructIRStandard:
    """Tests for IR construction with standard scheme."""

    def test_construct_ir_five_qubit_code(self, five_qubit_stabilizers, five_qubit_logical_z):
        """Test IR construction for five-qubit code."""
        code = StabCode(n=5, k=1, d=3)
        for stab in five_qubit_stabilizers:
            code.add_stab(stab)
        code.set_logical_Z(0, five_qubit_logical_z)
        code.scheme = "Standard"
        code.rounds = 2

        code.construct_IR_standard_scheme()

        # Should have IR list populated
        assert len(code._IRList) > 0

    def test_construct_ir_steane_code(self, steane_stabilizers, steane_logical_z):
        """Test IR construction for Steane code."""
        code = StabCode(n=7, k=1, d=3)
        for stab in steane_stabilizers:
            code.add_stab(stab)
        code.set_logical_Z(0, steane_logical_z)
        code.scheme = "Standard"
        code.rounds = 2

        code.construct_IR_standard_scheme()

        assert len(code._IRList) > 0


class TestCompileCircuitZStabilizer:
    """Tests for compiling circuits with Z-type stabilizers."""

    def test_z_stabilizer_propagation(self):
        """Test Z stabilizer circuit compilation."""
        code = StabCode(n=3, k=1, d=3)
        code.add_stab("ZZI")
        code.add_stab("IZZ")
        code.set_logical_Z(0, "ZZZ")
        code.scheme = "Standard"
        code.rounds = 2

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        # Should have circuit
        assert code.circuit is not None
        stim_str = str(code.stimcirc)
        # Z stabilizers use CNOT for propagation
        assert "CX" in stim_str or "CNOT" in stim_str


class TestCompileCircuitXStabilizer:
    """Tests for compiling circuits with X-type stabilizers."""

    def test_x_stabilizer_propagation(self):
        """Test X stabilizer circuit compilation."""
        code = StabCode(n=3, k=1, d=3)
        code.add_stab("XXI")
        code.add_stab("IXX")
        code.set_logical_Z(0, "ZZZ")
        code.scheme = "Standard"
        code.rounds = 2

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None
        stim_str = str(code.stimcirc)
        # X stabilizers use H-CNOT-H pattern
        assert "H" in stim_str


class TestCompileCircuitYStabilizer:
    """Tests for compiling circuits with Y-type stabilizers (NEW FIX)."""

    def test_y_stabilizer_propagation_simple(self):
        """Test Y stabilizer circuit compilation - simple case."""
        code = StabCode(n=2, k=1, d=1)
        code.add_stab("YI")
        code.set_logical_Z(0, "ZZ")
        code.scheme = "Standard"
        code.rounds = 1

        # This should NOT raise NotImplementedError anymore
        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None

    def test_y_stabilizer_propagation_multi_qubit(self):
        """Test Y stabilizer with multiple qubits."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("YYI")
        code.set_logical_Z(0, "ZZZ")
        code.scheme = "Standard"
        code.rounds = 1

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None
        stim_str = str(code.stimcirc)
        # Y = iXZ, so should have H gates (for X part) and CNOTs (for Z part)
        assert "H" in stim_str

    def test_y_stabilizer_single_y(self):
        """Test stabilizer with single Y operator."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("IYI")
        code.set_logical_Z(0, "ZZZ")
        code.scheme = "Standard"
        code.rounds = 1

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None


class TestCompileCircuitMixedStabilizers:
    """Tests for compiling circuits with mixed X/Y/Z stabilizers."""

    def test_mixed_xz_stabilizers(self):
        """Test mixed X and Z stabilizers."""
        code = StabCode(n=5, k=1, d=3)
        code.add_stab("XZZXI")
        code.add_stab("IXZZX")
        code.add_stab("XIXZZ")
        code.add_stab("ZXIXZ")
        code.set_logical_Z(0, "ZZZZZ")
        code.scheme = "Standard"
        code.rounds = 2

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None

    def test_mixed_xyz_stabilizers(self):
        """Test mixed X, Y, and Z stabilizers."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("XYZ")
        code.set_logical_Z(0, "ZZZ")
        code.scheme = "Standard"
        code.rounds = 1

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None

    def test_all_y_stabilizer(self):
        """Test stabilizer with all Y operators."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("YYY")
        code.set_logical_Z(0, "ZZZ")
        code.scheme = "Standard"
        code.rounds = 1

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None


class TestCommuteFunction:
    """Tests for the commute function used in StabCode."""

    def test_commute_imported_from_util(self):
        """Test that commute function works correctly."""
        # These should commute (even number of anticommuting positions)
        assert commute("XZZXI", "IXZZX") is True

        # Single qubit different paulis should anticommute
        assert commute("X", "Z") is False

    def test_stabilizers_commute_pairwise(self, five_qubit_stabilizers):
        """Test that all stabilizers commute pairwise."""
        for i, s1 in enumerate(five_qubit_stabilizers):
            for j, s2 in enumerate(five_qubit_stabilizers):
                assert commute(s1, s2) is True


class TestSchemeEnum:
    """Tests for the SCHEME enum."""

    def test_scheme_values(self):
        """Test SCHEME enum values."""
        assert SCHEME.STANDARD.value == 0
        assert SCHEME.SHOR.value == 1
        assert SCHEME.KNILL.value == 2
        assert SCHEME.FLAG.value == 3


class TestCircuitProperties:
    """Tests for circuit properties after compilation."""

    def test_circuit_property(self, five_qubit_stabilizers, five_qubit_logical_z):
        """Test circuit property returns CliffordCircuit."""
        code = StabCode(n=5, k=1, d=3)
        for stab in five_qubit_stabilizers:
            code.add_stab(stab)
        code.set_logical_Z(0, five_qubit_logical_z)
        code.scheme = "Standard"
        code.rounds = 2

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        # circuit property should return the compiled circuit
        assert code.circuit is not None

    def test_stimcirc_property(self, five_qubit_stabilizers, five_qubit_logical_z):
        """Test stimcirc property returns STIM circuit."""
        code = StabCode(n=5, k=1, d=3)
        for stab in five_qubit_stabilizers:
            code.add_stab(stab)
        code.set_logical_Z(0, five_qubit_logical_z)
        code.scheme = "Standard"
        code.rounds = 2

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        # stimcirc property should return the STIM circuit
        assert code.stimcirc is not None


class TestRepetitionCode:
    """Tests for repetition code (simple Z-only stabilizers)."""

    def test_repetition_code_d3(self):
        """Test distance-3 repetition code."""
        code = StabCode(n=3, k=1, d=3)
        code.add_stab("ZZI")
        code.add_stab("IZZ")
        code.set_logical_Z(0, "ZZZ")
        code.scheme = "Standard"
        code.rounds = 3

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None
        assert code.stimcirc is not None

    def test_repetition_code_d5(self):
        """Test distance-5 repetition code."""
        code = StabCode(n=5, k=1, d=5)
        code.add_stab("ZZIII")
        code.add_stab("IZZII")
        code.add_stab("IIZZI")
        code.add_stab("IIIZZ")
        code.set_logical_Z(0, "ZZZZZ")
        code.scheme = "Standard"
        code.rounds = 3

        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None
