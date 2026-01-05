"""
Tests for scalerqec.QEC.small module.

Tests the predefined small QEC codes: fivequbitCode, steaneCode, ShorCode.
"""
import pytest
from scalerqec.QEC.small import fivequbitCode, steaneCode, ShorCode
from scalerqec.util import commute


class TestFiveQubitCode:
    """Tests for the five-qubit [[5,1,3]] code."""

    def test_initialization(self):
        """Test five-qubit code initialization."""
        code = fivequbitCode("Standard")
        assert code.n == 5
        assert code.k == 1
        assert code.d == 3

    def test_stabilizers(self):
        """Test five-qubit code stabilizers."""
        code = fivequbitCode("Standard")
        assert len(code.stabilizers) == 4
        assert "XZZXI" in code.stabilizers
        assert "IXZZX" in code.stabilizers
        assert "XIXZZ" in code.stabilizers
        assert "ZXIXZ" in code.stabilizers

    def test_stabilizers_commute(self):
        """Test that all stabilizers commute pairwise."""
        code = fivequbitCode("Standard")
        for i, s1 in enumerate(code.stabilizers):
            for j, s2 in enumerate(code.stabilizers):
                assert commute(s1, s2) is True, \
                    f"Stabilizers {i} and {j} should commute"

    def test_logical_z_set(self):
        """Test that logical Z is set correctly."""
        code = fivequbitCode("Standard")
        assert code._logicalZ[0] == "ZZZZZ"

    def test_logical_z_commutes_with_stabilizers(self):
        """Test logical Z commutes with all stabilizers."""
        code = fivequbitCode("Standard")
        logical_z = code._logicalZ[0]
        for stab in code.stabilizers:
            assert commute(logical_z, stab) is True

    def test_compile_circuit(self):
        """Test compiling five-qubit code circuit."""
        code = fivequbitCode("Standard")
        code.rounds = 2
        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None
        assert code.stimcirc is not None


class TestSteaneCode:
    """Tests for the Steane [[7,1,3]] code."""

    def test_initialization(self):
        """Test Steane code initialization."""
        code = steaneCode("Standard")
        assert code.n == 7
        assert code.k == 1
        assert code.d == 3

    def test_stabilizers(self):
        """Test Steane code stabilizers."""
        code = steaneCode("Standard")
        assert len(code.stabilizers) == 6
        # X-type stabilizers
        assert "IIIXXXX" in code.stabilizers
        assert "IXXIIXX" in code.stabilizers
        assert "XIXIXIX" in code.stabilizers
        # Z-type stabilizers
        assert "IIIZZZZ" in code.stabilizers
        assert "IZZIIZZ" in code.stabilizers
        assert "ZIZIZIZ" in code.stabilizers

    def test_stabilizers_commute(self):
        """Test that all stabilizers commute pairwise."""
        code = steaneCode("Standard")
        for i, s1 in enumerate(code.stabilizers):
            for j, s2 in enumerate(code.stabilizers):
                assert commute(s1, s2) is True, \
                    f"Stabilizers {i} and {j} should commute"

    def test_is_css_code(self):
        """Test that Steane code is a CSS code (X and Z type stabilizers)."""
        code = steaneCode("Standard")
        x_type = [s for s in code.stabilizers if 'X' in s and 'Z' not in s]
        z_type = [s for s in code.stabilizers if 'Z' in s and 'X' not in s]
        # CSS code has only X-type or only Z-type stabilizers (no mixed)
        assert len(x_type) == 3
        assert len(z_type) == 3

    def test_logical_z_set(self):
        """Test that logical Z is set correctly."""
        code = steaneCode("Standard")
        assert code._logicalZ[0] == "ZZZZZZZ"

    def test_logical_z_commutes_with_stabilizers(self):
        """Test logical Z commutes with all stabilizers."""
        code = steaneCode("Standard")
        logical_z = code._logicalZ[0]
        for stab in code.stabilizers:
            assert commute(logical_z, stab) is True

    def test_compile_circuit(self):
        """Test compiling Steane code circuit."""
        code = steaneCode("Standard")
        code.rounds = 2
        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None
        assert code.stimcirc is not None


class TestShorCode:
    """Tests for the Shor [[9,1,3]] code."""

    def test_initialization(self):
        """Test Shor code initialization."""
        code = ShorCode("Standard")
        assert code.n == 9
        assert code.k == 1
        assert code.d == 3

    def test_stabilizers(self):
        """Test Shor code stabilizers."""
        code = ShorCode("Standard")
        assert len(code.stabilizers) == 8
        # Z-type stabilizers (6)
        assert "ZZIIIIIII" in code.stabilizers
        assert "IZZIIIIII" in code.stabilizers
        # X-type stabilizers (2)
        assert "XXXXXXIII" in code.stabilizers
        assert "XXXIIIXXX" in code.stabilizers

    def test_stabilizers_commute(self):
        """Test that all stabilizers commute pairwise."""
        code = ShorCode("Standard")
        for i, s1 in enumerate(code.stabilizers):
            for j, s2 in enumerate(code.stabilizers):
                assert commute(s1, s2) is True, \
                    f"Stabilizers {i} and {j} should commute"

    def test_logical_z_set(self):
        """Test that logical Z is set correctly."""
        code = ShorCode("Standard")
        assert code._logicalZ[0] == "ZZZZZZZZZ"

    def test_logical_z_commutes_with_stabilizers(self):
        """Test logical Z commutes with all stabilizers."""
        code = ShorCode("Standard")
        logical_z = code._logicalZ[0]
        for stab in code.stabilizers:
            assert commute(logical_z, stab) is True

    def test_compile_circuit(self):
        """Test compiling Shor code circuit."""
        code = ShorCode("Standard")
        code.rounds = 2
        code.construct_IR_standard_scheme()
        code.compile_stim_circuit_from_IR_standard()

        assert code.circuit is not None
        assert code.stimcirc is not None


class TestSmallCodesCompile:
    """Tests that all small codes compile without error."""

    def test_all_codes_compile_standard_scheme(self):
        """Test all codes compile with standard scheme."""
        codes = [
            fivequbitCode("Standard"),
            steaneCode("Standard"),
            ShorCode("Standard"),
        ]

        for code in codes:
            code.rounds = 2
            code.construct_IR_standard_scheme()
            code.compile_stim_circuit_from_IR_standard()

            assert code.circuit is not None, \
                f"{code.__class__.__name__} should compile successfully"

    def test_all_codes_have_valid_stim_circuit(self):
        """Test all codes produce valid STIM circuits."""
        codes = [
            fivequbitCode("Standard"),
            steaneCode("Standard"),
            ShorCode("Standard"),
        ]

        for code in codes:
            code.rounds = 2
            code.construct_IR_standard_scheme()
            code.compile_stim_circuit_from_IR_standard()

            stim_str = str(code.stimcirc)
            # Should have measurements
            assert "M" in stim_str, \
                f"{code.__class__.__name__} should have measurements"


class TestSmallCodesParameters:
    """Tests for small codes parameters."""

    def test_five_qubit_code_parameters(self):
        """Test [[5,1,3]] parameters."""
        code = fivequbitCode("Standard")
        # n - k = number of stabilizers
        assert code.n - code.k == len(code.stabilizers)

    def test_steane_code_parameters(self):
        """Test [[7,1,3]] parameters."""
        code = steaneCode("Standard")
        # n - k = number of stabilizers
        assert code.n - code.k == len(code.stabilizers)

    def test_shor_code_parameters(self):
        """Test [[9,1,3]] parameters."""
        code = ShorCode("Standard")
        # n - k = number of stabilizers
        assert code.n - code.k == len(code.stabilizers)


class TestSmallCodesStabilizerWeights:
    """Tests for stabilizer weights in small codes."""

    def test_five_qubit_code_stabilizer_weights(self):
        """Test weight of five-qubit code stabilizers."""
        code = fivequbitCode("Standard")
        for stab in code.stabilizers:
            weight = sum(1 for c in stab if c != 'I')
            # Five-qubit code stabilizers have weight 4
            assert weight == 4

    def test_steane_code_stabilizer_weights(self):
        """Test weight of Steane code stabilizers."""
        code = steaneCode("Standard")
        for stab in code.stabilizers:
            weight = sum(1 for c in stab if c != 'I')
            # Steane code stabilizers have weight 4
            assert weight == 4

    def test_logical_z_weights(self):
        """Test logical Z weights."""
        codes = [
            (fivequbitCode("Standard"), 5),
            (steaneCode("Standard"), 7),
            (ShorCode("Standard"), 9),
        ]
        for code, expected_weight in codes:
            logical_z = code._logicalZ[0]
            weight = sum(1 for c in logical_z if c != 'I')
            assert weight == expected_weight
