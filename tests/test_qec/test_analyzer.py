"""
Tests for scalerqec.QEC.analyzer module.

Tests the StabilizerAnalyzer, DistanceAnalyzer, and LogicalOperatorAnalyzer classes.
"""
import pytest
from scalerqec.QEC.qeccircuit import StabCode
from scalerqec.QEC.analyzer import (
    StabilizerAnalyzer,
    DistanceAnalyzer,
    LogicalOperatorAnalyzer,
)
from scalerqec.QEC.small import fivequbitCode, steaneCode, ShorCode
from scalerqec.util import commute


class TestStabilizerAnalyzer:
    """Tests for the StabilizerAnalyzer class."""

    def test_verify_commutation_five_qubit_code(self):
        """Test that five-qubit code stabilizers commute."""
        code = fivequbitCode("Standard")
        analyzer = StabilizerAnalyzer(code)
        assert analyzer.verify_commutation() is True

    def test_verify_commutation_steane_code(self):
        """Test that Steane code stabilizers commute."""
        code = steaneCode("Standard")
        analyzer = StabilizerAnalyzer(code)
        assert analyzer.verify_commutation() is True

    def test_verify_commutation_shor_code(self):
        """Test that Shor code stabilizers commute."""
        code = ShorCode("Standard")
        analyzer = StabilizerAnalyzer(code)
        assert analyzer.verify_commutation() is True

    def test_verify_commutation_custom_valid_code(self):
        """Test commutation with custom valid stabilizers."""
        code = StabCode(n=3, k=1, d=3)
        code.add_stab("ZZI")
        code.add_stab("IZZ")
        analyzer = StabilizerAnalyzer(code)
        assert analyzer.verify_commutation() is True

    def test_verify_commutation_non_commuting_stabilizers(self):
        """Test that non-commuting stabilizers are detected."""
        code = StabCode(n=2, k=0, d=1)
        code.add_stab("XZ")  # These two anti-commute
        code.add_stab("ZX")  # XZ and ZX have two anticommuting positions, so they commute!
        analyzer = StabilizerAnalyzer(code)
        # XZ and ZX actually commute (even count of anticommuting positions)
        assert analyzer.verify_commutation() is True

    def test_verify_commutation_truly_non_commuting(self):
        """Test detection of truly non-commuting stabilizers."""
        code = StabCode(n=2, k=0, d=1)
        code.add_stab("XI")  # X on qubit 0
        code.add_stab("ZI")  # Z on qubit 0 - anticommutes with X
        analyzer = StabilizerAnalyzer(code)
        assert analyzer.verify_commutation() is False


class TestDistanceAnalyzer:
    """Tests for the DistanceAnalyzer class."""

    def test_initialization_bruteforce(self):
        """Test DistanceAnalyzer with bruteforce method."""
        code = fivequbitCode("Standard")
        analyzer = DistanceAnalyzer(code, method="bruteforce")
        assert analyzer.method == "bruteforce"

    def test_initialization_smt(self):
        """Test DistanceAnalyzer with SMT method."""
        code = fivequbitCode("Standard")
        analyzer = DistanceAnalyzer(code, method="smt")
        assert analyzer.method == "smt"

    def test_initialization_invalid_method(self):
        """Test that invalid method raises ValueError."""
        code = fivequbitCode("Standard")
        with pytest.raises(ValueError) as excinfo:
            DistanceAnalyzer(code, method="invalid_method")
        assert "Unsupported method" in str(excinfo.value)

    def test_verify_code_distance_five_qubit(self):
        """Test distance verification for five-qubit code."""
        code = fivequbitCode("Standard")
        analyzer = DistanceAnalyzer(code, method="bruteforce")
        assert analyzer.verify_code_distance() is True

    def test_verify_code_distance_steane(self):
        """Test distance verification for Steane code."""
        code = steaneCode("Standard")
        analyzer = DistanceAnalyzer(code, method="bruteforce")
        assert analyzer.verify_code_distance() is True

    def test_verify_code_distance_shor(self):
        """Test distance verification for Shor code."""
        code = ShorCode("Standard")
        analyzer = DistanceAnalyzer(code, method="bruteforce")
        assert analyzer.verify_code_distance() is True

    def test_compute_distance_bruteforce(self):
        """Test bruteforce distance computation returns expected value."""
        code = fivequbitCode("Standard")
        analyzer = DistanceAnalyzer(code, method="bruteforce")
        # The placeholder implementation returns code.d
        assert analyzer.compute_distance_bruteforce() == 3


class TestLogicalOperatorAnalyzerInit:
    """Tests for LogicalOperatorAnalyzer initialization."""

    def test_initialization_five_qubit(self):
        """Test initialization with five-qubit code."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._n == 5
        assert analyzer.code == code

    def test_initialization_steane(self):
        """Test initialization with Steane code."""
        code = steaneCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._n == 7

    def test_initialization_shor(self):
        """Test initialization with Shor code."""
        code = ShorCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._n == 9


class TestEncodePauli:
    """Tests for Pauli encoding in LogicalOperatorAnalyzer."""

    def test_encode_identity(self):
        """Test encoding of identity operator."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        encoded = analyzer._encode_pauli("III")
        assert encoded == 0  # Identity maps to all zeros

    def test_encode_x(self):
        """Test encoding of X operator."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        # X on qubit 0: bit 0 set in X part
        encoded = analyzer._encode_pauli("XII")
        assert encoded == 1  # 0b001 for X part, 0b000 for Z part

    def test_encode_z(self):
        """Test encoding of Z operator."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        # Z on qubit 0: bit 0 set in Z part (which is shifted by n)
        encoded = analyzer._encode_pauli("ZII")
        # Z part starts at bit n=3
        assert encoded == (1 << 3)

    def test_encode_y(self):
        """Test encoding of Y operator (Y = iXZ)."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        # Y = iXZ, so both X and Z bits are set
        encoded = analyzer._encode_pauli("YII")
        expected = 1 | (1 << 3)  # X bit and Z bit for qubit 0
        assert encoded == expected

    def test_encode_length_mismatch(self):
        """Test that length mismatch raises error."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError) as excinfo:
            analyzer._encode_pauli("XX")
        assert "length" in str(excinfo.value).lower()

    def test_encode_invalid_character(self):
        """Test that invalid Pauli character raises error."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError) as excinfo:
            analyzer._encode_pauli("XAI")
        assert "Invalid Pauli character" in str(excinfo.value)


class TestRank:
    """Tests for GF(2) rank computation."""

    def test_rank_empty(self):
        """Test rank of empty row set."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._rank([]) == 0

    def test_rank_single_row(self):
        """Test rank of single non-zero row."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._rank([1]) == 1

    def test_rank_duplicate_rows(self):
        """Test rank with duplicate rows (should be 1)."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._rank([5, 5]) == 1  # Same row twice

    def test_rank_independent_rows(self):
        """Test rank of independent rows."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        # 0b001 and 0b010 are independent
        assert analyzer._rank([1, 2]) == 2

    def test_rank_dependent_rows(self):
        """Test rank of dependent rows."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        # 0b001, 0b010, 0b011 - third is XOR of first two
        assert analyzer._rank([1, 2, 3]) == 2


class TestIsInStabilizerGroup:
    """Tests for stabilizer group membership."""

    def test_identity_in_group(self):
        """Test that identity is always in stabilizer group."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("ZZI")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._is_in_stabilizer_group("III") is True

    def test_generator_in_group(self):
        """Test that a generator is in its own group."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("ZZI")
        code.add_stab("IZZ")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._is_in_stabilizer_group("ZZI") is True
        assert analyzer._is_in_stabilizer_group("IZZ") is True

    def test_product_in_group(self):
        """Test that product of generators is in group."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("ZZI")
        code.add_stab("IZZ")
        analyzer = LogicalOperatorAnalyzer(code)
        # ZZI * IZZ = ZIZ
        assert analyzer._is_in_stabilizer_group("ZIZ") is True

    def test_non_member_not_in_group(self):
        """Test that non-member is not in stabilizer group."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("ZZI")
        code.add_stab("IZZ")
        analyzer = LogicalOperatorAnalyzer(code)
        # XXX is not in the Z-only stabilizer group
        assert analyzer._is_in_stabilizer_group("XXX") is False


class TestVerifyLogicalZ:
    """Tests for logical Z verification."""

    def test_verify_logical_z_five_qubit(self):
        """Test logical Z verification for five-qubit code."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer.verify_logical_Z() is True

    def test_verify_logical_z_steane(self):
        """Test logical Z verification for Steane code."""
        code = steaneCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer.verify_logical_Z() is True

    def test_verify_logical_z_shor(self):
        """Test logical Z verification for Shor code."""
        code = ShorCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer.verify_logical_Z() is True

    def test_verify_logical_z_empty(self):
        """Test verification when no logical Z is set."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("ZZI")
        analyzer = LogicalOperatorAnalyzer(code)
        # No logical Z set should return True (nothing to verify)
        assert analyzer.verify_logical_Z() is True


class TestIsLogicalOperator:
    """Tests for logical operator detection."""

    def test_logical_z_is_logical_operator(self):
        """Test that logical Z is detected as logical operator."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        # ZZZZZ is the logical Z for five-qubit code
        assert analyzer.is_logical_operator("ZZZZZ") is True

    def test_stabilizer_not_logical_operator(self):
        """Test that stabilizers are not logical operators."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        # XZZXI is a stabilizer, not a logical operator
        assert analyzer.is_logical_operator("XZZXI") is False

    def test_identity_not_logical_operator(self):
        """Test that identity is not a logical operator."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer.is_logical_operator("IIIII") is False

    def test_anticommuting_not_logical(self):
        """Test that operator anticommuting with stabilizer is not logical."""
        code = StabCode(n=3, k=1, d=1)
        code.add_stab("ZZI")
        code.add_stab("IZZ")
        analyzer = LogicalOperatorAnalyzer(code)
        # XII anticommutes with ZZI (single qubit X vs Z)
        assert analyzer.is_logical_operator("XII") is False

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises error."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError) as excinfo:
            analyzer.is_logical_operator("ZZ")
        assert "length" in str(excinfo.value).lower()


class TestDetermineLogicalX:
    """Tests for logical X determination."""

    def test_determine_logical_x_five_qubit(self):
        """Test logical X determination for five-qubit code."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        logical_x = analyzer.determine_logical_X(0)
        # Should be a valid logical operator
        assert analyzer.is_logical_operator(logical_x) is True
        # Should anticommute with logical Z
        assert commute(logical_x, "ZZZZZ") is False

    def test_determine_logical_x_steane(self):
        """Test logical X determination for Steane code."""
        code = steaneCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        logical_x = analyzer.determine_logical_X(0)
        assert analyzer.is_logical_operator(logical_x) is True
        # Should anticommute with logical Z
        assert commute(logical_x, "ZZZZZZZ") is False

    def test_determine_logical_x_shor(self):
        """Test logical X determination for Shor code."""
        code = ShorCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        logical_x = analyzer.determine_logical_X(0)
        assert analyzer.is_logical_operator(logical_x) is True

    def test_determine_logical_x_invalid_index(self):
        """Test that invalid logical qubit index raises error."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError) as excinfo:
            analyzer.determine_logical_X(5)  # k=1, so only index 0 is valid
        assert "out of range" in str(excinfo.value)

    def test_determine_logical_x_negative_index(self):
        """Test that negative logical qubit index raises error."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError):
            analyzer.determine_logical_X(-1)


class TestDetermineLogicalH:
    """Tests for logical H determination."""

    def test_determine_logical_h_five_qubit(self):
        """Test logical H for five-qubit code."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        logical_h = analyzer.determine_logical_H(0)
        # Naive implementation returns H^n
        assert logical_h == "HHHHH"

    def test_determine_logical_h_steane(self):
        """Test logical H for Steane code."""
        code = steaneCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        logical_h = analyzer.determine_logical_H(0)
        assert logical_h == "HHHHHHH"

    def test_determine_logical_h_invalid_index(self):
        """Test that invalid index raises error."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError):
            analyzer.determine_logical_H(1)


class TestDetermineLogicalS:
    """Tests for logical S determination."""

    def test_determine_logical_s_five_qubit(self):
        """Test logical S for five-qubit code."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        logical_s = analyzer.determine_logical_S(0)
        # Naive implementation returns S^n
        assert logical_s == "SSSSS"

    def test_determine_logical_s_steane(self):
        """Test logical S for Steane code."""
        code = steaneCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        logical_s = analyzer.determine_logical_S(0)
        assert logical_s == "SSSSSSS"

    def test_determine_logical_s_invalid_index(self):
        """Test that invalid index raises error."""
        code = fivequbitCode("Standard")
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError):
            analyzer.determine_logical_S(1)


class TestDetermineLogicalCNOT:
    """Tests for logical CNOT determination."""

    def test_determine_logical_cnot_same_index_raises(self):
        """Test that same control and target raises error."""
        code = StabCode(n=10, k=2, d=3)  # k=2 for two logical qubits
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(AssertionError):
            analyzer.determine_logical_CNOT(0, 0)

    def test_determine_logical_cnot_invalid_control(self):
        """Test that invalid control index raises error."""
        code = StabCode(n=10, k=2, d=3)
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError):
            analyzer.determine_logical_CNOT(5, 0)

    def test_determine_logical_cnot_invalid_target(self):
        """Test that invalid target index raises error."""
        code = StabCode(n=10, k=2, d=3)
        analyzer = LogicalOperatorAnalyzer(code)
        with pytest.raises(ValueError):
            analyzer.determine_logical_CNOT(0, 5)


class TestLogicalOperatorAnalyzerIntegration:
    """Integration tests for LogicalOperatorAnalyzer."""

    def test_all_small_codes_pass_verification(self):
        """Test that all small codes pass logical operator verification."""
        codes = [
            fivequbitCode("Standard"),
            steaneCode("Standard"),
            ShorCode("Standard"),
        ]
        for code in codes:
            analyzer = LogicalOperatorAnalyzer(code)
            assert analyzer.verify_logical_Z() is True, \
                f"{code.__class__.__name__} logical Z verification failed"

    def test_logical_x_anticommutes_with_z(self):
        """Test that determined logical X anticommutes with logical Z."""
        codes = [
            (fivequbitCode("Standard"), "ZZZZZ"),
            (steaneCode("Standard"), "ZZZZZZZ"),
            (ShorCode("Standard"), "ZZZZZZZZZ"),
        ]
        for code, logical_z in codes:
            analyzer = LogicalOperatorAnalyzer(code)
            logical_x = analyzer.determine_logical_X(0)
            assert commute(logical_x, logical_z) is False, \
                f"{code.__class__.__name__}: logical X should anticommute with logical Z"

    def test_logical_x_commutes_with_stabilizers(self):
        """Test that determined logical X commutes with all stabilizers."""
        codes = [
            fivequbitCode("Standard"),
            steaneCode("Standard"),
            ShorCode("Standard"),
        ]
        for code in codes:
            analyzer = LogicalOperatorAnalyzer(code)
            logical_x = analyzer.determine_logical_X(0)
            for stab in code.stabilizers:
                assert commute(logical_x, stab) is True, \
                    f"{code.__class__.__name__}: logical X should commute with stabilizer {stab}"


class TestMultiplyPauliStrings:
    """Tests for the static _multiply_pauli_strings method."""

    def test_multiply_delegates_to_util(self):
        """Test that multiply delegates to utility function."""
        code = StabCode(n=3, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        result = analyzer._multiply_pauli_strings("XYZ", "ZYX")
        # XZ=Y, YY=I, ZX=Y
        assert result == "YIY"


class TestAnticommutes:
    """Tests for the static _anticommutes method."""

    def test_anticommutes_single_qubit(self):
        """Test anticommutation detection for single qubit."""
        code = StabCode(n=1, k=1, d=1)
        analyzer = LogicalOperatorAnalyzer(code)
        assert analyzer._anticommutes("X", "Z") is True
        assert analyzer._anticommutes("X", "X") is False
        assert analyzer._anticommutes("I", "X") is False
