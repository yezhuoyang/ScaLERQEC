"""
Tests for scalerqec.util.pauli module.

Tests the commute(), anticommute(), and multiply_pauli_strings() functions.
"""
import pytest
from scalerqec.util import commute, anticommute, multiply_pauli_strings


class TestCommute:
    """Tests for the commute() function."""

    def test_commute_identity_with_all(self):
        """Identity commutes with all Pauli operators."""
        assert commute("I", "I") is True
        assert commute("I", "X") is True
        assert commute("I", "Y") is True
        assert commute("I", "Z") is True

    def test_commute_same_pauli(self):
        """Same Pauli operators commute with themselves."""
        assert commute("X", "X") is True
        assert commute("Y", "Y") is True
        assert commute("Z", "Z") is True

    def test_commute_different_pauli_single_qubit(self):
        """Different non-identity Paulis on same qubit anticommute."""
        # X and Z anticommute
        assert commute("X", "Z") is False
        # X and Y anticommute
        assert commute("X", "Y") is False
        # Y and Z anticommute
        assert commute("Y", "Z") is False

    def test_commute_multi_qubit_same_position(self):
        """Multi-qubit strings with one anticommuting position."""
        # Single anticommuting position -> anticommute
        assert commute("XI", "ZI") is False
        assert commute("IX", "IZ") is False
        assert commute("IXI", "IZI") is False

    def test_commute_multi_qubit_two_anticommuting_positions(self):
        """Two anticommuting positions -> commute (even count)."""
        # XZ and ZX have two anticommuting positions
        assert commute("XZ", "ZX") is True
        # XY and YX have two anticommuting positions
        assert commute("XY", "YX") is True

    def test_commute_multi_qubit_identity_padding(self):
        """Identity positions don't affect commutation."""
        assert commute("XII", "ZII") is False
        assert commute("IXI", "IZI") is False
        assert commute("IIX", "IIZ") is False

    def test_commute_five_qubit_code_stabilizers(self, five_qubit_stabilizers):
        """All stabilizers of the 5-qubit code should commute pairwise."""
        stabs = five_qubit_stabilizers
        for i, s1 in enumerate(stabs):
            for j, s2 in enumerate(stabs):
                assert commute(s1, s2) is True, f"Stabilizers {i} and {j} should commute"

    def test_commute_steane_code_stabilizers(self, steane_stabilizers):
        """All stabilizers of the Steane code should commute pairwise."""
        stabs = steane_stabilizers
        for i, s1 in enumerate(stabs):
            for j, s2 in enumerate(stabs):
                assert commute(s1, s2) is True, f"Stabilizers {i} and {j} should commute"

    def test_commute_logical_z_with_stabilizers(self, five_qubit_stabilizers, five_qubit_logical_z):
        """Logical Z should commute with all stabilizers."""
        logical_z = five_qubit_logical_z
        for i, stab in enumerate(five_qubit_stabilizers):
            assert commute(logical_z, stab) is True, f"Logical Z should commute with stabilizer {i}"

    def test_commute_length_mismatch_raises(self):
        """Mismatched lengths should raise AssertionError."""
        with pytest.raises(AssertionError):
            commute("X", "XX")
        with pytest.raises(AssertionError):
            commute("XYZ", "XY")


class TestAnticommute:
    """Tests for the anticommute() function."""

    def test_anticommute_is_inverse_of_commute(self):
        """anticommute() should be the logical inverse of commute()."""
        test_pairs = [
            ("I", "I"),
            ("X", "X"),
            ("X", "Z"),
            ("Y", "Z"),
            ("XZ", "ZX"),
            ("XZZXI", "IXZZX"),
        ]
        for s1, s2 in test_pairs:
            assert anticommute(s1, s2) == (not commute(s1, s2))

    def test_anticommute_single_qubit(self):
        """Single qubit anticommutation tests."""
        assert anticommute("X", "Z") is True
        assert anticommute("X", "Y") is True
        assert anticommute("Y", "Z") is True
        assert anticommute("X", "X") is False
        assert anticommute("I", "X") is False


class TestMultiplyPauliStrings:
    """Tests for the multiply_pauli_strings() function."""

    def test_multiply_identity_left(self):
        """I * P = P for any Pauli P."""
        assert multiply_pauli_strings("I", "I") == "I"
        assert multiply_pauli_strings("I", "X") == "X"
        assert multiply_pauli_strings("I", "Y") == "Y"
        assert multiply_pauli_strings("I", "Z") == "Z"

    def test_multiply_identity_right(self):
        """P * I = P for any Pauli P."""
        assert multiply_pauli_strings("X", "I") == "X"
        assert multiply_pauli_strings("Y", "I") == "Y"
        assert multiply_pauli_strings("Z", "I") == "Z"

    def test_multiply_same_pauli(self):
        """P * P = I for any non-identity Pauli P."""
        assert multiply_pauli_strings("X", "X") == "I"
        assert multiply_pauli_strings("Y", "Y") == "I"
        assert multiply_pauli_strings("Z", "Z") == "I"

    def test_multiply_different_paulis(self):
        """Test multiplication of different Paulis (ignoring phase)."""
        # X * Z = Y (up to phase)
        assert multiply_pauli_strings("X", "Z") == "Y"
        assert multiply_pauli_strings("Z", "X") == "Y"
        # X * Y = Z (up to phase)
        assert multiply_pauli_strings("X", "Y") == "Z"
        assert multiply_pauli_strings("Y", "X") == "Z"
        # Y * Z = X (up to phase)
        assert multiply_pauli_strings("Y", "Z") == "X"
        assert multiply_pauli_strings("Z", "Y") == "X"

    def test_multiply_multi_qubit_strings(self):
        """Test multiplication of multi-qubit Pauli strings."""
        # II * II = II
        assert multiply_pauli_strings("II", "II") == "II"
        # XX * XX = II
        assert multiply_pauli_strings("XX", "XX") == "II"
        # XZ * ZX = YY
        assert multiply_pauli_strings("XZ", "ZX") == "YY"
        # XIZ * ZYI = YYZ
        assert multiply_pauli_strings("XIZ", "ZYI") == "YYZ"

    def test_multiply_five_qubit_stabilizers(self, five_qubit_stabilizers):
        """Multiplying stabilizers should give another element of the group."""
        s1, s2 = five_qubit_stabilizers[0], five_qubit_stabilizers[1]
        result = multiply_pauli_strings(s1, s2)
        # Result should be a valid 5-qubit Pauli string
        assert len(result) == 5
        assert all(c in "IXYZ" for c in result)

    def test_multiply_length_mismatch_raises(self):
        """Mismatched lengths should raise AssertionError."""
        with pytest.raises(AssertionError):
            multiply_pauli_strings("X", "XX")
        with pytest.raises(AssertionError):
            multiply_pauli_strings("XYZ", "XY")

    def test_multiply_case_insensitive(self):
        """Function should handle lowercase input."""
        # The function uppercases internally
        assert multiply_pauli_strings("x", "z") == "Y"
        assert multiply_pauli_strings("i", "x") == "X"


class TestPauliAlgebraConsistency:
    """Tests for algebraic consistency of Pauli operations."""

    def test_commutation_implies_product_order_invariant(self):
        """If [A,B]=0, then AB = BA (up to phase, which we ignore)."""
        # For Paulis that commute, product should be same either way
        # (ignoring phase, which our function does)
        pairs_that_commute = [
            ("II", "II"),
            ("XX", "XX"),
            ("XZ", "ZX"),  # Two anticommuting positions -> commute
        ]
        for a, b in pairs_that_commute:
            assert commute(a, b) is True
            # Product order shouldn't matter for our phase-ignoring multiply
            assert multiply_pauli_strings(a, b) == multiply_pauli_strings(b, a)

    def test_stabilizer_product_commutes_with_all(self, five_qubit_stabilizers):
        """Product of two stabilizers should commute with all stabilizers."""
        s1, s2 = five_qubit_stabilizers[0], five_qubit_stabilizers[1]
        product = multiply_pauli_strings(s1, s2)

        for stab in five_qubit_stabilizers:
            assert commute(product, stab) is True, \
                f"Product {product} should commute with {stab}"
