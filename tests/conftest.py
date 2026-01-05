"""
Shared pytest fixtures for ScaLER tests.
"""
import pytest


# ============================================================================
# Pauli String Fixtures
# ============================================================================

@pytest.fixture
def five_qubit_stabilizers():
    """Stabilizers for the [[5,1,3]] five-qubit code."""
    return [
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    ]


@pytest.fixture
def five_qubit_logical_z():
    """Logical Z operator for the five-qubit code."""
    return "ZZZZZ"


@pytest.fixture
def steane_stabilizers():
    """Stabilizers for the [[7,1,3]] Steane code."""
    return [
        "IIIXXXX",
        "IXXIIXX",
        "XIXIXIX",
        "IIIZZZZ",
        "IZZIIZZ",
        "ZIZIZIZ",
    ]


@pytest.fixture
def steane_logical_z():
    """Logical Z operator for the Steane code."""
    return "ZZZZZZZ"


@pytest.fixture
def shor_stabilizers():
    """Stabilizers for the [[9,1,3]] Shor code."""
    return [
        "ZZIIIIIII",
        "IZZIIIIII",
        "IIIZZIIIII"[:-1],  # "IIIZZIIII"
        "IIIIZZIII",
        "IIIIIIZZI",
        "IIIIIIIZZ",
        "XXXXXXIII",
        "IIIXXXXXX",
    ]


@pytest.fixture
def shor_logical_z():
    """Logical Z operator for the Shor code."""
    return "ZZZZZZZZZ"


# ============================================================================
# Binomial Test Data Fixtures
# ============================================================================

@pytest.fixture
def binomial_test_cases_small():
    """Test cases for small N (exact computation)."""
    return [
        # (N, W, p, expected_approx)
        (10, 3, 0.2, None),  # Will compute expected
        (10, 0, 0.5, None),
        (10, 10, 0.9, None),
        (5, 2, 0.3, None),
    ]


@pytest.fixture
def binomial_test_cases_large():
    """Test cases for large N (Poisson approximation)."""
    return [
        # (N, W, p)
        (300, 5, 0.01),
        (500, 2, 0.004),
        (1000, 10, 0.01),
    ]
