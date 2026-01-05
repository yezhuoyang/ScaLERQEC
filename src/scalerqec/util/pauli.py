"""
Utility functions for Pauli operator manipulation.
"""


def commute(stab1: str, stab2: str) -> bool:
    """
    Check if two Pauli operators (stabilizer generators) commute.

    Two Pauli strings commute if and only if the number of positions
    where they anti-commute (both non-identity and different) is even.

    Args:
        stab1 (str): The first Pauli string (e.g., "XZZXI").
        stab2 (str): The second Pauli string (e.g., "IXZZX").

    Returns:
        bool: True if the operators commute, False otherwise.

    Raises:
        AssertionError: If the two strings have different lengths.

    Examples:
        >>> commute("XZZXI", "IXZZX")
        True
        >>> commute("IXYZ", "IYZX")
        False
    """
    assert len(stab1) == len(stab2), "Pauli strings must be of the same length."
    anti_commute_count = sum(
        1 for a, b in zip(stab1, stab2) if a != "I" and b != "I" and a != b
    )
    return anti_commute_count % 2 == 0


def anticommute(stab1: str, stab2: str) -> bool:
    """
    Check if two Pauli operators anti-commute.

    Args:
        stab1 (str): The first Pauli string.
        stab2 (str): The second Pauli string.

    Returns:
        bool: True if the operators anti-commute, False otherwise.
    """
    return not commute(stab1, stab2)


def multiply_pauli_strings(p: str, q: str) -> str:
    """
    Multiply two Pauli strings, ignoring global phase.

    Uses the standard Pauli multiplication table modulo phases:
        I*P = P*I = P
        X*X = Y*Y = Z*Z = I
        X*Z = Z*X = Y (up to phase)
        X*Y = Y*X = Z (up to phase)
        Y*Z = Z*Y = X (up to phase)

    Args:
        p (str): First Pauli string.
        q (str): Second Pauli string.

    Returns:
        str: The product Pauli string (ignoring global phase).

    Raises:
        AssertionError: If the two strings have different lengths.
    """
    assert len(p) == len(q), "Pauli strings must have same length."

    def mul_single(a: str, b: str) -> str:
        a = a.upper()
        b = b.upper()
        if a == "I":
            return b
        if b == "I":
            return a
        if a == b:
            return "I"
        # Remaining combinations (up to phase):
        # XZ, ZX -> Y
        # XY, YX -> Z
        # YZ, ZY -> X
        if (a, b) in (("X", "Z"), ("Z", "X")):
            return "Y"
        if (a, b) in (("X", "Y"), ("Y", "X")):
            return "Z"
        if (a, b) in (("Y", "Z"), ("Z", "Y")):
            return "X"
        raise ValueError(f"Unexpected Pauli pair ({a}, {b})")

    return "".join(mul_single(ca, cb) for ca, cb in zip(p, q))
