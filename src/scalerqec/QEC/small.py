from .qeccircuit import StabCode


class fivequbitCode(StabCode):
    """
    The most famous five-qubit code
    5 physical qubits, 1 logical qubit, distance 3

    The stabilizers are:
    XZZXI
    IXZZX
    XIXZZ
    ZXIXZ
    The logical operators can be chosen as:
    Logical X: XXXXX
    Logical Z: ZZZZZ

    This code can correct any single-qubit error.

    The minimum weight of non-trivial logical operators is 3, which can be:
    - XXXII
    - IXXXI
    - IIXXX
    - XIIXX
    """

    def __init__(self, scheme: str):
        super().__init__(5, 1, 3)
        self.scheme = scheme
        self.construct_stabilizers()
        self.set_logical_Z(0, "ZZZZZ")

    def construct_stabilizers(self):
        """
        Construct the stabilizers for the five-qubit code.
        """
        self.add_stab("XZZXI")
        self.add_stab("IXZZX")
        self.add_stab("XIXZZ")
        self.add_stab("ZXIXZ")


class steaneCode(StabCode):
    """
    The Steane code, which is also the minimal CSS code based on the classical [7,4,3] Hamming code.
    7 physical qubits, 1 logical qubit, distance 3

    The stabilizers are:
    IIIXXXX
    IXXIIXX
    XIXIXIX
    IIIZZZZ
    IZZIIZZ
    ZIZIZIZ
    The logical operators can be chosen as:
    Logical X: XXXXXXX
    Logical Z: ZZZZZZZ
    This code can correct any single-qubit error.
    The minimum weight of non-trivial logical operators is 3, which can be:
    - XXXIIII
    - IXXXIII
    - IIXXXII
    - IIIXXXI
    - IIIIIXX
    """

    def __init__(self, scheme: str):
        super().__init__(7, 1, 3)
        self.scheme = scheme
        self.construct_stabilizers()
        self.set_logical_Z(0, "ZZZZZZZ")

    def construct_stabilizers(self):
        """
        Construct the stabilizers for the Steane code.
        """
        self.add_stab("IIIXXXX")
        self.add_stab("IXXIIXX")
        self.add_stab("XIXIXIX")
        self.add_stab("IIIZZZZ")
        self.add_stab("IZZIIZZ")
        self.add_stab("ZIZIZIZ")


class ShorCode(StabCode):
    """
    The Shor code in standard text book representation.
    9 physical qubits, 1 logical qubit, distance 3
    The stabilizers are:
    ZZIIIIIII
    IZZIIIIII
    IIIZZIIII
    IIIIZZIII
    IIIIIIZZI
    IIIIIIIZZ
    XXXXXXIII
    XXXIIIXXX
    The logical operators can be chosen as:
    Logical X: XXXXXXXXX
    Logical Z: ZZZZZZZZZ
    This code can correct any single-qubit error.
    The minimum weight of non-trivial logical operators is 3, which can be:
    - XXXIIIIII
    - IXXIIIIII
    - IIXXIIIII
    - IIIXXIIII
    - IIIIXXIII
    - IIIIIXXII
    - IIIIIIXXI
    """

    def __init__(self, scheme: str):
        super().__init__(9, 1, 3)
        self.scheme = scheme
        self.construct_stabilizers()
        self.set_logical_Z(0, "ZZZZZZZZZ")

    def construct_stabilizers(self):
        """
        Construct the stabilizers for the Shor code.
        """
        self.add_stab("ZZIIIIIII")
        self.add_stab("IZZIIIIII")
        self.add_stab("IIIZZIIII")
        self.add_stab("IIIIZZIII")
        self.add_stab("IIIIIIZZI")
        self.add_stab("IIIIIIIZZ")
        self.add_stab("XXXXXXIII")
        self.add_stab("XXXIIIXXX")
