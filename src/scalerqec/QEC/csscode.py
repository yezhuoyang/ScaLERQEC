from .qeccircuit import StabCode




class cssCode(StabCode):
    """
    A class representing a CSS quantum error correction code.

    The code is constructed by two classical linear codes C1 and C2
    """


    def __init__(self, n: int, k: int, d: int):
        super().__init__(n, k, d)


    def set_C1(self, C1: list[str]) -> None:
        """
        Set the classical code C1 used for X stabilizers.
        """
        for stab in C1:
            self.add_stab(stab.replace('0', 'I').replace('1', 'X'))


    def set_C2(self, C2: list[str]) -> None:
        """
        Set the classical code C2 used for Z stabilizers.
        """
        for stab in C2:
            self.add_stab(stab.replace('0', 'I').replace('1', 'Z'))