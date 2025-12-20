


from enum import Enum
from scalerqec.Clifford.clifford import CliffordCircuit
import numpy as np
from scalerqec.QEC.noisemodel import NoiseModel

class SCHEME(Enum):
    STANDARD = 0
    SHOR = 1
    KNILL = 2
    FLAG = 3



def commute(stab1: str, stab2: str) -> bool:
    """
    Check if two stabilizer generators commute.

    Args:
        stab1 (str): The first stabilizer generator.
        stab2 (str): The second stabilizer generator.

    Returns:
        bool: True if the stabilizers commute, False otherwise.
    """
    assert len(stab1) == len(stab2), "Stabilizers must be of the same length."
    anti_commute_count = sum(1 for a, b in zip(stab1, stab2) if a != 'I' and b != 'I' and a != b)
    return anti_commute_count % 2 == 0



"""
Current types of IR instructions.
TODO: Support repeat, conditional operations, etc. The IR should be stored as a tree structure.
"""
class IRType(Enum):
    PROP = 0
    DETECTOR = 1
    OBSERVABLE = 2
    IF_THEN = 3
    WHILE = 4
    REPEAT_UNTIL = 5
    REPEAT = 6


class IRInstruction:
    """
    A class representing an intermediate representation (IR) instruction for quantum circuits.
    """
    def __init__(self, instr_type) -> None:
        self._instr_type = instr_type



class StabPropInstruction(IRInstruction):
    """
    A class representing an intermediate representation (IR) instruction for quantum circuits.
    """
    def __init__(self, round: int, stabindex: int, dest: str, stab: str, is_observable: bool=False, observable_index: int=-1) -> None:
        super().__init__(IRType.PROP)
        self._round = round
        self._stabindex = stabindex
        self._dest = dest
        self._stab = stab
        self._is_observable = is_observable
        self._observable_index = observable_index

    @property
    def round(self) -> int:
        """
        Get the round number of the stabilizer propagation.

        Returns:
            int: The round number.
        """
        return self._round
    

    def is_observable(self) -> bool:
        """
        Check if the stabilizer propagation is for an observable.

        Returns:
            bool: True if it is an observable, False otherwise.
        """
        return self._is_observable


    def get_observable_index(self) -> int:
        """
        Get the index of the observable if applicable.

        Returns:
            int: The observable index.
        """
        return self._observable_index


    def get_stabindex(self) -> int:
        """
        Get the stabilizer index of the stabilizer propagation.

        Returns:
            int: The stabilizer index.
        """
        return self._stabindex


    def __str__(self) -> str:
        if self._is_observable:
            return f"{self._dest} = Prop {self._stab}"
        else:
            return f"{self._dest} = Prop[r={self._round}, s={self._stabindex}] {self._stab}"

    @property
    def dest(self) -> str:
        """
        Get the destination qubit/observable/detector from the instruction.

        Returns:
            str: The destination string.
        """
        return self._dest

    @property
    def stab(self) -> str:
        """
        Get the stabilizer from the instruction.

        Returns:
            str: The stabilizer string.
        """
        return self._stab


class ParityInstruction(IRInstruction):
    """
    Base class for parity-based IR instructions (detectors, observables).
    """

    def __init__(self, ir_type: IRType, dest: str, args: list[str]) -> None:
        super().__init__(ir_type)
        self._dest = dest
        self._args = args

    def __str__(self) -> str:
        return f"{self._dest} = Parity {' '.join(self._args)}"

    @property
    def dest(self) -> str:
        return self._dest

    @property
    def args(self) -> list[str]:
        return self._args


class DetectorInstruction(ParityInstruction):    
    """
    A class representing a detector instruction in the intermediate representation (IR) of a quantum circuit.
    """

    def __init__(self, dest: str, args: list[str]) -> None:
        super().__init__(IRType.DETECTOR, dest, args)


class ObservableInstruction(ParityInstruction):
    """
    A class representing an observable instruction in the intermediate representation (IR) of a quantum circuit.
    """

    def __init__(self, dest: str, args: list[str]) -> None:
        super().__init__(IRType.OBSERVABLE, dest, args)



class IF_THENInstruction(IRInstruction):
    """
    A class representing an IF-THEN instruction in the intermediate representation (IR) of a quantum circuit.
    """
    def __init__(self, condition: str, then_branch: list[IRInstruction]) -> None:
        super().__init__(IRType.IF_THEN)
        self._condition = condition
        self._then_branch = then_branch


class WHILEInstruction(IRInstruction):
    """
    A class representing a WHILE instruction in the intermediate representation (IR) of a quantum circuit.
    """
    def __init__(self, condition: str, body: list[IRInstruction]) -> None:
        super().__init__(IRType.WHILE)
        self._condition = condition
        self._body = body


class REPEAT_UNTILInstruction(IRInstruction):
    """
    A class representing a REPEAT-UNTIL instruction in the intermediate representation (IR) of a quantum circuit.
    """
    def __init__(self, body: list[IRInstruction], until_condition: str) -> None:
        super().__init__(IRType.REPEAT_UNTIL)
        self._body = body
        self._until_condition = until_condition



class REPEATInstruction(IRInstruction):
    """
    A class representing a REPEAT instruction in the intermediate representation (IR) of a quantum circuit.
    """
    def __init__(self, body: list[IRInstruction], times: int) -> None:
        super().__init__(IRType.REPEAT)
        self._body = body
        self._times = times



class QECStab:
    """
    A class representing a quantum error-correcting code (QECC) using the stabilizer formalism.
    """
    def __init__(self, n: int, k: int, d: int) -> None:
        self._n = n
        self._k = k
        self._d = d
        self._stabs = []
        self._scheme = SCHEME.STANDARD
        self._circuit = None
        self._stimcirc=None
        self._IRList = []
        self._rounds = 3*d
        #Define the k different logical Z operators
        self._logicalZ = {}
        self._paritymatrix = None
        self._noisemodel = None
        self._IR_compiled = False
        self._circuit_compiled = False


    def is_IR_compiled(self) -> bool:
        """
        Check if the intermediate representation (IR) has been compiled.

        Returns:
            bool: True if the IR is compiled, False otherwise.
        """
        return self._IR_compiled


    def is_circuit_compiled(self) -> bool:
        """
        Check if the quantum error-correcting circuit has been compiled.

        Returns:
            bool: True if the circuit is compiled, False otherwise.
        """
        return self._circuit_compiled


    @property
    def noisemodel(self) -> NoiseModel:
        """
        Get the noise model associated with the QECC.

        Returns:
            NoiseModel: The noise model.
        """
        return self._noisemodel
    

    @noisemodel.setter
    def noisemodel(self, noisemodel: NoiseModel) -> None:
        """
        Set the noise model associated with the QECC.

        Args:
            noisemodel (NoiseModel): The noise model to set.
        """
        self._noisemodel = noisemodel



    def init_by_parity_check_matrix(self, paritymatrix: np.ndarray) -> None:
        """
        Initialize the QECC stabilizer structures using a given parity check matrix.

        Args:
            paritymatrix (np.ndarray): The parity check matrix.
        """
        self._paritymatrix = paritymatrix
        self._n = paritymatrix.shape[1]
        self._k = self._n - paritymatrix.shape[0]
        self._stabs = []
        pass



    def construct_parity_check_matrix(self) -> None:
        """
        Construct the standard XZ parity check matrix for the quantum error-correcting code.

        Returns:
            The parity check matrix.
        """
        pass



    def get_parity_check_matrix(self) -> None:
        """
        Get the standard XZ parity check matrix for the quantum error-correcting code.

        Returns:
            The parity check matrix.
        """
        return self._paritymatrix




    @property
    def circuit(self) -> None:
        """
        Get the Clifford circuit for the quantum error-correcting code.

        Returns:
            The Clifford circuit.
        """
        return self._circuit


    @property
    def stimcirc(self) -> None:
        """
        Get the stimulus circuit for the quantum error-correcting code.

        Returns:
            The stimulus circuit.
        """
        return self._stimcirc


    def set_logical_Z(self, index: int, logicalZ: str) -> None:
        """
        Set the logical Z operator for a given logical qubit.

        Args:
            index (int): The index of the logical qubit.
            logicalZ (str): A string representation of the logical Z operator.
        """
        assert len(logicalZ) == self._n, "Logical Z length must match number of qubits."
        assert all(c in 'IXYZ' for c in logicalZ), "Logical Z must only contain I, X, Y, and Z."

        self._logicalZ[index] = logicalZ

    
    @property
    def rounds(self) -> int:
        """
        Get the number of error correction rounds.

        Returns:
            int: The number of rounds.
        """
        return self._rounds


    @rounds.setter
    def rounds(self, rounds: int) -> None:
        """
        Set the number of error correction rounds.

        Args:
            rounds (int): The number of rounds to set.
        """
        self._rounds = rounds


    def add_stab(self, stab: str) -> None:
        """
        Add a stabilizer generator to the code.

        Args:
            stab (str): A string representation of the stabilizer generator.
        """
        assert len(stab) == self._n, "Stabilizer length must match number of qubits."
        assert all(c in 'IXYZ' for c in stab), "Stabilizer must only contain I, X, Y, Z."

        self._stabs.append(stab)


    @property
    def scheme(self) -> SCHEME:
        """
        Get the error correction scheme for the code.

        Returns:
            SCHEME: The error correction scheme.
        """
        return self._scheme


    @scheme.setter
    def scheme(self, scheme: str) -> None:
        """
        Set the error correction scheme for the code.

        Args:
            scheme (SCHEME): The error correction scheme to use.
        """
        match scheme:
            case "Standard":
                self._scheme = SCHEME.STANDARD
            case "Shor":
                self._scheme = SCHEME.SHOR
            case "Knill":
                self._scheme = SCHEME.KNILL
            case "Flag":
                self._scheme = SCHEME.FLAG
            case _:
                raise ValueError(f"Unknown scheme: {scheme}")
            

    def construct_circuit(self):
        """
        Construct the quantum error-correcting circuit based on the stabilizers and scheme.

        There is a two step compilation:
             First, compile the stabilizers into an intermediate representation (IR) of the circuit.
             Second, translate the IR into a Clifford circuit.
             In IR, there is no concept of qubits, only Pauli operators, detectors, observables, and their relationships.
        The IR has the form:

        
        c0 = Prop XYZIX
        c1 = Prop IXYZI
        d0 = Parity c0 c1
        o0 = Parity c0   
        """
        match self._scheme:
            case SCHEME.STANDARD:
                self.construct_IR_standard_scheme()
                self.compile_stim_circuit_from_IR_standard()
                if self._noisemodel is not None:
                    self._circuit = self._noisemodel.reconstruct_clifford_circuit(self._circuit)
                    self._stimcirc = self._circuit._stimcircuit
            case _:
                raise NotImplementedError(f"Scheme {self._scheme} not implemented yet.")



    def construct_IR_standard_scheme(self):
        """
        Construct the quantum error-correcting circuit using the standard scheme.
        Now, we will create the intermediate representation (IR) for the circuit.
        """
        if self._IR_compiled:
            return
        current_measurement_idx = 0
        current_detector_idx = 0
        prev_stab_meas_addr = {}
        for r in range(self._rounds):
            stabidx=0
            for stab in self._stabs:
                dest = f"c{current_measurement_idx}"
                instr = StabPropInstruction(r, stabidx, dest, stab)
                self._IRList.append(instr)
                current_measurement_idx += 1
                stabidx += 1
                """
                Since the second round, add detectors comparing with previous round
                """
                if r > 0:
                    prev_dest = prev_stab_meas_addr[stab]
                    detector_dest = f"d{current_detector_idx}"
                    detector_instr = DetectorInstruction(detector_dest, [prev_dest, dest])
                    self._IRList.append(detector_instr)
                    current_detector_idx += 1
                prev_stab_meas_addr[stab] = dest
        #Logical observables
        for logical_idx in range(self._k):
            logicalZ = self._logicalZ[logical_idx]

            dest = f"c{current_measurement_idx}"
            instr = StabPropInstruction(0, 0, dest, logicalZ, is_observable=True, observable_index=logical_idx)

            self._IRList.append(instr)
            current_measurement_idx += 1
            observable_dest = f"o{logical_idx}"
            observable_instr = ObservableInstruction(observable_dest, [dest])
            self._IRList.append(observable_instr)
        self._IR_compiled = True

    def show_IR(self):
        """
        Display the intermediate representation of the quantum error-correcting circuit.

        The IR has the form:
        """
        for irinst in self._IRList:
            print(irinst)


    def compile_stim_circuit_from_IR_standard(self) -> None:
        """
        Compile the stim circuit from the intermediate representation (IR).

        Returns:
            str: The compiled stim circuit as a string.
        """
        #Convension: Stabilizer k stored in qubit n+k-1
        #Observable k stored in qubit n+num_syndromes+k-1

        if not self._IR_compiled:
            raise RuntimeError("IR not compiled yet.")
        if self._circuit_compiled:
            return str(self._stimcirc)
        self._circuit = CliffordCircuit(self._n+len(self._stabs) + self._k)
        parity_match_group = []
        observable_parity_group = []

        dest_to_measure_index = {}
        current_measure_index = 0
        for irinst in self._IRList:
            if isinstance(irinst, StabPropInstruction):
                stab = irinst.stab
                dest_index = int(irinst.dest[1:])
                if irinst.is_observable():
                    helper_qubit_index = self._n + len(self._stabs) + irinst.get_observable_index()
                else:
                    helper_qubit_index = self._n + irinst.get_stabindex()                    

                self._circuit.add_reset(helper_qubit_index)
                for qubit_index, pauli in enumerate(stab):
                    match pauli:
                        case 'X':
                            self._circuit.add_hadamard(qubit_index)
                            self._circuit.add_cnot(control=qubit_index, target=helper_qubit_index)
                            self._circuit.add_hadamard(qubit_index)
                        case 'Z':
                            self._circuit.add_cnot(control=qubit_index, target=helper_qubit_index)
                        case 'I':
                            continue
                        case 'Y':   
                            raise NotImplementedError("Y parity propagation not supported.")
                        
                self._circuit.add_measurement(helper_qubit_index)
                dest_to_measure_index[irinst.dest] = current_measure_index
                current_measure_index += 1

            elif isinstance(irinst, DetectorInstruction):
                args = irinst.args
                args_measure_indices = [dest_to_measure_index[arg] for arg in args]
                parity_match_group.append(args_measure_indices)


            elif isinstance(irinst, ObservableInstruction):
                args = irinst.args
                args_indices = [dest_to_measure_index[arg] for arg in args]
                observable_parity_group.append(args_indices)


        self._circuit.parityMatchGroup=parity_match_group
        self._circuit.observable=observable_parity_group[0]
        self._circuit.compile_detector_and_observable()
        self._stimcirc = self._circuit._stimcircuit
        self._circuit_compiled = True


def test_commute():
    assert commute("IXYZ", "IYZX") == False
    assert commute("XZZI", "ZXXI") == False
    assert commute("IIII", "ZZZZ") == True
    assert commute("XIZY", "YZXI") == True



if __name__ == "__main__":
    noise_model = NoiseModel(error_rate=0.001)
    qeccirc= QECStab(n=5,k=1,d=3)
    qeccirc.noisemodel = noise_model
    #Specify your stabilizers
    # Stabilizer generators
    qeccirc.add_stab("XZZXI")
    qeccirc.add_stab("IXZZX")
    qeccirc.add_stab("XIXZZ")
    qeccirc.add_stab("ZXIXZ")
    qeccirc.set_logical_Z(0, "ZZZZZ")
    #Set stabilizer parity measurement scheme, round of repetition
    qeccirc.scheme="Standard"
    qeccirc.rounds=z
    qeccirc.construct_circuit()
    stim_circuit = qeccirc.stimcirc
    print(stim_circuit)