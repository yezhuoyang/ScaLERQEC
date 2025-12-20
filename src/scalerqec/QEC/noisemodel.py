#A Noise model class, the purpose is to rewrite stim program and Clifford circuit to support noise model
#Rewrite all stim program/Clifford circuit to support the noise model
from ..Clifford.clifford import *
from enum import Enum

class ErrorType(Enum):
    MEASUREMENT = 0
    RESET = 1
    CNOT = 2
    HADAMARD = 3
    PHASE = 4
    PAULIX = 5
    PAULIY = 6
    PAULIZ = 7
    CZ = 8



class NoiseModel:
    """
    A class representing a noise model for quantum error correction simulations.
    """


    def __init__(self, error_rate: float) -> None:
        self._error_rate = error_rate

        self._has_MEASUREMENT_error = True
        self._has_RESET_error = True
        self._has_CNOT_error = True
        self._has_CZ_error = True
        self._has_HADAMARD_error = True
        self._has_PHASE_error = True
        self._has_PAULIX_error = True
        self._has_PAULIY_error = True
        self._has_PAULIZ_error = True


    @property
    def error_rate(self) -> float:
        """
        Get the error rate of the noise model.

        Returns:
            The error rate as a float.
        """
        return self._error_rate

    @error_rate.setter
    def error_rate(self, value: float) -> None:
        """
        Set the error rate of the noise model.

        Args:
            value: The new error rate as a float.
        """
        self._error_rate = value



    def disable_error(self, error_type: str) -> None:
        """
        Disable a specific type of error in the noise model.

        Args:
            error_type: The type of error to disable.
        """
        if error_type == "MEASUREMENT":
            self._has_MEASUREMENT_error = False
        elif error_type == "RESET":
            self._has_RESET_error = False
        elif error_type == "CNOT":
            self._has_CNOT_error = False
        elif error_type == "CZ":
            self._has_CZ_error = False
        elif error_type == "H":
            self._has_HADAMARD_error = False
        elif error_type == "P":
            self._has_PHASE_error = False
        elif error_type == "X":
            self._has_PAULIX_error = False
        elif error_type == "Y":
            self._has_PAULIY_error = False
        elif error_type == "Z":
            self._has_PAULIZ_error = False



    def rewrite_stim_program(self, stim_program) -> str:
        """
        Rewrite a given stim program to incorporate the noise model.

        Args:
            stim_program: The original stim program to be modified.

        Returns:
            A new stim program with the noise model applied.
        """
        # Placeholder for actual implementation
        return stim_program



    def reconstruct_clifford_circuit(self, clifford_circuit: CliffordCircuit) -> CliffordCircuit:
        """
        Reconstruct a given Clifford circuit to incorporate the noise model.

        Args:
            clifford_circuit: The original Clifford circuit to be modified.

        Returns:
            A new Clifford circuit with the noise model applied.
        """
        # Placeholder for actual implementation
        
        num_qubits = clifford_circuit.qubitnum
        new_circuit = CliffordCircuit(num_qubits)

        new_circuit.error_rate = self._error_rate
        gate_list = clifford_circuit.gatelists

        for gate in gate_list:

            if isinstance(gate, TwoQGate):
                # Apply CNOT gate with noise

                match gate.name:                    
                    case "CNOT":
                        if self._has_CNOT_error:
                            new_circuit.add_depolarize(gate.control)
                            new_circuit.add_depolarize(gate.target)
                        new_circuit.add_cnot(gate.control, gate.target)
                    case "CZ":
                        if self._has_CZ_error:
                            new_circuit.add_depolarize(gate.control)
                            new_circuit.add_depolarize(gate.target)
                        new_circuit.add_cz(gate.control, gate.target)

            elif isinstance(gate, SingleQGate):
                # Apply single-qubit gate with noise
                match gate.name:                    
                    case "H":
                        if self._has_HADAMARD_error:
                            new_circuit.add_depolarize(gate.qubitindex)
                        new_circuit.add_hadamard(gate.qubitindex)
                    case "P":
                        if self._has_PHASE_error:
                            new_circuit.add_depolarize(gate.qubitindex)
                        new_circuit.add_phase(gate.qubitindex)
                    case "X":
                        if self._has_PAULIX_error:
                            new_circuit.add_depolarize(gate.qubitindex)
                        new_circuit.add_paulix(gate.qubitindex)
                    case "Y":
                        if self._has_PAULIY_error:
                            new_circuit.add_depolarize(gate.qubitindex)
                        new_circuit.add_pauliy(gate.qubitindex)
                    case "Z":
                        if self._has_PAULIZ_error:
                            new_circuit.add_depolarize(gate.qubitindex)
                        new_circuit.add_pauliz(gate.qubitindex)

            elif isinstance(gate, Measurement):
                # Apply measurement with noise
                if self._has_MEASUREMENT_error:
                    new_circuit.add_depolarize(gate.qubitindex)
                new_circuit.add_measurement(gate.qubitindex)
                # Placeholder for adding noise after measurement

            elif isinstance(gate, Reset):
                # Apply reset with noise
                if self._has_RESET_error:
                    new_circuit.add_depolarize(gate.qubitindex)
                new_circuit.add_reset(gate.qubitindex)
                # Placeholder for adding noise after reset


        new_circuit.parityMatchGroup=clifford_circuit.parityMatchGroup
        new_circuit.observable=clifford_circuit.observable
        new_circuit.compile_detector_and_observable()

        return new_circuit



    def uniform_depolarization_single(stim_program: str) -> str:
        """
        Apply uniform depolarization noise to single-qubit operations in the stim program.

        Args:
            stim_program (str): The original stim program. 

        Returns:
            str: The modified stim program with depolarization noise applied.

        # Placeholder for actual implementation
        """
        return stim_program

