import stim



oneQGate_ = ["H", "P", "X", "Y", "Z"]
oneQGateindices={"H":0, "P":1, "X":2, "Y":3, "Z":4}


twoQGate_ = ["CNOT", "CZ"]
twoQGateindices={"CNOT":0, "CZ":1}

pauliNoise_ = ["I","X", "Y", "Z"]
pauliNoiseindices={"I":0,"X":1, "Y":2, "Z":3}


class SingleQGate:
    def __init__(self, gateindex: int, qubitindex: int):
        self._name = oneQGate_[gateindex]
        self._qubitindex = qubitindex

    @property
    def qubitindex(self) -> int:
        return self._qubitindex

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name + "[" + str(self._qubitindex) + "]"


class TwoQGate:
    def __init__(self, gateindex: int, control: int, target: int):
        self._name = twoQGate_[gateindex]
        self._control = control
        self._target = target

    @property
    def control(self) -> int:
        return self._control

    @property
    def target(self) -> int:
        return self._target

    @property
    def name(self) -> str:
        return self._name


    def __str__(self):
        return self._name + "[" + str(self._control) + "," + str(self._target)+ "]"


class pauliNoise:
    def __init__(self, noiseindex: int, qubitindex: int):
        self._name="n"+str(noiseindex)
        self._noiseindex : int= noiseindex
        self._qubitindex : int = qubitindex
        self._noisetype : int = 0

    @property
    def noisetype(self) -> int:
        return self._noisetype

    @noisetype.setter
    def noisetype(self, noisetype: int):
        self._noisetype = noisetype


    def __str__(self) -> str:
        return self._name +"("+pauliNoise_[self._noisetype] +")" +"[" + str(self._qubitindex) + "]"


class Measurement:
    def __init__(self,measureindex: int, qubitindex: int):
        self._name="M"+str(measureindex)
        self._qubitindex = qubitindex
        self._measureindex=measureindex

    @property
    def qubitindex(self) -> int:
        return self._qubitindex

    def __str__(self) -> str:
        return self._name + "[" + str(self._qubitindex) + "]"



class Reset:
    def __init__(self, qubitindex: int):
        self._name="R"
        self._qubitindex = qubitindex

    @property
    def qubitindex(self) -> int:
        return self._qubitindex

    def __str__(self) -> str:
        return self._name + "[" + str(self._qubitindex) + "]"



#Class: CliffordCircuit
class CliffordCircuit:


    def __init__(self, qubit_num : int):
        self._qubit_num : int = qubit_num
        self._totalnoise : int=0
        self._totalMeas : int=0
        self._totalgates : int=0
        self._gatelists: list[SingleQGate | TwoQGate | pauliNoise | Measurement | Reset] = []
        self._error_rate : float=0
        self._index_to_noise: dict[int, pauliNoise] = {}
        self._index_to_measurement: dict[int, Measurement] = {}

        #self._index_to_measurement={}

        self._shownoise: bool = False
        self._syndromeErrorTable: dict[str, int] = {}
        #Store the repeat match group
        #For example, if we require M0=M1, M2=M3, then the match group is [[0,1],[2,3]]
        self._parityMatchGroup: list[list[int]] = []
        self._observable: list[int] = []

        self._measIdx_to_parityIdx: dict[int, list[int]] = {}

        self._stim_str: str | None = None
        self._stimcircuit: stim.Circuit = stim.Circuit()


        #self._error_channel


    @property
    def gatelists(self) -> list:
        return self._gatelists


    @property
    def qubitnum(self) -> int:
        return self._qubit_num


    @qubitnum.setter
    def qubitnum(self, qubit_num : int):
        self._qubit_num = qubit_num

    def get_measIdx_to_parityIdx(self,measIdx : int) -> list[int]:
        return self._measIdx_to_parityIdx[measIdx]


    @property
    def stim_str(self) -> str | None:
        return self._stim_str


    @stim_str.setter
    def stim_str(self, stim_str: str):
        self._stim_str=stim_str


    @property
    def error_rate(self) -> float:
        return self._error_rate

    @error_rate.setter
    def error_rate(self, error_rate: float):
        self._error_rate=error_rate

    @property
    def stimcircuit(self):
        return self._stimcircuit


    @stimcircuit.setter
    def stimcircuit(self, stim_circuit):
        if isinstance(stim_circuit, str):
            self._stim_str=stim_circuit
            self._stimcircuit= stim.Circuit(stim_circuit)
        elif isinstance(stim_circuit, stim.Circuit):
            self._stimcircuit=stim_circuit
            self._stim_str= str(stim_circuit)

    @property
    def observable(self):
        return self._observable
    
    @observable.setter
    def observable(self, observablemeasurements):
        self._observable=observablemeasurements


    @property
    def parityMatchGroup(self):
        return self._parityMatchGroup

    @parityMatchGroup.setter
    def parityMatchGroup(self, parityMatchGroup):
        self._parityMatchGroup=parityMatchGroup


    @property
    def qubit_num(self):
        return self._qubit_num

    @property
    def totalnoise(self):
        return self._totalnoise
    
    @property
    def totalMeas(self):
        return self._totalMeas


    '''
    Read the circuit from a file
    Example of the file:

    NumberOfQubit 6
    cnot 1 2
    cnot 1 3
    cnot 1 0
    M 0
    cnot 1 4
    cnot 2 4
    M 4
    cnot 2 5
    cnot 3 5
    M 5
    R 4
    R 5
    cnot 1 4
    cnot 2 4
    M 4
    cnot 2 5
    cnot 3 5
    M 5

    '''
    def read_circuit_from_file(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                if line.startswith("NumberOfQubit"):
                    # Extract the number of qubits
                    self._qubit_num = int(line.split()[1])
                else:
                    # Parse the gate operation
                    parts = line.split()
                    gate_type = parts[0]
                    qubits = list(map(int, parts[1:]))
                    
                    if gate_type == "cnot":
                        self.add_cnot(qubits[0], qubits[1])
                    elif gate_type == "M":
                        self.add_measurement(qubits[0])
                    elif gate_type == "R":
                        self.add_reset(qubits[0])
                    elif gate_type == "H":
                        self.add_hadamard(qubits[0])
                    elif gate_type == "P":
                        self.add_phase(qubits[0])
                    elif gate_type == "CZ":
                        self.add_cz(qubits[0], qubits[1])
                    elif gate_type == "X":
                        self.add_paulix(qubits[0])
                    elif gate_type == "Y":
                        self.add_pauliy(qubits[0])
                    elif gate_type == "Z":
                        self.add_pauliz(qubits[0])
                    else:
                        raise ValueError(f"Unknown gate type: {gate_type}")

    
    '''
    Compile from a stim circuit string.
    '''
    def compile_from_stim_circuit_str(self, stim_str : str):
        #self._totalnoise=0
        self._totalnoise=0
        self._totalMeas=0
        self._totalgates=0       

        lines = stim_str.splitlines()
        output_lines = []
        maxum_q_index=0
        '''
        First, read and compute the parity match group and the observable
        '''
        parityMatchGroup=[]
        observable=[]

        
        measure_index_to_line={}
        measure_line_to_measure_index={}             
        current_line_index=0
        current_measure_index=0
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                # Skip empty lines (optional: you could also preserve them)
                current_line_index+=1
                continue
            
            # Keep lines that we do NOT want to split
            if (stripped_line.startswith("TICK") or
                stripped_line.startswith("DETECTOR(") or
                stripped_line.startswith("QUBIT_COORDS(") or                
                stripped_line.startswith("OBSERVABLE_INCLUDE(")):
                current_line_index+=1
                continue

            tokens = stripped_line.split()
            gate = tokens[0]

            if gate == "M":
                measure_index_to_line[current_measure_index]=current_line_index
                measure_line_to_measure_index[current_line_index]=current_measure_index
                current_measure_index+=1

            current_line_index+=1
        

        current_line_index=0
        measure_stack: list[int] = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("DETECTOR("):
                meas_index_str = [token.strip() for token in stripped_line.split() if token.strip().startswith("rec")]
                meas_index = [int(x[4:-1]) for x in meas_index_str]
                parityMatchGroup.append([measure_line_to_measure_index[measure_stack[idx]] for idx in meas_index])
                current_line_index+=1
                continue
            elif stripped_line.startswith("OBSERVABLE_INCLUDE("):
                meas_index_str = [token.strip() for token in stripped_line.split() if token.strip().startswith("rec")]
                meas_index = [int(x[4:-1]) for x in meas_index_str]
                observable=[measure_line_to_measure_index[measure_stack[idx]] for idx in meas_index]
                current_line_index+=1
                continue


            tokens = stripped_line.split()
            gate = tokens[0]
            if gate == "M":
                measure_stack.append(current_line_index)
            current_line_index+=1

        '''
        Insert gates
        '''
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                # Skip empty lines (optional: you could also preserve them)
                continue

            # Keep lines that we do NOT want to split
            if (stripped_line.startswith("TICK") or
                stripped_line.startswith("DETECTOR(") or
                stripped_line.startswith("QUBIT_COORDS(") or     
                stripped_line.startswith("OBSERVABLE_INCLUDE(")):
                output_lines.append(stripped_line)
                continue

            tokens = stripped_line.split()
            gate = tokens[0]


            if gate == "CX":
                control = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>control else control
                target = int(tokens[2])
                maxum_q_index=maxum_q_index if maxum_q_index>target else target
                self.add_depolarize(control)
                self.add_depolarize(target)
                self.add_cnot(control, target)


            elif gate == "M":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_depolarize(qubit)
                self.add_measurement(qubit)

            elif gate == "H":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_depolarize(qubit)
                self.add_hadamard(qubit)            

            elif gate == "S":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_depolarize(qubit)
                self.add_phase(qubit)    

            
            elif gate == "R":
                qubits = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubits else qubits
                self.add_depolarize(qubits)
                self.add_reset(qubits)
            
        '''
        Finally, compiler detector and observable
        '''
        self._parityMatchGroup=parityMatchGroup
        self._observable=observable
        self._qubit_num=maxum_q_index+1
        self.compile_detector_and_observable()    




    def save_circuit_to_file(self, filename):
        pass



    def set_noise_type(self, noiseindex: int, noisetype: int):
        self._index_to_noise[noiseindex].noisetype = noisetype


    def reset_noise_type(self):
        for i in range(self._totalnoise):
            self._index_to_noise[i].noisetype = 0

    def show_all_noise(self):
        for i in range(self._totalnoise):
            print(self._index_to_noise[i])


    def add_xflip_noise(self, qubit : int):
        self._stimcircuit.append("X_ERROR", [qubit], self._error_rate)
        noise = pauliNoise(self._totalnoise, qubit)
        self._gatelists.append(noise)
        self._index_to_noise[self._totalnoise] = noise
        self._totalnoise+=1           


    def add_depolarize(self, qubit : int):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)
        noise = pauliNoise(self._totalnoise, qubit)
        self._gatelists.append(noise)
        self._index_to_noise[self._totalnoise] = noise
        self._totalnoise+=1        

    def add_cnot(self, control : int, target : int):   
        self._gatelists.append(TwoQGate(twoQGateindices["CNOT"], control, target))
        self._stimcircuit.append("CNOT", [control, target])


    def add_hadamard(self, qubit: int):      
        self._gatelists.append(SingleQGate(oneQGateindices["H"], qubit))
        self._stimcircuit.append("H", [qubit])


    def add_phase(self, qubit: int):         
        self._gatelists.append(SingleQGate(oneQGateindices["P"], qubit))
        self._stimcircuit.append("S", [qubit])

    def add_cz(self, qubit1: int, qubit2 : int):
        self._gatelists.append(TwoQGate(twoQGateindices["CZ"], qubit1, qubit2))     


    def add_paulix(self, qubit : int):
        self._gatelists.append(SingleQGate(oneQGateindices["X"], qubit))
        self._stimcircuit.append("X", [qubit])

    def add_pauliy(self, qubit: int):
        self._gatelists.append(SingleQGate(oneQGateindices["Y"], qubit))
        self._stimcircuit.append("Y", [qubit])

    def add_pauliz(self, qubit : int):
        self._gatelists.append(SingleQGate(oneQGateindices["Z"], qubit))
        self._stimcircuit.append("Z", [qubit])


    def add_measurement(self, qubit: int):
        meas = Measurement(self._totalMeas, qubit)
        self._gatelists.append(meas)
        self._stimcircuit.append("M", [qubit])
        #self._stimcircuit.append("DETECTOR", [stim.target_rec(-1)])
        self._index_to_measurement[self._totalMeas] = meas
        self._measIdx_to_parityIdx[self._totalMeas]=[]
        self._totalMeas+=1


    def compile_detector_and_observable(self):
        totalMeas=self._totalMeas
        #print(totalMeas)
        detectorIdx=0
        for paritygroup in self._parityMatchGroup:
            #print(paritygroup)
            #print([k-totalMeas for k in paritygroup])
            self._stimcircuit.append("DETECTOR", [stim.target_rec(k-totalMeas) for k in paritygroup])
            for k in paritygroup:
                self._measIdx_to_parityIdx[k].append(detectorIdx)
            detectorIdx+=1
        self._stimcircuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(k-totalMeas) for k in self._observable], 0)

    def add_reset(self, qubit: int):
        self._gatelists.append(Reset(qubit))
        self._stimcircuit.append("R", [qubit])

    def setShowNoise(self, show: bool):
        self._shownoise=show

    def __str__(self):
        str=""
        for gate in self._gatelists:
            if isinstance(gate, pauliNoise) and not self._shownoise:
                continue
            str+=gate.__str__()+"\n"
        return str


    def get_yquant_latex(self):
        """
        Convert the circuit (stored in self._gatelists) into a yquant LaTeX string.
        This version simply prints each gate (or noise box) in the order they appear,
        without grouping or any fancy logic.
        """
        lines = []
        # Begin the yquant environment
        lines.append("\\begin{yquant}")
        lines.append("")
        
        # Declare qubits and classical bits.
        # Note: Literal braces in the LaTeX code are escaped by doubling them.
        lines.append("% -- Qubits and classical bits --")
        lines.append("qubit {{$\\ket{{q_{{\\idx}}}}$}} q[{}];".format(self._qubit_num))
        lines.append("cbit {{$c_{{\\idx}} = 0$}} c[{}];".format(self._totalMeas))
        lines.append("")
        lines.append("% -- Circuit Operations --")
        
        # Process each gate in the order they were added.
        for gate in self._gatelists:
            if isinstance(gate, pauliNoise):
                # Print the noise box only if noise output is enabled.
                if self._shownoise:
                    lines.append("[fill=red!80]")
                    # The following format string produces, e.g.,:
                    # "box {$n_{8}$} q[2];"
                    lines.append("box {{$n_{{{}}}$}} q[{}];".format(gate._noiseindex, gate._qubitindex))
            elif isinstance(gate, TwoQGate):
                # Two-qubit gate (e.g., CNOT or CZ).
                if gate._name == "CNOT":
                    # Note: yquant syntax for a CNOT is: cnot q[target] | q[control];
                    line = "cnot q[{}] | q[{}];".format(gate._target, gate._control)
                elif gate._name == "CZ":
                    line = "cz q[{}] | q[{}];".format(gate._target, gate._control)
                lines.append(line)
            elif isinstance(gate, SingleQGate):
                # Single-qubit gate.
                if gate._name == "H":
                    line = "h q[{}];".format(gate._qubitindex)

                lines.append(line)
            elif isinstance(gate, Measurement):
                # Measurement is output as three separate lines.
                lines.append("measure q[{}];".format(gate._qubitindex))
                lines.append("cnot c[{}] | q[{}];".format(gate._measureindex, gate._qubitindex))
                lines.append("discard q[{}];".format(gate._qubitindex))
            elif isinstance(gate, Reset):
                # Reset is output as an initialization command.
                lines.append("init {{$\\ket0$}} q[{}];".format(gate._qubitindex))
            else:
                continue
        
        lines.append("")
        lines.append("\\end{yquant}")
        
        return "\n".join(lines)


def example_basic():
    """Basic example without noise."""
    circ = CliffordCircuit(3)
    circ.error_rate = 0.1
    circ.add_hadamard(0)
    circ.add_cnot(0, 1)
    circ.add_cnot(0, 2)
    circ.add_measurement(1)
    circ.add_measurement(2)
    # Convert scaler circuit to stim circuit
    stimcirc = circ.stimcircuit
    print(stimcirc)


def example_with_noise():
    """Example with depolarizing noise."""
    circ = CliffordCircuit(3)
    circ.error_rate = 0.1
    circ.add_depolarize(0)
    circ.add_hadamard(0)
    circ.add_depolarize(0)
    circ.add_depolarize(1)
    circ.add_cnot(0, 1)
    circ.add_cnot(0, 2)
    circ.add_depolarize(1)
    circ.add_measurement(1)
    circ.add_measurement(2)
    # Convert scaler circuit to stim circuit
    stimcirc = circ.stimcircuit
    print(stimcirc)


if __name__ == "__main__":
    example_basic()
    example_with_noise()
