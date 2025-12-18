import stim
import pymatching
from .interface import *
from pathlib import Path
import numpy as np
from .stimparser import rewrite_stim_code




oneQGate_ = ["H", "P", "X", "Y", "Z"]
oneQGateindices={"H":0, "P":1, "X":2, "Y":3, "Z":4}


twoQGate_ = ["CNOT", "CZ"]
twoQGateindices={"CNOT":0, "CZ":1}

pauliNoise_ = ["I","X", "Y", "Z"]
pauliNoiseindices={"I":0,"X":1, "Y":2, "Z":3}


class SingeQGate:
    def __init__(self, gateindex, qubitindex):
        self._name = oneQGate_[gateindex]
        self._qubitindex = qubitindex

    def __str__(self):
        return self._name + "[" + str(self._qubitindex) + "]"


class TwoQGate:
    def __init__(self, gateindex, control, target):
        self._name = twoQGate_[gateindex]
        self._control = control
        self._target = target

    def __str__(self):
        return self._name + "[" + str(self._control) + "," + str(self._target)+ "]"


class pauliNoise:
    def __init__(self, noiseindex, qubitindex):
        self._name="n"+str(noiseindex)
        self._noiseindex= noiseindex
        self._qubitindex = qubitindex
        self._noisetype=0


    def set_noisetype(self, noisetype):
        self._noisetype=noisetype


    def __str__(self):
        return self._name +"("+pauliNoise_[self._noisetype] +")" +"[" + str(self._qubitindex) + "]"


class Measurement:
    def __init__(self,measureindex ,qubitindex):
        self._name="M"+str(measureindex)
        self._qubitindex = qubitindex
        self._measureindex=measureindex

    def __str__(self):
        return self._name + "[" + str(self._qubitindex) + "]"


class Reset:
    def __init__(self, qubitindex):
        self._name="R"
        self._qubitindex = qubitindex

    def __str__(self):
        return self._name + "[" + str(self._qubitindex) + "]"



#Class: CliffordCircuit
class CliffordCircuit:


    def __init__(self, qubit_num):
        self._qubit_num = qubit_num
        self._totalnoise=0
        self._totalMeas=0
        self._totalgates=0
        self._gatelists=[]
        self._error_rate=0
        self._index_to_noise={}
        self._index_to_measurement={}

        #self._index_to_measurement={}

        self._shownoise=False
        self._syndromeErrorTable={}
        #Store the repeat match group
        #For example, if we require M0=M1, M2=M3, then the match group is [[0,1],[2,3]]
        self._parityMatchGroup=[]
        self._observable=[]

        self._stim_str=None
        self._stimcircuit=stim.Circuit()


        #self._error_channel


    def set_stim_str(self, stim_str):
        self._stim_str=stim_str


    def set_error_rate(self, error_rate):
        self._error_rate=error_rate

    def get_stim_circuit(self):
        return self._stimcircuit


    def set_observable(self, observablemeasurements):
        self._observable=observablemeasurements


    def get_observable(self):
        return self._observable


    def set_parityMatchGroup(self, parityMatchGroup):
        self._parityMatchGroup=parityMatchGroup

    def get_parityMatchGroup(self):
        return self._parityMatchGroup

    def get_qubit_num(self):
        return self._qubit_num
    
    def get_totalnoise(self):
        return self._totalnoise

    def get_totalMeas(self):
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
    def compile_from_stim_circuit_str(self, stim_str):
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
        measure_stack=[]
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("DETECTOR("):
                meas_index = [token.strip() for token in stripped_line.split() if token.strip().startswith("rec")]
                meas_index = [int(x[4:-1]) for x in meas_index]
                parityMatchGroup.append([measure_line_to_measure_index[measure_stack[x]] for x in meas_index])
                current_line_index+=1
                continue
            elif stripped_line.startswith("OBSERVABLE_INCLUDE("):
                meas_index = [token.strip() for token in stripped_line.split() if token.strip().startswith("rec")]
                meas_index = [int(x[4:-1]) for x in meas_index]
                observable=[measure_line_to_measure_index[measure_stack[x]] for x in meas_index]
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
                self.add_cnot(control, target)


            elif gate == "M":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_measurement(qubit)

            elif gate == "H":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_hadamard(qubit)            

            elif gate == "S":
                qubit = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubit else qubit
                self.add_phase(qubit)    

            
            elif gate == "R":
                qubits = int(tokens[1])
                maxum_q_index=maxum_q_index if maxum_q_index>qubits else qubits
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



    def set_noise_type(self, noiseindex, noisetype):
        self._index_to_noise[noiseindex].set_noisetype(noisetype)


    def reset_noise_type(self):
        for i in range(self._totalnoise):
            self._index_to_noise[i].set_noisetype(0)

    def show_all_noise(self):
        for i in range(self._totalnoise):
            print(self._index_to_noise[i])


    def add_xflip_noise(self, qubit):
        self._stimcircuit.append("X_ERROR", [qubit], self._error_rate)
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1           



    def add_depolarize(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1        


    def add_cnot_no_noise(self, control, target):
        self._gatelists.append(TwoQGate(twoQGateindices["CNOT"], control, target))
        self._stimcircuit.append("CNOT", [control, target])        



    def add_cnot(self, control, target):
        self._stimcircuit.append("DEPOLARIZE1", [control], self._error_rate)
        self._gatelists.append(pauliNoise(self._totalnoise, control))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(pauliNoise(self._totalnoise, target))
        self._stimcircuit.append("DEPOLARIZE1", [target], self._error_rate)
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(TwoQGate(twoQGateindices["CNOT"], control, target))
        self._stimcircuit.append("CNOT", [control, target])


    def add_hadamard(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)        
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1        
        self._gatelists.append(SingeQGate(oneQGateindices["H"], qubit))
        self._stimcircuit.append("H", [qubit])

    def add_phase(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)   
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1      
        self._gatelists.append(SingeQGate(oneQGateindices["P"], qubit))
        self._stimcircuit.append("S", [qubit])

    def add_cz(self, qubit1, qubit2):
        self._gatelists.append(pauliNoise(self._totalnoise, qubit1))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(pauliNoise(self._totalnoise, qubit1))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1
        self._gatelists.append(TwoQGate(twoQGateindices["CZ"], qubit1, qubit2))     


    def add_paulix(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)   
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1     
        self._gatelists.append(SingeQGate(oneQGateindices["X"], qubit))
        self._stimcircuit.append("X", [qubit])

    def add_pauliy(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)  
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1    
        self._gatelists.append(SingeQGate(oneQGateindices["Y"], qubit))
        self._stimcircuit.append("Y", [qubit])

    def add_pauliz(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)  
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1    
        self._gatelists.append(SingeQGate(oneQGateindices["Z"], qubit))
        self._stimcircuit.append("Z", [qubit])


    def add_measurement_no_noise(self, qubit):
        self._gatelists.append(Measurement(self._totalMeas,qubit))
        self._stimcircuit.append("M", [qubit])
        #self._stimcircuit.append("DETECTOR", [stim.target_rec(-1)])
        self._index_to_measurement[self._totalMeas]=self._gatelists[-1]
        self._totalMeas+=1



    def add_measurement(self, qubit):
        self._stimcircuit.append("DEPOLARIZE1", [qubit], self._error_rate)  
        self._gatelists.append(pauliNoise(self._totalnoise, qubit))
        self._index_to_noise[self._totalnoise]=self._gatelists[-1]
        self._totalnoise+=1   
        self._gatelists.append(Measurement(self._totalMeas,qubit))
        self._stimcircuit.append("M", [qubit])
        #self._stimcircuit.append("DETECTOR", [stim.target_rec(-1)])
        self._index_to_measurement[self._totalMeas]=self._gatelists[-1]
        self._totalMeas+=1


    def compile_detector_and_observable(self):
        totalMeas=self._totalMeas
        #print(totalMeas)
        for paritygroup in self._parityMatchGroup:
            #print(paritygroup)
            #print([k-totalMeas for k in paritygroup])
            self._stimcircuit.append("DETECTOR", [stim.target_rec(k-totalMeas) for k in paritygroup])

        self._stimcircuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(k-totalMeas) for k in self._observable], 0)

        #print(self._stimcircuit)


    def add_reset(self, qubit):
        self._gatelists.append(Reset(qubit))
        self._stimcircuit.append("R", [qubit])

    def setShowNoise(self, show):
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
            elif isinstance(gate, SingeQGate):
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




from .qepg import return_samples,return_samples_many_weights,return_detector_matrix


import math


def binomial_weight(N, W, p):
    if N<200:
        return math.comb(N, W) * (p**W) * ((1 - p)**(N - W))
    else:
        lam = N * p
        # PMF(X=W) = e^-lam * lam^W / W!
        # Evaluate in logs to avoid overflow for large W, then exponentiate
        log_pmf = (-lam) + W*math.log(lam) - math.lgamma(W+1)
        return math.exp(log_pmf)





import time


def sample_time():
    distance=3
    p=0.00001
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(p)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()    
    start = time.perf_counter()      # high‑resolution timer
    result=return_samples(str(new_stim_circuit),2,10000)
    end = time.perf_counter()
    print(f"Elapsed wall‑clock time: {end - start:.4f} seconds")

    states, observables = [], []

    for j in range(0,10000):
        states.append(result[j][:-1])
        observables.append([result[j][-1]])

    # Configure a decoder using the circuit.
    detector_error_model = new_stim_circuit.detector_error_model(decompose_errors=False)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    shots=10000
    predictions = matcher.decode_batch(states)
    num_errors = 0
    for shot in range(shots):
        actual_for_shot = observables[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1


    print("Logical error rate when w=3")
    print(num_errors/shots)

def stratified_sampling():
    index=0


    distance=3
    p=0.001
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(p)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()      

    total_noise=circuit.get_totalnoise()


    # Configure a decoder using the circuit.
    detector_error_model = new_stim_circuit.detector_error_model(decompose_errors=False)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)


    wlist = [2,3,4,5,6,7]        # [2, 3, ..., 20]
    shotlist = [100000] * len(wlist)   # repeat 10000 same number of times

    print("Average number of noise: ")
    print(total_noise*p)

    result=return_samples_many_weights(str(new_stim_circuit),wlist,shotlist)



    LER=0
 

    for i in range(len(wlist)):
        states, observables = [], []

        for j in range(0,shotlist[i]):
            states.append(result[i][j][:-1])
            observables.append([result[i][j][-1]])


        shots=len(states)
        predictions = matcher.decode_batch(states)
        num_errors = 0
        for shot in range(shots):
            actual_for_shot = observables[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1


        print("Logical error rate when w="+str(wlist[i]))
        print(num_errors/shots)
        LER+=binomial_weight(total_noise, wlist[i], p)*(num_errors/shots)

    
    print(LER)



def read_file(path: str | Path) -> str:
    """Return the entire contents of *path* as a single string."""
    path = Path(path)
    try:
        # `with` closes the file automatically, even on errors
        with path.open("r", encoding="utf-8") as f:
            data = f.read()           # read the whole file at once
        return data
    except FileNotFoundError:
        print(f"File {path} not found.")
        raise
    except UnicodeDecodeError:
        print(f"Could not decode {path}; try a different encoding.")
        raise


def get_circuit():
    index=0

    distance=50
    p=0.0001
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(p)

    stim_circuit_str=read_file("stimprograms\cnot0")

    circuitmatrix=return_detector_matrix(stim_circuit_str)

    print("Circuit matrix:")

    print(circuitmatrix)

 


#get_circuit()



if __name__ == "__main__":
    stratified_sampling()
#sample_time()