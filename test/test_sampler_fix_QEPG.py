import stim
import numpy as np


from QEPG.QEPG import return_samples,return_samples_many_weights,return_detector_matrix,return_samples_many_weights_separate_obs_with_QEPG,compile_QEPG
from .test_QEPG_by_stim import *
import pymatching
from scalerqec.LERcalculator import *
from scalerqec.Clifford.clifford import *


'''
Convert integer to bool list 
'''
def convert_int_to_bool_list(value, length):
    """
    Convert an integer to a list of booleans of a given length.
    """
    return [bool(value & (1 << i)) for i in reversed(range(length))]



def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors


def stim_ground_truth_LER(circuit_file_path):
    stim_str=""
    with open(circuit_file_path, "r", encoding="utf-8") as f:
        stim_str = f.read()
    

    circuit=CliffordCircuit(4)   
    circuit.set_error_rate(0.1)  
    circuit.compile_from_stim_circuit_str(stim_str)
    stimcircuit=circuit.get_stim_circuit()

    detector_error_model = stimcircuit.detector_error_model(decompose_errors=False)
    shots=50000000
    num_erros=count_logical_errors(stimcircuit, shots)
    print(num_erros/shots)




def LER_small_circuit(circuit_file_path):

    stim_str=""
    with open(circuit_file_path, "r", encoding="utf-8") as f:
        stim_str = f.read()
    

    circuit=CliffordCircuit(4)   
    circuit.set_error_rate(0.1)  
    circuit.compile_from_stim_circuit_str(stim_str)
    stimcircuit=circuit.get_stim_circuit()


    total_noise = circuit.get_totalnoise()
    print("circuit noise number")
    print(total_noise)
    total_detector =len(circuit.get_parityMatchGroup())
    print("total number of detectors")
    print(total_detector)


    
    # Configure a decoder using the circuit.
    detector_error_model = stimcircuit.detector_error_model(decompose_errors=False)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)


    # TODO: Generate a ground truth table of the pymatching decoder output:
    all_inputs = []
    for i in range(0,1<<total_detector):
        # Convert the integer to a list of booleans
        bool_list = convert_int_to_bool_list(i, total_detector)
        # Print the list of booleans
        all_inputs.append(bool_list)

    all_predictions = matcher.decode_batch(all_inputs)

    # Print the predictions
    for i in range(len(all_inputs)):
        print("input: ", all_inputs[i], " prediction: ", all_predictions[i])


    graph=compile_QEPG(str(stimcircuit))

    wlist = [0,1,2,3,4,5,6,7]        # [2, 3, ..., 20]
    shotlist = [5000] * len(wlist)   # repeat 10000 same number of times


    result=return_samples_many_weights_separate_obs_with_QEPG(graph,wlist,shotlist)


    LER=0
    p=0.1

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
    





if __name__ == "__main__":
    # Example usage
    circuit_file_path = "stimprograms/small/1cnot"
    LER_small_circuit(circuit_file_path)
    #stim_ground_truth_LER(circuit_file_path)



    #value=0b1101
    #length=4
    #bool_list = convert_int_to_bool_list(value, length)
    #print(bool_list)  # Output: [True, False, True, True]

