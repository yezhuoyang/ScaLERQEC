'''
Generate the ground truth of logical error rate for surface code and several benchmark
'''
import stim
import pymatching
import numpy as np
from scaler.LERcalculator import *

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




def surface_groundTruth():
    distance=7
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)          



    new_stim_circuit=circuit.get_stim_circuit()        

    num_shots = 1000000
    num_logical_errors = count_logical_errors(new_stim_circuit, num_shots)



    error_rate= num_logical_errors / num_shots
    print(f"Distance: {distance}, Qubit number: {circuit.get_qubit_num()}, Total Measure: {circuit.get_totalMeas()}, Total Noise: {circuit.get_totalnoise()}, Error rate: {error_rate:.9e}")




surface_groundTruth()