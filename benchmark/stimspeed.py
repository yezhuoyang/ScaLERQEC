import stim
import pymatching
from scaler.LERcalculator import *
import time




if __name__ == "__main__":

    distance=50

    print("Initialize circuit")

    circuit=CliffordCircuit(2)
    circuit.set_error_rate(0.001)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()    


    print("Compilation done!")
    start = time.perf_counter()      # high‑resolution timer

    sampler=new_stim_circuit.compile_sampler()

    end = time.perf_counter()
    print(f"Sampler takes time: {end - start:.4f} seconds")
    print("Sampler done!")

    start = time.perf_counter()      # high‑resolution timer


    print(sampler.sample(shots=10000))

    end = time.perf_counter()

    print(f"Elapsed wall‑clock time: {end - start:.4f} seconds")



#stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
#stim_circuit=rewrite_stim_code(str(stim_circuit))
#circuit.set_stim_str(stim_circuit)
#circuit.compile_from_stim_circuit_str(stim_circuit)          


    


#new_stim_circuit=circuit.get_stim_circuit()        






