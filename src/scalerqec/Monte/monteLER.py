from ..Clifford.clifford import *
import pymatching
from ..Clifford.stimparser import *
import time
import os
from contextlib import redirect_stdout

from ..qepg import compile_QEPG,return_samples_Monte_separate_obs_with_QEPG
from ..QEC.noisemodel import NoiseModel
from ..QEC.qeccircuit import QECStab

import sinter
import os




def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=False)
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



def format_with_uncertainty(value, std):
    """
    Format a value and its standard deviation in the form:
    1.23(±0.45)×10^k
    """
    if value == 0:
        return f"0(+{std:.2e})"
    exponent = int(np.floor(np.log10(abs(value))))
    coeff = value / (10**exponent)
    std_coeff = std / (10**exponent)
    return f"{coeff:.2f}(+{std_coeff:.2f})*10^{exponent}"




SAMPLE_GAP_INITIAL = 100
MAX_SAMPLE_GAP = 500000


'''
Use stim and Monte Calo sampling method to estimate the logical error rate
The sampler will finally decide how many samples to used.
Shot is the initial guess of how many samples to used.
We also need to estimate the uncertainty of the LER.
'''
class stimLERcalc:
    def __init__(self,time_budget = 10,samplebudget=100000 ,MIN_NUM_LE_EVENT=3):
        self._num_LER=0
        self._sample_used=0
        self._sample_needed=0
        self._uncertainty=0
        self._estimated_LER=0
        self._samplebudget=samplebudget
        self._MIN_NUM_LE_EVENT = MIN_NUM_LE_EVENT
        self._QEPG=None
        self._time_budget=time_budget



    def calculate_LER_from_QECircuit(self,qeccirc:QECStab, noise_model: NoiseModel , repeat=1):
        """
        Calculate the logical error rate from a QECStab object using Monte Carlo sampling.
        """
        qeccirc.construct_IR_standard_scheme()
        qeccirc.compile_stim_circuit_from_IR_standard()

        noisy_circuit = noise_model.reconstruct_clifford_circuit(qeccirc.circuit) 
        stim_circuit = noisy_circuit.stimcircuit
        self._QEPG=compile_QEPG(str(stim_circuit))


        detector_error_model = stim_circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)       

        error_rate = noise_model.error_rate
        Ler_list=[]
        samples_list=[]
        time_list=[]
        ler_count_list=[]
        for i in range(repeat):
            start = time.perf_counter()
            ler_count=0
            sampleused=0


            detector_result,obsresult=return_samples_Monte_separate_obs_with_QEPG(self._QEPG,error_rate,SAMPLE_GAP_INITIAL)
            predictions_result = matcher.decode_batch(detector_result)
            observables =  np.asarray(obsresult).ravel()                    # (shots,)
            predictions = np.asarray(predictions_result).ravel()
            num_errors = np.count_nonzero(observables != predictions)
            # 3. count mismatches in vectorised form ---------------------------------


            ler_count+=num_errors
            sampleused+=SAMPLE_GAP_INITIAL
            while ler_count<self._MIN_NUM_LE_EVENT and sampleused<self._samplebudget:

                if ler_count==0:
                    current_sample_gap=sampleused*10
                    current_sample_gap=min(current_sample_gap, MAX_SAMPLE_GAP)
                else:
                    current_sample_gap=min(int(self._MIN_NUM_LE_EVENT/ler_count)*sampleused, MAX_SAMPLE_GAP)

                detector_result,obsresult=return_samples_Monte_separate_obs_with_QEPG(self._QEPG,error_rate,current_sample_gap)
                predictions_result = matcher.decode_batch(detector_result)
                observables =  np.asarray(obsresult).ravel()                    # (shots,)
                predictions = np.asarray(predictions_result).ravel()
                num_errors = np.count_nonzero(observables != predictions)
                ler_count+=num_errors
                sampleused+=current_sample_gap

            ler_count_list.append(ler_count)
            Ler_list.append(ler_count/sampleused)
            samples_list.append(sampleused)
            elapsed = time.perf_counter() - start
            time_list.append(elapsed)


        ler_count_average=np.mean(ler_count_list)
        #print("Average number of logical errors: ", ler_count_average)
        std_ler_count=np.std(ler_count_list)
        self._estimated_LER=np.mean(Ler_list)
        self._sample_used=np.mean(samples_list)
        """
        Standard deviation
        """
        std_ler=np.std(Ler_list)
        std_sample=np.std(samples_list)
        #self.calculate_standard_error()
        time_mean=np.mean(time_list)
        time_std=np.std(time_list)
        print("Time(STIM): ", format_with_uncertainty(time_mean, time_std))
        print("PL(STIM): ", format_with_uncertainty(self._estimated_LER, std_ler))
        print("Nerror(STIM): ", format_with_uncertainty(ler_count_average, std_ler_count))
        print("Sample(STIM): ", format_with_uncertainty(self._sample_used, std_sample))      




    def calculate_LER_from_my_random_sampler(self, samplebudget, filepath, pvalue, repeat=1):
        circuit=CliffordCircuit(2)
        circuit.error_rate=pvalue
        self._samplebudget=samplebudget

        stim_str=""
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()
        self._QEPG=compile_QEPG(stim_str)

        stim_circuit=rewrite_stim_code(stim_str)
        circuit.stimcircuit = stim_circuit
        circuit.compile_from_stim_circuit_str(stim_circuit)           
        new_stim_circuit=circuit.stimcircuit      


        detector_error_model = new_stim_circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)       

        Ler_list=[]
        samples_list=[]
        time_list=[]
        ler_count_list=[]
        for i in range(repeat):
            
            ler_count=0
            sampleused=0
            start = time.time()


            detector_result,obsresult=return_samples_Monte_separate_obs_with_QEPG(self._QEPG,pvalue,SAMPLE_GAP_INITIAL)
            predictions_result = matcher.decode_batch(detector_result)
            observables =  np.asarray(obsresult).ravel()                    # (shots,)
            predictions = np.asarray(predictions_result).ravel()
            num_errors = np.count_nonzero(observables != predictions)
            # 3. count mismatches in vectorised form ---------------------------------


            ler_count+=num_errors
            sampleused+=SAMPLE_GAP_INITIAL
            while ler_count<self._MIN_NUM_LE_EVENT and sampleused<self._samplebudget:

                if ler_count==0:
                    current_sample_gap=sampleused*10
                    current_sample_gap=min(current_sample_gap, MAX_SAMPLE_GAP)
                else:
                    current_sample_gap=min(int(self._MIN_NUM_LE_EVENT/ler_count)*sampleused, MAX_SAMPLE_GAP)

                detector_result,obsresult=return_samples_Monte_separate_obs_with_QEPG(self._QEPG,pvalue,current_sample_gap)
                predictions_result = matcher.decode_batch(detector_result)
                observables =  np.asarray(obsresult).ravel()                    # (shots,)
                predictions = np.asarray(predictions_result).ravel()
                num_errors = np.count_nonzero(observables != predictions)
                ler_count+=num_errors
                sampleused+=current_sample_gap

            ler_count_list.append(ler_count)
            Ler_list.append(ler_count/sampleused)
            samples_list.append(sampleused)
            elapsed = time.time() - start
            time_list.append(elapsed)

        ler_count_average=np.mean(ler_count_list)
        #print("Average number of logical errors: ", ler_count_average)
        std_ler_count=np.std(ler_count_list)

        self._estimated_LER=np.mean(Ler_list)
        self._sample_used=np.mean(samples_list)
        """
        Standard deviation
        """
        std_ler=np.std(Ler_list)
        std_sample=np.std(samples_list)
        #self.calculate_standard_error()
        time_mean=np.mean(time_list)
        time_std=np.std(time_list)
        
        print("Time(STIM): ", format_with_uncertainty(time_mean, time_std))
        print("PL(STIM): ", format_with_uncertainty(self._estimated_LER, std_ler))
        print("Nerror(STIM): ", format_with_uncertainty(ler_count_average, std_ler_count))
        print("Sample(STIM): ", format_with_uncertainty(self._sample_used, std_sample))        
        return self._estimated_LER



    def calculate_LER_from_file_sinter(self,samplebudget,filepath,pvalue, repeat=1):
        circuit=CliffordCircuit(2)
        circuit.set_error_rate(pvalue)
        self._samplebudget=samplebudget

        stim_str=""
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()

        stim_circuit=rewrite_stim_code(stim_str)
        circuit.stimcircuit = stim_circuit
        circuit.compile_from_stim_circuit_str(stim_circuit)           
        new_stim_circuit=circuit.stimcircuit      

             
        Ler_list=[]
        samples_list=[]
        time_list=[]
        ler_count_list=[]
        for i in range(repeat):
            
            start = time.time()
            self._num_LER=0
            self._sample_used=0

            mytask=sinter.Task(
                            circuit=new_stim_circuit,
                            json_metadata={
                                'p': pvalue,
                                'd': 0,
                            },
                        )            
            samples = sinter.collect(
                num_workers=os.cpu_count(),
                max_shots=samplebudget,
                max_errors=self._MIN_NUM_LE_EVENT,
                tasks=[mytask],
                decoders=['pymatching'],
            )

            self._num_LER=samples[0].errors
            ler_count_list.append(self._num_LER)
            self._sample_used=samples[0].shots
            

            Ler_list.append(self._num_LER/self._sample_used)
            samples_list.append(self._sample_used)
            elapsed = time.time() - start
            time_list.append(elapsed)

        ler_count_average=np.mean(ler_count_list)
        #print("Average number of logical errors: ", ler_count_average)
        std_ler_count=np.std(ler_count_list)

        self._estimated_LER=np.mean(Ler_list)
        self._sample_used=np.mean(samples_list)
        """
        Standard deviation
        """
        std_ler=np.std(Ler_list)
        std_sample=np.std(samples_list)
        #self.calculate_standard_error()
        time_mean=np.mean(time_list)
        time_std=np.std(time_list)
        
        print("Time(STIM): ", format_with_uncertainty(time_mean, time_std))
        print("PL(STIM): ", format_with_uncertainty(self._estimated_LER, std_ler))
        print("Nerror(STIM): ", format_with_uncertainty(ler_count_average, std_ler_count))
        print("Sample(STIM): ", format_with_uncertainty(self._sample_used, std_sample))        
        return self._estimated_LER
            


    def calculate_LER_from_file(self,samplebudget,filepath,pvalue, repeat=1):
        circuit=CliffordCircuit(2)
        circuit.error_rate=pvalue
        self._samplebudget=samplebudget

        stim_str=""
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()

        stim_circuit=rewrite_stim_code(stim_str)
        circuit.stimcircuit = stim_circuit
        circuit.compile_from_stim_circuit_str(stim_circuit)           
        new_stim_circuit=circuit.stimcircuit   

        
        sampler = new_stim_circuit.compile_detector_sampler()
        detector_error_model = new_stim_circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)        


        Ler_list=[]
        samples_list=[]
        time_list=[]
        ler_count_list=[]
        for i in range(repeat):
            
            start = time.time()
            self._num_LER=0
            self._sample_used=0
            current_sample_gap=SAMPLE_GAP_INITIAL
            while self._num_LER<self._MIN_NUM_LE_EVENT:
                if self._num_LER==0 and self._sample_used>0:
                    current_sample_gap*=2
                    current_sample_gap=min(current_sample_gap, MAX_SAMPLE_GAP)
                elif self._num_LER>0:
                    current_sample_gap=min(int(self._MIN_NUM_LE_EVENT/self._num_LER)*self._sample_used, MAX_SAMPLE_GAP)
                self._sample_used+=current_sample_gap
                #self._num_LER+=count_logical_errors(new_stim_circuit,SAMPLE_GAP)


                detection_events, observable_flips = sampler.sample(current_sample_gap, separate_observables=True)
                predictions = matcher.decode_batch(detection_events)
                    # 3. count mismatches in vectorised form ---------------------------------
                num_errors = np.count_nonzero(observable_flips != predictions)
                self._num_LER+=num_errors

                self._estimated_LER=self._num_LER/self._sample_used
                #self.calculate_standard_error()
                # print("Current LER: ", self._num_LER)
                # print("Current logical error rate: ", self._num_LER/self._sample_used)
                # print("Current stdandard error: ", self._uncertainty)
                # print("Current sample used: ", self._sample_used)

                if self._sample_used>self._samplebudget:
                    #print("Sample budget reached, stop sampling")
                    if(self._num_LER>0):
                        self._sample_needed=int(self._sample_used*(100/self._num_LER))
                    else:
                        self._sample_needed=-1
                    break
                self._sample_needed=self._sample_used
            ler_count_list.append(self._num_LER)
            Ler_list.append(self._estimated_LER)
            samples_list.append(self._sample_used)
            elapsed = time.time() - start
            time_list.append(elapsed)

        ler_count_average=np.mean(ler_count_list)
        #print("Average number of logical errors: ", ler_count_average)
        std_ler_count=np.std(ler_count_list)

        self._estimated_LER=np.mean(Ler_list)
        self._sample_used=np.mean(samples_list)
        """
        Standard deviation
        """
        std_ler=np.std(Ler_list)
        std_sample=np.std(samples_list)
        #self.calculate_standard_error()
        time_mean=np.mean(time_list)
        time_std=np.std(time_list)
        
        print("Time(STIM): ", format_with_uncertainty(time_mean, time_std))
        print("PL(STIM): ", format_with_uncertainty(self._estimated_LER, std_ler))
        print("Nerror(STIM): ", format_with_uncertainty(ler_count_average, std_ler_count))
        print("Sample(STIM): ", format_with_uncertainty(self._sample_used, std_sample))        
        return self._estimated_LER
    

    def calculate_standard_error(self):
        """
        Calculate the standard error of the LER.
        """
        self._estimated_LER=self._num_LER/self._sample_used
        self._uncertainty = np.sqrt(self._estimated_LER*(1-self._estimated_LER)) / self._sample_used
        return self._uncertainty


    def get_sample_used(self):
        return self._sample_used



if __name__ == "__main__":

    base_dir = "C:/Users/username/Documents/Sampling/stimprograms/"
    result_dir = "C:/Users/username/Documents/Sampling/"


    p=0.0005
    code_type="hexagon"
    rel="hexagon/hexagon"
    dlist=[11]
    for d in dlist:
        stim_path = base_dir+rel+str(d)
        # 3) build your output filename:
        out_fname =  result_dir+str(p)+"-"+str(code_type)+str(d)+"-resultMonte.txt"     # e.g. "surface3-result.txt"
        # 4) redirect prints for just this file:
        with open(out_fname, "w") as outf, redirect_stdout(outf):
            print(f"---- Processing {stim_path} ----")

            calculator=stimLERcalc(20)
            # pass the string path into your function:
            ler = calculator.calculate_LER_from_my_random_sampler(500000000, str(stim_path), p,5)



    p=0.001
    code_type="hexagon"
    rel="hexagon/hexagon"
    dlist=[15]
    for d in dlist:
        stim_path = base_dir+rel+str(d)
        # 3) build your output filename:
        out_fname =  result_dir+str(p)+"-"+str(code_type)+str(d)+"-resultMonte.txt"     # e.g. "surface3-result.txt"
        # 4) redirect prints for just this file:
        with open(out_fname, "w") as outf, redirect_stdout(outf):
            print(f"---- Processing {stim_path} ----")

            calculator=stimLERcalc(10)
            # pass the string path into your function:
            ler = calculator.calculate_LER_from_my_random_sampler(500000000, str(stim_path), p,5)

