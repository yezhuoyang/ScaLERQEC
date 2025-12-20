
from ..qepg import return_samples_many_weights_separate_obs, compile_QEPG, return_samples_many_weights_separate_obs_with_QEPG
from ..Clifford.clifford import *
import pymatching
import time
from ..QEC.noisemodel import NoiseModel
from ..QEC.qeccircuit import QECStab
from ..util import binomial_weight, subspace_size, format_with_uncertainty 


MIN_NUM_LE_EVENT = 50
SAMPLE_GAP=100


'''
Use strafified sampling algorithm to calculate the logical error rate
'''
class stratifiedLERcalc:
    def __init__(self, error_rate=0, sampleBudget=10000, num_subspace=30):
        self._num_detector=0
        self._num_noise=0
        self._error_rate=error_rate
        self._cliffordcircuit=CliffordCircuit(4)  

        self._LER=0
        """
        Use a dictionary to store the estimated subspace logical error rate
        """
        self._estimated_subspaceLER={}
        self._subspace_LE_count={}       
        self._subspace_sample_used={}             


        self._sampleBudget=sampleBudget
        self._num_subspace=num_subspace
        self._minW=0
        self._maxW=0

        self._stim_str_after_rewrite=""

        self._sample_used=0
        self._uncertainty=0


        self._circuit_level_code_distance=1


        self._QEPG_graph=None


    def parse_from_file(self,filepath):
        """
        Read the circuit, parse from the file
        """
        stim_str=""
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()
        
        self._cliffordcircuit.error_rate = self._error_rate  
        self._cliffordcircuit.compile_from_stim_circuit_str(stim_str)
        self._num_noise = self._cliffordcircuit.totalnoise
        self._num_detector=len(self._cliffordcircuit.parityMatchGroup)
        self._stim_str_after_rewrite=stim_str

        # Configure a decoder using the circuit.
        self._detector_error_model = self._cliffordcircuit.stimcircuit.detector_error_model(decompose_errors=True)
        self._matcher = pymatching.Matching.from_detector_error_model(self._detector_error_model)

        self._QEPG_graph=compile_QEPG(stim_str)


    def sample_all_subspace(self, shots_each_subspace=1000000):
        """
        Aggressively sample all the subspace.
        This function is only used to test the correctness of the algorithm.
        """
        wlist = list(range(0, self._num_noise + 1))
        slist = [shots_each_subspace] * len(wlist)
        detector_result,obsresult=return_samples_many_weights_separate_obs(self._stim_str_after_rewrite,wlist,slist)
        predictions_result = self._matcher.decode_batch(detector_result)


        for w in wlist:
            self._subspace_LE_count[w]=0
            self._estimated_subspaceLER[w]=0
            self._subspace_sample_used[w]=shots_each_subspace

        begin_index=0
        for w_idx, (w, quota) in enumerate(zip(wlist, slist)):
            observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
            # 2. batch-decode (decode_batch should accept ndarray) -------------------
               # shape (shots,) or (shots,1)
            predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()

            # 3. count mismatches in vectorised form ---------------------------------
            num_errors = np.count_nonzero(observables != predictions)

            self._subspace_LE_count[w]+=num_errors
            self._estimated_subspaceLER[w] = self._subspace_LE_count[w] / self._subspace_sample_used[w]

            #print(f"Subspace logical error {self._estimated_subspaceLER[w]}")
            #print(f"Logical error rate when w={w}: {self._estimated_subspaceLER[w]*binomial_weight(self._num_noise, w,self._error_rate):.6g}")
            begin_index+=quota


    def determine_range_to_sample(self,epsilon=0.01):
        """
        We need to be exact about the range of w we want to sample. 
        We don't want to sample too many subspaces, especially those subspaces with tiny binomial weights.
        This should comes from the analysis of the weight of each subspace.

        We use the standard deviation to approimxate the range
        """
        sigma=int(np.sqrt(self._error_rate*(1-self._error_rate)*self._num_noise))
        if sigma==0:
            sigma=1
        ep=int(self._error_rate*self._num_noise)
        self._minW=max(1,ep-5*sigma)
        self._maxW=max(2,ep+5*sigma)


    def subspace_sampling(self):
        """
        Sample around the subspaces.
        """
        self.determine_range_to_sample()
        """
        wlist store the subset of weights we need to sample and get
        correct logical error rate.
        
        In each subspace, we stop sampling until 100 logical error events are detected, or we hit the total budget.
        """
        wlist_need_to_sample = list(range(self._minW, self._maxW + 1))
        self._sample_used=0
        for weight in wlist_need_to_sample:
            self._subspace_LE_count[weight]=0
            self._subspace_sample_used[weight]=0

        # print("Weights need to sample: ")
        # print(wlist_need_to_sample)

        min_non_zero_weight=1e9
        while True:
            slist=[]
            wlist=[]
            """
            Case 1 to end the while loop: We have consumed all of our sample budgets
            """
            if(self._sample_used>self._sampleBudget):
                break

            for weight in wlist_need_to_sample:
                if(weight<=(self._circuit_level_code_distance-1)/2):
                    continue
                """
                If the subspace has been sampled enough, but logical error rate is still zero,
                also, the number of samples used in the subspace is comparable with the size of the subspace,
                we declare that the code distance is larger than the current weight.
                """
                if(weight+1<wlist_need_to_sample[-1]):
                    # if(weight>min_non_zero_weight):
                    #     continue
                    if(self._subspace_LE_count[weight]==0):
                        if(subspace_size(self._num_noise, weight)<2*self._subspace_sample_used[weight]):
                            self._circuit_level_code_distance=weight
                            continue                            
                        if(self._subspace_LE_count[weight+1]>=MIN_NUM_LE_EVENT and 2*self._subspace_sample_used[weight+1]<self._subspace_sample_used[weight]):
                            self._circuit_level_code_distance=weight
                            continue
                        else:
                            slist.append(max(2*self._subspace_sample_used[weight+1],SAMPLE_GAP))
                            wlist.append(weight)
                            self._subspace_sample_used[weight]+=max(2*self._subspace_sample_used[weight+1],SAMPLE_GAP)
                            self._sample_used+=max(2*self._subspace_sample_used[weight+1],SAMPLE_GAP)
                            continue

                if(self._subspace_LE_count[weight]>0):
                    min_non_zero_weight=min(weight,min_non_zero_weight)

                if(self._subspace_LE_count[weight]<MIN_NUM_LE_EVENT):
                    if(self._subspace_LE_count[weight]>=1):
                        sample_num_required=int(MIN_NUM_LE_EVENT/self._subspace_LE_count[weight])* self._subspace_sample_used[weight]
                        slist.append(sample_num_required)
                        self._subspace_sample_used[weight]+=sample_num_required  
                        self._sample_used+=sample_num_required
                    else:                   
                        slist.append(SAMPLE_GAP)
                        self._subspace_sample_used[weight]+=SAMPLE_GAP
                        self._sample_used+=SAMPLE_GAP
                    wlist.append(weight)
            """
            Case 2 to end the while loop: We have get 100 logical error events for all these subspaces
            """
            if(len(wlist)==0):
                break

            # print("wlist: ",wlist)
            # print("slist: ",slist)
            #detector_result,obsresult=return_samples_many_weights_separate_obs(self._stim_str_after_rewrite,wlist,slist)
            detector_result,obsresult=return_samples_many_weights_separate_obs_with_QEPG(self._QEPG_graph,wlist,slist)
            predictions_result = self._matcher.decode_batch(detector_result)
            #print("Result get!")

            
            begin_index=0
            for w_idx, (w, quota) in enumerate(zip(wlist, slist)):

                observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
                predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()

                # 3. count mismatches in vectorised form ---------------------------------
                num_errors = np.count_nonzero(observables != predictions)

                self._subspace_LE_count[w]+=num_errors
                self._estimated_subspaceLER[w] = self._subspace_LE_count[w] / self._subspace_sample_used[w]

                #print(f"Logical error rate when w={w}: {self._estimated_subspaceLER[w]*binomial_weight(self._num_noise, w,self._error_rate):.6g}")

                begin_index+=quota
            # print(self._subspace_LE_count)
            # print(self._subspace_sample_used)
        # print("Samples used:{}".format(self._sample_used))
        # print("circuit level code distance:{}".format(self._circuit_level_code_distance))

    # ----------------------------------------------------------------------
    # Calculate logical error rate
    # The input is a list of rows with logical errors
    def calculate_LER(self):
        self._LER=0
        for weight in range(1,self._num_noise+1):
            if weight in self._estimated_subspaceLER.keys():
                self._LER+=self._estimated_subspaceLER[weight]*binomial_weight(self._num_noise, weight,self._error_rate)
        return self._LER    

    def get_LER_subspace_no_weight(self,weight):
        return self._estimated_subspaceLER[weight]


    def get_LER_subspace(self,weight):
        return self._estimated_subspaceLER[weight]*binomial_weight(self._num_noise, weight,self._error_rate)


    def calculate_LER_from_file(self,filepath,pvalue):
        pass



    def clear_all(self):
        pass



    def calc_LER_from_QECcircuit(self, qeccirc:QECStab, noise_model:NoiseModel,repeat=1):
        qeccirc.construct_IR_standard_scheme()
        qeccirc.compile_stim_circuit_from_IR_standard()
        noisy_circuit = noise_model.reconstruct_clifford_circuit(qeccirc.circuit) 
        self._error_rate = noise_model.error_rate
        self._circuit_level_code_distance=qeccirc.d
        self._cliffordcircuit =  noisy_circuit
        self._num_noise = self._cliffordcircuit.totalnoise
        self._num_detector=len(self._cliffordcircuit.parityMatchGroup)
        self._stim_str_after_rewrite=str(self._cliffordcircuit.stimcircuit)
        # Configure a decoder using the circuit.
        self._detector_error_model = self._cliffordcircuit.stimcircuit.detector_error_model(decompose_errors=True)
        self._matcher = pymatching.Matching.from_detector_error_model(self._detector_error_model)
        self._QEPG_graph=compile_QEPG(self._stim_str_after_rewrite)


        ler_list=[]
        sample_used_list=[]
        time_list=[]

        for i in range(repeat):
            starttime = time.perf_counter()

            self.subspace_sampling()
            self.calculate_LER()
            ler_list.append(self._LER)
            sample_used_list.append(self._sample_used)  
            endtime = time.perf_counter()
            time_list.append(endtime - starttime)
        
        
        average_LER=sum(ler_list)/len(ler_list)
        average_sample_used=sum(sample_used_list)/len(sample_used_list)
        time_mean=sum(time_list)/len(time_list)
        ler_std = np.std(ler_list)
        sample_used_std = np.std(sample_used_list)
        time_std = np.std(time_list)
        print("Samples(ours): ", format_with_uncertainty(average_sample_used, sample_used_std))
        print("Time(our): ", format_with_uncertainty(time_mean, time_std))
        print("PL(ours): ", format_with_uncertainty(average_LER, ler_std))


if __name__ == "__main__":
    # tmp=stratifiedLERcalc(0.001,sampleBudget=15000000,num_subspace=5)
    # filepath="C:/Users/username/Documents/Sampling/stimprograms/small/simple"
    # tmp.parse_from_file(filepath)
    # tmp.sample_all_subspace(11*1000000)

    # LER=tmp.calculate_LER()

    # print(LER)

    # for weight in range(1,12):
    #     #print("LER in the subspace {} is {}".format(weight,tmp.get_LER_subspace_no_weight(weight)))    
    #     print("LER in the subspace {} is {}".format(weight,tmp.get_LER_subspace(weight)))


    qeccirc= QECStab(n=3,k=1,d=3)
    # Stabilizer generators
    qeccirc.add_stab("ZZI")
    qeccirc.add_stab("IZZ")
    qeccirc.set_logical_Z(0, "ZZZ")
    noise_model = NoiseModel(0.001) #Set the noise model
    # Set stabilizer parity measurement scheme, round of repetition
    qeccirc.scheme="Standard"
    qeccirc.rounds=2


    stratifiedcalculator = stratifiedLERcalc()
    stratifiedcalculator.calc_LER_from_QECcircuit(qeccirc, noise_model,repeat=1)

