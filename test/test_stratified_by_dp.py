#Test our stratified algorithm by comparing it with dp algorithm
#This test only work for small scale

from scalerqec.Stratified.stratifiedLERcalc import stratifiedLERcalc
from scalerqec.stimLER import stimLERcalc
from scalerqec.Symbolic.symbolicLER import symbolicLER





all_test_files=["1cnot","1cnot1R","1cnoth","2cnot","2cnot2R","cnot1","cnot1","cnot01","cnot01h01","cnoth0","cnoth01","simple","simpleh","simpleMultiObs","repetition3r2","surface3r1"]



error_rate=0.001
absolute_error=0.05
sample_size=100000
num_subspace=3


def test_by_file_name(filepath):
    symbolic_calculator=symbolicLER(error_rate)
    ground_truth=symbolic_calculator.calculate_LER_from_file(filepath,error_rate)
    print("Exact ground truth: ",ground_truth)


    tmp=stratifiedLERcalc(error_rate,sampleBudget=sample_size,num_subspace=num_subspace)
    tmp.parse_from_file(filepath)
    tmp.sample_all_subspace(sample_size)

    num_noise=symbolic_calculator.get_totalnoise()
    for weight in range(1,num_noise):
        print("Weight: ", weight)
        ground_truth=symbolic_calculator.evaluate_LER_subspace(error_rate,weight)
        print("Ground truth: ", ground_truth)
        my_result=tmp.get_LER_subspace(weight)
        print("My result: ", my_result)
        if abs(my_result-ground_truth)>0:
            if ground_truth>0:
                assert (abs(my_result-ground_truth)/ground_truth)<absolute_error
            else:
                assert False




if __name__ == "__main__":


    rootfilepath="C:/Users/username/Documents/Sampling/stimprograms/small/"

    for test_file in all_test_files:
        filepath=rootfilepath+test_file
        print("Test file: ", filepath)
        test_by_file_name(filepath)