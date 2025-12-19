#Test our strafied algoairhtm by comparing with STIM
#This test can go to larger scale



from scalerqec.Stratified.stratifiedLERcalc import stratifiedLERcalc
from scalerqec.Monte.monteLER import stimLERcalc
from scalerqec.Symbolic.symbolic import symbolicLER



error_rate=0.01
absolute_error=0.05
sample_size=800000
stim_sample_size=800000
num_subspace=6


all_test_files=["1cnot","1cnot1R","1cnoth","2cnot","2cnot2R","cnot1","cnot1","cnot01","cnot01h01","cnoth0","cnoth01","simple","simpleh","simpleMultiObs"]


def test_by_file_name(filepath):
    stimcalculator=stimLERcalc()
    ground_truth=stimcalculator.calculate_LER_from_file(stim_sample_size,filepath,error_rate)
    print("STIM result: ",ground_truth)


    tmp=stratifiedLERcalc(error_rate,sampleBudget=sample_size,num_subspace=num_subspace)
    tmp.parse_from_file(filepath)
    tmp.subspace_sampling()

    stratifiedresult=tmp.calculate_LER()
    print("Stratified result: "+str(stratifiedresult))

    if abs(stratifiedresult-ground_truth)>0:
        if ground_truth>0:
            assert (abs(stratifiedresult-ground_truth)/ground_truth)<absolute_error
        else:
            assert False




def test_all():
    for test_file in all_test_files:
        filepath="C:/Users/username/Documents/Sampling/stimprograms/small/"+test_file
        print("Testing file: ",test_file)
        test_by_file_name(filepath)


if __name__ == "__main__":


    test_all()