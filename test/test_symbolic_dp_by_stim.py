#Test the correctness of dp algorithm of LER calculation by comparing the result with STIM



from scaler.StratifiedLERcalc import StratifiedLERcalc
from scaler.monteLER import MonteLERcalc
from scaler.symbolicLER import SymbolicLERcalc



error_rate=0.001
absolute_error=0.05
sample_size=1000000000


all_test_files=["1cnot","1cnot1R","1cnoth","2cnot","2cnot2R","cnot1","cnot1","cnot01","cnot01h01","cnoth0","cnoth01","simple","simpleh","simpleMultiObs","repetition3r2","surface3r1","surface3r2"]




def test_by_file_name(filepath):
    symbolic_calculator=SymbolicLERcalc(error_rate)
    ground_truth=symbolic_calculator.calculate_LER_from_file(filepath,error_rate)
    print("Exact ground truth: ",ground_truth)

    stimcalculator=MonteLERcalc()
    stimresult=stimcalculator.calculate_LER_from_file(sample_size,filepath,error_rate)
    print("STIM result: ",stimresult)

    if abs(stimresult-ground_truth)>0:
        if ground_truth>0:
            assert (abs(stimresult-ground_truth)/ground_truth)<absolute_error
        else:
            assert False



def test_all():
    for test_file in all_test_files:
        filepath="C:/Users/username/Documents/Sampling/stimprograms/small/"+test_file
        test_by_file_name(filepath)



if __name__ == "__main__":

    test_all()








