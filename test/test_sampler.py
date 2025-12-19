#Test the correctness of the sampler in QEPG
from scalerqec.qepg import return_samples_with_noise_vector, return_samples_many_weights, return_detector_matrix
from scalerqec.Clifford.clifford import *
from test.test_QEPG_by_stim import transpile_stim_with_noise_vector
import pymatching



error_rate=0.001
absolute_error=0.05
sample_size=5000
num_subspace=3
weight=8



def convert_Floyd_sample_to_noise_vector(sample,num_noise):
    """
    Input:  Floyd samples with the form [(7, 1), (1, 2)]
    Output: The seven'th noise has type X, the first one has type Y
    """
    noise_vector=[0]*num_noise*3
    for i in range(len(sample)):
        if sample[i][1]==1:
            noise_vector[sample[i][0]]=1
        elif sample[i][1]==2:
            noise_vector[sample[i][0]+num_noise]=1
        elif sample[i][1]==3:
            noise_vector[sample[i][0]+2*num_noise]=1
    return noise_vector
    

     
def test_by_file_name(filepath):
    stim_str=""
    with open(filepath, "r", encoding="utf-8") as f:
        stim_str = f.read()
    noise_vector,samples=return_samples_with_noise_vector(stim_str,weight,sample_size)
    print("Floyd samples: ",noise_vector)

    print("samples output: ",samples)
    circuit=CliffordCircuit(3)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()
    num_noise = circuit.get_totalnoise()


    detectorMatrix=np.array(return_detector_matrix(str(stim_str)))
    print("Detector matrix: ", detectorMatrix)
    detectorMatrix = detectorMatrix.T          # or: np.transpose(detector_matrix)


    noise_vector=[convert_Floyd_sample_to_noise_vector(x,num_noise) for x in noise_vector]

    print("Noise vector: ", noise_vector)


    for i in range(len(noise_vector)):
        detector_result=transpile_stim_with_noise_vector(str(new_stim_circuit),noise_vector[i],num_noise)
        print("-------------Detector result from stim: -------------")
        print(detector_result)


        #print(dectectorMatrix.shape, noise_vector.shape)
        mydetectorresult=samples[i]


        print("-------------My Detector result: -------------")
        print(mydetectorresult)

        assert((detector_result==list(mydetectorresult)))
        
        print("-------------------Pass test!-------------------------------")


"""
We simply enumerate all the noise vector with weight 1, and check if the result is correct
"""
def calc_W1_LER_subspace_by_stim(filepath):
    stim_str=""
    with open(filepath, "r", encoding="utf-8") as f:
        stim_str = f.read()

    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.001)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()
    num_noise = circuit.get_totalnoise()

    detector_error_model = new_stim_circuit.detector_error_model(decompose_errors=False)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    LER_count=0

    all_x_inputs=[]
    all_x_outputs=[]
    for current_index in range(num_noise):
        noise_vector=[0]*(num_noise*3)
        noise_vector[current_index]=1
        detector_result=transpile_stim_with_noise_vector(str(new_stim_circuit),noise_vector,num_noise)

        all_x_inputs.append(detector_result[:-1])
        all_x_outputs.append(detector_result[-1])

    all_x_predictions = matcher.decode_batch(all_x_inputs)
    for shot in range(len(all_x_inputs)):
        if all_x_outputs[shot]!=all_x_predictions[shot]:
            LER_count+=1

    print( "LER count: ", LER_count)


    all_y_inputs=[]
    all_y_outputs=[]
    for current_index in range(num_noise):
        noise_vector=[0]*(num_noise*3)
        noise_vector[num_noise+current_index]=1
        detector_result=transpile_stim_with_noise_vector(str(new_stim_circuit),noise_vector,num_noise)

        all_y_inputs.append(detector_result[:-1])
        all_y_outputs.append(detector_result[-1])

    all_y_predictions = matcher.decode_batch(all_y_inputs)
    for shot in range(len(all_y_inputs)):
        if all_y_outputs[shot]!=all_y_predictions[shot]:
            LER_count+=1

    print( "Y LER count: ", LER_count)

    all_z_inputs=[]
    all_z_outputs=[]
    for current_index in range(num_noise):
        noise_vector=[0]*(num_noise*3)
        noise_vector[num_noise*2+current_index]=1
        detector_result=transpile_stim_with_noise_vector(str(new_stim_circuit),noise_vector,num_noise)

        all_z_inputs.append(detector_result[:-1])
        all_z_outputs.append(detector_result[-1])

    all_z_predictions = matcher.decode_batch(all_z_inputs)
    for shot in range(len(all_z_inputs)):
        if all_z_outputs[shot]!=all_z_predictions[shot]:
            LER_count+=1

    print( "LER count: ", LER_count)
    print( "LER count prob: ", LER_count/(3*num_noise))






"""
We simply enumerate all the noise vector with weight 1, and check if the result is correct
"""
def calc_W2_LER_subspace_by_stim(filepath):
    stim_str=""
    with open(filepath, "r", encoding="utf-8") as f:
        stim_str = f.read()

    circuit=CliffordCircuit(3)
    circuit.set_error_rate(0.001)
    circuit.compile_from_stim_circuit_str(stim_str)
    new_stim_circuit=circuit.get_stim_circuit()
    num_noise = circuit.get_totalnoise()

    detector_error_model = new_stim_circuit.detector_error_model(decompose_errors=False)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    LER_count=0
    combination_list = [(1,1), (1,2),(1,3), (2,1),(2,2),(2,3), (3,1),(3,2),(3,3)]

    all_inputs=[]
    all_outputs=[]

    print("Num noise: ", num_noise)
    for index_1 in range(num_noise):
        print("Index 1: ", index_1)
        for index_2 in range(index_1+1,num_noise):
            for (type_1,type_2) in combination_list:
                noise_vector=[0]*(num_noise*3)
                if type_1==1:
                    noise_vector[index_1]=1
                elif type_1==2:
                    noise_vector[index_1+num_noise]=1
                elif type_1==3:
                    noise_vector[index_1+2*num_noise]=1

                if type_2==1:
                    noise_vector[index_2]=1
                elif type_2==2:
                    noise_vector[index_2+num_noise]=1
                elif type_2==3:
                    noise_vector[index_2+2*num_noise]=1


                detector_result=transpile_stim_with_noise_vector(str(new_stim_circuit),noise_vector,num_noise)

                all_inputs.append(detector_result[:-1])
                all_outputs.append(detector_result[-1])
    
    all_predictions = matcher.decode_batch(all_inputs)
    for shot in range(len(all_predictions)):
        if all_outputs[shot]!=all_predictions[shot]:
            LER_count+=1

    print( "LER count: ", LER_count)
    print( "LER count prob: ", LER_count/(num_noise*(num_noise-1)*9))
   


if __name__ == "__main__":


    filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon3"
    calc_W1_LER_subspace_by_stim(filepath)
    #test_by_file_name(filepath)