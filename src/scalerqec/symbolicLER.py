import sys
from sympy import symbols, binomial, Rational, simplify, latex
from .clifford import *
from .stimparser import *
from .QEPGpython import *
import pymatching

# ----------------------------------------------------------------------
# Physical-error model
p = symbols('p')
q = 1 - p                     #   probability of "no error"
Px = Py = Pz = p / 3          #   probabilities of  X  Y  Z



def vec_to_idx(vec):
    """
    Converts a binary vector (A tuple or a list of 0s and 1s) to an integer index.
    The first element of the vector is treated as the most significant bit(MSB).
    Example: (1,0,1) -> 1*2^2 + 0*2^1 + 1*2^0 =5
    
    Args:
        vec: A tuple or list of binary digits(0 or 1).

    Returns:
        An integer representation of the binary vector.
        Returns 0 for an empty vector.
    """
    idx=0
    for bit in vec:
        idx = (idx<<1) | bit
    return idx


def idx_to_vec(idx, dimension):
    """
    Converts and integer index back to the binary vector of a given dimension.
    The first element of the vector is the most significant bit (MSB).
    Example: (5,3) - > (1,0,1)

    Args:
        idx: The non-negative integer index.
        dimension: The desired non-negative dimension(length) of the output binary vector

    Returns:
        A tuple of binary digits representing the index.
    """
    bits = []
    temp_idx = idx
    for _ in range(dimension):
        bits.append(temp_idx & 1)
        temp_idx >>= 1
    return tuple(reversed(bits))



def idx_to_bool_list(idx, dimension):
    """
    Converts and integer index back to the boolean vector of a given dimension.
    The first element of the vector is the most significant bit (MSB).
    Example: (5,3) - > [True,False,True]

    Args:
        idx: The non-negative integer index.
        dimension: The desired non-negative dimension(length) of the output binary vector

    Returns:
        A list of boolean representing the binary digits given the index.
    """
    bool_list = []
    temp_idx = idx
    for _ in range(dimension):
        bool_list.append(True if (temp_idx & 1) else False)
        temp_idx >>= 1
    return list(reversed(bool_list))


def xor_vec(vec_a, vec_b):
    """
    Performs element-wise XOR addition on two binary vectors.

    Args:
        vec_a: The first binary vector (tuple or list of 0s and 1s).
        vec_b: The second binary vector (tuple or list of 0s and 1s).

    Returns:
        A tuple representing the element-wise XOR of two vectors.
    """

    return tuple(a^b for a,b in zip(vec_a,vec_b))


MAX_degree=100
MAX_weight=12

'''
Use symbolic algorithm to calculate the probability.
Simply enumerate all possible cases
'''
class symbolicLER:
    def __init__(self,error_rate=0):
        self._num_detector=0
        self._num_noise=0
        self._error_rate=error_rate
        self._dp=None
        self._cliffordcircuit=CliffordCircuit(4)  
        self._graph=None
        self._all_predictions=None

        self._PROP_X = None
        self._PROP_Y = None
        self._PROP_Z = None

        self._error_row_indices = []
        self._subspace_LER={}

    def get_totalnoise(self):
        """
        Get the total number of noise in the circuit
        """
        return self._num_noise


    def parse_from_file(self,filepath):
        """
        Read the circuit, parse from the file
        """
        stim_str=""
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()
        

        self._cliffordcircuit.set_error_rate(self._error_rate)  
        self._cliffordcircuit.compile_from_stim_circuit_str(stim_str)
        self._num_noise = self._cliffordcircuit.get_totalnoise()
        self._num_detector=len(self._cliffordcircuit.get_parityMatchGroup())

        self._total_detector_outcome=(1<<(self._num_detector+1))

        self._graph=QEPGpython(self._cliffordcircuit)
        self._graph.backword_graph_construction()


    def generate_pymatching_table(self):
        """
        For all detector result, generate the prediction through pymatching.

        _all_predictions store all prediction by matching
        """
        # Configure a decoder using the circuit.
        stimcircuit=self._cliffordcircuit.get_stim_circuit()
        detector_error_model = stimcircuit.detector_error_model(decompose_errors=False)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

        all_inputs = []

        print("Total detector outcome: ", 1<<self._num_detector)
        for i in range(0,1<<self._num_detector):
            print("i=",i)
            print(1<<self._num_detector)
            # Convert the integer to a list of booleans
            bool_list = idx_to_bool_list(i, self._num_detector)
            # Print the list of booleans
            all_inputs.append(bool_list)
        #print(all_inputs)
        self._all_predictions = matcher.decode_batch(all_inputs)


    
    def calc_error_row_indices(self):
        """
        Based on the prediction result by pymatching of all possible input,
        build a list including all row indices that cause logical error
        """
        self._error_row_indices = []
        for row in range(0,self._total_detector_outcome):
            full_bool_list = idx_to_bool_list(row, self._num_detector+1)     
            """
            Get the current observable outcome represented by 
            the row index.
            """ 
            current_obs=full_bool_list[-1]
            ind_no_obs=vec_to_idx(full_bool_list[:-1])
            """
            Return the exact prediction result from the decoder, which
            is computed by function generate_pymatching_table(self)
            """
            real_obs=self._all_predictions[ind_no_obs][0]
            if(current_obs!=real_obs):
                self._error_row_indices.append(row)
        
        #print(self._error_row_indices)



    def initialize_single_pauli_propagation(self):
        """
        Calculate and store the table of the propagation result of single pauli error
        """
        self._PROP_X = []
        self._PROP_Y = []
        self._PROP_Z = []
        for noiseidx in range(self._num_noise):
            self._PROP_X.append(tuple(self._graph.sample_x_error(noiseidx)))
            self._PROP_Y.append(tuple(self._graph.sample_y_error(noiseidx)))
            self._PROP_Z.append(tuple(self._graph.sample_z_error(noiseidx)))

    

    def initialize_dp(self):
        """
        Given the circuit information, initialize the dp table for running the algorithm
        """
        self._dp = [ [ [0]*self._total_detector_outcome for _ in range(self._num_noise+1) ]
                                for _ in range(self._num_noise+1) ]
        


    def verify_table(self,i):
        sum=0
        for j in range(0,i+1):
            for vec_index in range(self._total_detector_outcome):
                sum+=self._dp[i][j][vec_index]
            # print(self._dp[i][j][vec_index])
        sum=simplify(sum)
        # print(i,sum)
        assert sum==1


    def dynamic_calculation_of_dp(self):
        # ----------------------------------------------------------------------
        # DP tables
        MAX_I = self._num_noise
        self.initialize_dp()


        col_size=self._num_detector+1
        # ----------------------------------------------------------------------
        # Fill   dp[i][j][·]   using the recurrence in Eq. (1)
        for i in range(0, MAX_I+1):
            self._dp[i][0][0] = (1-p)**i

            for j in range(1, i+1):           # j ≤ i
                if j > MAX_weight:
                    continue

                print("MAX_I=",MAX_I,"i=",i,"j=",j)
                for vec_idx in range(self._total_detector_outcome):

                    vec = idx_to_vec(vec_idx,col_size)

                    # 1) “no error’’ branch
                    acc = q * self._dp[i-1][j][vec_idx]

                    if j >= 1:
                        for (prob, prop) in ((Px, self._PROP_X[i-1]),
                                            (Py, self._PROP_Y[i-1]),
                                            (Pz, self._PROP_Z[i-1])):
                            

                            prev_vec = xor_vec(prop, vec)
                            acc += prob * self._dp[i-1][j-1][ vec_to_idx(prev_vec) ]

                    self._dp[i][j][vec_idx] = simplify(acc)
            if MAX_weight>=self._num_noise:
                self.verify_table(i)




    def calculate_LER_brute_force(self):
        """
        Enumerate all possible noise input to get the exact LER polynomial
        """
        pass


    # ----------------------------------------------------------------------
    # Calculate logical error rate
    # The input is a list of rows with logical errors
    def calculate_LER(self):
        self._LER=0
        for weight in range(1,self._num_noise+1):
            subLER=0
            for rowindex in self._error_row_indices:
                subLER+=self._dp[self._num_noise][weight][rowindex]

            self._subspace_LER[weight]=simplify(subLER)
            self._LER+=simplify(subLER)
        self._LER=simplify(self._LER).expand()
        #LER=LER.series(p, 0, MAX_degree).removeO()    # no .expand()

        print("LER polynomial: ", latex(self._LER))
        return self._LER


    def evaluate_LER(self,pval):
        return self._LER.evalf(subs={p:pval})


    def evaluate_LER_subspace(self,pval,weight):
        return self._subspace_LER[weight].evalf(subs={p:pval})


    def subspace_LER(self,weight):
        """
        Get the subspace LER for a given weight
        """
        if weight in self._subspace_LER:
            bernolli_coeff = binomial(self._num_noise, weight) * (p**weight) * ((1-p)**(self._num_noise-weight))
            subspaceLER=simplify(self._subspace_LER[weight]/bernolli_coeff)
            return subspaceLER.expand()
        else:
            raise ValueError("Subspace LER for weight {} is not calculated.".format(weight))


    
    def calculate_LER_from_file(self,filepath,pvalue):
        """
        The most import function. Read the stim circuit, calculate the exact polynomial 
        of LER, and evaluate with the value of p(physical error rate).

        Following steps are included:
             Step1:   Parse the circuit from the file, inject depolarization noise
             Step2:   Compile the STIM detector graph, generate the entire prediction table
             Stem3:   Construct the QEPG graph
             Step4:   Calculate all row indices in the table that will cause logical error
             Step5:   Use dynamic algorithm to calculate the probability of measuring any possible outcomes
             Step6:   Sum up all probability in the row with logical error

        Args:
             filepath: The path of the file with the STIM circuit
             pvalue:   The physical error rate

        Returns:
             A floating value which store the final logical error rate
        """
        self._error_rate=pvalue
        self.parse_from_file(filepath)
        print("---Step2: Generate the prediction table---")
        self.generate_pymatching_table()
        print("---Step2: construction QEPG--------------")
        self.initialize_single_pauli_propagation()
        print("---Step3: calculating error indices--------------")
        self.calc_error_row_indices()
        print("---Step4: dynamic algorithm--------------")
        self.dynamic_calculation_of_dp()
        self.calculate_LER()
        return self.evaluate_LER(pvalue)



if __name__=="__main__":


    tmp=symbolicLER(0.001)
    filepath="C:/Users/username/Documents/Sampling/stimprograms/small/simple"
    print(tmp.calculate_LER_from_file(filepath,0.001))


    num_noise=tmp._num_noise

    # for weight in range(1,11):
    #     print("LER in the subspace {} is {}".format(weight,tmp.evaluate_LER_subspace(0.001,weight)))        


    for weight in range(1,12):
        print("SubspaceLER {} is {}".format(weight,tmp.subspace_LER(weight)))        
