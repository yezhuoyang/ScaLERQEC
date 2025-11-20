
from QEPG.QEPG import return_samples,return_samples_many_weights,return_detector_matrix,return_samples_many_weights_numpy,return_samples_many_weights_separate_obs, compile_QEPG, return_samples_many_weights_separate_obs_with_QEPG, return_samples_with_fixed_QEPG
from ScaLER.clifford import *
import math
import pymatching
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import binom
from contextlib import redirect_stdout

import warnings
from scipy.optimize import OptimizeWarning
import pickle
import time



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


# Define the inverse transform: y → 1/2 * 1 / (1 + e^y)
def inv_logit_half(y):
    return 0.5 / (1 + np.exp(y))

def binomial_weight(N, W, p):
    return binom.pmf(W, N, p)
    #return math.comb(N, W) * (p**W) * ((1 - p)**(N - W))

def linear_function(x, a, b):
    """
    Linear function for curve fitting.
    """
    return a * x + b


# def modified_linear_function(x, a, b,c,d):
#     """
#     Linear function for curve fitting.
#     """
#     return a * x + b+c/(x-d)


def modified_linear_function_with_d(x, a, b, c, d):
    eps   = 1e-12
    delta = (x - d)**0.5
    delta = np.where(np.abs(delta) < eps, np.sign(delta)*eps, delta)
    return a * x + b + c / delta



# Strategy A: keep the model safe near the pole
def modified_linear_function(d):
    def tempfunc(x,a,b,c,d=d):
        return modified_linear_function_with_d(x, a, b, c, d)
    return tempfunc



def modified_sigmoid_function(x, a, b,c,d):
    """
    Modified sigmoid function for curve fitting.
    This function is used to fit the S-curve.
    """
    z = a*x + b + c/((x - d)**0.5)
    # ignore overflows in exp → exp(z) becomes np.inf, so 0.5/(1+inf) = 0.0
    with np.errstate(over='ignore'):
        y = 0.5 / (1 + np.exp(z))
    return y

def quadratic_function(x, a, b,c):
    """
    Linear function for curve fitting.
    """
    return a * x**2+b*x + c


def poly_function(x, a, b,c,d):
    """
    Linear function for curve fitting.
    """
    return a * x**3+b*x**2 + c*x+d



# Redefine turning point where the 2nd term is still significant in dy/dw
def refined_sweat_spot(alpha, beta, t, ratio=0.05):
    # We define turning point by solving: 1/alpha = ratio * (1/2) * beta / (w - t)^{3/2}
    # => (w - t)^{3/2} = (ratio * beta * alpha) / 2
    # => w = t + [(ratio * beta * alpha / 2)]^{2/3}
    return t + ((ratio * beta * alpha / 2) ** (2 / 3))




"""
Return the estimated sigma of y(w)
"""
def sigma_estimator(N,M):
    return np.sqrt(N**2*(N-M)/(M*(N-1)*(N-2*M)**2))


"""
Return the estimated sigma of Pw
"""
def subspace_sigma_estimator(N,M):
    return np.sqrt(M*(N-M)/(N-1))/N

def bias_estimator(N, M):
    """
    Bias = E[y(w)] - y(w)
    Estimated by: (1/2) * f''(P_w) * Var(P_w)
    where f(x) = ln(1/(2x) - 1)
    """
    # Pw = M / N
    # var_Pw = sigma_estimator(N, M)**2
    # f2 = 4 / (1 - 2 * Pw)**2 + 1 / Pw**2
    # bias = 0.5 * f2 * var_Pw
    # return 0
    return 1/2*(N/M)*(N-4*M)/(N-2*M)**2*(N-M)/(N-1)


def show_bias_estimator(N, M):
    """
    Bias = E[y(w)] - y(w)
    Estimated by: (1/2) * f''(P_w) * Var(P_w)
    where f(x) = ln(1/(2x) - 1)
    """
    Pw = M / N
    return (1 - Pw) / (2 * Pw * N)


def subspace_size(num_noise, weight):
    """
    Calculate the size of the subspace
    """
    return math.comb(num_noise, weight)

# def scurve_function(x, mu, sigma):
#     cdf_values = 0.5*norm.cdf(x, loc=mu, scale=sigma)
#     return cdf_values


def scurve_function(x, center, sigma):
    return 0.5/(1+np.exp(-(x - center) / sigma))
    #return 0*x



def scurve_function_with_distance(x, cd, mu, sigma):
    """
    Piece-wise S-curve:
        0                          for x < cd
        0.5 * Φ((x - μ) / σ)       for x ≥ cd
    where Φ is the standard normal CDF.
    """
    x = np.asarray(x)                      # ensure array
    y = 0.5 * norm.cdf(x, loc=mu, scale=sigma)
    return np.where(x < cd, 0.0, y)        # vectorised “if”




def evenly_spaced_ints(minw, maxw, N):
    if N == 1:
        return [minw]
    if N > (maxw - minw + 1):
        return list(range(minw, maxw + 1))
     
    # Use high-resolution linspace, round, then deduplicate
    raw = np.linspace(minw, maxw, num=10 * N)
    rounded = sorted(set(map(int, raw)))
    
    # Pick N evenly spaced indices from the unique set
    indices = np.linspace(0, len(rounded) - 1, num=N, dtype=int)
    return [rounded[i] for i in indices]


def r_squared(y_true, y_pred, clip=False):
    """
    Compute the coefficient of determination (R²).

    Parameters
    ----------
    y_true : array-like
        Observed data.
    y_pred : array-like
        Model-predicted data (same length as y_true).
    clip : bool, default False
        If True, negative R² values are clipped to 0 so the
        score lies strictly in the interval [0, 1].
    Returns
    -------
    float
        The R² statistic.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    ss_res = np.sum((y_true - y_pred) ** 2)        # residual sum of squares
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) # total sum of squares

    # Handle the degenerate case where variance is zero
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    r2 = 1.0 - ss_res / ss_tot
    return max(0.0, r2) if clip else r2


'''
Use strafified sampling + Scurve fitting  algorithm to calculate the logical error rate
'''
class stratified_Scurve_LERcalc:

    def __init__(self, error_rate=0, sampleBudget=10000, k_range=3, num_subspace=5,beta=4):
        self._num_detector=0
        self._num_noise=0
        self._error_rate=error_rate
        self._cliffordcircuit=CliffordCircuit(4)  

        self._LER=0
        """
        Use a dictionary to store the estimated subspace logical error rate,
        how many samples have been used in each subspace
        """
        self._estimated_subspaceLER={}
        self._subspace_LE_count={}
        self._estimated_subspaceLER_second={}
        self._subspace_sample_used={}

        self._sampleBudget=sampleBudget
        self._sample_used=0
        self._circuit_level_code_distance=1
        self._t=1
        self._num_subspace=num_subspace
        """
        minw and maxw store the range of subspace we need to fit.
        This is determined by the uncertainty value
        """
        self._minw=1
        self._maxw=10000000000000
        """
        self._saturatew is the weight of the subspace where the 
        logical error get satureated
        """
        self._saturatew=10000000000000       
        self._has_logical_errorw=0
        self._estimated_wlist=[]

        self._stim_str_after_rewrite=""

        self._mu=0
        self._sigma=0

        #In the area we are interested in, the maximum value of the logical error rate
        self._rough_value_for_subspace_LER=0

        self._stratified_succeed=False

        self._k_range=k_range
        self._QEPG_graph=None

        self._R_square_score=0
        self._beta=beta

        self._sweat_spot=None


        self._MIN_NUM_LE_EVENT = 100
        self._SAMPLE_GAP=100
        self._MAX_SAMPLE_GAP=1000000
        self._MAX_SUBSPACE_SAMPLE=5000000

        self._ratio=0.05
        self._max_PL= 0.15
        


    def set_sample_bound(self, MIN_NUM_LE_EVENT,SAMPLE_GAP, MAX_SAMPLE_GAP, MAX_SUBSPACE_SAMPLE):
        """
        Set the sample bound for the subspace sampling
        """
        self._MIN_NUM_LE_EVENT=MIN_NUM_LE_EVENT
        self._SAMPLE_GAP=SAMPLE_GAP
        self._MAX_SAMPLE_GAP=MAX_SAMPLE_GAP
        self._MAX_SUBSPACE_SAMPLE=MAX_SUBSPACE_SAMPLE


    def clear_all(self):
        self._estimated_subspaceLER={}
        self._subspace_LE_count={}
        self._estimated_subspaceLER_second={}
        self._subspace_sample_used={}
        self._sample_used=0
        self._LER=0        
        self._estimated_wlist=[]
        self._saturatew=10000000000000               
        self._minw=self._t+1
        self._maxw=10000000000000
        self._cliffordcircuit=CliffordCircuit(4)  
        self._R_square_score=0


    def calc_logical_error_rate_with_fixed_w(self, shots, w):
        """
        Calculate the logical error rate with fixed w
        """
        result= return_samples_with_fixed_QEPG(self._QEPG_graph,w,shots)
        # self._sample_used+=shots
        # if w not in self._subspace_LE_count.keys():
        #     self._subspace_LE_count[w]=0
        #     self._subspace_sample_used[w]=shots
        # else:
        #     self._subspace_sample_used[w]+=shots
        arr=np.asarray(result)
        states=arr[:,:-1]
        observables=arr[:,-1]
        predictions =np.squeeze(self._matcher.decode_batch(states))
        num_errors = np.count_nonzero(observables != predictions)
        # self._subspace_LE_count[w]+=num_errors
        return num_errors/shots


    '''
    Use binary search to determine the exact number of errors 
    that give saturate logical error rate
    We just try 10 samples

    TODO: Restructure the function.
    Add the threshold as an input parameter.
    '''
    def binary_search_upper(self,low,high, shots):
        left=low
        right=high
        epsion=self._max_PL
        while left<right:
            mid=(left+right)//2
            er=self.calc_logical_error_rate_with_fixed_w(shots,mid)
            if er>epsion:
                right=mid
            else:
                left=mid+1
        return left


    def binary_search_lower(self,low,high, shots=5000):
        left=low
        right=high
        epsion=0.002
        while left<right:
            mid=(left+right)//2
            er=self.calc_logical_error_rate_with_fixed_w(shots,mid)
            if er>epsion:
                right=mid
            else:
                left=mid+1
        return left


    def determine_lower_w(self):
        self._has_logical_errorw=self.binary_search_lower(self._t+1,self._num_noise//10)
        #self._has_logical_errorw=self._t+100


    def determine_saturated_w(self,shots=1000):
        """
        Use binary search to determine the minw and maxw
        """
        #self._saturatew=self._num_detector//30
        self._saturatew=self.binary_search_upper(self._minw,self._num_noise//10,shots)
        #print("Self._saturatew: ",self._saturatew)



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
        self._stim_str_after_rewrite=stim_str

        # Configure a decoder using the circuit.
        self._detector_error_model = self._cliffordcircuit.get_stim_circuit().detector_error_model(decompose_errors=True)
        self._matcher = pymatching.Matching.from_detector_error_model(self._detector_error_model)

        self._QEPG_graph=compile_QEPG(stim_str)


    def determine_range_to_sample(self):
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
        self._minw=max(self._t+1,ep-self._k_range*sigma)
        self._maxw=max(self._num_subspace,ep+self._k_range*sigma)
        self._maxw=min(self._maxw,self._num_noise)



    def subspace_sampling(self):
        """
        wlist store the subset of weights we need to sample and get
        correct logical error rate.
        
        In each subspace, we stop sampling until 100 logical error events are detected, or we hit the total budget.
        """
        #wlist_need_to_sample = list(range(self._minw, self._maxw + 1))
        #wlist_need_to_sample=evenly_spaced_ints(self._sweat_spot,self._saturatew,self._num_subspace)
        
        
        wlist_need_to_sample=evenly_spaced_ints(self._sweat_spot,self._has_logical_errorw,self._num_subspace)
        
        
        #print("wlist_need_to_sample: ",wlist_need_to_sample)
        for weight in wlist_need_to_sample:
            if not weight in self._estimated_wlist:
                self._estimated_wlist.append(weight)
                self._subspace_LE_count[weight]=0
                self._subspace_sample_used[weight]=0

        #print(wlist_need_to_sample)
        self._sample_used=0
        total_LE_count=0
        while True:
            x_list = [x for x in self._estimated_subspaceLER.keys() if (self._estimated_subspaceLER[x] < 0.5 and self._estimated_subspaceLER[x]>0)]

            slist=[]
            wlist=[]
            """
            Case 1 to end the while loop: We have consumed all of our sample budgets
            """
            if(self._sample_used>self._sampleBudget):
                break

            for weight in wlist_need_to_sample:
                """
                When we declare the circuit level code distance, we don't need to sample these subspaces
                """
                if(self._subspace_sample_used[weight]>self._MAX_SUBSPACE_SAMPLE):
                    continue


                if(self._subspace_LE_count[weight]<self._MIN_NUM_LE_EVENT):
                    if(self._subspace_LE_count[weight]>=1):
                        """
                        For larger subspaces, when we have already get some logical error, 
                        we can estimate how many we still need to sample
                        """
                        sample_num_required=int(self._MIN_NUM_LE_EVENT/self._subspace_LE_count[weight])* self._subspace_sample_used[weight]
                        if sample_num_required>self._MAX_SAMPLE_GAP:
                            sample_num_required=self._MAX_SAMPLE_GAP
                        slist.append(sample_num_required)
                        self._subspace_sample_used[weight]+=sample_num_required  
                        self._sample_used+=sample_num_required
                    else:        
                        """
                        For larger subspaces, if we have not get any logical error, then we double the sample size
                        """
                        sample_num_required=max(self._SAMPLE_GAP,self._subspace_sample_used[weight]*10)
                        if sample_num_required>self._MAX_SAMPLE_GAP:
                            sample_num_required=self._MAX_SAMPLE_GAP
                        slist.append(sample_num_required)
                        self._subspace_sample_used[weight]+=sample_num_required
                        self._sample_used+=sample_num_required
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
            # print("Result get!")

            
            begin_index=0
            for w_idx, (w, quota) in enumerate(zip(wlist, slist)):

                observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
                predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()

                # 3. count mismatches in vectorised form ---------------------------------
                num_errors = np.count_nonzero(observables != predictions)
                total_LE_count+=num_errors
                self._subspace_LE_count[w]+=num_errors
                self._estimated_subspaceLER[w] = self._subspace_LE_count[w] / self._subspace_sample_used[w]


                # print(f"Logical error rate when w={w}: {self._estimated_subspaceLER[w]*binomial_weight(self._num_noise, w,self._error_rate):.6g}")

                begin_index+=quota
            # print(self._subspace_LE_count)
            # print("Subspace LE count: ",self._subspace_LE_count)
            # print("self._subspace_sample_used: ",self._subspace_sample_used)

        
        # print("Samples used:{}".format(self._sample_used))
        #print("circuit level code distance:{}".format(self._circuit_level_code_distance))
        #print(self._subspace_LE_count)



    def subspace_sampling_to_fit_curve(self,sampleBudget):

        """
        After we determine the minw and maxw, we generate an even distribution of points 
        between minw and maxw.

        The goal is for the curve fitting in the next step to get more accurate.
        """

        wlist=evenly_spaced_ints(self._has_logical_errorw,self._saturatew,self._num_subspace)
        for weight in wlist:
            if not (weight in self._estimated_wlist):
                self._estimated_wlist.append(weight)
        slist=[sampleBudget//self._num_subspace]*len(wlist)


        #detector_result,obsresult=return_samples_many_weights_separate_obs(self._stim_str_after_rewrite,wlist,slist)
        detector_result,obsresult=return_samples_many_weights_separate_obs_with_QEPG(self._QEPG_graph,wlist,slist)
        predictions_result = self._matcher.decode_batch(detector_result)

        for w,s in zip(wlist,slist):
            if not w in self._subspace_LE_count.keys():
                self._subspace_LE_count[w]=0
                self._subspace_sample_used[w]=s
                self._estimated_subspaceLER[w]=0
            else:
                self._subspace_sample_used[w]+=s        


        begin_index=0
        for w_idx, (w, quota) in enumerate(zip(wlist, slist)):

            observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
            predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()

            # 3. count mismatches in vectorised form ---------------------------------
            num_errors = np.count_nonzero(observables != predictions)

            self._subspace_LE_count[w]+=num_errors
            self._estimated_subspaceLER[w]=self._subspace_LE_count[w]/self._subspace_sample_used[w]
            #print("Logical error rate when w={} ".format(w)+str(self._estimated_subspaceLER[w]))
            begin_index+=quota


    def calculate_R_square_score(self):
        y_observed = [self._estimated_subspaceLER[x] for x in self._estimated_wlist]
        y_predicted = [scurve_function(x,self._mu,self._sigma) for x in self._estimated_wlist]
        #y_predicted = [scurve_function_with_distance(x,self._mu,self._sigma) for x in self._estimated_wlist]
        r2 = r_squared(y_observed, y_predicted)
        #print("R^2 score: ", r2)
        return r2


    # ----------------------------------------------------------------------
    # Calculate logical error rate
    # The input is a list of rows with logical errors
    def calculate_LER(self):
        self._LER=0
        for weight in range(1,self._num_noise+1):
            if weight in self._estimated_subspaceLER.keys():
                self._LER+=self._estimated_subspaceLER[weight]*binomial_weight(self._num_noise, weight,self._error_rate)
        return self._LER    


    def get_LER_subspace(self,weight):
        return self._estimated_subspaceLER[weight]*binomial_weight(self._num_noise, weight,self._error_rate)




    def fit_linear_area(self):
        x_list = [x for x in self._estimated_subspaceLER.keys() if (self._estimated_subspaceLER[x] < 0.5 and self._estimated_subspaceLER[x]>0)]

        #y_list = [np.log(0.5/self._estimated_subspaceLER[x]-1) for x in x_list]
        y_list = [np.log(0.5/self._estimated_subspaceLER[x]-1)-bias_estimator(self._subspace_sample_used[x],self._subspace_LE_count[x]) for x in x_list]
        sigma_list= [sigma_estimator( self._subspace_sample_used[x],self._subspace_LE_count[x]) for x in x_list]


        popt, pcov = curve_fit(
            linear_function,
            x_list,
            y_list,
            sigma=sigma_list,          # <-- use sigma_list as weights
        )


        sigma=int(np.sqrt(self._error_rate*(1-self._error_rate)*self._num_noise))
        if sigma==0:
            sigma=1
        ep=int(self._error_rate*self._num_noise)
        self._minw=max(self._t+1,ep-self._k_range*sigma)
        self._maxw=max(2,ep+self._k_range*sigma)



        self._a,self._b= popt[0] , popt[1]

        #Plot the fitted line
        x_fit = np.linspace(min(x_list), max(x_list), 1000)

        y_fit = linear_function(x_fit, self._a, self._b)

        #print("Fitted parameters: a={}, b={}".format(self._a,self._b))
        plt.figure()
        plt.plot(x_fit, y_fit, label='Fitted line', color='orange')
        plt.scatter(x_list, y_list, label='Data points', color='blue')


        # ── NEW: highlight linear-fit window ───────────────────────────────────
        plt.axvline(self._minw, color='red',  linestyle='--', linewidth=1.2, label=r'$w_{\min}$')
        plt.axvline(self._maxw, color='green', linestyle='--', linewidth=1.2, label=r'$w_{\max}$')
        plt.axvspan(self._minw, self._maxw, color='gray', alpha=0.15)  # translucent fill
        # ───────────────────────────────────────────────────────────────────────


        alpha = -1 / self._a
        mu = alpha * self._b

        # ── NEW: annotate alpha and mu ─────────────────────────────────────
        textstr = '\n'.join((
            r'$\alpha=%.4f$' % alpha,
            r'$\mu=%.4f$' % mu
        ))
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        # ───────────────────────────────────────────────────────────────────

        plt.xlabel('Weight')
        plt.ylabel('Linear')
        plt.title('Linear Fit of S-curve')
        plt.legend()
        plt.savefig("total_linear_fit.pdf", dpi=300)
        plt.close()


 
    def fit_log_S_model(self,filename,time=None):
        x_list = [x for x in self._estimated_subspaceLER.keys() if (self._estimated_subspaceLER[x] < 0.5 and self._estimated_subspaceLER[x]>0 and self._subspace_LE_count[x]>=(self._MIN_NUM_LE_EVENT//5))]

        sigma_list= [sigma_estimator( self._subspace_sample_used[x],self._subspace_LE_count[x]) for x in x_list]
        y_list = [np.log(0.5/self._estimated_subspaceLER[x]-1)-bias_estimator(self._subspace_sample_used[x],self._subspace_LE_count[x]) for x in x_list]

        print("Saturated weight: ",self._saturatew)
        print("LE count: ",self._subspace_LE_count)
        print("Sample used: ",self._subspace_sample_used)


        non_zero_indices=[x for x in x_list if self._estimated_subspaceLER[x]>0]
        upper_bound_code_distance=min(non_zero_indices) if len(non_zero_indices)>0 else self._circuit_level_code_distance*10

        center = self._saturatew /2 
        sigma    = self._saturatew/7          # centre in the middle of that span
        b=self._b
        a=self._a
        c=self._beta
        alpha= -1/self._a
        initial_guess  = (a, b ,alpha)

        #print("Initial guess d: ",int((self._circuit_level_code_distance+upper_bound_code_distance)/2))
        sigma=int(np.sqrt(self._error_rate*(1-self._error_rate)*self._num_noise))
        if sigma==0:
            sigma=1
        ep=int(self._error_rate*self._num_noise)
        self._minw=max(self._t+1,ep-self._k_range*sigma)
        self._maxw=max(2,ep+self._k_range*sigma)


        self._num_detector
        self._num_noise


        beta=alpha
        initial_guess  = (a, b ,beta)
        # ── lower bounds for [param1, param2, param3, param4]


        lower = [ min(self._a*5,self._a*0.2), min(self._b*0.2,self._b*5),  min(beta*0.2,beta*5)]

        # ── upper bounds for [param1, param2, param3, param4]
        upper = [ max(self._a*5,self._a*0.2), max(self._b*0.2,self._b*5) , max(beta*0.2,beta*5)]


        popt, pcov = curve_fit(
            modified_linear_function(self._t),
            x_list,
            y_list,
            p0=initial_guess,          # len(initial_guess) must be 4 and within the bounds above
            bounds=(lower, upper),     # <-- tuple with two arrays
            maxfev=50_000              # or max_nfev in newer SciPy
        )

        self._codedistance = 0
        # Extract the best-fit parameter (alpha)
        self._a,self._b,self._c= popt[0] , popt[1], popt[2]


        #print("circuit d:",self._circuit_level_code_distance)
        y_list =  [np.log(0.5/self._estimated_subspaceLER[x]-1)-bias_estimator(self._subspace_sample_used[x],self._subspace_LE_count[x]) for x in x_list]
        y_predicted = [modified_linear_function_with_d(x,self._a,self._b,self._c,self._t) for x in x_list]
        #y_predicted = [scurve_function_with_distance(x,self._mu,self._sigma) for x in self._estimated_wlist]
        self._R_square_score = r_squared(y_list, y_predicted)
        #print("R^2 score: ", self._R_square_score)

        #Plot the fitted line
        x_fit = np.linspace(self._t+1, max(x_list), 1000)

        y_fit = modified_linear_function_with_d(x_fit, self._a, self._b,self._c,self._t)

        self.calc_logical_error_rate_after_curve_fitting()
        
        alpha= -1/self._a
        self._sweat_spot = int(refined_sweat_spot(alpha, self._c, self._t, ratio=self._ratio))
        if self._sweat_spot<ep:
            self._sweat_spot=ep
        if self._sweat_spot<=self._t:
            self._sweat_spot=self._t+1
        #self._sweat_spot=self._t+100
        

        sweat_spot_y = modified_linear_function_with_d(self._sweat_spot, self._a, self._b, self._c, self._t)

        sample_cost_list= [self._subspace_sample_used[x] for x in x_list]

        #print("Fitted parameters: a={}, b={}, c={}, d={}".format(self._a, self._b, self._c, self._d))

        # Setup the plot
        fig, ax = plt.subplots(figsize=(7, 5))


        # Plot histogram-style bars for the y values
        bar_container = ax.bar(
            x_list,
            y_list,
            width=0.6,  # Adjust width if needed
            align='center',
            color='orange',
            edgecolor='orange',
            label='Data histogram'
        )

        # Overlay error bars on top of bars
        ax.errorbar(
            x_list,
            y_list,
            yerr=sigma_list,
            fmt='o',
            color='black',
            capsize=3,
            markersize=1,
            elinewidth=1,
            label='Error bars'
        )


        # Fit curve
        ax.plot(x_fit, y_fit, label=f'Fitted line, R2={self._R_square_score:.4f}', color='blue', linestyle='--')

        # Sweat spot marker
        ax.scatter(self._sweat_spot, sweat_spot_y, color='purple', marker='o', s=50, label='Sweet Spot')
        ax.text(self._sweat_spot*1.1, sweat_spot_y*1.1, 'Sweet Spot', ha='center',color='purple', fontsize=10)

        # Region: Fault-tolerant (green)
        ax.axvspan(0, self._t, color='green', alpha=0.15)
        ax.text(self._t / 2, max(y_list) * 1.8, 'Fault\ntolerant', ha='center', color='green',fontsize=8)


        ax.axvspan(self._t, self._saturatew, color='yellow', alpha=0.10)
        ax.text((self._t+ self._saturatew) / 2, max(y_list)*1.2, 'Curve fitting', ha='center', fontsize=15)

        # Region: Critical area (gray)
        ax.axvspan(self._minw, self._maxw, color='gray', alpha=0.2)
        ax.axvline(self._minw, color='red', linestyle='--', linewidth=1.2, label=r'$w_{\min}$')
        ax.axvline(self._maxw, color='green', linestyle='--', linewidth=1.2, label=r'$w_{\max}$')
        ax.text((self._minw + self._maxw) / 2, max(y_list) * 1.8, r'$5\sigma$ Critical Region', ha='center', fontsize=10)

        ax.axvspan(self._saturatew,self._saturatew+12, color='red', alpha=0.15)
        ax.text(self._saturatew+6, max(y_list) * 2.8, 'Saturation', ha='center',color='red', fontsize=10)

        # Sample cost annotations (scientific notation)
        num_points_to_annotate = 5
        indices = np.linspace(0, len(x_list) - 1, num=num_points_to_annotate, dtype=int)
        for i in indices:
            x, y, s = x_list[i], y_list[i], sample_cost_list[i]
            if s > 0:
                s_str = "{0:.1e}".format(s)
                base, exp = s_str.split('e')
                label = r'${0}\times 10^{{{1}}}$'.format(base, int(exp))
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=7)

        # Side annotation box
        text_lines = [
            r'$N_{LE}^{Clip}=%d$' % self._MIN_NUM_LE_EVENT,
            r'$N_{sub}^{Gap}=%d$' % self._MAX_SAMPLE_GAP,
            r'$N_{sub}^{Max}=%d$' % self._MAX_SUBSPACE_SAMPLE,
            r'$N_{total}=%d$' % self._sample_used,
            r'$r_{sweat}=%.2f$' % self._ratio,
            r'$\alpha=%.4f$' % alpha,
            r'$\mu =%.4f$' % (alpha * self._b),
            r'$\beta=%.4f$' % self._c,
            r'$w_{\min}=%d$' % self._minw,
            r'$w_{\max}=%d$' % self._maxw,
            r'$w_{sweet}=%d$' % self._sweat_spot,
            r'$\#\mathrm{detector}=%d$' % self._num_detector,
            r'$\#\mathrm{noise}=%d$' % self._num_noise,
            r'$P_L={0}\times 10^{{{1}}}$'.format(*"{0:.2e}".format(self._LER).split('e'))
        ]
        if time is not None:
            text_lines.append(r'$\mathrm{Time}=%.2f\,\mathrm{s}$' % time)

        fig.subplots_adjust(right=0.75)
        fig.text(0.78, 0.5, '\n'.join(text_lines),
                 fontsize=7, va='center', ha='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))

        # Final formatting
        ax.set_xlabel('Weight')
        ax.set_ylabel(r'$\log\left(\frac{0.5}{\mathrm{LER}} - 1\right)$')
        ax.set_title('Fitted log-S-curve')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(filename, format='pdf', bbox_inches='tight')  # `dpi` optional
        plt.close()


    '''
    Fit the distribution by 1/2-e^{alpha/W}
    '''
    def fit_Scurve(self):
        if self._stratified_succeed:
            self._saturatew=self._maxw

        center = self._saturatew /2 
        sigma    = self._saturatew/7          # centre in the middle of that span
        initial_guess  = (center, sigma )


        popt, pcov = curve_fit(
            scurve_function, 
            self._estimated_wlist, 
            [self._estimated_subspaceLER[x] for x in self._estimated_wlist], 
            p0=initial_guess
        )

        self._codedistance = 0
        # Extract the best-fit parameter (alpha)
        self._mu,self._sigma = popt[0] , popt[1]
        return self._codedistance,self._mu,self._sigma


    def ground_truth_subspace_sampling(self):
        """
        Sample around the subspaces.
        This is the ground truth value to test the accuracy of the curve fitting.
        """
        sigma=int(np.sqrt(self._error_rate*(1-self._error_rate)*self._num_noise))
        if sigma==0:
            sigma=1
        ep=int(self._error_rate*self._num_noise)
        minw=max(self._t+1,ep-self._k_range*sigma)
        maxw=max(self._num_subspace,ep+self._k_range*sigma)
        maxw=min(maxw,self._num_noise)
        wlist_need_to_sample = list(range(minw, maxw + 1))
        self._ground_sample_used=0
        self._ground_estimated_subspaceLER={}
        self._ground_subspace_LE_count={}
        self._ground_subspace_sample_used={}
        for weight in wlist_need_to_sample:
            self._ground_subspace_LE_count[weight]=0
            self._ground_subspace_sample_used[weight]=0        

        while True:
            slist=[]
            wlist=[]        

            if(self._ground_sample_used>self._sampleBudget):
                break
            
            for weight in wlist_need_to_sample:
                if(self._ground_subspace_sample_used[weight]>self._MAX_SUBSPACE_SAMPLE):
                    continue
                if(self._ground_subspace_LE_count[weight]==0):
                    wlist.append(weight)
                    sample_num_required=min(self._MAX_SAMPLE_GAP,self._ground_subspace_sample_used[weight]*10)
                    self._ground_subspace_sample_used[weight]+=sample_num_required  
                    self._ground_sample_used+=sample_num_required
                    slist.append(sample_num_required)
                    continue
                if(self._ground_subspace_LE_count[weight]<self._MIN_NUM_LE_EVENT):
                    sample_num_required=int(self._MIN_NUM_LE_EVENT/self._ground_subspace_LE_count[weight])* self._ground_subspace_sample_used[weight]
                    if sample_num_required>self._MAX_SAMPLE_GAP:
                        sample_num_required=self._MAX_SAMPLE_GAP
                    self._ground_subspace_sample_used[weight]+=sample_num_required  
                    self._ground_sample_used+=sample_num_required
                    wlist.append(weight)
                    slist.append(sample_num_required)

            if(len(wlist)==0):
                break
            #detector_result,obsresult=return_samples_many_weights_separate_obs(self._stim_str_after_rewrite,wlist,slist)
            
            print("Ground truth wlist: ",wlist)
            print("Ground truth slist: ",slist)
            
            detector_result,obsresult=return_samples_many_weights_separate_obs_with_QEPG(self._QEPG_graph,wlist,slist)
            predictions_result = self._matcher.decode_batch(detector_result)

            
            begin_index=0
            for w_idx, (w, quota) in enumerate(zip(wlist, slist)):

                observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
                predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()

                # 3. count mismatches in vectorised form ---------------------------------
                num_errors = np.count_nonzero(observables != predictions)

                self._ground_subspace_LE_count[w]+=num_errors
                self._ground_estimated_subspaceLER[w] = self._ground_subspace_LE_count[w] / self._ground_subspace_sample_used[w]


                print(f"Logical error rate when w={w}: {self._ground_estimated_subspaceLER[w]*binomial_weight(self._num_noise, w,self._error_rate):.6g}")

                begin_index+=quota
            print(self._ground_subspace_LE_count)
            print(self._ground_subspace_sample_used)
        print("Samples used:{}".format(self._ground_sample_used))




    def calc_logical_error_rate_after_curve_fitting(self):
        #self.fit_Scurve()
        self._LER=0

        sigma=int(np.sqrt(self._error_rate*(1-self._error_rate)*self._num_noise))
        if sigma==0:
            sigma=1
        ep=int(self._error_rate*self._num_noise)
        self._minw=max(self._t+1,ep-self._k_range*sigma)
        self._maxw=max(2,ep+self._k_range*sigma)

        for weight in range(self._minw,self._maxw+1):
            """
            If the weight is in the estimated list, we use the estimated value
            Else, we use the curve fitting value

            If the weight is less than the minw, we just declare it as 0
            """
            if weight in self._estimated_subspaceLER.keys():
                self._LER+=self._estimated_subspaceLER[weight]*binomial_weight(self._num_noise, weight,self._error_rate)
                #print("Weight: ",weight," LER: ",self._estimated_subspaceLER[weight]*binomial_weight(self._num_noise, weight,self._error_rate))
            else:
                fitted_subspace_LER=modified_sigmoid_function(weight,self._a,self._b,self._c,self._t)
                self._LER+=fitted_subspace_LER*binomial_weight(self._num_noise,weight,self._error_rate)
                #print("Weight: ",weight," LER: ",fitted_subspace_LER*binomial_weight(self._num_noise, weight,self._error_rate))
            #self._LER+=scurve_function(weight,self._mu,self._sigma)*binomial_weight(self._num_noise,weight,self._error_rate)
        return self._LER




    def plot_scurve(self, filename,title="S-curve"):
        """Plot the S-curve and its discrete estimate."""
        keys   = list(self._estimated_subspaceLER.keys())
        values = [self._estimated_subspaceLER[k] for k in keys]
        sigma_list= [subspace_sigma_estimator(self._subspace_sample_used[k],self._subspace_LE_count[k]) for k in keys]
        fig, ax = plt.subplots()

        # bars ── discrete estimate
        ax.bar(keys, values,
            color='tab:orange',
            alpha=0.8,
            label='Estimated subspace LER by sampling')

        ax.errorbar(
            keys,
            values,
            yerr=sigma_list,
            fmt='none',
            ecolor='black',
            capsize=3,
            elinewidth=1,
            label='LER error bars'
        )

        # smooth S-curve
        x = np.linspace(self._t + 0.1, self._saturatew, 1000)
        y = modified_sigmoid_function(x, self._a, self._b, self._c, self._t)
        ax.plot(x, y,
                color='tab:blue',
                linewidth=2.0,
                label='Fitted S-curve',linestyle='--')

        # Fault-tolerant area
        ax.axvspan(0, self._t, color='green', alpha=0.15)
        ax.text(self._t / 2, max(values)/2, 'Fault\ntolerant', ha='center', color='green', fontsize=8)


        ax.axvspan(self._t, self._saturatew, color='yellow', alpha=0.10)
        ax.text((self._t+ self._saturatew) / 2, max(values)/2, 'Curve fitting', ha='center', fontsize=10)


        ax.axvspan(self._saturatew,self._saturatew+12, color='red', alpha=0.15)
        ax.text(self._saturatew+6, max(values)/2, 'Saturation', ha='center',color='red', fontsize=10)

        # Region: Critical area (gray)
        ax.axvspan(self._minw, self._maxw, color='gray', alpha=0.2)
        ax.axvline(self._minw, color='red', linestyle='--', linewidth=1.2, label=r'$w_{\min}$')
        ax.axvline(self._maxw, color='green', linestyle='--', linewidth=1.2, label=r'$w_{\max}$')
        ax.text((self._minw + self._maxw) / 2, max(values)/2, r'$5\sigma$ Critical Region', ha='center', fontsize=10)


        # Labels and legend
        ax.set_xlabel('Weight')
        ax.set_ylabel('Logical Error Rate in subspace')
        ax.set_title(f"S-curve of {title} (PL={self._LER:.2e})")
        ax.legend()

        # Integer ticks on x-axis
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Layout and save
        plt.tight_layout()
        fig.savefig(filename+".pdf", format='pdf', bbox_inches='tight')  # `dpi` optional
        plt.close(fig)


    def set_t(self, t):
        """
        Set the t value for the S-curve fitting.
        This is used to determine the range of subspace we need to sample.
        """
        self._t = t

    
    """
    In this function, we just try to sample the linear area
    """
    def fast_calculate_LER_from_file(self,filepath,pvalue,codedistance,figname,titlename, repeat=1):
        self._error_rate=pvalue
        self._circuit_level_code_distance=codedistance
        ler_list=[]
        sample_used_list=[]
        r_squared_list=[]
        Nerror_list=[]
        time_list=[]
        for i in range(repeat):
            self.clear_all()
            self.parse_from_file(filepath)
            start = time.time()
            self.determine_lower_w()
            self.determine_saturated_w()
            self.subspace_sampling()
            self.fit_linear_area()
            tmptime=time.time()


            self.fit_log_S_model(figname+"-R"+str(i)+"Final.pdf",tmptime-start)
            self.calc_logical_error_rate_after_curve_fitting()

            end = time.time()
            time_list.append(end - start)

            self.plot_scurve(figname+".pdf",titlename)
            r_squared_list.append(self._R_square_score)
            self._sample_used=np.sum(list(self._subspace_sample_used.values()))
            # print("Final LER: ",self._LER)
            # print("Total samples used: ",self._sample_used)
            ler_list.append(self._LER)
            sample_used_list.append(self._sample_used)
            Nerror_list.append(sum(self._subspace_LE_count.values()))

        # Compute means
        self._LER = np.mean(ler_list)
        self._sample_used = np.mean(sample_used_list)

        # Compute standard deviations
        ler_std = np.std(ler_list)
        sample_used_std = np.std(sample_used_list)
        r2_mean = np.mean(r_squared_list)
        r2_std = np.std(r_squared_list)
        Nerror_mean = np.mean(Nerror_list)
        Nerror_std = np.std(Nerror_list)

        time_mean = np.mean(time_list)
        time_std = np.std(time_list)

        # Print with scientific ± formatting
        print("k: ", self._k_range)
        print("beta: ",self._beta)
        print("Subspaces: ", self._num_subspace)
        print("R2: ", format_with_uncertainty(r2_mean, r2_std))
        print("Samples(ours): ", format_with_uncertainty(self._sample_used, sample_used_std))
        print("Time(our): ", format_with_uncertainty(time_mean, time_std))
        print("PL(ours): ", format_with_uncertainty(self._LER, ler_std))
        print("Nerror(ours): ", format_with_uncertainty(Nerror_mean, Nerror_std))



    def sample_all_subspaces(self, Nclip, Budget, save_path=None):
            """
            Sample all subspaces from minw to maxw
            return the result of these samples as two dictionary
            """
            self.determine_saturated_w()
            wlist_to_sample = np.arange(self._t+1, self._saturatew, step=1)
            sample_used={}
            ler_count={}
            subspaceLER={}
            for w in wlist_to_sample:
                sample_used[w]=0
                ler_count[w]=0
                subspaceLER[w]=0
            """
            First round of sampling
            """
            slist=[8000]*len(wlist_to_sample)
            wlist=wlist_to_sample
            detector_result,obsresult=return_samples_many_weights_separate_obs_with_QEPG(self._QEPG_graph,wlist,slist)
            predictions_result = self._matcher.decode_batch(detector_result)      
            begin_index=0        
            for w_idx, (w, quota) in enumerate(zip(wlist, slist)):
                observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
                predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()
                # 3. count mismatches in vectorised form ---------------------------------
                num_errors = np.count_nonzero(observables != predictions)
                ler_count[w]+=num_errors
                sample_used[w]+=quota
                subspaceLER[w]=ler_count[w]/sample_used[w]
                begin_index+=quota


            while True:
                slist=[]
                wlist=[]

                for weight in wlist_to_sample:
                    if(sample_used[weight]>Budget):
                        continue
                    if(ler_count[weight]<Nclip):
                        wlist.append(weight)
                        if(ler_count[weight]==0):
                            slist.append(min(10000,sample_used[weight]*10))
                        else:
                            slist.append(min(10000,int(sample_used[weight]*Nclip/(Nclip-ler_count[weight]))))

                if len(wlist)==0:
                    break

                detector_result,obsresult=return_samples_many_weights_separate_obs_with_QEPG(self._QEPG_graph,wlist,slist)
                predictions_result = self._matcher.decode_batch(detector_result)            

                begin_index=0        
                for w_idx, (w, quota) in enumerate(zip(wlist, slist)):
                    observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
                    predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()
                    # 3. count mismatches in vectorised form ---------------------------------
                    num_errors = np.count_nonzero(observables != predictions)
                    ler_count[w]+=num_errors
                    sample_used[w]+=quota
                    subspaceLER[w]=ler_count[w]/sample_used[w]
                    begin_index+=quota
                

            result = {
                "ler_count": ler_count,
                "sample_used": sample_used,
                "subspaceLER": subspaceLER
            }

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(result, f)


            return ler_count,sample_used,subspaceLER


    def load_all_sample_result(self,filepath):
        with open(filepath, 'rb') as f:
            result = pickle.load(f)

        ler_count = result["ler_count"]
        sample_used = result["sample_used"]
        subspaceLER = result["subspaceLER"]

        """Plot the S-curve and its discrete estimate."""
        keys   = list(subspaceLER.keys())
        values = [subspaceLER[k] for k in keys]

        fig=plt.figure()
        # bars ── discrete estimate
        plt.bar(keys, values,
                color='tab:blue',         # pick any color you like
                alpha=0.8,
                label='Estimated subspace LER by sampling')
    
        plt.axvline(x=self._error_rate*self._num_noise, color="red", linestyle="--", linewidth=1.2, label="Average Error number") # vertical line at x=0.5


        plt.xlabel('Weight')
        plt.ylabel('Logical Error Rate in subspace')
        plt.title("Scurve plot")
        plt.legend()                     # <- shows the two labels

        # ── Force integer ticks on the X axis ──────────────────────────
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        figname="AllSubspace.pdf"
        plt.tight_layout()               # optional: nicely fit everything
        plt.savefig(figname, dpi=300)
        #plt.show()
        plt.close(fig)



        """Fit the curve and plot the fitted line."""
        #print("circuit d:",self._circuit_level_code_distance)
        x_list = [x for x in subspaceLER.keys() if (subspaceLER[x] < 0.5 and subspaceLER[x]>0 and ler_count[x]>100)]

        sigma_list= [sigma_estimator( sample_used[x],ler_count[x]) for x in x_list]

        y_list = [np.log(0.5/subspaceLER[x]-1) for x in x_list]


        initial_guess  = (0, 0 ,0)

        lower = [ -np.inf, -np.inf,  -np.inf]
        # ── upper bounds for [param1, param2, param3, param4]
        upper = [ np.inf, np.inf , np.inf ]



        popt, pcov = curve_fit(
            modified_linear_function(self._t),
            x_list,
            y_list,
            p0=initial_guess,          # len(initial_guess) must be 4 and within the bounds above
            bounds=(lower, upper),     # <-- tuple with two arrays
            maxfev=50_000              # or max_nfev in newer SciPy
        )

        # Extract the best-fit parameter (alpha)
        a,b,c= popt[0] , popt[1], popt[2]


        #print("circuit d:",self._circuit_level_code_distance)
        y_list = [np.log(0.5/subspaceLER[x]-1) for x in x_list]
        y_predicted = [modified_linear_function_with_d(x,a,b,c,self._t) for x in x_list]
        #y_predicted = [scurve_function_with_distance(x,self._mu,self._sigma) for x in self._estimated_wlist]
        R_square_score = r_squared(y_list, y_predicted)
        #print("R^2 score: ", self._R_square_score)

        #Plot the fitted line
        x_fit = np.linspace(self._t+1, max(x_list), 1000)

        y_fit = modified_linear_function_with_d(x_fit, a, b, c,self._t)
        
        alpha= -1/a

        sample_cost_list= [sample_used[x] for x in x_list]

        #print("Fitted parameters: a={}, b={}, c={}, d={}".format(self._a, self._b, self._c, self._d))
        plt.figure()
        plt.errorbar(
            x_list,
            y_list,
            yerr=sigma_list,
            fmt='o',
            color='orange',
            label='Data points with error bars',
            capsize=3,
            markersize=4,
            elinewidth=1
        )
        plt.plot(x_fit, y_fit, label=f'Fitted line, R2={R_square_score}', color='blue')



        # Select 5 uniformly spaced indices from the available data points
        num_points_to_annotate = 5
        indices = np.linspace(0, len(x_list) - 1, num=num_points_to_annotate, dtype=int)

        # Annotate only the selected points
        for i in indices:
            x, y, s = x_list[i], y_list[i], sample_cost_list[i]
            if s == 0:
                continue
            s_str = "{0:.1e}".format(s)  # Scientific notation like 1.2e+03
            base, exp = s_str.split('e')
            exp = int(exp)
            label = r'${0}\times 10^{{{1}}}$'.format(base, exp)
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=7)


        text_lines = [                   
            r'$\alpha=%.4f$' % alpha,
            r'$\mu =%.4f$' % (alpha*b),
            r'$\beta=%.4f$' % c,
            r'$\#\mathrm{detector}=%d$' % self._num_detector,
            r'$\#\mathrm{noise}=%d$' % self._num_noise
        ]

        textstr = '\n'.join(text_lines)

        fig = plt.gcf()
        fig.subplots_adjust(right=0.75)  # Make room on the right
        fig.text(0.78, 0.5, textstr,
                fontsize=7,
                va='center', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
        # ──────────────────────────────────────────────────────────────────

        plt.xlabel('Weight')
        plt.ylabel(r'$\log\left(\frac{0.5}{\mathrm{LER}} - 1\right)$')
        plt.title('Linear Fit of S-curve')
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig("yfunction.pdf")
        plt.close()




    def calculate_LER_from_file(self,filepath,pvalue,codedistance,figname,titlename, repeat=1):
        self._error_rate=pvalue
        self._circuit_level_code_distance=codedistance
        ler_list=[]
        sample_used_list=[]
        r_squared_list=[]
        Nerror_list=[]
        time_list=[]
        for i in range(repeat):
            self.clear_all()
            self.parse_from_file(filepath)
            start = time.time()
            #self.determine_range_to_sample()
            #self.subspace_sampling()
            self.determine_lower_w()

            #self.ground_truth_subspace_sampling()
            #self._has_logical_errorw=self._t+4
            self.determine_saturated_w()


            self.subspace_sampling_to_fit_curve(1000*self._num_subspace)
            '''
            Fit the curve first time just to get the estimated sweat spot
            '''
            self.fit_linear_area()
            tmptime=time.time()
            self.fit_log_S_model(figname+"-R"+str(i)+"First.pdf",tmptime-start)
            '''
            Second round of samples
            '''
            self.subspace_sampling()

            self.fit_linear_area()
            tmptime=time.time()
            self.fit_log_S_model(figname+"-R"+str(i)+"Final.pdf",tmptime-start)

            self.calc_logical_error_rate_after_curve_fitting()

            end = time.time()
            time_list.append(end - start)

            self.plot_scurve(figname,titlename)
            r_squared_list.append(self._R_square_score)
            self._sample_used=np.sum(list(self._subspace_sample_used.values()))
            # print("Final LER: ",self._LER)
            # print("Total samples used: ",self._sample_used)
            ler_list.append(self._LER)
            sample_used_list.append(self._sample_used)
            Nerror_list.append(sum(self._subspace_LE_count.values()))

        # Compute means
        self._LER = np.mean(ler_list)
        self._sample_used = np.mean(sample_used_list)

        # Compute standard deviations
        ler_std = np.std(ler_list)
        sample_used_std = np.std(sample_used_list)
        r2_mean = np.mean(r_squared_list)
        r2_std = np.std(r_squared_list)
        Nerror_mean = np.mean(Nerror_list)
        Nerror_std = np.std(Nerror_list)

        time_mean = np.mean(time_list)
        time_std = np.std(time_list)

        # Print with scientific ± formatting
        print("k: ", self._k_range)
        print("beta: ",self._beta)
        print("Subspaces: ", self._num_subspace)
        print("R2: ", format_with_uncertainty(r2_mean, r2_std))
        print("Samples(ours): ", format_with_uncertainty(self._sample_used, sample_used_std))
        print("Time(our): ", format_with_uncertainty(time_mean, time_std))
        print("PL(ours): ", format_with_uncertainty(self._LER, ler_std))
        print("Nerror(ours): ", format_with_uncertainty(Nerror_mean, Nerror_std))



if __name__ == "__main__":


    p = 0.001
    sample_budget = 100_000_0000

    for d in range(7,9,2):
        t = (d - 1) // 2

        stim_path = f"C:/Users/username/Documents/Sampling/stimprograms/surface/surface{d}"
        figname = f"Surface{d}"
        titlename = f"Surface{d}"
        output_filename = f"Surface{d}.txt"

        tmp = stratified_Scurve_LERcalc(p, sampleBudget=sample_budget, k_range=5, num_subspace=6, beta=4)
        tmp.set_t(t)
        tmp.set_sample_bound(
            MIN_NUM_LE_EVENT=100,
            SAMPLE_GAP=100,
            MAX_SAMPLE_GAP=5000,
            MAX_SUBSPACE_SAMPLE=50000
        )
        with open(output_filename, "w") as f:
            with redirect_stdout(f):
                tmp.calculate_LER_from_file(stim_path, p, 0, figname, titlename, 1)