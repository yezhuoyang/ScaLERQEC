
# Python interface to construct a clifford circuit with noise

In current version, scaler will automatically add single qubit uniform depolarization noise before all gates. An example of constructing a Clifford circuit
is given as following:

```python
circ= CliffordCircuit(3)
circ.set_error_rate(0.1)
circ.add_depolarize(0)
circ.add_hadamard(0)
circ.add_depolarize(0)
circ.add_depolarize(1)
circ.add_cnot(0,1)
circ.add_cnot(0,2)
circ.add_depolarize(1)
circ.add_measurement(1)
circ.add_measurement(2)
#Convert scaler circuit to stim circuit
stimcirc=circ.get_stim_circuit()
print(stimcirc)
```

To compile the QEPG graph for any customized circuit and get samples, scaler provides you with the following interfaces:


```python
from scalerqec.qepg import compile_QEPG, return_samples_many_weights_separate_obs_with_QEPG
import pymatching
import numpy as np

#Compile the QEPG graph from the circuit we just created
QEPG_graph=compile_QEPG(str(circ.get_stim_circuit()))

wlist=[2,3,4,5]
slist=[1000,2000,3000,5000]
#Return a list of detector result samples and obserble result with fixed list of error weight and shots
detector_result,obsresult=return_samples_many_weights_separate_obs_with_QEPG(QEPG_graph,wlist=wlist,slist=slist)

#Construct the pymatching instance, use the detector error model constructed by STIM
DEM = circ.get_stim_circuit().detector_error_model(decompose_errors=False)
matcher=pymatching.Matching.from_detector_error_model(detector_error_model)

#Calculat the predicted result by Pymatching
predictions_result = matcher.decode_batch(detector_result)


subspace_LE_count={}
estimated_subspaceLER{}
for w in wlist:
    subspace_LE_count[w]=0
    estimated_subspaceLER[w]=0

begin_index=0
for w_idx, (w, quota) in enumerate(zip(wlist, slist)):
    observables =  np.asarray(obsresult[begin_index:begin_index+quota])                    # (shots,)
    # 2. batch-decode (decode_batch should accept ndarray) -------------------
        # shape (shots,) or (shots,1)
    predictions = np.asarray(predictions_result[begin_index:begin_index+quota]).ravel()

    # 3. count mismatches in vectorised form ---------------------------------
    num_errors = np.count_nonzero(observables != predictions)

    subspace_LE_count[w]+=num_errors
    estimated_subspaceLER[w] = subspace_LE_count[w] / slist[w_idx]
    begin_index+=quota
```


# Python interface to construct a QEC circuit with noise

Scaler support direct construction of Surface code/ Repetition code/ Five qubit code/ Steane code with different stabilizer measurment schemes(Shor's scheme, Knill scheme, Standard scheme).
User can construct any stabilizer code circuit by the following provided interface:



```python
from scalerqec.QEC.qeccircuit import QECStab
qeccirc= QECStab(n=5,k=1,d=3)
#Specify your stabilizers
# Stabilizer generators
qeccirc.add_stab("XZZXI")
qeccirc.add_stab("IXZZX")
qeccirc.add_stab("XIXZZ")
qeccirc.add_stab("ZXIXZ")
qeccirc.set_logical_Z(0, "ZZZZZ")
#Set stabilizer parity measurement scheme, round of repetition
qeccirc.scheme="Standard" 
qeccirc.rounds=2
qeccirc.construct_circuit()
stim_circuit = qeccirc.stimcirc
print(stim_circuit)
```



# How to run tests?


All test script are kept under ../test/ folder. You can test the correct ness of our QEPG implementation, test with ground truth for small scale circuit. 


