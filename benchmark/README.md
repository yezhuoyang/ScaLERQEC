# How to run benchmark circuit and reproduce the table?

All our benchmark circuit are stored undered stimprograms/ folder. To reproduce one circuit, for example, for surface code with distance 7, execute the following python script:



```python
from scaler import StratifiedScurveLERcalc
d=7
p = 0.001
repeat=5
sample_budget = 100_000_0000
t = (d - 1) // 2
stim_path = f"your/path/stimprograms/surface/surface7"
figname = f"Surface{d}"
titlename = f"Surface{d}"
output_filename = f"Surface{d}.txt"
testinstance = StratifiedScurveLERcalc(p, sampleBudget=sample_budget, k_range=5, num_subspace=6, beta=4)
testinstance.set_t(t)
testinstance.set_sample_bound(
    MIN_NUM_LE_EVENT=100,
    SAMPLE_GAP=100,
    MAX_SAMPLE_GAP=5000,
    MAX_SUBSPACE_SAMPLE=50000
)
with open(output_filename, "w") as f:
    with redirect_stdout(f):
        testinstance.calculate_LER_from_file(stim_path, p, 0, figname, titlename, repeat)
```



# Symbolic calculator of LER

In this part, I explain how to get the ground truth of logical error rate by Symbolic calculator. Reader can reproduce Table 2



```python
from scaler.stratifiedScurve import StratifiedScurveLERcalc
from contextlib import redirect_stdout
from scaler.symbolicLER import SymbolicLERcalc


if __name__ == "__main__":

    testinstance=SymbolicLERcalc(0.001)
    filepath="your/file/path/to/circuit"
    print(testinstance.calculate_LER_from_file(filepath,0.001))
    p=0.001

    num_noise=testinstance._num_noise

    for weight in range(1,num_noise):
        print("LER in the subspace {} is {}".format(weight,testinstance.evaluate_LER_subspace(p,weight)))        


    for weight in range(1,num_noise):
        print("SubspaceLER {} is {}".format(weight,testinstance.subspace_LER(weight)))     
```




# Use Monte random fault injection we implemeted to test any circuit

In this part, I explain how to test any circuit with the widely use random fault injection method.



```python
from contextlib import redirect_stdout
from scaler.stimLER import MonteLERcalc

if __name__ == "__main__":


    p=0.001
    filepath="C:/Users/username/Documents/ScaLER/stimprograms/surface/surface3"
    d=3
    repeat=5
    sampleBudget=500000
    # 3) build your output filename:
    out_fname ="resultMonte.txt"     # e.g. "surface3-result.txt"
    # 4) redirect prints for just this file:

    with open(out_fname, "w") as outf, redirect_stdout(outf):

        calculator=MonteLERcalc(MIN_NUM_LE_EVENT=10)
        # pass the string path into your function:
        ler = calculator.calculate_LER_from_my_random_sampler(sampleBudget,filepath, p, repeat)    
```


# Use Stim and Sinter to test any circuit


You can also test the circuit with Stim optimized by Sinter. 


```python
from contextlib import redirect_stdout
from scaler.stimLER import MonteLERcalc

if __name__ == "__main__":


    p=0.001
    filepath=f"your/path/stimprograms/surface/surface7"
    d=3
    repeat=5
    sampleBudget=500000
    # 3) build your output filename:
    out_fname ="resultSinter.txt"     # e.g. "surface3-result.txt"
    # 4) redirect prints for just this file:

    with open(out_fname, "w") as outf, redirect_stdout(outf):

        calculator=MonteLERcalc(MIN_NUM_LE_EVENT=10)
        # pass the string path into your function:   
        ler  = calculator.calculate_LER_from_file_sinter(sampleBudget,filepath, p, repeat)
```


# Use ScaLER to test any circuit

In this part, I explain how to use ScaLER to test and input circuit. I will explain how to change hyper parameters. In a python script, run the folloing code:


```python
from scaler.stratifiedScurve import StratifiedScurveLERcalc
from contextlib import redirect_stdout

if __name__ == "__main__":

    d=7
    p = 0.001
    repeat=5
    sample_budget = 100_000_0000
    t = (d - 1) // 2
    stim_path = f"Relative/Path/ScaLER/stimprograms/surface/surface{d}"
    figname = f"Surface{d}"
    titlename = f"Surface{d}"
    output_filename = f"Surface{d}.txt"
    testinstance = StratifiedScurveLERcalc(p, sampleBudget=sample_budget, k_range=5, num_subspace=6, beta=4)
    testinstance.set_t(t)
    testinstance.set_sample_bound(
        MIN_NUM_LE_EVENT=100,
        SAMPLE_GAP=100,
        MAX_SAMPLE_GAP=5000,
        MAX_SUBSPACE_SAMPLE=50000
    )
    with open(output_filename, "w") as f:
        with redirect_stdout(f):
            testinstance.calculate_LER_from_file(stim_path, p, 0, figname, titlename, repeat)
```

Under the same directory, you will see a output figure which shows the fitted curve, and a output.txt file which shows all results. 






