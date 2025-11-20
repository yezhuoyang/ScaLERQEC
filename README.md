# ScaLER

ScaLER is a scalable framework for estimating logical error rates (LER) of quantum error-correcting (QEC) circuits.
It combines optimized C++ backends (QEPG) with high-level Python interfaces for QEC experimentation, benchmarking, symbolic analysis, and Monte-Carlo fault injection.


🚀 Installation
🔧 Option 1 — Install via pip (recommended)

```bash
pip install scaler
```

This installs:

the Python package scaler

the compiled C++ backend scaler.qepg

all Python modules for LER calculation, sampling, symbolic analysis, etc.

Then in Python:


```python
import scaler
import scaler.qepg
```


🔧 Option 2 — Install from source

Clone the repository:

```bash
git clone https://github.com/yourname/ScaLER.git
cd ScaLER
```

Build and install:

```bash
pip install .
```

This compiles the C++ backend using pybind11 and places the compiled extension under:


scaler/qepg.*.so or .pyd


📚 Project Structure

After installation, the package structure is:
```bash
scaler/
    qepg               # compiled C++ backend (pybind11)
    clifford.py
    LERcalculator.py
    stratifiedScurveLER.py
    stimparser.py
    symbolicLER.py
    symbolicNaive.py
    monteLER.py
    ...
```

2️⃣ Using the C++ QEPG Backend from Python

```python
import scaler.qepg as qepg

graph = qepg.compile_QEPG(open("circuit.stim").read())

samples = qepg.return_samples_with_fixed_QEPG(graph, weight=3, shots=10_000)

print(samples)
```

3️⃣ Running Monte Carlo Fault-Injection

```python
from scaler.stimLER import stimLERcalc
from contextlib import redirect_stdout

p = 0.001
filepath = "stimprograms/surface/surface3"
sample_budget = 500_000

with open("resultMonte.txt", "w") as f, redirect_stdout(f):
    calc = stimLERcalc(MIN_NUM_LE_EVENT=10)
    ler = calc.calculate_LER_from_my_random_sampler(sample_budget, filepath, p, repeat=5)
```



4️⃣ Running Symbolic LER Analysis (Ground Truth)
from scaler.symbolicLER import symbolicLER

calc = symbolicLER(0.001)
filepath = "path/to/circuit"

print(calc.calculate_LER_from_file(filepath, 0.001))

num_noise = calc._num_noise
for w in range(1, num_noise):
    print("LER in subspace", w, "=", calc.evaluate_LER_subspace(0.001, w))



📌 TODO (Roadmap)

- [ ] Support installation via `pip install`
- [ ] Add cross-platform installation support (including macOS)
- [ ] Write full documentation
- [ ] Implement dynamic-circuit support(Compatible with IBM)
- [ ] Higher-level, easier interface to generate QEC program
- [ ] Support testing code switching
- [ ] Add more noise models
- [ ] Support injecting quantum errors by type
- [ ] Add CUDA backend support



🧰 Development Notes (for contributors)


## 1. Installation (Development Only)

At the moment, ScaLER is installed **from source**. 

### 1.1. Prerequisites

Common to all platforms:

- Python ≥ 3.9 (3.11+ recommended)
- A C++20-compatible compiler
- `pip` and a virtual environment (`venv`, `conda`, etc.)

Additional dependencies:

- **Boost** (for `boost::dynamic_bitset`)
- **pybind11** (handled automatically as a build dependency, but the C++ compiler must be able to see its headers)
- **Eigen3** Library


#### Windows (MSVC)

1. Install [Visual Studio Build Tools] or full Visual Studio with C++ toolchain.
2. Install Boost (MSVC flavor), e.g. via Chocolatey:

```bash
choco install boost-msvc-14.3 -y
``` 

The boost header file will be stored under the path "C:\local\boost_1_87_0\boost". Add this path into VScode cpp include path in your development process. 

We also use vcpkg and install the Eigen3 library for matrix operations.

We use pybind11 to convert the samples from C++ objects to python objects. To install using vcpkg, run the following command:

```bash
vcpkg install pybind11
```


#### macOS (Apple Silicon / Intel)

Install Xcode command-line tools:

```bash
xcode-select --install
```

Install Homebrew (if you don’t have it):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
``` 

Install pybind11 and Eigen via Homebrew:

```bash
brew install eigen pybind11 boost
``` 

This provides headers in:

/opt/homebrew/include (ARM)
/usr/local/include (Intel)


#### Linux (Ubuntu / Debian-like)

Roughly:

```bash
sudo apt install libeigen3-dev libboost-dev
pip install pybind11
```

Boost headers go into /usr/include/boost.


## Locating Python headers


Also need to add the path of the python header file

```bash
py -c "from sysconfig import get_paths as gp; print(gp()['include'])"
```

## Building the C++ QEPG backend


Run the following command to build the QEPG package with pybinding:

```bash
py setup.py build_ext --inplace
```

Run the following command to clear the previously compiled output:

```bash
py setup.py clean --all    
```


We also need to convert C++ object to python object directly. So "Python.h" needs to be added to the search path. Typically, it is under:


```bash
C:\Users\username\miniconda3\include
```


# How to compile run python script


To compile QEPG python package by pybind11:

```bash
(Under QEPG folder)./compilepybind.ps1
```

The python code is divided into different modules. For example, to run the test_by_stim.py file under test module, stay at the root folder and execute:

```bash
(Under Sampling folder)py -m test.test_by_stim   
```

# How to run tests?


All test script are kept under test/ folder. You can test the correct ness of our QEPG implementation, test with ground truth for small scale circuit. 



# Pip install instructions

Just directly run the following pip install command:

```bash
pip install .
```



# How to run benchmark circuit and reproduce the table?

All our benchmark circuit are stored undered stimprograms/ folder. To reproduce one circuit, for example, for surface code with distance 7, execute the following python script:



```python
from scaler import stratified_Scurve_LERcalc
d=7
p = 0.001
repeat=5
sample_budget = 100_000_0000
t = (d - 1) // 2
stim_path = f"your/path/stimprograms/surface/surface7"
figname = f"Surface{d}"
titlename = f"Surface{d}"
output_filename = f"Surface{d}.txt"
testinstance = stratified_Scurve_LERcalc(p, sampleBudget=sample_budget, k_range=5, num_subspace=6, beta=4)
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
from scaler.stratifiedScurve import stratified_Scurve_LERcalc
from contextlib import redirect_stdout
from scaler.symbolicLER import symbolicLER


if __name__ == "__main__":

    testinstance=symbolicLER(0.001)
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
from scaler.stimLER import stimLERcalc

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

        calculator=stimLERcalc(MIN_NUM_LE_EVENT=10)
        # pass the string path into your function:
        ler = calculator.calculate_LER_from_my_random_sampler(sampleBudget,filepath, p, repeat)    
```


# Use Stim and Sinter to test any circuit


You can also test the circuit with Stim optimized by Sinter. 


```python
from contextlib import redirect_stdout
from scaler.stimLER import stimLERcalc

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

        calculator=stimLERcalc(MIN_NUM_LE_EVENT=10)
        # pass the string path into your function:   
        ler  = calculator.calculate_LER_from_file_sinter(sampleBudget,filepath, p, repeat)
```


# Use ScaLER to test any circuit

In this part, I explain how to use ScaLER to test and input circuit. I will explain how to change hyper parameters. In a python script, run the folloing code:


```python
from scaler.stratifiedScurve import stratified_Scurve_LERcalc
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
    testinstance = stratified_Scurve_LERcalc(p, sampleBudget=sample_budget, k_range=5, num_subspace=6, beta=4)
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







