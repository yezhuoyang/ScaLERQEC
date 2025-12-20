# ScaLERQEC

<p align="center">
  <img src="Figures/logo.png" alt="Our logo" width="350"/>
</p>
<p align="center">
  <em>Figure 1: Our logo.</em>
</p> 

ScaLERQEC is a scalable framework for estimating logical error rates (LER) of quantum error-correcting (QEC) circuits at scale.
It combines optimized C++ backends (QEPG) with high-level Python interfaces for QEC experimentation, benchmarking, symbolic analysis, and Monte-Carlo fault injection.
ScaLER is compatible with STIM, but use completely different approach to test logical error rate. 


🚀 Installation
🔧 Option 1 — Install via pip (recommended)

```bash
pip install scalerqec
```

This installs:

the Python package scalerqec

the compiled C++ backend scalerqec.qepg

all Python modules for LER calculation, sampling, symbolic analysis, etc.

Then in Python:


```python
import scalerqec
import scalerqec.qepg
```


🔧 Option 2 — Install from source

Clone the repository:

```bash
git clone https://github.com/yourname/ScaLERQEC.git
cd ScaLERQEC
```

Build and install:

```bash
pip install .
```

This compiles the C++ backend using pybind11 and places the compiled extension under:


scalerqec/qepg.*.so or .pyd


📚 Project Structure

After installation, the package structure is:
```bash
scalerqec/
    qepg               # compiled QEPG graph and samping method from C++ backend (by pybind11)
    Clifford/          # Clifford circuit
    Monte/             # Monte Carlo sampling method
    QEC/               # High-level description of quantum error correction circuit 
    Stratified/        # Stratified fault injection
    Symbolic/          # Symbolic method
    ...
```

# Construct QEC circuit by Stabilizer


In ScalerQEC, user can construct a circuit by stabilizer formalism. 


```python
from scalerqec.QEC.qeccircuit import QECStab
from scalerqec.QEC.noisemodel import NoiseModel

qeccirc= QECStab(n=3,k=1,d=3)
# Stabilizer generators
qeccirc.add_stab("ZZI")
qeccirc.add_stab("IZZ")
#Set the first logical Z
qeccirc.set_logical_Z(0, "ZZZ")
noise_model = NoiseModel(0.001) #Set the noise model with physical error rate
# Set stabilizer parity measurement scheme, round of repetition
# We support Standard/Shor/Knill/Flag scheme
qeccirc.scheme="Standard"
# How many rounds of stabilizer measurement?
qeccirc.rounds=2
# Construct IR and stim circuit
qeccirc.construct_circuit()
```

We design a IR representation for qeccirc circuit that is much more easier to debug by Clifford circuit. Call the following interface:

```python
qeccirc.show_IR()
```

The output of IR representation of the above circuit is:

```bash
c0 = Prop[r=0, s=0] ZZI
c1 = Prop[r=0, s=1] IZZ
c2 = Prop[r=1, s=0] ZZI
d0 = Parity c0 c2
c3 = Prop[r=1, s=1] IZZ
d1 = Parity c1 c3
c4 = Prop ZZZ
o0 = Parity c4
```



1️⃣ Main method: Test Logical by Statefied fault-sampling and curve fitting:



<p align="center">
  <img src="Figures/diagra.png" alt="diag" width="550"/>
</p>
<p align="center">
  <em>Figure 2: Diagram for the main method in ScaLERQEC</em>
</p> 


You propose a novel method which tests the logical error rate by stratified sampling and curve fitting. With fixed QEC circuit
and the noise model, we provide a simple interface for this method.

```python
from scalerqec.Stratified import stratified_Scurve_LERcalc
calculator = stratified_Scurve_LERcalc()
figname="Repetition"  
titlename="Repetition" 
stratifiedcalculator.calc_LER_from_QECcircuit(qeccirc, noise_model,figname,titlename, repeat=3)
```


| <img src="Figures/Surface7-R0Final.png" alt="Curve in the Log Space" width="300"/> | <img src="Figures/Surface7.png" alt="Curve in the original space" width="300"/> |
|:---------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
| *Figure 1: Subspace error rate in the log space* | *Figure 2: Same, but plot in original space* |




2️⃣ Using the C++ QEPG Backend from Python


<p align="center">
  <img src="Figures/prop.png" alt="QEPG" width="350"/>
</p>
<p align="center">
  <em>Figure 2: Illustration of how we compile a QEPG graph in ScaLERQEC.</em>
</p> 

ScalerQEC compile any STIM circuit to QEPG graph.


```python
import scalerqec.qepg as qepg

graph = qepg.compile_QEPG(open("circuit.stim").read())

samples = qepg.return_samples_with_fixed_QEPG(graph, weight=3, shots=10_000)

print(samples)
```

3️⃣ Running Monte Carlo Fault-Injection


We support standard Monte Carlo testing through the following interface:


```python
from scalerqec.Monte.monteLER import stimLERcalc
montecalculator = stimLERcalc()
symbcalculator.calc_LER_of_QECircuit(qeccirc, noise_model)
```


4️⃣ Running Symbolic LER Analysis (Ground Truth)


ScalerQEC has a novel method which calculate the exact symbolic polynomial representation of a given QEC circuit under a uniform noise model.


```python
from scalerqec.Symbolic.symbolicLER import symbolicLER
symbcalculator = symbolicLER()
symbcalculator.calc_LER_of_QECircuit(qeccirc, noise_model)
```


📌 TODO (Roadmap)

- [x] Support installation via `pip install`
- [x] Higher-level, easier interface to generate QEC program
- [x] Add cross-platform installation support (including macOS)
- [x] Python interface to construct QEC circuit
- [ ] Support LDPC code and LDPC code decoder
- [ ] Get rid of Boost package, use binary representation
- [ ] Add CUDA backend support and compare with STIM
- [ ] SIMD support and compare with STIM
- [ ] Constructing and testing magic state distillation/Cultivation
- [ ] Compatible with Qiskit
- [ ] Visualize results better and visualize QEPG graph
- [ ] HotSpot analysis(What is the reason for logical error?)
- [ ] Write full documentation
- [ ] Implement dynamic-circuit support(Compatible with IBM)
- [ ] Support testing code switching such as lattice surgery, LDPC code switching protocol
- [ ] Add more realistic noise models(Decoherence noise, Correlated noise)
- [ ] Support injecting quantum errors by type(Hook Error, Gate error, Propagated error, etc)
- [ ] Static analysis pass of circuit(Learn symmetric structure)
- [ ] Test Pauli measurement based fault-tolerant circuit 


🧰 Development Notes (for contributors)


## 1. Installation (Development Only)

At the moment, ScaLERQEC is installed **from source**. 

### 1.1. Prerequisites

Common to all platforms:

- Python ≥ 3.9 (3.11+ recommended)
- A C++20-compatible compiler
- `pip` and a virtual environment (`venv`, `conda`, etc.)

Additional dependencies:

- **Boost** (for `boost::dynamic_bitset`)
- **pybind11** (handled automatically as a build dependency, but the C++ compiler must be able to see its headers)


#### Windows (MSVC)

1. Install [Visual Studio Build Tools] or full Visual Studio with C++ toolchain.
2. Install Boost (MSVC flavor), e.g. via Chocolatey:

```bash
choco install boost-msvc-14.3 -y
``` 

The boost header file will be stored under the path "C:\local\boost_1_87_0\boost". Add this path into VScode cpp include path in your development process. 

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

Install pybind11 via Homebrew:

```bash
brew install pybind11 boost
``` 

This provides headers in:

/opt/homebrew/include (ARM)
/usr/local/include (Intel)


#### Linux (Ubuntu / Debian-like)

Roughly:

```bash
sudo apt install libboost-dev
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

# Pip install instructions

Just directly run the following pip install command:

```bash
pip install .
```

