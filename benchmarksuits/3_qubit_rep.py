from scalerqec.QEC.qeccircuit import StabCode
from scalerqec.Monte.monteLER import MonteLERcalc
from scalerqec.Stratified.stratifiedLER import StratifiedLERcalc
from scalerqec.Symbolic.symbolicLER import SymbolicLERcalc
from scalerqec.QEC.noisemodel import NoiseModel

n = 3
k = 1
d = 3
qeccirc= StabCode(n=n,k=k,d=0)

# Repetition code stabilizers
qeccirc.add_stab("ZZI")
qeccirc.add_stab("IZZ")

# Logical operators
qeccirc.set_logical_Z(0, "ZZZ")

# Stabilizer parity measurement scheme & round of repetition
qeccirc.scheme="Standard"
qeccirc.rounds=2

qeccirc.construct_circuit()
#qeccirc.show_IR()

# Export to temporary stim file
import tempfile
stim_circuit = qeccirc.stimcirc
stim_str = str(stim_circuit)
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".stim", delete=False)
tmp.write(stim_str)
tmp.flush()
stim_path = tmp.name
tmp.close()

# Error model
p = 0.01
noise_model = NoiseModel(p)

### (1) MC
print("---------Monte-Carlo Logical-Z LER---------")
calc = MonteLERcalc(MIN_NUM_LE_EVENT=100)
calc.calculate_LER_from_StabCode(qeccirc, noise_model, repeat=3)

### (2) Stratified
print("---------Stratified Logical-Z LER---------")
est = StratifiedLERcalc(error_rate=p, sampleBudget=100_000, num_subspace=6)
est.calculate_LER_from_StabCode(qeccirc, noise_model, repeat=3)

### (3) Symbolic
print("---------Symbolic Logical-Z LER---------")
sym = SymbolicLERcalc()        
ler = sym.calculate_LER_from_StabCode(qeccirc, noise_model)
print("Symbolic Logical-Z LER =", ler)