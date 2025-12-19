from scalerqec.QEC.qeccircuit import QECStab
from scalerqec.monteLER import stimLERcalc
from scalerqec.stratifiedScurveLER import stratified_Scurve_LERcalc
from scalerqec.symbolicLER import symbolicLER

qeccirc= QECStab(n=5,k=1,d=3)

# Stabilizer generators
qeccirc.add_stab("XZZXI")
qeccirc.add_stab("IXZZX")
qeccirc.add_stab("XIXZZ")
qeccirc.add_stab("ZXIXZ")
qeccirc.set_logical_Z(0, "ZZZZZ")

# Set stabilizer parity measurement scheme, round of repetition
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

# Params
p = 1e-3
samplebudget = 2_000_000

### (1) MC
calc = stimLERcalc(MIN_NUM_LE_EVENT=20)
ler = calc.calculate_LER_from_my_random_sampler(samplebudget, stim_path, p, repeat=3)
print("Monte-Carlo Logical-Z LER:", ler)

### (2) Symbolic
sym = symbolicLER(error_rate=p)        
ler = sym.calculate_LER_from_file(stim_path, p)
print("Symbolic Logical-Z LER =", ler)

### (3) Stratified
d = 3
t = (d - 1) // 2
est = stratified_Scurve_LERcalc(error_rate=p, sampleBudget=samplebudget, k_range=5, num_subspace=6, beta=4)
est.set_t(t)
est.set_sample_bound(MIN_NUM_LE_EVENT=100, SAMPLE_GAP=100, MAX_SAMPLE_GAP=5000, MAX_SUBSPACE_SAMPLE=50_000)
est.calculate_LER_from_file(stim_path, p, codedistance=3, figname="5q", titlename="5-qubit", repeat=1)