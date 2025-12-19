'''
Surface code array for Z error
Author: A.W
'''
from scalerqec.stimparser import rewrite_stim_code
from pathlib import Path
import stim

def square_3_circ(error_rate: float, rounds: int, distance : int = 3) -> str:
    def generate_dot(w: int, h: int) -> str:
        cnt = 0
        s = ""
        for i in range(w):
            for j in range(h):
                s += "QUBIT_COORDS(%d, %d) %d\n" % (j, i, cnt)
                cnt += 1
        return s
    
    w = 2*(distance-1) + 1 + 2
    h = distance

    from typing import Callable
    '''Convert 2D coordinate into 1D qubit order'''
    T : Callable[[int, int], int] = lambda x, y: y*h + x
    TARGET : Callable[[list], str] = lambda qlist: "".join([' %d' % T(x, y) for x, y in qlist])
    X_ERROR : Callable[[float, list], str] = lambda error_rate, qlist: 'X_ERROR(%lf)' % (error_rate) + TARGET(qlist) # X errors
    R : Callable[[list], str]  = lambda qlist: 'R' + TARGET(qlist) # reset to |0>.
    TICK = 'TICK' # used as a barrier for different layer of circuit
    DEPOLARIZE1 : Callable[[float, list], str] = lambda error_rate, qlist: 'DEPOLARIZE1(%lf)' % (error_rate) + TARGET(qlist) # single-qubit Depolarization errors
    DEPOLARIZE2 : Callable[[float, list], str] = lambda error_rate, qlist: 'DEPOLARIZE2(%lf)' % (error_rate) + TARGET(qlist) # two-qubit Depolarization errors
    H : Callable[[list], str]  = lambda qlist: 'H' + TARGET(qlist) # H gate
    CX : Callable[[list], str]  = lambda qlist: 'CX' + TARGET(qlist) # CX gate
    # MR : Callable[[list], str]  = lambda qlist: 'MR' + TARGET(qlist) # MR gate, measure and reset
    # M : Callable[[list], str]  = lambda qlist: 'M' + TARGET(qlist) # M gate, measure and reset
    LF : Callable[[tuple], tuple] = lambda qbit: (qbit[0], qbit[1]-1)
    RF : Callable[[tuple], tuple] = lambda qbit: (qbit[0], qbit[1]+1)
    UP : Callable[[tuple], tuple] = lambda qbit: (qbit[0]-1, qbit[1])
    DW : Callable[[tuple], tuple] = lambda qbit: (qbit[0]+1, qbit[1])
    
    meas_list = [] # measure list
    def MR(qlist: list, flag: int) -> str:
        '''flag: X stabilizer: 1, Z stabilizer: 2, logical qubit: 3.'''
        _ = [ meas_list.append(qbit+(flag,)) for qbit in qlist ]
        return 'MR' + TARGET(qlist)

    def M(qlist: list, flag: int) -> str:
        '''flag: X stabilizer: 1, Z stabilizer: 2, logical qubit: 3.'''
        _ = [ meas_list.append(qbit+(flag,)) for qbit in qlist ]
        return 'M' + TARGET(qlist)
    M1 = lambda qbit, flag: max(loc for loc, val in enumerate(meas_list) if val == (qbit+(flag,))) - len(meas_list)
    M2 = lambda qbit, flag: max([loc for loc, val in enumerate(meas_list) if val == (qbit+(flag,))][:-1]) - len(meas_list)

    import os
    newline = os.linesep
    
    # error_rate = 0.005
    before_measure_flip_error = error_rate
    before_round_data_depolarization = error_rate
    after_clifford_depolarization = error_rate

    # for this model, 
    allqbits = []
    for i in range(w):
        for j in range(h):
            allqbits.append((j, i))
    dqbits = []
    for i in range(1, w, 2):
        for j in range(h):
            dqbits.append((j, i))
    mqbits = list(set(allqbits) - set(dqbits) - {(h-1, 0), (0, w-1)})
    mxqbits_four = [] # X syndrome qubit
    for i in range(2, w-1, 4):
        for j in range(0, h-1, 2):
            mxqbits_four.append((j, i))
    for i in range(4, w-1, 4):
        for j in range(1, h-1, 2):
            mxqbits_four.append((j, i))
    mxqbits_two = []
    for i in range(2, w-1, 4):
        mxqbits_two.append((h-1,i))
    for i in range(4, w-1, 4):
        mxqbits_two.append((0,i))
    mzqbits_four = [] # Z syndrome qubit, in this model, (0,0) is a Z parity qubit
    for j in range(1, h-1, 2):
        for i in range(2, w-1, 4):
            mzqbits_four.append((j, i))
    for j in range(0, h-1, 2):
        for i in range(4, w-1, 4):
            mzqbits_four.append((j, i))
    mzqbits_two = []
    for j in range(1, h-1, 2):
        mzqbits_two.append((j,w-1))
    for j in range(0, h-1, 2):
        mzqbits_two.append((j, 0))

    circ = generate_dot(w, h)
    circ += R(dqbits) + newline  # reset data qubits
    # circ += M(dqbits) + newline # |0> returns 0
    # circ += "OBSERVABLE_INCLUDE(%d) rec[%d]" % (4,-3) + newline 
    circ += R(mqbits) + newline # reset measurement/parity qubits
    circ += TICK + newline
    circ += DEPOLARIZE1(before_round_data_depolarization, dqbits) + newline
    # first start X measurements
    qlist = [qb for qb in mxqbits_four + mxqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, DW(i)])
    for i in mxqbits_two:
        qlist.extend([i, RF(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, LF(i), DW(i), LF(DW(i))])
    for i in mxqbits_two:
        qlist.extend([i, LF(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, RF(i), DW(i), RF(DW(i))])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, DW(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = [qb for qb in mxqbits_four + mxqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = [qb for qb in mxqbits_two]
    for i in mxqbits_four:
        qlist.extend([i])
        qlist.extend([DW(i)])
    circ += X_ERROR(before_measure_flip_error, qlist) + newline + MR(qlist, 1) + newline
    # second, start Z measurements
    qlist = [DW(i) for i in mzqbits_four+mzqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four+mzqbits_two:
        qlist.extend([DW(i), i])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four:
        qlist.extend([LF(i), i, LF(DW(i)), DW(i)])
    for i in mzqbits_two:
        if i[1] == w-1:
            qlist.extend([LF(i), i, LF(DW(i)), DW(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four:
        qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
    for i in mzqbits_two:
        if i[1] == 0:
            qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four+mzqbits_two:
        qlist.extend([DW(i), i])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = [DW(i) for i in mzqbits_four+mzqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four + mzqbits_two:
        qlist.extend([i])
        qlist.extend([DW(i)])
    circ += X_ERROR(before_measure_flip_error, qlist) + newline + MR(qlist, 2) + newline
    '''Detector returns 0 if no error detected.'''
    # first X measurement
    qlist = [qb for qb in mxqbits_two]
    for i in mxqbits_four:
        qlist.extend([i])
        qlist.extend([DW(i)])
    # for qb in qlist:
    #     circ += "DETECTOR(%d,%d,0) rec[%d]" % (qb[0],qb[1],M1(qb, 1)) + newline
    # second Z measurement
    circ += "SHIFT_COORDS(%d, %d, %d)" % (0,0,1) + newline
    qlist = []
    for i in mzqbits_four + mzqbits_two:
        qlist.extend([i])
        qlist.extend([DW(i)])
    for qb in qlist:
        circ += "DETECTOR(%d,%d,0) rec[%d]" % (qb[0],qb[1],M1(qb, 2)) + newline
    circ += "REPEAT %d {" % rounds + newline
    circ += TICK + newline + DEPOLARIZE1(before_round_data_depolarization, dqbits) + newline
    # first start X measurements
    qlist = [qb for qb in mxqbits_four + mxqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, DW(i)])
    for i in mxqbits_two:
        qlist.extend([i, RF(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, LF(i), DW(i), LF(DW(i))])
    for i in mxqbits_two:
        qlist.extend([i, LF(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, RF(i), DW(i), RF(DW(i))])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mxqbits_four:
        qlist.extend([i, DW(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = [qb for qb in mxqbits_four + mxqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = [qb for qb in mxqbits_two]
    for i in mxqbits_four:
        qlist.extend([i])
        qlist.extend([DW(i)])
    circ += X_ERROR(before_measure_flip_error, qlist) + newline + MR(qlist, 1) + newline
    # second, start Z measurements
    qlist = [DW(i) for i in mzqbits_four+mzqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four+mzqbits_two:
        qlist.extend([DW(i), i])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four:
        qlist.extend([LF(i), i, LF(DW(i)), DW(i)])
    for i in mzqbits_two:
        if i[1] == w-1:
            qlist.extend([LF(i), i, LF(DW(i)), DW(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four:
        qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
    for i in mzqbits_two:
        if i[1] == 0:
            qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four+mzqbits_two:
        qlist.extend([DW(i), i])
    circ += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = [DW(i) for i in mzqbits_four+mzqbits_two]
    circ += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
    qlist = []
    for i in mzqbits_four + mzqbits_two:
        qlist.extend([i])
        qlist.extend([DW(i)])
    circ += X_ERROR(before_measure_flip_error, qlist) + newline + MR(qlist, 2) + newline
    # first assert X measurements preserves
    circ += "SHIFT_COORDS(%d, %d, %d)" % (0,0,1) + newline
    qlist = [qb for qb in mxqbits_two]
    for i in mxqbits_four:
        qlist.extend([i])
        qlist.extend([DW(i)])
    for qb in qlist:
        circ += "DETECTOR(%d,%d,0) rec[%d] rec[%d]" % (qb[0],qb[1],M1(qb, 1), M2(qb, 1)) + newline
    # second assert Z measurements preserves
    circ += "SHIFT_COORDS(%d, %d, %d)" % (0,0,1) + newline
    qlist = []
    for i in mzqbits_four + mzqbits_two:
        qlist.extend([i])
        qlist.extend([DW(i)])
    for qb in qlist:
        circ += "DETECTOR(%d,%d,0) rec[%d] rec[%d]" % (qb[0],qb[1],M1(qb, 2), M2(qb, 2)) + newline
    circ += "}" + newline
    circ += X_ERROR(before_measure_flip_error, dqbits) + newline + M(dqbits, 3) + newline
    # let me assert more stabilizers
    for qb in mzqbits_four:
        dqa, dqb, dqc, dqd = LF(qb), RF(qb), DW(LF(qb)), DW(RF(qb))
        circ += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(dqa, 3), M1(dqb, 3), M1(dqc, 3), M1(dqd, 3), M1(qb, 2)) + newline
    for qb in mzqbits_two:
        if qb[1] == 0:
            circ += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(RF(qb), 3), M1(DW(RF(qb)), 3), M1(qb, 2)) + newline
        else:
            circ += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(LF(qb), 3), M1(DW(LF(qb)), 3), M1(qb, 2)) + newline
    # assert the Z logical operation
    qlist = []
    for i in range(1, w, 2):
        qlist.append(((i-1)//2, i))
    circ += "OBSERVABLE_INCLUDE(0)"
    for qb in qlist:
        circ += " rec[%d]" % M1(qb, 3)
    circ += newline
    # for i, qb in enumerate(dqbits):
    #     circ += "OBSERVABLE_INCLUDE(%d) rec[%d]" % (i+1,M1(qb, 3)) + newline
    return circ


def generate_circuit(filepath: str | Path, distance: int = 3) -> Path:
    """
    Build a rotated‑surface‑code memory circuit and write it to `filepath`.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Where to write the circuit text (e.g. 'circuits/my_surface_code.stim').
    distance : int, optional
        Code distance; default is 3.

    Returns
    -------
    pathlib.Path
        Absolute path to the file that was written.
    """
    # 1. Make the circuit and rewrite it

    stim_circuit = stim.Circuit(square_3_circ(0.005,distance*3, distance)).flattened()
    circuit_text = rewrite_stim_code(str(stim_circuit))

    # 2. Resolve the target path and create parent dirs if needed
    filepath = Path(filepath).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 3. Write the text
    with filepath.open("w", encoding="utf-8") as f:
        f.write(circuit_text)

    return filepath




if __name__ == '__main__':

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square3"
    # generate_circuit(filepath,3)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square5"
    # generate_circuit(filepath,5)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square7"
    # generate_circuit(filepath,7)   

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square9"
    # generate_circuit(filepath,9)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square11"
    # generate_circuit(filepath,11)       
    
    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square13"
    # generate_circuit(filepath,13)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square15"
    # generate_circuit(filepath,15)   

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square17"
    # generate_circuit(filepath,17)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square19"
    # generate_circuit(filepath,19)  


    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square21"
    # generate_circuit(filepath,21)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/square/square23"
    # generate_circuit(filepath,23)  

    from pathlib import Path
    home = Path(__file__).resolve().parent.parent
    filepath = home / "stimprograms" / "square" / "square25" # [script directory parent]/stimprograms/square/square25
    generate_circuit(filepath,25)  