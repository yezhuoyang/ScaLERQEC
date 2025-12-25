'''
Surface code array for Z error
Author: A.W
'''

from typing import Callable
import os
from scalerqec.stimparser import rewrite_stim_code
from pathlib import Path
import stim

def hexagon_3_circ(error_rate: float, rounds: int, distance : int = 3) -> str:
    
    w = 3*(distance-1) + 1 + 2 + 1
    h = (distance-1) + 1

    # generate surface code array
    qbit_dict = {}
    cnt = 0
    for j in range(h-1):
        qbit_dict[(j, 0)] = cnt
        cnt += 1
    for i in range(1, w-1):
        for j in range(h):
            qbit_dict[(j, i)] = cnt
            cnt += 1
    for j in range(1, h):
        qbit_dict[(j, w-1)] = cnt
        cnt += 1
    # print(qbit_dict)
    
    '''Convert 2D coordinate into 1D qubit order'''
    T : Callable[[int, int], int] = lambda x, y: qbit_dict[(x, y)]
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

    newline = os.linesep
    
    # error_rate = 0.005
    before_measure_flip_error = error_rate
    before_round_data_depolarization = error_rate
    after_clifford_depolarization = error_rate

    # for this model, 
    dqbits = []
    for i in range(2, w-1, 3):
        for j in range(0, h, 1):
            dqbits.append((j, i))

    mxqbits_four = [] # X syndrome qubit
    for i in range(4, w-1, 6):
        for j in range(0, h-1, 2):
            mxqbits_four.append((j, i))
    for i in range(7, w-1, 6):
        for j in range(1, h-1, 2):
            mxqbits_four.append((j, i))

    mxqbits_two = []
    for i in range(6, w-1, 6):
        mxqbits_two.append((0,i))
    for i in range(3, w-1, 6):
        mxqbits_two.append((h-1,i))

    mxsignal = []
    for i in mxqbits_four:
        mxsignal.extend([i, DW(i), LF(i), LF(DW(i))])
    for i in mxqbits_two:
        mxsignal.extend([i, RF(i)])
    
    
    mzqbits_four = [] # Z syndrome qubit, in this model, (0,0) is a Z parity qubit
    for j in range(0, h-1, 2):
        for i in range(6, w-1, 6):
            mzqbits_four.append((j, i))
    for j in range(1, h-1, 2):
        for i in range(3, w-1, 6):
            mzqbits_four.append((j, i))

    mzqbits_two = []
    for j in range(1, h-1, 2):
        mzqbits_two.append((j,w-1))
    for j in range(0, h-1, 2):
        mzqbits_two.append((j, 0))

    mzsignal = []
    for i in mzqbits_four:
        mzsignal.extend([i, DW(i), RF(i), RF(DW(i))])
    for i in mzqbits_two:
        if i[1] == 0:
            mzsignal.extend([i, DW(i), RF(i), RF(DW(i))])
        else:
            mzsignal.extend([i, DW(i)])

    mqbits = list(set(mxqbits_four + mxqbits_two + mzqbits_four + mzqbits_two))

    allqubits = list(qbit_dict.keys())
    # for qb in dqbits:
    #     allqubits.append(qb)
    # for qb in mxqbits_four + mzqbits_four + mzqbits_two:
    #     allqubits.extend([qb, UP(qb), DW(qb)])
    # for qb in mxqbits_two:
    #     allqubits.append(qb)
    # allqubits = list(set(allqubits))

    qbit_idx = {}
    for qb, idx in qbit_dict.items():
        qbit_idx[idx] = qb

    circ = ""
    for i in qbit_idx.keys():
        circ += "QUBIT_COORDS(%d, %d) %d\n" % (qbit_idx[i][0], qbit_idx[i][1], i)

    circ += R(allqubits) + newline  # reset data qubits
    circ += TICK + newline
    circ += DEPOLARIZE1(before_round_data_depolarization, dqbits) + newline

    def X_stabilizer():
        '''Construct X stabilizers'''
        s = ""
        qlist = [qb for qb in mxqbits_four + mxqbits_two]
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:
            qlist.extend([i, DW(i)])
        for i in mxqbits_two:
            qlist.extend([i, RF(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:
            qlist.extend([i, LF(i), DW(i), LF(DW(i))])
        for i in mxqbits_two:
            qlist.extend([i, LF(i), RF(i), RF(RF(i))])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:
            qlist.extend([LF(i), LF(LF(i)), LF(DW(i)), LF(LF(DW(i))), i, RF(i), DW(i), RF(DW(i))])
        for i in mxqbits_two:
            qlist.extend([i, RF(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:
            qlist.extend([i, LF(i), DW(i), LF(DW(i))])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mxqbits_four:
            qlist.extend([i, DW(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = [qb for qb in mxqbits_four + mxqbits_two]
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        ##################################
        qlist = mxsignal
        s += X_ERROR(before_measure_flip_error, qlist) + newline + MR(qlist, 1) + newline
        return s
    
    def Z_stabilizer():
        '''Construct Z stabilizers'''
        s = ""
        qlist = list(set(mzsignal)-set(mzqbits_four)-set(mzqbits_two))
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:
            qlist.extend([DW(i), i])
        for i in mzqbits_two:
            if i[1] == w-1:
                qlist.extend([DW(i), i])
            else:
                qlist.extend([DW(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:
            qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
        for i in mzqbits_two:
            if i[1] == w-1:
                qlist.extend([LF(i), i, LF(DW(i)), DW(i)])
            else:
                qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:
            qlist.extend([LF(i), i, LF(DW(i)), DW(i), RF(RF(i)), RF(i), RF(RF(DW(i))), RF(DW(i))])
        for i in mzqbits_two:
            if i[1] == w-1:
                qlist.extend([DW(i), i])
            else:
                pass
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:
            qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
        for i in mzqbits_two:
            if i[1] == w-1:
                pass
            else:
                qlist.extend([RF(RF(i)), RF(i), RF(RF(DW(i))), RF(DW(i))])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_four:
            qlist.extend([DW(i), i])
        for i in mzqbits_two:
            if i[1] == w-1:
                pass
            else:
                qlist.extend([RF(i), i, RF(DW(i)), DW(i)])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = []
        for i in mzqbits_two:
            if i[1] == w-1:
                pass
            else:
                qlist.extend([DW(i), i])
        s += CX(qlist) + newline + DEPOLARIZE2(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = list(set(mzsignal)-set(mzqbits_four)-set(mzqbits_two))
        s += H(qlist) + newline + DEPOLARIZE1(after_clifford_depolarization, qlist) + newline + TICK + newline
        qlist = mzsignal
        s += X_ERROR(before_measure_flip_error, qlist) + newline + MR(qlist, 2) + newline
        return s
    
    def X_measure(flag : int = 0) -> str:
        '''flag: 0, initial; 1, compare to before'''
        s = ""
        qlist = mxsignal
        for qb in qlist:
            if flag == 0:
                s += "DETECTOR(%d,%d,0) rec[%d]" % (qb[0],qb[1],M1(qb, 1)) + newline # may cause conflict with Z-reset
            elif flag == 1:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d]" % (qb[0],qb[1],M1(qb, 1),M2(qb, 1)) + newline
        return s
    def Z_measure(flag : int = 0) -> str:
        '''flag: 0, initial; 1, compare to before'''
        s = ""
        qlist = mzsignal
        for qb in qlist:
            if flag == 0:
                s += "DETECTOR(%d,%d,0) rec[%d]" % (qb[0],qb[1],M1(qb, 2)) + newline
            elif flag == 1:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d]" % (qb[0],qb[1],M1(qb, 2), M2(qb, 2)) + newline
        return s

    circ += X_stabilizer()
    # circ += X_measure(0)

    circ += Z_stabilizer()
    circ += Z_measure(0)
    # Enter the repetition section
    def rep(rounds):
        s = ""
        if rounds > 0:
            s += "REPEAT %d {" % rounds + newline
            s += TICK + newline + DEPOLARIZE1(before_round_data_depolarization, dqbits) + newline
            s += X_stabilizer()
            s += Z_stabilizer()
            ''' first assert X measurements preserves '''
            s += "SHIFT_COORDS(%d, %d, %d)" % (0,0,1) + newline
            s += X_measure(1)
            ''' second assert Z measurements preserves '''
            s += "SHIFT_COORDS(%d, %d, %d)" % (0,0,1) + newline
            s += Z_measure(1)
            s += "}" + newline
        return s

    circ += rep(rounds)

    circ += X_ERROR(before_measure_flip_error, dqbits) + newline + M(dqbits, 3) + newline

    def assert_Z_stabilizers():
        ''' let me assert Z stabilizers '''
        s = ""
        for qb in mzqbits_four:
            dqa, dqb, dqc, dqd = LF(qb), RF(RF(qb)), LF(DW(qb)), RF(RF(DW(qb)))
            s += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(dqa, 3), M1(dqb, 3), M1(dqc, 3), M1(dqd, 3), M1(qb, 2)) + newline
        for qb in mzqbits_two:
            if qb[1] == 0:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(RF(RF(qb)), 3), M1(RF(RF(DW(qb))), 3), M1(qb, 2)) + newline
            else:
                s += "DETECTOR(%d,%d,0) rec[%d] rec[%d] rec[%d]" % (qb[0],qb[1],M1(LF(qb), 3), M1(LF(DW(qb)), 3), M1(qb, 2)) + newline
        return s

    def assert_Z_logical():
        ''' assert the Z logical operation '''
        s = ""
        qlist = []
        for i in range(2, w, 3):
            qlist.append(((i-2)//3, i))
        s += "OBSERVABLE_INCLUDE(0)"
        for qb in qlist:
            s += " rec[%d]" % M1(qb, 3)
        s += newline
        return s

    circ += assert_Z_stabilizers()
    circ += assert_Z_logical()
    
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

    stim_circuit = stim.Circuit(hexagon_3_circ(0.005,distance*3, distance)).flattened()
    circuit_text = rewrite_stim_code(str(stim_circuit))

    # 2. Resolve the target path and create parent dirs if needed
    filepath = Path(filepath).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 3. Write the text
    with filepath.open("w", encoding="utf-8") as f:
        f.write(circuit_text)

    return filepath

if __name__ == '__main__':
    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon5"
    # generate_circuit(filepath,5)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon7"
    # generate_circuit(filepath,7)   

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon9"
    # generate_circuit(filepath,9)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon11"
    # generate_circuit(filepath,11)       
    
    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon13"
    # generate_circuit(filepath,13)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon15"
    # generate_circuit(filepath,15)   

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon17"
    # generate_circuit(filepath,17)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon19"
    # generate_circuit(filepath,19)  
    
    from pathlib import Path
    home = Path(__file__).resolve().parent.parent
    filepath = home / "stimprograms" / "hexagon" / "hexagon21" # [script directory parent]/stimprograms/hexagon/hexagon21
    generate_circuit(filepath,21)  

    filepath = home / "stimprograms" / "hexagon" / "hexagon23"
    generate_circuit(filepath,23)   

    filepath = home / "stimprograms" / "hexagon" / "hexagon25"
    generate_circuit(filepath,25)  

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon27"
    # generate_circuit(filepath,27)       
    
    # filepath="C:/Users/username/Documents/Sampling/stimprograms/hexagon/hexagon29"
    # generate_circuit(filepath,29)  
