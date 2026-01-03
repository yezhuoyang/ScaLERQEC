# Class of Quantum Error Correction Program

This project implements a **multi-layer compiler stack** for fault-tolerant quantum programs:

```text
LogicQ + MigicQ  ──►  LogicQIR  ──►  QStabIR  ──►  Clifford / STIM
(high)              (logical-IR)    (stabilizer IR)   (physical circuit)
```

* **LogicQ**: high-level fault-tolerant programming language (types + logical operations).
* **LogicQIR**: code-aware IR with explicit stabilizers, QECCycles, and logical gate structure.
* **QStabIR**: low-level stabilizer IR (syndrome extraction, decoding, correction).
* **Clifford / STIM**: concrete circuit representation with dynamic classical control.

The goal is to let users write **logical programs** while the compiler takes care of QEC details.

---

# LogicQ — High-Level Fault-Tolerant Language

LogicQ is the **user-facing language**. At this level:

* You work with **code types** (`surface`, `steane`, …).
* You allocate **code blocks** with logical qubit counts and distances.
* You apply **logical operations** (`LogicH`, `LogicCNOT`, `LogicMeasure`, …).
* You call **high-level protocols** such as magic-state distillation and injection.

You **do not** specify stabilizers or QECCycles at this level — those are handled by the compiler.

---

## LogicQ Syntax

### 1. Code type and block declarations

| Syntax                  | Meaning                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| `Type surface:`         | Declare a QEC code *type* (e.g., surface code family). The stabilizers are a function of `d`. |
| `surface q1 [n1,k1,d1]` | Instantiate a surface code block `q1` with parameters ((n_1, k_1, d_1)).                      |
| `surface q2 [n2,k2,d2]` | Another block of the same code family.                                                        |
| `surface t0 [n3,k3,d3]` | A block used as a magic-state ancilla (e.g., `T`-state block).                                |

Typical interpretation:

* `Type surface:` defines how to construct stabilizers as a function of `d`.
* `surface q1 [n1,k1,d1]` declares a *logical register* backed by a `StabCode` instance.

### 2. Logical gate operations

| Syntax                           | Meaning                                                  |
| -------------------------------- | -------------------------------------------------------- |
| `q1[i] = LogicH q1[i]`           | Apply logical Hadamard on block `q1`, logical qubit `i`. |
| `q2[j] = LogicCNOT q1[i], q2[j]` | Apply logical CNOT with control `q1[i]`, target `q2[j]`. |
| `q1[i] = LogicX q1[i]`           | Logical bit-flip on `q1[i]` (if supported).              |
| `q1[i] = LogicZ q1[i]`           | Logical phase-flip on `q1[i]` (if supported).            |

(Your current compiler implements logical X/Z/H in LogicQIR; LogicQ maps these to the corresponding LogicQIR `Logical` instructions.)

### 3. Magic-state operations (MagicQ-style)

| Syntax                      | Meaning                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `t0 = Distill15to1_T[d=25]` | Run a 15-to-1 T-state distillation protocol at distance `d=25`. Returns a magic-(T) handle `t0`. |
| `InjectT q1[0], t0`         | Perform gate teleportation / T injection on logical qubit `q1[0]` using magic state `t0`.        |

Later, the LogicQ → LogicQIR compiler expands these into explicit distillation and injection subcircuits with specific codes and stabilizers.

### 4. Logical measurement

| Syntax                            | Meaning                                                                                 |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| `c = LogicMeasure q1[i]`          | Measure logical Z (or specified logical basis) of `q1[i]`. Returns a classical bit `c`. |
| `c = LogicMeasureX q1[i]`         | (Possible extension) Measure logical X.                                                 |
| `c = LogicMeasureXZ q1[i], q2[j]` | (Possible extension) Measure joint logical operator (e.g., Bell measurement).           |

---

## Example — LogicQ Program

A small illustrative LogicQ program using surface code blocks and T distillation:

```text
Type surface:
    # Stabilizers are a function of the distance d
    # This type describes the stabilizer structure (primal/dual plaquettes, boundaries, etc.)

surface q1 [n1, k1, d1]   # First logical data block
surface q2 [n2, k2, d2]   # Second logical data block
surface t0 [n3, k3, d3]   # Magic T state block

# Apply logical H on q1[0]
q1[0] = LogicH q1[0]

# Prepare a high-fidelity magic T state via distillation
t0 = Distill15to1_T[d=25]     # returns a magic_T handle on block t0

# Inject a logical T gate into q1[0] using magic state t0
InjectT q1[0], t0

# Apply a logical CNOT from q1[0] to q2[1]
q2[1] = LogicCNOT q1[0], q2[1]

# Measure both logical qubits
c1 = LogicMeasure q1[0]
c2 = LogicMeasure q2[1]
```

The **LogicQ → LogicQIR** compiler:

* Chooses concrete stabilizer codes for `surface` blocks based on `d`.
* Creates explicit stabilizer lists, QECCycles, and logical operators.
* Lowers `LogicH`, `LogicCNOT`, `LogicMeasure`, and `InjectT` into the LogicQIR structures below.

---

# LogicQIR — Logical, Code-Aware IR

LogicQIR is the next layer: an IR that is:

* **code-aware** (explicit stabilizers),
* structured in terms of:

  * code blocks,
  * QECCycles,
  * transversal gates,
  * logical gates,
  * logical measurements.

This is what your current `LogicQIRProgram` / `LogicQCompiler` operate on.

### LogicQIR Code Block Syntax

| Syntax                          | Meaning                                              |
| ------------------------------- | ---------------------------------------------------- |
| `[[n,k,d,'scheme']] q1 { ... }` | Declare a QEC code block with parameters and scheme. |
| `XZZXI;`                        | One stabilizer generator (per line, `;`-terminated). |

Example:

```text
[[5,1,3,'Standard']] q1 {
    XZZXI;
    IXZZX;
    XIXZZ;
    ZXIXZ;
}
```

### LogicQIR Instruction Syntax

| Category            | Syntax                          | Meaning                                                                            |
| ------------------- | ------------------------------- | ---------------------------------------------------------------------------------- |
| QECCycle            | `QECCycle q1`                   | One abstract QEC cycle on code block `q1`.                                         |
| Transversal H       | `Transversal H q1[0]`           | H on all physical qubits of block `q1`.                                            |
| Transversal CNOT    | `Transversal CNOT q1[0], q2[0]` | CNOT between corresponding physical qubits of `q1` and `q2` (assumed transversal). |
| Logical gate        | `Logical X q1[0]`               | Logical X on logical qubit 0 of block `q1` using `LogicalOperatorAnalyzer`.        |
| Logical gate        | `Logical Z q1[0]`               | Logical Z (currently naive `Z^n` in the example).                                  |
| Logical gate        | `Logical H q1[0]`               | Logical H pattern, lowered to HInstruction.                                        |
| Logical measurement | `c1 = Measure Logic Z q1[0]`    | Measure logical Z on `q1[0]` → classical bit `c1`.                                 |

---

## Example — LogicQIR Program

This mirrors the example in your current test driver:

```text
[[5,1,3,'Standard']] q1 {
    XZZXI;
    IXZZX;
    XIXZZ;
    ZXIXZ;
}

[[7,1,3,'Standard']] q2 {
    IIIXXXX;
    IXXIIXX;
    XIXIXIX;
    IIIZZZZ;
    IZZIIZZ;
    ZIZIZIZ;
}

# Logical operations on q1
Logical X q1[0]
QECCycle q1

Logical Z q1[0]
QECCycle q1

Logical H q1[0]
QECCycle q1

# Logical operations on q2
Logical X q2[0]
QECCycle q2

Logical Z q2[0]
QECCycle q2

Logical H q2[0]
QECCycle q2

# Logical Z measurements
c1 = Measure Logic Z  q1[0]
QECCycle q1

c2 = Measure Logic Z  q2[0]
QECCycle q2
```

Your `LogicQCompiler` then compiles this LogicQIR into QStabIR.

---

# QStabIR — Stabilizer-Level IR

QStabIR is the **stabilizer execution IR** used internally by `StabCode`:

* Expanded QECCycles → stabilizer propagation gadgets.
* Logical gates → explicit Pauli or Clifford actions on physical indices.
* Logical measurements → `Prop` and `Decode` sequences.

## QStabIR Instruction Set

| Category         | Instruction Form         | Meaning                                                            |
| ---------------- | ------------------------ | ------------------------------------------------------------------ |
| Propagation      | `c = Prop XZZXI`         | Measure a stabilizer or logical observable into classical bit `c`. |
| Decode           | `E = Decode c0,c1,...`   | Map syndrome bits to an error label/operation.                     |
| Correct          | `Correct E`              | Apply correction associated with `E`.                              |
| Logical X        | `X i , j , k`            | Apply X on listed physical qubits.                                 |
| Logical Y        | `Y i , j , k`            | Apply Y on listed physical qubits.                                 |
| Logical Z        | `Z i , j , k`            | Apply Z on listed physical qubits.                                 |
| Logical H        | `H i , j , k`            | Apply H on listed physical qubits.                                 |
| Logical S        | `S i , j , k`            | Apply S on listed physical qubits.                                 |
| Compound Pauli   | `X[1]Z[4]Y[6]`           | Apply mixed Pauli string on specified qubits.                      |
| Transversal CNOT | `CNOT 0->5 , 1->6 , ...` | Index-wise physical CNOT pairs (transversal assumption).           |

---

## Example — QStabIR Generated by LogicQCompiler

For a single QECCycle on a 5-qubit code `q1`:

```text
# QECCycle q1 (cycle 0)

q1_s0_0 = Prop XZZXI
q1_s0_1 = Prop IXZZX
q1_s0_2 = Prop XIXZZ
q1_s0_3 = Prop ZXIXZ

E_q1_0 = Decode q1_s0_0,q1_s0_1,q1_s0_2,q1_s0_3
Correct E_q1_0
```

For a logical `H q1[0]` where `determine_logical_H` returns `H` on all 5 physical qubits:

```text
H 0 , 1 , 2 , 3 , 4
```

For a transversal CNOT between two 5-qubit blocks `q1` and `q3`:

```text
CNOT 0->5 , 1->6 , 2->7 , 3->8 , 4->9
```

For a logical Z measurement `c1 = Measure Logic Z q1[0]` (using logical Z = `ZZZZZ`):

```text
c1 = Prop ZZZZZ
```

---

# Clifford / STIM Output

The final backend lowers QStabIR to a real circuit:

* `StabPropInstruction` → reset + CNOT/H gadgets + measurement.
* `DetectorInstruction` / `ObservableInstruction` → STIM `DETECTOR` / `OBSERVABLE_INCLUDE`.
* Noise model (if present) is folded into the Clifford circuit.

A typical **standard scheme** compilation pipeline is:

```python
qeccirc = StabCode(n=5, k=1, d=3)
qeccirc.add_stab("XZZXI")
qeccirc.add_stab("IXZZX")
qeccirc.add_stab("XIXZZ")
qeccirc.add_stab("ZXIXZ")
qeccirc.set_logical_Z(0, "ZZZZZ")
qeccirc.scheme = "Standard"
qeccirc.rounds = 3
qeccirc.construct_circuit()
stim_circuit = qeccirc.stimcirc
print(stim_circuit)
```

The resulting STIM-style circuit contains:

* qubit resets,
* Clifford gates (H, CX),
* measurements,
* detectors and logical observables.

---

## Summary of Layers

| Layer           | Main Abstraction                  | User-Facing Syntax                                                                                | Example Section                               |
| --------------- | --------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| LogicQ          | Logical algorithms over QEC types | `Type surface`, `LogicH`, `LogicCNOT`, `LogicMeasure`, `Distill15to1_T`, `InjectT`                | “LogicQ — High-Level Fault-Tolerant Language” |
| LogicQIR        | Code-aware logical IR             | `[[n,k,d,'scheme']] q1 { ... }`, `QECCycle`, `Logical X/Z/H`, `Transversal CNOT`, `Measure Logic` | “LogicQIR — Logical, Code-Aware IR”           |
| QStabIR         | Stabilizer execution IR           | `Prop`, `Decode`, `Correct`, `X/Y/Z/H/S`, `CompoundPaulis`, `CNOT`                                | “QStabIR — Stabilizer-Level IR”               |
| Clifford / STIM | Physical circuit                  | H, CX, M, R, DETECTOR, OBSERVABLE_INCLUDE                                                         | “Clifford / STIM Output”                      |
