# Class of Quantum Error Correction Program

This project implements a modular, end-to-end compiler stack for **fault-tolerant quantum programs**, starting from a high-level language (LogicQ) and lowering through multiple intermediate representations down to a **Clifford circuit with dynamic classical control**.

The design is code-centric and QEC-aware:

* Users express quantum programs in terms of **logical qubits** and **QEC code blocks**.
* The compiler expands logical operations into:

  * stabilizer propagation gadgets,
  * QEC cycles,
  * logical gate implementations,
  * transversal operations when available.
  
* The backend produces:

  * a stabilizer-level IR (QStabIR),
  * a Clifford circuit or STIM circuit suitable for simulation or backend execution.

The goal is to allow users to write programs at the logical level, while still enabling fully explicit fault-tolerant compilation at lower abstraction layers.

---

# LogicQ — High-Level Fault-Tolerant Language

LogicQ is the **highest-level programming abstraction**. At this level:

Users specify:

* the QEC code family,
* code parameters ((n,k,d)),
* logical program structure,
* logical gates and logical measurements.

The compiler handles:

* Code construction
* Logical operators
* QECCycles
* Fault-tolerant execution sequencing

Users do **not** need to specify:

* stabilizers,
* syndrome extraction procedures,
* physical gate layouts.

---

## Supported LogicQ Syntax

| Category               | Syntax                          | Description                        |
| ---------------------- | ------------------------------- | ---------------------------------- |
| Code block definition  | `[[n,k,d,'scheme']] q1 { ... }` | Declare a QEC code block           |
| Stabilizer declaration | `XXXXZI;`                       | Define stabilizer generators       |
| QECCycle               | `QECCycle q1`                   | Execute one error-correction round |
| Transversal gate       | `Transversal H q1[0]`           | Apply transversal logical H        |
| Transversal CNOT       | `Transversal CNOT q1[0], q2[0]` | Assumed transversal between blocks |
| Logical gate           | `Logical X q1[0]`               | Logical operator via analyzer      |
| Logical measurement    | `c1 = Measure Logic Z q1[0]`    | Propagate logical operator         |

---

## Example — LogicQ Program

```text
[[5,1,3,'standard']] q1 {
    XZZXI;
    IXZZX;
    XIXZZ;
    ZXIXZ;
}

[[5,1,3,'standard']] q2 {
    XZZXI;
    IXZZX;
    XIXZZ;
    ZXIXZ;
}

# Apply a logical H to q1
Logical H q1[0]
QECCycle q1

# Apply transversal CNOT between blocks
Transversal CNOT q1[0], q2[0]
QECCycle q1
QECCycle q2

# Logical Z measurement on q1
c1 = Measure Logic Z q1[0]
```

---

# LogicQIR — Structural Logical IR

LogicQIR is the **second-stage IR**, where abstract logical constructs are made explicit.

Compared with LogicQ, this layer:

| Difference from LogicQ        | Meaning                                   |
| ----------------------------- | ----------------------------------------- |
| Stabilizers are explicit      | Code block fully defined in operator form |
| QECCycle boundaries enforced  | Scheduling between logical operations     |
| Logical CNOT must be compiled | Cross-block structure is explicit         |

LogicQIR serves as the input to the QStabIR compiler.

---

## Supported LogicQIR Instruction Set

| Category            | IR Form                         | Meaning                             |
| ------------------- | ------------------------------- | ----------------------------------- |
| QECCyle             | `QECCycle q1`                   | Expand to stabilizer propagation    |
| Transversal         | `Transversal H q1[0]`           | Apply H on all physical qubits      |
| Transversal CNOT    | `Transversal CNOT q1[0], q2[0]` | Index-wise CNOT pairs               |
| Logical gate        | `Logical X q1[0]`               | Lower using LogicalOperatorAnalyzer |
| Logical measurement | `c1 = Measure Logic Z q1[0]`    | Map to Pauli measurement pattern    |

---

## Example — LogicQIR Output

```text
Logical X q1[0]
QECCycle q1

Transversal H q2[0]
QECCycle q2

Transversal CNOT q1[0], q2[0]
QECCycle q1
QECCycle q2

c1 = Measure Logic Z q1[0]
```

---

# QStabIR — Stabilizer-Level IR

QStabIR is the **fully expanded stabilizer execution IR**.

At this level:

* No “logical” abstractions remain.

* QECCycle expands into:

  * Pauli propagation
  * syndrome extraction
  * decoding
  * correction

* Logical gates expand into:

  * Pauli strings
  * transversal operations
  * compound Pauli actions

* Measurements are expressed as:

  * propagation gadgets
  * observable decoding

---

## QStabIR Instruction Set

| Category         | Instruction            | Meaning                         |
| ---------------- | ---------------------- | ------------------------------- |
| Propagation      | `c = Prop XYZII`       | Stabilizer syndrome measurement |
| Decode           | `E = Decode c0,c1,...` | Classical decoding of syndrome  |
| Correct          | `Correct E`            | Apply correction                |
| Logical X        | `X i , j , k`          | Physical logical X action       |
| Logical Z        | `Z i , j , k`          | Physical logical Z action       |
| Logical H        | `H i , j , k`          | Physical H action               |
| Compound Pauli   | `X[1]Z[3]Y[5]`         | Mixed-Pauli operator            |
| Transversal CNOT | `CNOT i->j , ...`      | Index-wise physical mapping     |

---

## Example — QStabIR Output

```text
# QECCycle(q1)

q1_s0_0 = Prop XZZXI
q1_s0_1 = Prop IXZZX
q1_s0_2 = Prop XIXZZ
q1_s0_3 = Prop ZXIXZ

E_q1_0 = Decode q1_s0_0,q1_s0_1,q1_s0_2,q1_s0_3
Correct E_q1_0

# Logical H(q1)

H 0 , 1 , 2 , 3 , 4

# Transversal CNOT(q1,q2)

CNOT 0->5 , 1->6 , 2->7 , 3->8 , 4->9
```

---

# Clifford-Level Output

The final stage lowers QStabIR into:

* a Clifford circuit
* optionally a STIM circuit
* with explicit:

  * measurement ancillae
  * reset operations
  * detector construction
  * classical-feedback correction

This stage performs:

| Function                | Description                         |
| ----------------------- | ----------------------------------- |
| Parity gadget synthesis | Convert Pauli propagation → circuit |
| Detector construction   | Stabilizer repeat consistency       |
| Observable encoding     | Logical-Z measurement record        |
| Backend-aware lowering  | (STIM / QASM / executors)           |

---

## Example — Clifford Circuit Output (STIM-style)

```text
R 4
CX 0 4
CX 1 4
M 4
DETECTOR rec[-1] rec[-5]
OBSERVABLE_INCLUDE rec[-2]
```

(Actual output depends on scheme and backend.)

---

# Summary

This compiler pipeline supports:

| Layer           | Role                        | User Sees                                |
| --------------- | --------------------------- | ---------------------------------------- |
| LogicQ          | Logical-program abstraction | Logical qubits, QECCycles, logical gates |
| LogicQIR        | Structural FT schedule      | Stabilizers + logical structure          |
| QStabIR         | Fault-tolerant execution IR | Syndrome propagation + decoding          |
| Clifford Output | Physical implementation     | Real executable circuit                  |

The design cleanly separates:

* mathematical logical intent,
* stabilizer semantics,
* hardware-level execution,
* and code-specific compilation rules.

