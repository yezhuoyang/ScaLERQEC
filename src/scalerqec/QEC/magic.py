from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from scalerqec.QEC.logic import CodeBlock


@dataclass(frozen=True)
class StabilizerSpec:
    pauli: str
    support: Tuple[int, ...]


class MGDDistillation:
    """
    15-to-1 Reed–Muller magic state distillation (state-level simulation).
    We add state tomography of the (conditional) output qubit by measuring it in X/Y/Z
    bases, conditioned on acceptance (syn == 0).
    """

    def __init__(self):
        self.stabilizers: List[StabilizerSpec] = self._rm15_generators()

    @staticmethod
    def _rm15_generators() -> List[StabilizerSpec]:
        x1 = (7, 8, 9, 10, 11, 12, 13, 14)
        x2 = (3, 4, 5, 6, 11, 12, 13, 14)
        x3 = (1, 2, 5, 6, 9, 10, 13, 14)
        x4 = (0, 2, 4, 6, 8, 10, 12, 14)
        z1, z2, z3, z4 = x1, x2, x3, x4
        z12 = (11, 12, 13, 14)
        z13 = (9, 10, 13, 14)
        z14 = (8, 10, 12, 14)
        z23 = (5, 6, 13, 14)
        z24 = (4, 6, 12, 14)
        z34 = (2, 6, 10, 14)
        return [
            StabilizerSpec("X", x1),
            StabilizerSpec("X", x2),
            StabilizerSpec("X", x3),
            StabilizerSpec("X", x4),
            StabilizerSpec("Z", z1),
            StabilizerSpec("Z", z2),
            StabilizerSpec("Z", z3),
            StabilizerSpec("Z", z4),
            StabilizerSpec("Z", z12),
            StabilizerSpec("Z", z13),
            StabilizerSpec("Z", z14),
            StabilizerSpec("Z", z23),
            StabilizerSpec("Z", z24),
            StabilizerSpec("Z", z34),
        ]

    @staticmethod
    def _prepare_raw_T_states(qc: QuantumCircuit, data: QuantumRegister) -> None:
        for i in range(len(data)):
            qc.h(data[i])
            qc.t(data[i])

    @staticmethod
    def _measure_css_stabilizer(
        qc: QuantumCircuit,
        data: QuantumRegister,
        anc,
        syn_bit,
        stab: StabilizerSpec,
    ) -> None:
        qc.reset(anc)
        if stab.pauli == "Z":
            for j in stab.support:
                qc.cx(data[j], anc)
            qc.measure(anc, syn_bit)
        elif stab.pauli == "X":
            qc.h(anc)
            for j in stab.support:
                qc.cx(anc, data[j])
            qc.h(anc)
            qc.measure(anc, syn_bit)
        else:
            raise ValueError("stab.pauli must be 'X' or 'Z'.")

    def construct_circuit(self, out_basis: str = "Z") -> QuantumCircuit:
        """
        out_basis in {'X','Y','Z'} determines how we measure the output qubit,
        but ONLY when accepted (syn == 0).
        """
        if out_basis not in {"X", "Y", "Z"}:
            raise ValueError("out_basis must be one of {'X','Y','Z'}.")
        data = QuantumRegister(15, "f")
        anc = QuantumRegister(1, "a")
        syn = ClassicalRegister(14, "syn")
        out = ClassicalRegister(1, "out")
        qc = QuantumCircuit(data, anc, syn, out, name=f"RM15_15to1_T_{out_basis}")
        self._prepare_raw_T_states(qc, data)
        for k, stab in enumerate(self.stabilizers):
            self._measure_css_stabilizer(qc, data, anc[0], syn[k], stab)
        # Conditionally measure output in requested basis
        with qc.if_test((syn, 0)):
            # Apply transversal basis change gates
            if out_basis == "X":
                for i in range(15):
                    qc.h(data[i])
            elif out_basis == "Y":
                for i in range(15):
                    qc.sdg(data[i])
                    qc.h(data[i])
            # Z basis: do nothing
            # Now measure logical Z (product of all Z)
            qc.reset(anc[0])
            for j in range(15):
                qc.cx(data[j], anc[0])
            qc.measure(anc[0], out[0])
        return qc

    @staticmethod
    def _parse_counts(counts: Dict[str, int]) -> List[Tuple[int, int, int]]:
        """
        Robust parsing:
          - Identify syn as the 14-bit substring and out as the 1-bit substring,
            regardless of which appears first.
        Returns (syn_val, out_bit, weight).
        """
        parsed: List[Tuple[int, int, int]] = []
        for key, w in counts.items():
            parts = key.split()
            if len(parts) == 2:
                a, b = parts
                if len(a) == 14 and len(b) == 1:
                    syn_str, out_str = a, b
                elif len(a) == 1 and len(b) == 14:
                    out_str, syn_str = a, b
                else:
                    raise ValueError(f"Unexpected key format: {key}")
            elif len(parts) == 1:
                s = parts[0]
                # Total classical bits should be 15 (=14 syn + 1 out).
                if len(s) != 15:
                    raise ValueError(f"Unexpected key length {len(s)} for key: {key}")
                # If no spaces, we'll assume last bit is out and preceding 14 are syn.
                syn_str, out_str = s[:-1], s[-1:]
            syn_val = int(syn_str, 2)
            out_bit = int(out_str, 2)
            parsed.append((syn_val, out_bit, w))
        return parsed

    @staticmethod
    def _accepted_expectation(counts: Dict[str, int]) -> Tuple[float, float]:
        """
        Returns (accept_rate, E[(-1)^out | accept]).
        """
        parsed = MGDDistillation._parse_counts(counts)
        total = sum(w for _, _, w in parsed)
        acc = 0
        num = 0
        for syn_val, out_bit, w in parsed:
            if syn_val == 0:
                acc += w
                num += w * (1 if out_bit == 0 else -1)
        accept_rate = acc / total if total else 0.0
        exp = num / acc if acc else 0.0
        return accept_rate, exp

    @staticmethod
    def _accepted_shots(counts: Dict[str, int]) -> int:
        parsed = MGDDistillation._parse_counts(counts)
        return sum(w for syn_val, _, w in parsed if syn_val == 0)

    @staticmethod
    def reconstruct_density_matrix(x: float, y: float, z: float) -> np.ndarray:
        identity = np.eye(2, dtype=complex)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        rho = 0.5 * (identity + x * pauli_x + y * pauli_y + z * pauli_z)
        return rho

    @staticmethod
    def ideal_T_state() -> np.ndarray:
        c = math.cos(math.pi / 4.0)
        s = math.sin(math.pi / 4.0)
        phase = complex(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))  # e^{i pi/4}
        return np.array([c, phase * s], dtype=complex)

    @staticmethod
    def fidelity_to_T(rho: np.ndarray) -> float:
        t = MGDDistillation.ideal_T_state()
        return float(np.real(np.conjugate(t).T @ (rho @ t)))

    def run_tomography(self, shots: int = 20000) -> None:
        sim = AerSimulator()  # ideal (noiseless) simulator for now
        exps: Dict[str, float] = {}
        accs: Dict[str, float] = {}
        acc_shots: Dict[str, int] = {}
        for basis in ["X", "Y", "Z"]:
            qc = self.construct_circuit(out_basis=basis)
            result = sim.run(qc, shots=shots).result()
            counts = result.get_counts(qc)
            acc_rate, exp = self._accepted_expectation(counts)
            accs[basis] = acc_rate
            exps[basis] = exp
            acc_shots[basis] = self._accepted_shots(counts)
        x, y, z = exps["X"], exps["Y"], exps["Z"]
        rho = self.reconstruct_density_matrix(x, y, z)
        F = self.fidelity_to_T(rho)
        print("Accepted shots:")
        for basis in ["X", "Y", "Z"]:
            print(f"  [{basis}] {acc_shots[basis]} / {shots}")
        print("Accepted-shot expectations:")
        print(f"  <X>={x:.6f}, <Y>={y:.6f}, <Z>={z:.6f}")
        print("Acceptance rates:")
        print(
            f"  acc(X)={accs['X']:.6f}, acc(Y)={accs['Y']:.6f}, acc(Z)={accs['Z']:.6f}"
        )
        print("Reconstructed rho:")
        print(rho)
        print(f"Fidelity to |T> (conditioned on accept): {F:.8f}")


class MagicFactoryIR:
    """
    The logical version of IR
    """

    def __init__(self):
        pass


class SuccessCondition(MagicFactoryIR):
    """
    Success condition for RUS
    """

    def __init__(self, conditions: list):
        self._conditions = conditions  # List of conditions to check


class LogicProp(MagicFactoryIR):
    """
    Logical Propagation operation
    """

    def __init__(self, dest: str, stabilizer: str):
        self._dest = dest  # Destination qubit
        self._stabilizer = stabilizer  # Stabilizer string representation


class RepeatUntilSuccess(MagicFactoryIR):
    """
    Repeat-Until construct for logical circuits
    """

    def __init__(self, condition: str, body: list[MagicFactoryIR]):
        self._condition = condition  # Condition to check
        self._body = body  # Body of the loop


class MagicFactory:
    """
    The class of Magic state distillation/Cultivation factory
    """

    def __init__(self, name: str, code_block: CodeBlock):
        self._name = name
        self._code_block = code_block

    def add_IR(self, ir: MagicFactoryIR) -> None:
        """
        The Factory is described by a series of IR instructions
        """
        pass

    def compile_to_qiskit(self) -> None:
        """
        TODO: Should be able to compile the logical level IR to Qiskit circuit
        and test the correctness
        """
        pass


if __name__ == "__main__":
    distiller = MGDDistillation()
    distiller.run_tomography(shots=5000)
