from __future__ import annotations

from enum import Enum
import ast

from .qeccircuit import StabCode
from .analyzer import LogicalOperatorAnalyzer


"----------------------------------------------------------------------------------------------------------------"


class LogicQIRType(Enum):
    QECCYCLE = 1
    TRANSVERSAL = 2
    LOGIC = 3
    MEASURE = 4


class LogicQIRInstruction:
    """
    Base class for logical QEC IR instructions.
    """

    def __init__(self, ir_type: LogicQIRType):
        self.ir_type = ir_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ir_type={self.ir_type})"


class QECCycleInstruction(LogicQIRInstruction):
    """
    The IR representing a QECCycle operation on a code block.
    Inside a QECCycle, stabilizers are measured and errors are corrected.
    This is a high-level instruction that abstracts away the details of
    stabilizer measurements and corrections.
    """

    def __init__(self, block_name: str):
        super().__init__(LogicQIRType.QECCYCLE)
        self.block_name = block_name

    def __repr__(self) -> str:
        return f"QECCycle {self.block_name}"


class TransversalInstruction(LogicQIRInstruction):
    """
    The IR representing a transversal logical gate operation on one or more
    logical qubits. For transversal CNOT, add a transversal CNOT instruction
    with control and target logical qubits.
    """

    def __init__(self, gate: str, logical_qubits: list[str]):
        super().__init__(LogicQIRType.TRANSVERSAL)
        self.gate = gate
        self.logical_qubits = logical_qubits

    def __repr__(self) -> str:
        return f"Transversal {self.gate} {', '.join(self.logical_qubits)}"


class MeasureInstruction(LogicQIRInstruction):
    """
    Measuring a logical operator and propagating the result to a classical bit.

    Example textual forms supported by the parser:

        c1 = Measure Logic Z  q1[0]
        c2 = Measure Logic X  q2[1]
        c3 = Measure Logic XZ q1[0],q2[1]
    """

    def __init__(self, operator: str, logical_qubits: list[str], classical_bit: str):
        super().__init__(LogicQIRType.MEASURE)
        self.operator = operator
        self.logical_qubits = logical_qubits
        self.classical_bit = classical_bit

    def __repr__(self) -> str:
        return (
            f"{self.classical_bit} = Measure Logic "
            f"{self.operator} {', '.join(self.logical_qubits)}"
        )


class LogicInstruction(LogicQIRInstruction):
    """
    The IR representing a logical gate operation on a code block.
    This will be further compiled within the code block compiler to determine
    how to implement the logical gate.

    Examples:

        Logical X q1[0]
        Logical H q2[1]
    """

    def __init__(self, gate: str, logical_qubits: list[str]):
        super().__init__(LogicQIRType.LOGIC)
        self.gate = gate
        self.logical_qubits = logical_qubits

    def __repr__(self) -> str:
        return f"Logical {self.gate} {', '.join(self.logical_qubits)}"


class QECCodeBlock:
    """
    Representation of a QEC code block definition in LogicalQ IR:

        [[n,k,d,'scheme']] q1 {
            STAB1;
            STAB2;
            ...
        }

    This holds:
        - block name (e.g., "q1"),
        - code parameters (n, k, d, scheme),
        - the list of stabilizer strings,
        - a constructed StabCode instance (qeccircuit).
    """

    def __init__(
        self,
        name: str,
        n: int,
        k: int,
        d: int,
        scheme: str,
        stabilizers: list[str],
        code: StabCode,
    ):
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        self.scheme = scheme
        self.stabilizers = stabilizers
        self.code = code

    def __repr__(self) -> str:
        header = f"[[{self.n},{self.k},{self.d!r},{self.scheme!r}]] {self.name}"
        body = ";\n    ".join(self.stabilizers) + ";"
        return f"{header} {{\n    {body}\n}}"


class LogicQIRProgram:
    """
    A LogicalQ IR program consisting of:
        - a mapping of QEC code blocks (name → QECCodeBlock),
        - a sequence of logical QEC IR instructions.

    Example LogicalQ IR:

        [[5,1,3,'standard']] q1 {
            XXZII;
            ZIXZX;
            IZZXI;
            IXXZZ;
        }

        [[7,1,3,'standard']] q2 {
            IIIIIII;
            XXIIIIX;
            YYIIIIY;
            ZZIIIIZ;
            IXXIYYI;
        }

        TRANSVERSAL H q1[0]
        QECCycle q1

        Transversal CNOT q1[0], q2[1]

        Logical X q1[0]
        Logical H q2[1]

        QECCycle q1
        QECCycle q2

        c1 = Measure Logic Z  q1[0]
        c2 = Measure Logic X  q2[1]
        c3 = Measure Logic XZ q1[0],q2[1]
    """

    def __init__(self):
        self.instructions: list[LogicQIRInstruction] = []
        # Map from block name (e.g., "q1") to QECCodeBlock
        self.code_blocks: dict[str, QECCodeBlock] = {}

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_code_block_header(self, line: str) -> tuple[int, int, int, str, str]:
        """
        Parse a header line of the form:

            [[5,1,3,'standard']] q1 {

        Returns:
            (n, k, d, scheme, block_name)
        """
        if not line.startswith("[["):
            raise ValueError(f"Invalid QEC code block header: {line}")

        close_idx = line.find("]]")
        if close_idx == -1:
            raise ValueError(f"Missing ']]' in QEC code block header: {line}")

        meta_str = line[2:close_idx].strip()
        # meta_str is like: "5,1,3,'standard'"
        try:
            n, k, d, scheme = ast.literal_eval(f"[{meta_str}]")
        except Exception as e:
            raise ValueError(f"Failed to parse code parameters in: {line}") from e

        rest = line[close_idx + 2 :].strip()
        # Expected something like "q1 {" or "q1{"
        if rest.endswith("{"):
            rest = rest[:-1].strip()
        block_name = rest
        if not block_name:
            raise ValueError(f"Missing block name in QEC code block header: {line}")

        return int(n), int(k), int(d), str(scheme), block_name

    # ------------------------------------------------------------------
    # Public parsing API
    # ------------------------------------------------------------------

    def parse_from_string(self, ir_string: str) -> None:
        """
        Parse a LogicalQ IR program from a string representation.

        Supported forms (whitespace-insensitive, comments allowed after '#'):

            [[5,1,3,'standard']] q1 {
                XXZII;
                ZIXZX;
                ...
            }

            QECCycle q1

            Transversal H q1[0]
            Transversal CNOT q1[0], q2[1]

            Logical X q1[0]
            Logical H q2[1]

            c1 = Measure Logic Z  q1[0]
            c2 = Measure Logic X  q2[1]
            c3 = Measure Logic XZ q1[0],q2[1]
        """
        lines = ir_string.strip().split("\n")
        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.strip()
            i += 1

            if not line:
                continue

            # Strip end-of-line comments.
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            # 1) QEC code block header
            if line.startswith("[["):
                n, k, d, scheme, block_name = self._parse_code_block_header(line)

                # Collect stabilizers until we hit a line with "}"
                stabilizers: list[str] = []
                while i < len(lines):
                    stab_raw = lines[i]
                    stab_line = stab_raw.strip()
                    i += 1

                    if not stab_line:
                        continue

                    # Strip comments
                    if "#" in stab_line:
                        stab_line = stab_line.split("#", 1)[0].strip()
                        if not stab_line:
                            continue

                    if stab_line.startswith("}"):
                        # End of this code block
                        break

                    # Each stabilizer line is like "XXZII;" or "XXZII"
                    stab = stab_line.rstrip(";").strip()
                    if stab:
                        stabilizers.append(stab)

                # Construct a StabCode instance for this block
                code = StabCode(n, k, d)
                # Attach scheme as a dynamic attribute (or you can extend StabCode)
                code.scheme = scheme
                for s in stabilizers:
                    code.add_stab(s)

                block = QECCodeBlock(
                    name=block_name,
                    n=n,
                    k=k,
                    d=d,
                    scheme=scheme,
                    stabilizers=stabilizers,
                    code=code,
                )
                self.code_blocks[block_name] = block
                continue

            # 2) Measurement with assignment: "c1 = Measure Logic Z q1[0]"
            if "Measure" in line and "=" in line:
                lhs, rhs = [part.strip() for part in line.split("=", 1)]
                classical_bit = lhs
                if not rhs.startswith("Measure"):
                    raise ValueError(f"Unrecognized Measure form: {raw_line}")
                parts = rhs.split()
                # Expect: Measure Logic <operator> <qubits>
                if len(parts) < 4 or parts[1] != "Logic":
                    raise ValueError(f"Invalid Measure syntax: {raw_line}")
                operator = parts[2]
                logical_qubits_str = " ".join(parts[3:])
                logical_qubits = [
                    q.strip() for q in logical_qubits_str.split(",") if q.strip()
                ]
                self.add_instruction(
                    MeasureInstruction(operator, logical_qubits, classical_bit)
                )
                continue

            # 3) Standalone Measure (optional variant)
            if line.startswith("Measure"):
                parts = line.split()
                if len(parts) < 5 or parts[1] != "Logic":
                    raise ValueError(f"Invalid Measure syntax: {raw_line}")
                operator = parts[2]
                logical_qubits = [q.strip() for q in parts[3].split(",") if q.strip()]
                if len(parts) >= 5:
                    classical_bit = parts[4]
                else:
                    raise ValueError(f"Missing classical bit in Measure: {raw_line}")
                self.add_instruction(
                    MeasureInstruction(operator, logical_qubits, classical_bit)
                )
                continue

            # 4) QECCycle
            if line.startswith("QECCycle"):
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid QECCycle syntax: {raw_line}")
                block_name = parts[1]
                self.add_instruction(QECCycleInstruction(block_name))
                continue

            # 5) Transversal gates
            if line.startswith("Transversal"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid Transversal syntax: {raw_line}")
                gate = parts[1]
                # Remaining tokens contain comma-separated logical qubits
                logical_qubits_str = " ".join(parts[2:])
                logical_qubits = [
                    q.strip() for q in logical_qubits_str.split(",") if q.strip()
                ]
                self.add_instruction(TransversalInstruction(gate, logical_qubits))
                continue

            # 6) Logical gates
            if line.startswith("Logical"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid Logical syntax: {raw_line}")
                gate = parts[1]
                logical_qubits_str = " ".join(parts[2:])
                logical_qubits = [
                    q.strip() for q in logical_qubits_str.split(",") if q.strip()
                ]
                self.add_instruction(LogicInstruction(gate, logical_qubits))
                continue

            # If we reach here, the line is not recognized.
            raise ValueError(f"Unrecognized LogicalQ IR line: {raw_line}")

    def add_instruction(self, instruction: LogicQIRInstruction) -> None:
        """
        Add a logical QEC IR instruction to the program.
        """
        self.instructions.append(instruction)

    def __repr__(self) -> str:
        blocks_repr = "\n".join([repr(block) for block in self.code_blocks.values()])
        instr_repr = "\n".join([repr(instr) for instr in self.instructions])
        if blocks_repr and instr_repr:
            return blocks_repr + "\n\n" + instr_repr
        elif blocks_repr:
            return blocks_repr
        else:
            return instr_repr


"""
The final compiled QStab IR types:

    # Start QECCycle on q0
    c0 = Prop X[1]Z[4]
    c1 = Prop X[3]Y[2]Z[5]
    c2 = Prop Z[0]

    E = Decode c0,c1,c2
    Correct E

    # Transversal H:
    H 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14

    # Logical X:
    X 1 , 2 , 3 , 4 , 5 , 6 , 7

    # compound Pauli:
    X[1]Z[4]Y[6]

    # Transversal CNOT:
    CNOT 1->8, 2->9 , 3->10 , 4->11 , 5->12 , 6->13 , 7->14
"""


class QStabIRType(Enum):
    PROP = 0
    DECODE = 1
    CORRECT = 2
    COMPOUNDPAULIS = 3
    X = 4
    Y = 5
    Z = 6
    H = 7
    S = 8
    CNOT = 9


class QStabIRInstruction:
    def __init__(self, ir_type: QStabIRType):
        self.ir_type = ir_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ir_type={self.ir_type})"


class PropInstruction(QStabIRInstruction):
    def __init__(self, pauli_string: str, classical_bit: str):
        super().__init__(QStabIRType.PROP)
        self.pauli_string = pauli_string
        self.classical_bit = classical_bit

    def __repr__(self) -> str:
        return f"{self.classical_bit} = Prop {self.pauli_string}"


class DecodeInstruction(QStabIRInstruction):
    def __init__(self, classical_bits: list[str], error_variable: str):
        super().__init__(QStabIRType.DECODE)
        self.classical_bits = classical_bits
        self.error_variable = error_variable

    def __repr__(self) -> str:
        bits_str = ",".join(self.classical_bits)
        return f"{self.error_variable} = Decode {bits_str}"


class CorrectInstruction(QStabIRInstruction):
    def __init__(self, error_variable: str):
        super().__init__(QStabIRType.CORRECT)
        self.error_variable = error_variable

    def __repr__(self) -> str:
        return f"Correct {self.error_variable}"


class CompoundPaulisInstruction(QStabIRInstruction):
    def __init__(self, pauli_string: str):
        super().__init__(QStabIRType.COMPOUNDPAULIS)
        self.pauli_string = pauli_string

    def __repr__(self) -> str:
        return f"{self.pauli_string}"


class XInstruction(QStabIRInstruction):
    def __init__(self, qubit_indices: list[int]):
        super().__init__(QStabIRType.X)
        self.qubit_indices = qubit_indices

    def __repr__(self) -> str:
        indices_str = " , ".join(map(str, self.qubit_indices))
        return f"X {indices_str}"


class YInstruction(QStabIRInstruction):
    def __init__(self, qubit_indices: list[int]):
        super().__init__(QStabIRType.Y)
        self.qubit_indices = qubit_indices

    def __repr__(self) -> str:
        indices_str = " , ".join(map(str, self.qubit_indices))
        return f"Y {indices_str}"


class ZInstruction(QStabIRInstruction):
    def __init__(self, qubit_indices: list[int]):
        super().__init__(QStabIRType.Z)
        self.qubit_indices = qubit_indices

    def __repr__(self) -> str:
        indices_str = " , ".join(map(str, self.qubit_indices))
        return f"Z {indices_str}"


class HInstruction(QStabIRInstruction):
    def __init__(self, qubit_indices: list[int]):
        super().__init__(QStabIRType.H)
        self.qubit_indices = qubit_indices

    def __repr__(self) -> str:
        indices_str = " , ".join(map(str, self.qubit_indices))
        return f"H {indices_str}"


class SInstruction(QStabIRInstruction):
    def __init__(self, qubit_indices: list[int]):
        super().__init__(QStabIRType.S)
        self.qubit_indices = qubit_indices

    def __repr__(self) -> str:
        indices_str = " , ".join(map(str, self.qubit_indices))
        return f"S {indices_str}"


class CNOTInstruction(QStabIRInstruction):
    def __init__(self, control_target_pairs: list[tuple[int, int]]):
        super().__init__(QStabIRType.CNOT)
        self.control_target_pairs = control_target_pairs

    def __repr__(self) -> str:
        pairs_str = " , ".join([f"{c}->{t}" for c, t in self.control_target_pairs])
        return f"CNOT {pairs_str}"


class QStabIRProgram:
    """
    A QStab IR program consisting of a sequence of QStab IR instructions.
    """

    def __init__(self):
        self.instructions: list[QStabIRInstruction] = []

    def add_instruction(self, instruction: QStabIRInstruction) -> None:
        """
        Add a QStab IR instruction to the program.
        """
        self.instructions.append(instruction)

    def __repr__(self) -> str:
        return "\n".join([repr(instr) for instr in self.instructions])


class LogicQCompiler:
    """
    Compile LogicalQ program to general stabilizer IR (QStab IR).

    Compilation is rule-based: each LogicQIRInstruction subclass has its own
    compilation rule implemented in a dedicated method:

        - QECCycleInstruction    → _compile_qeccycle(...)
        - TransversalInstruction → _compile_transversal(...)
        - MeasureInstruction     → _compile_measure(...)
        - LogicInstruction       → _compile_logic(...)

    Logical X / H are implemented using LogicalOperatorAnalyzer on the
    per-block StabCode. Logical Z currently uses a simple placeholder
    (Z on all physical qubits) which is correct for the example small codes
    but not necessarily minimal weight.

    Logical CNOT and logical measurements remain unimplemented stubs.
    """

    def __init__(self):
        # Map from block name to base (virtual) physical qubit index.
        # These are "virtual" physical qubits; register allocation will later
        # map them to real hardware qubits.
        self._block_offsets: dict[str, int] = {}
        self._total_virtual_qubits: int = 0

        # Per-block QECCycle counter, used to generate distinct syndrome and
        # error variable names.
        self._qec_cycle_counter: dict[str, int] = {}

        # Per-block logical operator analyzers
        self._logical_analyzers: dict[str, LogicalOperatorAnalyzer] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compile(self, logicalQIRprogram: LogicQIRProgram) -> QStabIRProgram:
        """
        Compile a LogicalQ program into a general stabilizer IR.

        Args:
            logicalQIRprogram: The LogicalQ IR program to compile.

        Returns:
            The compiled QStabIR program.
        """
        # Assign virtual physical qubit ranges to each QEC code block.
        self._assign_block_offsets(logicalQIRprogram)

        qstab = QStabIRProgram()

        # Dispatch each instruction to its compilation rule.
        for instr in logicalQIRprogram.instructions:
            if isinstance(instr, QECCycleInstruction):
                self._compile_qeccycle(instr, logicalQIRprogram, qstab)
            elif isinstance(instr, TransversalInstruction):
                self._compile_transversal(instr, logicalQIRprogram, qstab)
            elif isinstance(instr, MeasureInstruction):
                self._compile_measure(instr, logicalQIRprogram, qstab)
            elif isinstance(instr, LogicInstruction):
                self._compile_logic(instr, logicalQIRprogram, qstab)
            else:
                raise TypeError(f"Unknown LogicQIRInstruction: {instr!r}")

        return qstab

    # ------------------------------------------------------------------
    # Block / qubit bookkeeping
    # ------------------------------------------------------------------

    def _assign_block_offsets(self, program: LogicQIRProgram) -> None:
        """
        Assign contiguous virtual physical-qubit ranges to each QEC code block.

        For now, blocks are laid out back-to-back in the order they appear in
        program.code_blocks (dict insertion order).
        """
        offset = 0
        self._block_offsets.clear()

        for block_name, block in program.code_blocks.items():
            self._block_offsets[block_name] = offset
            offset += block.n  # number of physical qubits in that code block

        self._total_virtual_qubits = offset

    @staticmethod
    def _parse_logical_qubit(label: str) -> tuple[str, int]:
        """
        Parse a logical qubit label of the form 'q1[0]' → ('q1', 0).

        If no '[idx]' is present, interpret the entire string as the block name
        and use logical index 0 by default.
        """
        label = label.strip()
        if "[" in label and label.endswith("]"):
            block_name, idx_str = label.split("[", 1)
            idx_str = idx_str[:-1]  # strip trailing ']'
            return block_name.strip(), int(idx_str)
        else:
            return label, 0

    def _block_physical_range(
        self, block_name: str, program: LogicQIRProgram
    ) -> list[int]:
        """
        Return the list of virtual physical qubit indices for a given block.

        Currently, we assume each code block occupies a contiguous range and
        that all physical qubits in that block are used for a single logical
        qubit (k=1). This can be refined later.
        """
        if block_name not in self._block_offsets:
            raise KeyError(f"Block {block_name!r} has no assigned offset.")
        if block_name not in program.code_blocks:
            raise KeyError(f"Block {block_name!r} not found in program.code_blocks.")

        base = self._block_offsets[block_name]
        n = program.code_blocks[block_name].n
        return list(range(base, base + n))

    def _get_analyzer(
        self, block_name: str, program: LogicQIRProgram
    ) -> LogicalOperatorAnalyzer:
        """
        Lazily construct and cache a LogicalOperatorAnalyzer for a given block.
        """
        if block_name in self._logical_analyzers:
            return self._logical_analyzers[block_name]

        block = program.code_blocks.get(block_name)
        if block is None:
            raise KeyError(f"Block {block_name!r} not found in program.code_blocks.")
        analyzer = LogicalOperatorAnalyzer(block.code)
        self._logical_analyzers[block_name] = analyzer
        return analyzer

    # ------------------------------------------------------------------
    # Compilation rules for each instruction kind
    # ------------------------------------------------------------------

    def _compile_qeccycle(
        self,
        instr: QECCycleInstruction,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Compilation rule for QECCycle:

            QECCycle q1

        Lowered into:

            c0 = Prop <stab_0>
            c1 = Prop <stab_1>
            ...
            E = Decode c0,c1,...
            Correct E
        """
        block_name = instr.block_name
        if block_name not in program.code_blocks:
            raise KeyError(f"QECCycle refers to unknown block {block_name!r}")

        block = program.code_blocks[block_name]
        stabilizers = block.stabilizers

        # Cycle index for this block.
        cycle_idx = self._qec_cycle_counter.get(block_name, 0)
        self._qec_cycle_counter[block_name] = cycle_idx + 1

        classical_bits: list[str] = []

        # 1) Propagate each stabilizer to a classical bit.
        for s_idx, stab in enumerate(stabilizers):
            c_name = f"{block_name}_s{cycle_idx}_{s_idx}"
            classical_bits.append(c_name)
            qstab.add_instruction(
                PropInstruction(pauli_string=stab, classical_bit=c_name)
            )

        # 2) Decode the full syndrome into an error "handle" E_block_cycle.
        error_var = f"E_{block_name}_{cycle_idx}"
        qstab.add_instruction(
            DecodeInstruction(classical_bits=classical_bits, error_variable=error_var)
        )

        # 3) Apply the correction represented by that error variable.
        qstab.add_instruction(CorrectInstruction(error_variable=error_var))

    def _compile_transversal_cnot(
        self,
        instr: TransversalInstruction,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Compilation rule for transversal CNOT:

            Transversal CNOT qControl[0], qTarget[0]

        Assumptions:
            - We only support exactly two logical qubit labels.
            - Each label refers to a block with k >= 1; for now we only support
              logical index 0 in each block.
            - The two blocks have the same number of physical qubits n.
            - The mapping is purely index-wise: for i in [0..n-1]:
                    CNOT (control_block_base + i) -> (target_block_base + i)

        This is a simplification that treats all CNOT as transversal between
        code blocks; more sophisticated mappings can be plugged in later.
        """
        if len(instr.logical_qubits) != 2:
            raise ValueError(
                f"Transversal CNOT expects two logical qubits, got: {instr.logical_qubits}"
            )

        control_label, target_label = instr.logical_qubits
        control_block_name, control_lidx = self._parse_logical_qubit(control_label)
        target_block_name, target_lidx = self._parse_logical_qubit(target_label)

        control_block = program.code_blocks.get(control_block_name)
        if control_block is None:
            raise KeyError(
                f"Transversal CNOT refers to unknown control block {control_block_name!r}"
            )
        target_block = program.code_blocks.get(target_block_name)
        if target_block is None:
            raise KeyError(
                f"Transversal CNOT refers to unknown target block {target_block_name!r}"
            )

        # For now, only support logical index 0 in each block.
        if control_lidx != 0 or target_lidx != 0:
            raise NotImplementedError(
                "Transversal CNOT currently only supports logical index 0 in each block."
            )

        # Get physical ranges.
        control_range = self._block_physical_range(control_block_name, program)
        target_range = self._block_physical_range(target_block_name, program)

        if len(control_range) != len(target_range):
            raise NotImplementedError(
                f"Transversal CNOT between blocks of different size is not supported "
                f"(control n={len(control_range)}, target n={len(target_range)})."
            )

        control_target_pairs: list[tuple[int, int]] = list(
            zip(control_range, target_range)
        )
        qstab.add_instruction(
            CNOTInstruction(control_target_pairs=control_target_pairs)
        )

    def _compile_transversal(
        self,
        instr: TransversalInstruction,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Compilation rule for transversal logical gates.

        Currently supports:

            Transversal H    q1[0]
            Transversal CNOT q1[0], q2[0]   (assumed transversal between blocks)

        For H, we apply H to every physical qubit in the code block containing
        the specified logical qubit.

        For CNOT, we assume a simple transversal layout: the i-th physical
        qubit of the control block is CNOT control for the i-th physical qubit
        of the target block. This requires both blocks to have the same
        number of physical qubits.
        """
        gate = instr.gate.upper()

        if gate == "H":
            # Expect exactly one logical qubit label, e.g. "q1[0]".
            if len(instr.logical_qubits) != 1:
                raise ValueError(
                    f"Transversal H expects one logical qubit, got: {instr.logical_qubits}"
                )
            block_name, logical_idx = self._parse_logical_qubit(instr.logical_qubits[0])

            # Basic sanity: we only support k=1 for now.
            block = program.code_blocks.get(block_name)
            if block is None:
                raise KeyError(f"Transversal H refers to unknown block {block_name!r}")
            if block.k != 1:
                raise NotImplementedError(
                    "Transversal H for codes with k != 1 is not implemented."
                )

            # Apply H to all physical qubits in this block.
            qubit_indices = self._block_physical_range(block_name, program)
            qstab.add_instruction(HInstruction(qubit_indices=qubit_indices))

        elif gate == "CNOT":
            self._compile_transversal_cnot(instr, program, qstab)

        else:
            raise NotImplementedError(
                f"Transversal gate {instr.gate!r} is not implemented."
            )

    def _compile_transversal_cnot_stub(
        self,
        instr: TransversalInstruction,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Stub for transversal CNOT compilation.

        A concrete implementation would:
            - parse control and target logical qubits,
            - map each to the corresponding physical-qubit range,
            - construct a list of (control, target) pairs,
            - emit a CNOTInstruction with these pairs.

        This mapping is code-dependent and therefore left unimplemented here.
        """
        raise NotImplementedError(
            "Transversal CNOT compilation is not implemented yet. "
            "Mapping logical CNOT to physical CNOT pairs is code- and layout-dependent."
        )

    def _compile_measure(
        self,
        instr: MeasureInstruction,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Compilation rule for logical measurements.

        Currently supports only single-qubit logical Z measurements:

            c1 = Measure Logic Z  q1[0]

        We map the logical Z operator of that logical qubit to a physical
        Pauli string on the code block:

            - If code.set_logical_Z(k, ...) has been called, we use that
              operator string.
            - Otherwise, we fall back to Z^n on the block as a naive choice.

        Then we emit:

            <c_bit> = Prop <pauli_string>

        at the QStab IR level.
        """
        # For now, only support single logical qubit measurement.
        if len(instr.logical_qubits) != 1:
            raise NotImplementedError(
                f"Logical measurement with multiple logical qubits "
                f"({instr.logical_qubits}) is not implemented yet."
            )

        op = instr.operator.upper()
        if op != "Z":
            raise NotImplementedError(
                f"Only logical Z measurement is supported for now, got operator {instr.operator!r}"
            )

        block_name, logical_idx = self._parse_logical_qubit(instr.logical_qubits[0])
        block = program.code_blocks.get(block_name)
        if block is None:
            raise KeyError(
                f"Logical measurement refers to unknown block {block_name!r}"
            )

        if logical_idx < 0 or logical_idx >= block.k:
            raise ValueError(
                f"Logical qubit index {logical_idx} out of range for block {block_name!r} (k={block.k})"
            )

        code = block.code

        # Try to use the logical Z set on the code (if any).
        logicalZ_dict = getattr(code, "_logicalZ", {})
        opstring = logicalZ_dict.get(logical_idx, None)

        # Naive fallback: use Z^n as a logical Z.
        if opstring is None:
            opstring = "Z" * block.n

        # For now, keep Prop at the block-local level (no explicit global
        # indices in the pauli_string). Register allocation / layout can
        # later refine this.
        qstab.add_instruction(
            PropInstruction(pauli_string=opstring, classical_bit=instr.classical_bit)
        )

    def _compile_logic(
        self,
        instr: LogicInstruction,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Compilation rule for generic logical gates inside a code block:

            Logical X q1[0]
            Logical Z q1[0]
            Logical H q2[1]

        We use LogicalOperatorAnalyzer on the block's StabCode to obtain a
        physical operator pattern, then lower it to QStabIR:

            - For X / Z only on some qubits: XInstruction / ZInstruction.
            - For mixed Paulis: CompoundPaulisInstruction.
            - For H: HInstruction on the 'H' sites.
        """
        if len(instr.logical_qubits) != 1:
            raise NotImplementedError(
                f"Logical gate {instr.gate!r} with multiple logical qubits "
                f"({instr.logical_qubits}) is not implemented yet."
            )

        block_name, logical_idx = self._parse_logical_qubit(instr.logical_qubits[0])
        block = program.code_blocks.get(block_name)
        if block is None:
            raise KeyError(f"Logical gate refers to unknown block {block_name!r}")

        if logical_idx < 0 or logical_idx >= block.k:
            raise ValueError(
                f"Logical qubit index {logical_idx} out of range for block {block_name!r} (k={block.k})"
            )

        gate = instr.gate.upper()
        analyzer = self._get_analyzer(block_name, program)

        if gate == "X":
            opstring = analyzer.determine_logical_X(logical_idx)
            self._emit_pauli_like_operator(opstring, block_name, program, qstab)

        elif gate == "Z":
            # Placeholder: use Z on all physical qubits in the block.
            # For your example small codes (5-qubit, Steane, Shor), Z^n is a valid
            # logical Z (though not necessarily minimal weight). You can later
            # replace this with a proper determine_logical_Z implementation.
            opstring = "Z" * block.n
            self._emit_pauli_like_operator(opstring, block_name, program, qstab)

        elif gate == "H":
            opstring = analyzer.determine_logical_H(logical_idx)
            self._emit_H_operator(opstring, block_name, program, qstab)

        else:
            raise NotImplementedError(
                f"Logical gate {instr.gate!r} (other than X/Z/H) is not implemented yet."
            )

    # ------------------------------------------------------------------
    # Helpers to turn operator strings into QStabIR instructions
    # ------------------------------------------------------------------

    def _emit_pauli_like_operator(
        self,
        opstring: str,
        block_name: str,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Emit one or more QStabIR instructions for a Pauli-type operator
        described by `opstring` (e.g., 'XIXZXI...') over the physical qubits
        of the given block.

        Strategy:
            - If only X/I → XInstruction
            - If only Z/I → ZInstruction
            - If only Y/I → YInstruction
            - Otherwise → CompoundPaulisInstruction "X[i]Z[j]Y[k]..."
        """
        if (
            block_name not in self._block_offsets
            or block_name not in program.code_blocks
        ):
            raise KeyError(
                f"Unknown block {block_name!r} in _emit_pauli_like_operator."
            )

        base = self._block_offsets[block_name]
        n = program.code_blocks[block_name].n
        if len(opstring) != n:
            raise ValueError(
                f"Operator string length {len(opstring)} does not match code length {n} for block {block_name!r}."
            )

        # Collect where each non-identity Pauli acts.
        non_id_terms: list[tuple[str, int]] = []
        x_indices: list[int] = []
        y_indices: list[int] = []
        z_indices: list[int] = []

        for local_idx, ch in enumerate(opstring):
            ch = ch.upper()
            if ch == "I":
                continue
            global_idx = base + local_idx
            non_id_terms.append((ch, global_idx))
            if ch == "X":
                x_indices.append(global_idx)
            elif ch == "Y":
                y_indices.append(global_idx)
            elif ch == "Z":
                z_indices.append(global_idx)
            else:
                # Unexpected symbol; treat as part of compound operator
                pass

        if not non_id_terms:
            # Identity operator – nothing to emit.
            return

        chars = {ch for ch, _ in non_id_terms}

        if chars == {"X"}:
            qstab.add_instruction(XInstruction(qubit_indices=x_indices))
        elif chars == {"Y"}:
            qstab.add_instruction(YInstruction(qubit_indices=y_indices))
        elif chars == {"Z"}:
            qstab.add_instruction(ZInstruction(qubit_indices=z_indices))
        else:
            # Mixed Pauli operator, use the string "X[i]Z[j]Y[k]..."
            pauli_str = "".join(f"{ch}[{idx}]" for ch, idx in non_id_terms)
            qstab.add_instruction(CompoundPaulisInstruction(pauli_string=pauli_str))

    def _emit_H_operator(
        self,
        opstring: str,
        block_name: str,
        program: LogicQIRProgram,
        qstab: QStabIRProgram,
    ) -> None:
        """
        Emit HInstruction(s) for an operator string consisting of 'H' and 'I'
        over the physical qubits of the given block.

        In the placeholder LogicalOperatorAnalyzer, determine_logical_H returns
        "H" * n, but this helper supports sparse patterns as well.
        """
        if (
            block_name not in self._block_offsets
            or block_name not in program.code_blocks
        ):
            raise KeyError(f"Unknown block {block_name!r} in _emit_H_operator.")

        base = self._block_offsets[block_name]
        n = program.code_blocks[block_name].n
        if len(opstring) != n:
            raise ValueError(
                f"Operator string length {len(opstring)} does not match code length {n} for block {block_name!r}."
            )

        h_indices: list[int] = []
        for local_idx, ch in enumerate(opstring):
            ch = ch.upper()
            if ch == "H":
                global_idx = base + local_idx
                h_indices.append(global_idx)
            elif ch == "I":
                continue
            else:
                # For now we only support H/I patterns here.
                raise ValueError(
                    f"Unexpected character {ch!r} in logical H pattern for block {block_name!r}."
                )

        if h_indices:
            qstab.add_instruction(HInstruction(qubit_indices=h_indices))


# ----------------------------------------------------------------------
# Example program and main entry point
# ----------------------------------------------------------------------


def _build_example_program() -> LogicQIRProgram:
    """
    Build a small example LogicalQ IR program using two code blocks:

      - q1: 5-qubit code [[5,1,3,'Standard']]
      - q2: Steane code [[7,1,3,'Standard']]

    For each block we do:

        Logical X q[0]
        QECCycle q


        Transversal CNOT q1[0], q2[0]
        QECCycle q1
        QECCycle q2


        Logical Z q[0]
        QECCycle q

        Logical H q[0]
        QECCycle q

    Then we measure logical Z on both blocks (each followed by a QECCycle):

        c1 = Measure Logic Z  q1[0]
        QECCycle q1

        c2 = Measure Logic Z  q2[0]
        QECCycle q2
    """
    ir_text = """
    [[5,1,3,'Standard']] q1 {
        XZZXI;
        IXZZX;
        XIXZZ;
        ZXIXZ;
    }

    [[5,1,3,'Standard']] q2 {
        XZZXI;
        IXZZX;
        XIXZZ;
        ZXIXZ;
    }

    # Logical operations on q1
    Logical X q1[0]
    QECCycle q1

    Transversal CNOT q1[0], q2[0]
    QECCycle q1
    QECCycle q2

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
    """

    program = LogicQIRProgram()
    program.parse_from_string(ir_text)

    # Attach logical Z for both example blocks so the analyzer has references.
    # 5-qubit code: logical Z = ZZZZZ
    block_q1 = program.code_blocks.get("q1")
    if block_q1 is not None:
        block_q1.code.set_logical_Z(0, "ZZZZZ")

    # Steane code (in this representation): logical Z = ZZZZZZZ
    block_q2 = program.code_blocks.get("q2")
    if block_q2 is not None:
        block_q2.code.set_logical_Z(0, "ZZZZZ")

    return program


if __name__ == "__main__":
    """
    Simple driver: build an example LogicalQ IR program, compile it, and
    print both the LogicalQ IR and the resulting QStab IR.
    """
    # 1) Build example LogicalQ program
    program = _build_example_program()

    print("=== Parsed LogicalQ IR ===")
    print(program)
    print()

    # 2) Compile to QStab IR
    compiler = LogicQCompiler()
    qstab_program = compiler.compile(program)

    print("=== Compiled QStab IR ===")
    print(qstab_program)
