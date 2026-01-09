#This is the file define the semantic of logical circuits
#This is useful for Magic state distillation which works on logical level
import re
from .qeccircuit import StabCode



"""

Synatatic design:

First, user has to allocate QEC code blocks to hold logical qubits.
All operations are defined on the allocated blocks


Type surface:
    The stabilizers should be a function of d
    This type describe the stabilizer structure.


surface q1 [n1,k1,d1]
surface q2 [n2,k2,d2]

surface t0 [n3,k3,d3]   # magic T state block

q1[0] = LogicH q1[0]

t0 = Distill15to1_T[d=25]     # returns a magic_T handle
InjectT q1[0], t0

q2[1] = LogicCNOT q1[0], q2[1]

c1 = LogicMeasure q1[0]
c2 = LogicMeasure q2[1]
"""



"""
A compiler translate LogicQ to a slightly lower-level IR which is more specific how to implement logical operations on specific QEC codes.


TRANSVERSAL H q1[0]            # Transversal H on logical qubit 0 of block q1
QECCycle q1                    # Perform one QEC cycle on block q1, including stabilizer measurements and corrections
Transversal CNOT q1[0], q2[1]  # Transversal CNOT between logical qubit 0 of block q1 and logical qubit 1 of block q2
Transversal H q2[1]
XXZZXIXII   q1[0]                # Also support direct Pauli operations, if needed

QECCycle q1                    # Perform one QEC cycle on block q1, including stabilizer measurements and corrections
QECCycle q2                    # Perform one QEC cycle on block q2, including stabilizer measurements and corrections

c1 = Prop LogicZ q1[0]          # Propagate logical Z operator to classical bit c1
c2 = Prop LogicZ q2[1]          # Propagate logical Z operator to classical bit c2


"""








"""
Magic state distillation protocol:

Now we have a language which works on the logical level. We can define the MGD protocol as follows:
15-to-1 Reed-Muller magic state distillation (logical-level).

Precondition:
  - surface f has k >= 15 logical qubits.
  - On each attempt, f[0..14] are initialized to 15 noisy |T> states
    (supplied externally; this IR does not prepare |T> states).

Postcondition:
  - returns a distilled magic_T handle backed by the encoded logical qubit.


  protocol Distill15to1_T(surface f, int d) -> magic_T:
    Repeat:

        # ---- X-type stabilizer checks ----
        c_x1 = LogicProp IIIIIIIXXXXXXXX
        c_x2 = LogicProp IIIXXXXIIIIXXXX
        c_x3 = LogicProp IXXIIXXIIXXIIXX
        c_x4 = LogicProp XIXIXIXIXIXIXIX

        # ---- Z-type stabilizer checks ----
        c_z1  = LogicProp IIIIIIIIZZZZZZZZ
        c_z2  = LogicProp IIIZZZZIIIIZZZZ
        c_z3  = LogicProp IZZIIZZIIZZIIZZ
        c_z4  = LogicProp ZIZIZIZIZIZIZIZ
        c_z12 = LogicProp IIIIIIIIIIZZZZ
        c_z13 = LogicProp IIIIIIIIZZIIIZZ
        c_z14 = LogicProp IIIIIIIIZIZIZIZ
        c_z23 = LogicProp IIIIIZZIIIIIIZZ
        c_z24 = LogicProp IIIIZIZIIIIIZIZ
        c_z34 = LogicProp IIZIIIZIIIZIIIZ

        Success = c_x1 == 0 && c_x2 == 0 && c_x3 == 0 && c_x4 == 0 &&
                  c_z1 == 0 && c_z2 == 0 && c_z3 == 0 && c_z4 == 0 &&
                  c_z12 == 0 && c_z13 == 0 && c_z14 == 0 &&
                  c_z23 == 0 && c_z24 == 0 && c_z34 == 0
        Until Success

        return 


"""


class CodeBlock:
    """
    The QEC code block to hold logical qubits
    TODO: The type should be StabCode
    """
    def __init__(self, type:str,name: str, n: int, k: int, d: int):
        self._name = name  # Name of the code block
        self._n = n      # Number of physical qubits
        self._k = k      # Number of logical qubits
        self._d = d      # Code distance
        self._type = type  # Type of the code, e.g., 'surface', 'color', 'LDPC'


    def __repr__(self):
        return f"{self._name}[{self._n},{self._k},{self._d}]"



class LogicalGate:
    def __init__(self, type: str):
        self._type = type  # Type of the logical gate, e.g., 'CNOT', 'H', 'T'


    @property
    def type(self):
        return self._type

    def __repr__(self):
        raise NotImplementedError("Subclasses must implement __repr__ method.")


class LogicalH(LogicalGate):
    def __init__(self, block: CodeBlock, index: int):
        super().__init__('H')
        self._block = block
        self._index = index

    def __repr__(self):
        return f"LogicalH {self._block._name}[{self._index}]"


class LogicalCNOT(LogicalGate):
    def __init__(self, control_block: CodeBlock, control_index: int, target_block: CodeBlock, target_index: int):
        super().__init__('CNOT')
        self._control_block = control_block
        self._target_block = target_block
        self._control_index = control_index
        self._target_index = target_index

    def __repr__(self):
        return f"LogicalCNOT {self._control_block._name}[{self._control_index}], {self._target_block._name}[{self._target_index}]"


class LogicalT(LogicalGate):
    def __init__(self, block: CodeBlock, index: int):
        super().__init__('T')
        self._block = block
        self._index = index

    def __repr__(self):
        return f"LogicalT {self._block._name}[{self._index}]"



class InjectT(LogicalGate):
    def __init__(self, dest_block: CodeBlock, dest_index: int, magic_T_handle: str):
        super().__init__('InjectT')
        self._dest_block = dest_block
        self._dest_index = dest_index
        self._magic_T_handle = magic_T_handle

    def __repr__(self):
        return f"InjectT {self._dest_block._name}[{self._dest_index}], {self._magic_T_handle}"



class LogicalMeasure(LogicalGate):    
    """
    Measure Logical Z
    TODO: Support more general logical measurement, such as MXX, MZX, etc.
    """
    def __init__(self, block: CodeBlock, cindex: int, index: int):
        super().__init__('Measure')
        self._block = block
        self._cindex = cindex
        self._index = index


    @property
    def index(self):
        return self._index
    
    @property
    def block(self):
        return self._block
    
    @property
    def cindex(self):
        return self._cindex

    def __repr__(self):
        return f"c[{self._cindex}] = LogicalMeasure {self._block._name}[{self._index}]"



class LogicalReset(LogicalGate):
    def __init__(self, block: CodeBlock, index: int):
        super().__init__('Reset')
        self._block = block
        self._index = index


    @property
    def index(self):
        return self._index

    @property
    def block(self):
        return self._block


    def __repr__(self):
        return f"LogicalReset {self._block._name}[{self._index}]"



class LogicalCircuit:
    """
    Class of Logical circuit
    User 
    """
    def __init__(self):
        self.gates = []  # List to hold logical gates in the circuit
        self._blocks = []  # List to hold code blocks in the circuit
        self._qec_types = []
        self._qec_type_names = []
        self._MGT_handles = []

    def add_qec_type(self, qec_type: str) -> None:
        self._qec_types.append(qec_type)
        self._qec_type_names.append(qec_type)

    def add_MGT_handle(self, handle: str) -> None:
        self._MGT_handles.append(handle)

    def add_block(self, block: CodeBlock) -> None:
        self._blocks.append(block)
        
    def add_gate(self, gate: LogicalGate) -> None:
        """
        Check if the gate is compatible with the code blocks
        """
        match gate.type:
            case 'H' | 'T' | 'Measure' | 'Reset':
                # Single-qubit gates
                if gate._index >= gate._block._k:  # type: ignore[attr-defined]
                    raise ValueError(f"Index {gate._index} out of range for block {gate._block._name} with {gate._block._k} logical qubits.")  # type: ignore[attr-defined]
                self.gates.append(gate)
            case 'CNOT':
                # Two-qubit gates
                if gate._control_index >= gate._control_block._k:  # type: ignore[attr-defined]
                    raise ValueError(f"Control index {gate._control_index} out of range for block {gate._control_block._name} with {gate._control_block._k} logical qubits.")  # type: ignore[attr-defined]
                if gate._target_index >= gate._target_block._k:  # type: ignore[attr-defined]
                    raise ValueError(f"Target index {gate._target_index} out of range for block {gate._target_block._name} with {gate._target_block._k} logical qubits.")  # type: ignore[attr-defined]
                self.gates.append(gate)
            case 'InjectT':
                if gate._dest_index >= gate._dest_block._k:  # type: ignore[attr-defined]
                    raise ValueError(f"Index {gate._dest_index} out of range for block {gate._dest_block._name} with {gate._dest_block._k} logical qubits.")  # type: ignore[attr-defined]
                if gate._MGT_handle not in self._MGT_handles:  # type: ignore[attr-defined]
                    raise ValueError(f"Magic T handle {gate._MGT_handle} not recognized.")  # type: ignore[attr-defined]
                self.gates.append(gate)
            case _:
                raise ValueError(f"Unsupported gate type: {gate.type}")

    def __repr__(self):
        output = ""
        for gate in self.gates:
            output += repr(gate) + "\n"
        return output




class LogicalParser:
    """
    Parser for logical circuits
    Parse input string into LogicalCircuit object and vice versa
    Example:

    
    Type surface

    surface q1 [n1,k1,d1]
    surface q2 [n2,k2,d2]
    surface t0 [n3,k3,d3]   # magic T state block

    q1[0] = LogicalH q1[0]
    t0 = Distill15to1_T[d=25]     # returns a magic_T handle
    InjectT q1[0], t0
    q2[1] = LogicalCNOT q1[0], q2[1]
    c[0] = LogicalMeasure q1[0]
    c[1] = LogicalMeasure q2[1]
    """


    # Patterns defined at class level to avoid AttributeError
    _index_re = re.compile(r"(\w+)\[(\d+)\]")
    _triplet_re = re.compile(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]")
    _param_re = re.compile(r"\[(\w+)\s*=\s*(\d+)\]")  # For [d=25]


    def __init__(self):
        self._qec_types = []
        self._qec_type_names = []
        self._blocksmap = {}


    def _parse_indexed(self, text: str) -> tuple[str, int]:
            """Helper to extract name and index from string like 'q1[0]'"""
            match = self._index_re.search(text)
            if not match:
                raise ValueError(f"Could not parse indexed reference: {text}")
            return match.group(1), int(match.group(2))

    def parse(self, logical_circ_string: str) -> LogicalCircuit:
        circuit = LogicalCircuit()
        # Remove comments and empty lines
        lines = []
        for raw_line in logical_circ_string.strip().split("\n"):
            clean_line = raw_line.split('#')[0].strip()
            if clean_line:
                lines.append(clean_line)

        # Pass 1: Types
        for line in lines:
            parts = line.split()
            if parts[0] == "Type":
                circuit.add_qec_type(parts[1])

        # Pass 2: Block Definitions
        for line in lines:
            parts = line.split()
            if parts[0] in circuit._qec_type_names:
                qec_type = parts[0]
                block_name = parts[1]
                # Use regex to find [n,k,d]
                triplet_match = self._triplet_re.search(line)
                if triplet_match:
                    n, k, d = map(int, triplet_match.groups())
                    block = CodeBlock(qec_type, block_name, n, k, d)
                    self._blocksmap[block_name] = block
                    circuit.add_block(block)

        # Pass 3: Gates and Assignments
        for line in lines:
            parts = line.replace(',', ' ').split() # Replace comma with space for easy splitting
            if parts[0] in circuit._qec_type_names or parts[0] == "Type":
                continue

            # Case 1: Assignment (e.g., c[0] = LogicalMeasure q1[0])
            if "=" in line:
                lhs, rhs_full = line.split("=", 1)
                lhs_parts = lhs.strip().split()
                rhs_parts = rhs_full.strip().split()
                
                # Check for classical assignment
                if lhs_parts[0].startswith("c["):
                    c_name, c_idx = self._parse_indexed(lhs_parts[0])
                    gate_op = rhs_parts[0]
                    
                    if gate_op == "LogicalMeasure":
                        b_name, b_idx = self._parse_indexed(rhs_parts[1])
                        circuit.add_gate(LogicalMeasure(self._blocksmap[b_name], c_idx, b_idx))
                    
                    else:
                        raise ValueError(f"Unsupported gate operation in assignment: {gate_op}")

            # Case 2: In-place Gates (e.g., InjectT q1[0], t0)
            else:
                gate_type = parts[0]
                if gate_type == "LogicalH":
                    name, idx = self._parse_indexed(parts[1])
                    circuit.add_gate(LogicalH(self._blocksmap[name], idx))
                
                elif gate_type == "LogicalCNOT":
                    ctrl_name, ctrl_idx = self._parse_indexed(parts[1])
                    tgt_name, tgt_idx = self._parse_indexed(parts[2])
                    circuit.add_gate(LogicalCNOT(
                        self._blocksmap[ctrl_name], ctrl_idx, 
                        self._blocksmap[tgt_name], tgt_idx
                    ))
                
                elif gate_type == "InjectT":
                    dest_name, dest_idx = self._parse_indexed(parts[1])
                    handle = parts[2]
                    circuit.add_gate(InjectT(self._blocksmap[dest_name], dest_idx, handle))

        return circuit





if __name__ == "__main__":

    parser = LogicalParser()

    logical_circ_str = """
    Type surface
    surface q1 [13,1,3]
    surface q2 [13,1,3]
    LogicalH q1[0]
    LogicalCNOT q1[0], q2[0]
    c[1] = LogicalMeasure q1[0]
    c[2] = LogicalMeasure q2[0]
    """

    logical_circuit = parser.parse(logical_circ_str)
    print(logical_circuit)