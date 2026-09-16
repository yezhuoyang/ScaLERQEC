"""Compare circuit transformations and propagation against Stim semantics."""

import numpy as np
import pytest
import stim

from scalerqec import qepg
from scalerqec.Clifford.stimparser import rewrite_stim_code
from scalerqec.Monte.noise_model_parser import _accumulate_noise
from scalerqec.QEC.noisemodel import SIDNoiseModel


@pytest.mark.parametrize(
    "gate", ["SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "S_DAG"]
)
def test_unitary_decompositions_match_stim(gate):
    rewritten = stim.Circuit(rewrite_stim_code(f"{gate} 0"))
    assert stim.Tableau.from_circuit(rewritten) == stim.Tableau.from_named_gate(gate)


@pytest.mark.parametrize(
    "gate,basis", [("MX", "X"), ("MY", "Y"), ("MRX", "X"), ("MRY", "Y")]
)
@pytest.mark.parametrize("sign", [1, -1])
def test_basis_measurements_preserve_postmeasurement_state(gate, basis, sign):
    # For non-reset measurements the matching eigenstate must survive;
    # measure-reset must prepare the + eigenstate of the requested basis.
    sim = stim.TableauSimulator()
    sim.set_state_from_stabilizers(
        [stim.PauliString(("+" if sign > 0 else "-") + basis)]
    )
    sim.do(stim.Circuit(rewrite_stim_code(f"{gate} 0")))
    assert sim.current_measurement_record() == [sign < 0]
    expected = 1 if gate.startswith("MR") else sign
    assert sim.peek_observable_expectation(stim.PauliString(basis)) == expected


def test_y_reset_prepares_y_eigenstate():
    sim = stim.TableauSimulator()
    sim.do(stim.Circuit(rewrite_stim_code("RY 0")))
    assert sim.peek_y(0) == 1


def test_sid_noise_interleaves_coalesced_gates():
    circuit = stim.Circuit("CX 0 1 0 2")
    expected = stim.Circuit(
        "DEPOLARIZE1(0.1) 0 1\nCX 0 1\nDEPOLARIZE1(0.1) 0 2\nCX 0 2"
    )
    assert SIDNoiseModel(0.1).inject_noise(circuit) == expected


def test_pauli_channel_composition_cross_terms():
    pending = {}
    _accumulate_noise(pending, 0, 1, 0, 0)
    _accumulate_noise(pending, 0, 0, 1, 0)
    assert pending[0] == (0, 0, 1)  # X followed by Y gives Z.
    pending = {0: (0.2, 0.3, 0.1)}
    _accumulate_noise(pending, 0, 0.1, 0.05, 0.2)
    np.testing.assert_allclose(pending[0], [0.235, 0.265, 0.185])


def test_repeated_measurement_detector_cancels_earlier_fault():
    # X before both M instructions flips both records and cancels in D0.
    circuit = "R 0\nM 0\nM 0\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    matrix = np.asarray(qepg.return_detector_matrix(circuit))
    np.testing.assert_array_equal(
        matrix, [[0, 1], [1, 1], [0, 1], [1, 1], [0, 0], [0, 0]]
    )


def test_python_repeated_measurement_propagation():
    from scalerqec.Clifford.clifford import CliffordCircuit
    from scalerqec.Clifford.QEPGpython import QEPGpython

    circuit = "R 0\nM 0\nM 0\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    clifford = CliffordCircuit(1)
    clifford.compile_from_stim_circuit_str(circuit)
    graph = QEPGpython(clifford)
    graph.backword_graph_construction()
    np.testing.assert_array_equal(
        graph._propMatrix, [[0, 1], [1, 1], [0, 1], [1, 1], [0, 0], [0, 0]]
    )


def test_recompilation_replaces_python_and_native_circuit():
    from scalerqec.Clifford.clifford import CliffordCircuit

    circuit = "R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"
    python_circuit = CliffordCircuit(1)
    for _ in range(2):
        python_circuit.compile_from_stim_circuit_str(circuit)
        assert python_circuit.totalnoise == 1
        assert python_circuit.stimcircuit.num_measurements == 1
    native_circuit = qepg.CliffordCircuit()
    for _ in range(2):
        native_circuit.compile_from_rewrited_stim_string(circuit)
        assert native_circuit.get_num_noise() == 1
