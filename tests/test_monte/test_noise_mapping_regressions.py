"""Regressions for noise that used to move, vanish, or change distribution."""

import numpy as np
import pytest
import stim

from scalerqec import qepg
from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.Clifford.QEPGpython import QEPGpython
from scalerqec.Clifford.stimparser import rewrite_stim_code
from scalerqec.Monte.noise_model_parser import extract_noise_model
from scalerqec.QEC.noisemodel import NoiseModel, SI1000NoiseModel
from scalerqec.Stratified.general_noise import LinearNoiseModel


@pytest.mark.parametrize(
    "noise,operations",
    [
        ("DEPOLARIZE2(.2) 0 1", "M 0 1"),
        ("DEPOLARIZE2(.2) 1 0", "CX 0 1\nM 0 1"),
        ("DEPOLARIZE2(.2) 0 1\nDEPOLARIZE2(.3) 0 1", "CX 0 1\nM 0 1"),
        ("DEPOLARIZE2(.2) 0 1", "R 0\nM 0 1"),
        ("DEPOLARIZE2(.2) 0 1", "H 0\nH 0\nM 0 1"),
        ("X_ERROR(.2) 1", "CZ 0 1\nM 0 1"),
        ("PAULI_CHANNEL_1(.1,.2,.3) 0", "M 0 1"),
        ("DEPOLARIZE2(.2) 0 1", "CZ 0 1\nM 0 1"),
    ],
)
@pytest.mark.parametrize("backend", ["python", "native"])
def test_legacy_noise_mapping_matches_exact_stim_oracle(noise, operations, backend):
    circuit = (
        f"R 0 1\n{noise}\n{operations}\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    model = extract_noise_model(circuit)
    normalized = rewrite_stim_code(circuit)
    compiled = CliffordCircuit(2)
    compiled.compile_from_stim_circuit_str(normalized)
    graph = QEPGpython(compiled)
    graph.backword_graph_construction()
    if backend == "native":
        graph._cpp_graph = qepg.compile_QEPG(normalized)
    else:
        graph._cpp_graph = None
    oracle = LinearNoiseModel(circuit, 0.1)
    # Test the complete joint distribution of D0 and L0, including correlation.
    exact = np.zeros(4)
    import itertools

    for history in itertools.product(*(range(len(f.weights)) for f in oracle._factors)):
        probability = np.prod(
            [f.probabilities(0.1)[a] for f, a in zip(oracle._factors, history)]
        )
        bits = np.logical_xor.reduce(
            [oracle._responses[j][a] for j, a in enumerate(history)]
        )
        exact[int(bits[0]) * 2 + int(bits[1])] += probability
    shots = 120_000
    det, obs = graph.sample_nonuniform_batch(
        model.noise_probs, shots, model.correlated_pairs, np.random.default_rng(312)
    )
    counts = (
        np.bincount(2 * det[:, 0].astype(int) + obs.astype(int), minlength=4) / shots
    )
    tolerance = 7 * np.sqrt(exact * (1 - exact) / shots) + 1 / shots
    assert np.all(np.abs(counts - exact) <= tolerance)


@pytest.mark.parametrize(
    "syntax",
    [
        "PAULI_CHANNEL_2(.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 0 1",
        "M(.1) 0",
        "E(.1) X0",
        "H_XY 0",
    ],
)
def test_legacy_mapper_never_silently_drops_unsupported_syntax(syntax):
    with pytest.raises(NotImplementedError, match="LinearNoiseModel"):
        extract_noise_model(f"R 0 1\n{syntax}\nM 0 1")


@pytest.mark.parametrize("basis", ["", "X", "Y"])
@pytest.mark.parametrize("fault", ["reset", "measurement"])
def test_preparation_and_measurement_errors_act_in_each_basis(basis, fault):
    noise = NoiseModel(0.2)
    noise.disable_error("MEASUREMENT" if fault == "reset" else "RESET")
    circuit = noise.inject_noise(
        stim.Circuit(f"R{basis} 0\nM{basis} 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    )
    model = LinearNoiseModel(circuit, 0.2)
    zero = lambda det: np.zeros((len(det), 1), dtype=np.bool_)
    assert model.enumerate_histories(zero, 0.2)["ler"] == pytest.approx(0.2)


def test_measure_reset_also_has_a_reset_fault():
    noise = NoiseModel(0.2)
    noise.disable_error("MEASUREMENT")
    circuit = noise.inject_noise(stim.Circuit("R 0\nMRX 0\nMX 0"))
    assert "MRX 0\nZ_ERROR(0.2) 0" in str(circuit)


@pytest.mark.parametrize("model", [NoiseModel(0.1), SI1000NoiseModel(0.1, p_idle=0)])
def test_operation_noise_interleaves_shared_qubit_gates(model):
    circuit = model.inject_noise(stim.Circuit("CX 0 1 0 2"))
    assert circuit == stim.Circuit(
        "CX 0 1\nDEPOLARIZE2(.1) 0 1\nCX 0 2\nDEPOLARIZE2(.1) 0 2"
    )


@pytest.mark.parametrize(
    "gate,flag",
    [
        ("CZ", "CZ"),
        ("CX", "CNOT"),
        ("S", "P"),
        ("X", "X"),
        ("Y", "Y"),
        ("Z", "Z"),
        ("H", "H"),
    ],
)
def test_disable_error_is_specific_to_the_requested_gate(gate, flag):
    model = NoiseModel(0.1)
    model.disable_error(flag)
    targets = "0 1" if gate in {"CX", "CZ"} else "0"
    original = stim.Circuit(f"{gate} {targets}")
    assert model.inject_noise(original) == original


def test_si1000_idle_rate_is_applied_to_inactive_qubits():
    model = SI1000NoiseModel(0.01, p_reset=0, p_meas=0, p_1q=0, p_2q=0, p_idle=0.003)
    original = stim.Circuit("R 0 1\nTICK\nH 0\nTICK\nM 0 1")
    expected = stim.Circuit("R 0 1\nTICK\nH 0\nDEPOLARIZE1(.003) 1\nTICK\nM 0 1")
    assert model.inject_noise(original) == expected
