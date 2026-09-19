"""Noise insertion contracts: all Stim Clifford gates and operation locations."""

import numpy as np
import pytest
import stim

from scalerqec.QEC import NoiseModel, SI1000NoiseModel
from scalerqec.Stratified import LinearNoiseModel


def test_deprecated_clifford_bridge_obeys_the_same_gate_rates():
    from scalerqec.Clifford.clifford import CliffordCircuit

    circuit = CliffordCircuit(2)
    circuit.add_hadamard(0)
    circuit.add_cnot(0, 1)
    circuit.add_measurement(1)
    noise = NoiseModel(0.01, p_1q=0.002, p_2q=0.03, p_meas=0.05)
    result = noise.reconstruct_clifford_circuit(circuit)
    assert result.stimcircuit == noise.apply(circuit.stimcircuit)
    with pytest.raises(ValueError, match="Unknown error type"):
        noise.disable_error("typo")


@pytest.mark.parametrize(
    "gate",
    [
        g.name
        for g in stim.gate_data().values()
        if g.is_unitary and (g.is_single_qubit_gate or g.is_two_qubit_gate)
    ],
)
def test_all_stim_unitary_gate_types_get_the_correct_channel(gate):
    data = stim.gate_data(gate)
    targets = [0, 1] if data.is_two_qubit_gate else [0]
    original = stim.Circuit()
    original.append(gate, targets, tag="source")
    saved = original.copy()
    noisy = NoiseModel(0.01, p_1q=0.002, p_2q=0.03).apply(original)
    assert original == saved
    assert noisy[0] == original[0]
    assert noisy[1].name == ("DEPOLARIZE2" if len(targets) == 2 else "DEPOLARIZE1")
    assert noisy[1].gate_args_copy() == [0.03 if len(targets) == 2 else 0.002]
    assert noisy[1].targets_copy() == original[0].targets_copy()


@pytest.mark.parametrize(
    "name", ["p_1q", "p_2q", "p_meas", "p_reset", "p_idle", "error_rate"]
)
@pytest.mark.parametrize("value", [-0.1, 1.1, np.inf, np.nan])
def test_invalid_probabilities_fail_early(name, value):
    args = {"error_rate": 0.01, name: value}
    with pytest.raises(ValueError, match="finite probability"):
        NoiseModel(**args)


def test_generic_matches_explicit_preset_overrides():
    args = {"p_1q": 0.002, "p_2q": 0.01, "p_meas": 0.05, "p_reset": 0, "p_idle": 0}
    circuit = stim.Circuit("RX 0\nH 1\nCX 0 1\nMRX 0\nM 1")
    assert NoiseModel(0.01, **args).apply(circuit) == SI1000NoiseModel(
        0.01, **args
    ).apply(circuit)


def test_repeat_tags_and_sequential_overlapping_pairs():
    circuit = stim.Circuit("REPEAT[loop] 2 {\nISWAP[tag] 0 1 1 2\n}")
    expected = stim.Circuit(
        "REPEAT[loop] 2 {\nISWAP[tag] 0 1\nDEPOLARIZE2(.1) 0 1\n"
        "ISWAP[tag] 1 2\nDEPOLARIZE2(.1) 1 2\n}"
    )
    assert NoiseModel(0.1).apply(circuit) == expected


@pytest.mark.parametrize(
    "measurement", ["MPP X0*X1 Z0*Z1", "MXX 0 1", "MYY 0 1", "MZZ 0 1"]
)
def test_parity_readout_noise_has_no_quantum_backaction(measurement):
    circuit = stim.Circuit(measurement)
    noisy = NoiseModel(0.01, p_meas=0.2).apply(circuit)
    assert len(noisy) == 1
    assert noisy[0].targets_copy() == circuit[0].targets_copy()
    assert noisy[0].gate_args_copy() == [0.2]
    twice = noisy + circuit
    twice.append(
        "DETECTOR",
        [stim.target_rec(-1), stim.target_rec(-1 - circuit.num_measurements)],
    )
    bits = twice.compile_detector_sampler(seed=8).sample(40000)
    assert abs(bits.mean() - 0.2) < 0.015
    with pytest.raises(ValueError, match="not linear"):
        NoiseModel(0.01).apply(noisy)


def test_idle_tracking_includes_pauli_targets_and_repeat_boundaries():
    original = stim.Circuit("R 0 1 2\nTICK\nREPEAT 2 {\nMPP X0*X1\nTICK\n}")
    noisy = NoiseModel(0, p_idle=0.03).apply(original)
    assert [op.targets_copy() for op in noisy if op.name == "DEPOLARIZE1"] == [
        [stim.GateTarget(2)],
        [stim.GateTarget(2)],
    ]


def test_feedback_is_an_ideal_pauli_frame_update():
    circuit = stim.Circuit("M 0\nCX rec[-1] 1\nCZ sweep[0] 2")
    assert NoiseModel(0.1, p_meas=0).apply(circuit) == circuit


@pytest.mark.parametrize("gate", ["SPP", "SPP_DAG"])
def test_pauli_rotations_use_documented_elementary_gate_decomposition(gate):
    circuit = stim.Circuit(f"{gate} X0*Y1*Z2")
    assert NoiseModel(0).apply(circuit) == circuit
    model = NoiseModel(0.01)
    noisy = model.apply(circuit)
    assert noisy == model.apply(circuit.decomposed())
    assert noisy.without_noise().to_tableau() == circuit.to_tableau()


def test_all_zero_channels_are_disabled_and_existing_noise_is_preserved():
    circuit = stim.Circuit("R 0\nH 0\nX_ERROR(.03) 0\nM 0")
    assert (
        NoiseModel(0.01, p_1q=0, p_2q=0, p_meas=0, p_reset=0, p_idle=0).apply(circuit)
        == circuit
    )
    with pytest.raises(TypeError):
        NoiseModel(0.01).apply("H 0")


def test_default_rates_follow_base_rate_setter_but_explicit_overrides_do_not():
    noise = NoiseModel(0.01, p_1q=0.002)
    noise.error_rate = 0.02
    assert noise.error_rate == 0.02
    assert noise.apply(stim.Circuit("H 0\nCX 0 1")) == stim.Circuit(
        "H 0\nDEPOLARIZE1(.002) 0\nCX 0 1\nDEPOLARIZE2(.02) 0 1"
    )
    with pytest.raises(ValueError):
        noise.error_rate = float("nan")


def test_scaled_locations_have_expected_rates_and_weight_two_mass():
    circuit = NoiseModel(0.01, p_1q=0.002, p_2q=0.01, p_meas=0.05, p_reset=0).apply(
        stim.Circuit("R 0 1\nS 0\nCX 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]")
    )
    model = LinearNoiseModel(circuit, 0.01)
    np.testing.assert_allclose(model.rates, [0.2, 1, 5])
    dep2 = model._factors[1]
    assert dep2.probabilities(0.01)[dep2.weights == 2].sum() == pytest.approx(0.006)
