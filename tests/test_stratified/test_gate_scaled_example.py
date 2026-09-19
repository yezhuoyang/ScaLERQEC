"""The exact p/5, p, 5p family requested in the hardware-user example."""

import numpy as np
import pymatching
import pytest

from examples.gate_scaled_noise import memory_circuit, surface_model
from scalerqec.Stratified import LinearNoiseModel


def independent_distribution(p):
    """Enumerate final bit patterns by elementary propagation, not fault tables.

    Qubit 0 is the low bit. Its first X/Y error propagates to all three data
    qubits. An (a,b) error after the first CX becomes (a,b,a) after the second
    CX. Each two-qubit depolarizing channel has three nonzero binary flip
    patterns, each with probability 4p/15. Z components do not flip Z readout.
    """
    distribution = np.zeros(8)
    distribution[0] = 1

    def combine(patterns):
        nonlocal distribution
        updated = np.zeros(8)
        for pattern, probability in patterns:
            updated[np.arange(8) ^ pattern] += probability * distribution
        distribution = updated

    combine([(0, 1 - 2 * p / 15), (7, 2 * p / 15)])
    combine([(0, 1 - 4 * p / 5), (5, 4 * p / 15), (2, 4 * p / 15), (7, 4 * p / 15)])
    combine([(0, 1 - 4 * p / 5), (1, 4 * p / 15), (4, 4 * p / 15), (5, 4 * p / 15)])
    for q in range(3):
        combine([(0, 1 - 5 * p), (1 << q, 5 * p)])
    return distribution


def fixed_decoder():
    return pymatching.Matching.from_detector_error_model(
        memory_circuit(0.01).detector_error_model(decompose_errors=True)
    )


def test_gate_rates_domain_and_original_noise_locations():
    model = LinearNoiseModel(memory_circuit(0.01), 0.01)
    np.testing.assert_allclose(
        [f.rates[0] for f in model._factors], [0.2, 1, 1, 5, 5, 5]
    )
    assert model.max_p == pytest.approx(0.2)
    for p in [0, 0.001, 0.04, 0.2]:
        actual, expected = list(model.circuit_at(p)), list(memory_circuit(p))
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert a.name == b.name and a.targets_copy() == b.targets_copy()
            np.testing.assert_allclose(
                a.gate_args_copy(), b.gate_args_copy(), rtol=1e-14
            )
        pair = model._factors[1]
        assert pair.probabilities(p)[pair.weights == 2].sum() == pytest.approx(
            3 * p / 5
        )
    with pytest.raises(ValueError, match="between"):
        model.circuit_at(0.201)


def test_existing_stabir_builder_accepts_all_three_gate_rate_overrides():
    model = surface_model(0.01, distance=3, rounds=1)
    np.testing.assert_allclose(model.rates, [0.2, 1, 5])
    assert model.max_p == pytest.approx(0.2)
    assert model.num_detectors > 0 and model.num_observables == 1


@pytest.mark.parametrize("p", [0, 0.002, 0.02, 0.2])
def test_full_ler_against_independent_bit_propagation(p):
    model = LinearNoiseModel(memory_circuit(0.01), 0.01)
    decoder = fixed_decoder()
    patterns = (np.arange(8)[:, None] >> np.arange(3)) & 1
    det = (patterns[:, :2] ^ patterns[:, 1:]).astype(bool)
    failed = decoder.decode_batch(det)[:, 0] != patterns[:, 0]
    expected = independent_distribution(p)[failed].sum()
    assert model.enumerate_histories(decoder, p)["ler"] == pytest.approx(
        expected, abs=1e-13
    )
    d, obs = (
        memory_circuit(p)
        .compile_detector_sampler(seed=260925)
        .sample(100000, separate_observables=True)
    )
    observed = np.mean(decoder.decode_batch(d) != obs)
    assert (
        abs(observed - expected)
        <= 7 * np.sqrt(expected * (1 - expected) / len(d)) + 1e-14
    )
