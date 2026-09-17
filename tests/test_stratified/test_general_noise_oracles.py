"""Independent Pauli algebra and regressions for misleading rare-event SEs."""

import itertools
import math

import numpy as np
import pytest

from benchmark.general_noise_matrix import cases, make_circuit
from benchmark.general_noise_oracles import (
    FAMILIES,
    SPECS,
    LookupDecoder,
    channels,
    circuit_at,
    code_spec,
    exact_joint,
    exact_profile_variance,
    signature,
)
from scalerqec.Stratified import LinearNoiseModel
from scalerqec.Stratified.confidence import _maximum_log_likelihood


@pytest.mark.parametrize(
    "name", list(SPECS) + ["repetition_3", "repetition_5", "repetition_7"]
)
@pytest.mark.parametrize("family", FAMILIES)
def test_all_fault_columns_and_weight_masses_against_independent_pauli_algebra(
    name, family
):
    model = LinearNoiseModel(circuit_at(name, family, 0.05), 0.05)
    stabs, logical = code_spec(name)
    checks = stabs + [logical]
    if family == "readout":
        expected = [[0, 1 << j] for j in range(len(checks))]
        weights = [[0, 0] for _ in checks]
    else:
        expected, weights = [], []
        for channel in channels(len(logical), family):
            effects, ws = [0], [0]
            for pauli in channel.paulis:
                full = ["I"] * len(logical)
                for q, a in zip(channel.qubits, pauli):
                    full[q] = a
                effects.append(signature(full, checks))
                ws.append(sum(a != "I" for a in full))
            expected.append(effects)
            weights.append(ws)
    assert len(model._responses) == len(expected)
    powers = 1 << np.arange(len(checks))
    for response, factor, effects, ws in zip(
        model._responses, model._factors, expected, weights
    ):
        np.testing.assert_array_equal(response.astype(int) @ powers, effects)
        np.testing.assert_array_equal(factor.weights, ws)
    for p in [0, 0.001, 0.05, 0.15]:
        joint = exact_joint(name, family, p)
        assert joint.sum() == pytest.approx(1, abs=2e-14)
        np.testing.assert_allclose(
            model.weight_distribution(p), joint.sum(axis=1), atol=1e-15, rtol=3e-13
        )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("p", [0.0, 0.01, 0.5])
def test_maximum_likelihood_bound_against_exhaustive_histories(family, p):
    model = LinearNoiseModel(circuit_at("repetition_3", family, 0.05), 0.05)
    brute = np.full(model.max_weight + 1, -np.inf)
    second_joint = np.zeros((model.max_weight + 1, 8))
    reference = [f.probabilities(0.05) for f in model._factors]
    target = [f.probabilities(p) for f in model._factors]
    for outcomes in itertools.product(*(range(len(f.weights)) for f in model._factors)):
        q = math.prod(probs[a] for probs, a in zip(reference, outcomes))
        prob = math.prod(probs[a] for probs, a in zip(target, outcomes))
        if q > 0 and prob > 0:
            w = sum(f.weights[a] for f, a in zip(model._factors, outcomes))
            brute[w] = max(brute[w], math.log(prob) - math.log(q))
            bits = np.logical_xor.reduce(
                [r[a] for r, a in zip(model._responses, outcomes)]
            )
            state = int(bits.astype(int) @ (1 << np.arange(3)))
            second_joint[w, state] += prob * prob / q
    np.testing.assert_allclose(
        _maximum_log_likelihood(model, p, model.max_weight), brute, atol=1e-13
    )
    np.testing.assert_allclose(
        _maximum_log_likelihood(model, p, 0), brute[:1], atol=1e-13
    )
    np.testing.assert_allclose(
        exact_joint("repetition_3", family, p, second_moment_reference=0.05),
        second_joint,
        atol=1e-13,
        rtol=2e-13,
    )


def test_exact_variance_reduces_to_bernoulli_at_reference():
    # Readout has only W=0; at p=p0 the profile is ordinary Bernoulli MC.
    p, n = 0.05, 1000
    decoder = LookupDecoder(exact_joint("five_qubit", "readout", p))
    truth = float(decoder.failure_mass(exact_joint("five_qubit", "readout", p)).sum())
    assert exact_profile_variance("five_qubit", "readout", p, p, n) == pytest.approx(
        truth * (1 - truth) / n
    )


@pytest.mark.parametrize("family", ["nonuniform_single", "heralded_pauli"])
def test_high_ess_does_not_certify_unobserved_failures(family):
    name, p0, target = "repetition_7", 0.05, 0.001
    decoder = LookupDecoder(exact_joint(name, family, p0))
    model = LinearNoiseModel(circuit_at(name, family, p0), p0)
    profile = model.sample_profile(decoder, shots_per_weight=5000, seed=130927)
    exact = float(decoder.failure_mass(exact_joint(name, family, target)).sum())
    estimate = profile.evaluate(target)
    assert estimate.minimum_ess > 4000
    assert abs(estimate.ler - exact) > 10 * estimate.standard_error
    assert not profile.failures[profile.weights == 3].any()
    bounds = profile.confidence_bounds(target)
    assert 3 in bounds.zero_failure_weights
    assert bounds.lower <= exact <= bounds.upper


@pytest.mark.parametrize("family", FAMILIES)
def test_confidence_bounds_include_exact_oracle_and_omitted_weights(family):
    name, p0 = "five_qubit", 0.05
    decoder = LookupDecoder(exact_joint(name, family, p0))
    model = LinearNoiseModel(circuit_at(name, family, p0), p0)
    profile = model.sample_profile(decoder, shots_per_weight=100, max_weight=2, seed=12)
    for p in [0.0, 0.01, 0.5]:
        truth = float(decoder.failure_mass(exact_joint(name, family, p)).sum())
        bounds = profile.confidence_bounds(p)
        assert 0 <= bounds.lower <= truth <= bounds.upper <= 1
        tighter_confidence = profile.confidence_bounds(p, confidence=0.999)
        assert tighter_confidence.lower <= bounds.lower
        assert tighter_confidence.upper >= bounds.upper


@pytest.mark.parametrize("confidence", [0, 1, -1, np.nan, np.inf])
def test_invalid_confidence(confidence):
    model = LinearNoiseModel("R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1)
    profile = model.sample_profile(
        lambda d: np.zeros((len(d), 1), dtype=bool), shots_per_weight=2
    )
    with pytest.raises(ValueError, match="confidence"):
        profile.confidence_bounds(0.01, confidence=confidence)


def test_bounds_handle_tiny_reference_and_zero_rate_channels(tmp_path):
    model = LinearNoiseModel(
        "R 0\nX_ERROR(1e-310) 0\nZ_ERROR(0) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
        1e-310,
    )
    profile = model.sample_profile(
        lambda d: np.zeros((len(d), 1), dtype=bool), shots_per_weight=100
    )
    for p in [0.0, 0.5, 1.0]:
        bounds = profile.confidence_bounds(p)
        assert bounds.lower <= p <= bounds.upper
    path = tmp_path / "profile.npz"
    profile.save(path)
    assert type(profile).load(path).confidence_bounds(0.5) == profile.confidence_bounds(
        0.5
    )


def test_matrix_uniform_transform_preserves_measurements_and_detectors():
    spec = next(s for s in cases() if s["name"] == "stim_rotated_z_d3_uniform_single")
    circuit = make_circuit(spec)
    assert circuit.num_measurements > 0
    assert circuit.num_detectors == 24
    circuit.detector_error_model(decompose_errors=True)
