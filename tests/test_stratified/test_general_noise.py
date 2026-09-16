"""Independent exact and statistical checks for general-noise profiling."""

import math

import numpy as np
import pytest
import stim

from scalerqec.Stratified.general_noise import GeneralNoiseProfile, LinearNoiseModel


def zero_decoder(det):
    return np.zeros((len(det), 1), dtype=np.bool_)


def majority_decoder(det):
    # d0=x0^x1, d1=x1^x2; the correction on x0 is d0 & ~d1.
    return (det[:, 0] & ~det[:, 1])[:, None]


def repetition(noise):
    return stim.Circuit(f"""R 0 1 2
{noise}
M 0 1 2
DETECTOR rec[-3] rec[-2]
DETECTOR rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-3]
""")


def assert_mc(model, decoder, p, expected, seed=42, shots=100_000):
    det, obs = (
        model.circuit_at(p)
        .compile_detector_sampler(seed=seed)
        .sample(shots, separate_observables=True)
    )
    actual = np.any(decoder(det) != obs, axis=1).mean()
    sigma = math.sqrt(expected * (1 - expected) / shots)
    assert abs(actual - expected) < 7 * sigma + 1 / shots


def test_nonuniform_weight_only_counterexample():
    model = LinearNoiseModel(
        "R 0 1\nX_ERROR(.1) 0\nX_ERROR(.2) 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]", 0.1
    )
    for p in [0, 0.01, 0.2, 0.5]:
        exact = model.enumerate_histories(zero_decoder, p)
        assert exact["ler"] == pytest.approx(p)
        if p > 0:
            assert exact["conditional_ler"][1] == pytest.approx(
                (1 - 2 * p) / (3 - 4 * p)
            )
    profile = model.sample_profile(zero_decoder, shots_per_weight=12_000, seed=9)
    for p in [0, 1e-6, 0.01, 0.1, 0.2, 0.5]:
        result = profile.evaluate(p)
        assert abs(result.ler - p) <= 7 * result.standard_error + 1e-14
        assert result.missing_probability_mass < 1e-14


def test_depolarize2_physical_weights_and_analytic_ler():
    model = LinearNoiseModel(
        repetition("DEPOLARIZE2(.08) 0 1\nDEPOLARIZE1(.02) 2"), 0.1
    )
    assert model.max_weight == 3
    np.testing.assert_array_equal(np.bincount(model._factors[0].weights), [1, 6, 9])
    for p in [0, 1e-6, 0.03, 0.1, 0.4, 1.25]:
        a, b = 0.8 * p, 0.2 * p
        # Two flips in the pair: 4/15*a; exactly one: 8/15*a,
        # with a third-qubit bit flip of probability 2*b/3.
        expected = 4 * a / 15 + 8 * a / 15 * (2 * b / 3)
        exact = model.enumerate_histories(majority_decoder, p)
        assert exact["ler"] == pytest.approx(expected, abs=1e-15)
        np.testing.assert_allclose(
            exact["weight_mass"],
            np.convolve([1 - a, 0.4 * a, 0.6 * a], [1 - b, b]),
            atol=1e-14,
        )
    profile = model.sample_profile(majority_decoder, shots_per_weight=8000, seed=22)
    for p in [0.0001, 0.01, 0.1, 0.4]:
        a, b = 0.8 * p, 0.2 * p
        expected = 4 * a / 15 + 8 * a / 15 * (2 * b / 3)
        result = profile.evaluate(p)
        assert abs(result.ler - expected) < 7 * result.standard_error
        assert_mc(model, majority_decoder, p, expected)


@pytest.mark.parametrize(
    "noise",
    [
        "X_ERROR(.03) 0\nY_ERROR(.05) 1\nZ_ERROR(.07) 2",
        "PAULI_CHANNEL_1(.02,.03,.04) 0 1 2",
        "DEPOLARIZE2(.1) 0 1\nDEPOLARIZE2(.06) 1 2",
        "PAULI_CHANNEL_2(.01,.02,.03,.04,.05,.06,.07,.02,.01,.01,.02,.03,.01,.02,.03) 0 1",
        "E(.1) X0 X1\nELSE_CORRELATED_ERROR(.2) Y1 Y2\nELSE_CORRELATED_ERROR(.3) Z0",
        "HERALDED_ERASE(.15) 0 1",
        "HERALDED_PAULI_CHANNEL_1(.03,.04,.05,.06) 0",
    ],
)
def test_channel_semantics_against_actual_stim(noise):
    model = LinearNoiseModel(repetition(noise), 0.1)
    for p in [0.03, 0.12]:
        exact = model.enumerate_histories(majority_decoder, p)
        np.testing.assert_allclose(
            exact["weight_mass"], model.weight_distribution(p), atol=1e-14
        )
        assert exact["weight_mass"].sum() == pytest.approx(1)
        assert_mc(model, majority_decoder, p, exact["ler"])
    profile = model.sample_profile(majority_decoder, shots_per_weight=5000, seed=40)
    result = profile.evaluate(0.04)
    truth = model.enumerate_histories(majority_decoder, 0.04)["ler"]
    assert abs(result.ler - truth) <= 7 * result.standard_error + 1e-14


def test_heralded_identity_and_record_errors_have_zero_pauli_weight():
    model = LinearNoiseModel(
        "R 0\nHERALDED_PAULI_CHANNEL_1(.1,0,0,0) 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    assert model.max_weight == 1  # zero-probability Pauli outcomes remain harmless
    assert model.enumerate_histories(zero_decoder, 0.2)["ler"] == pytest.approx(0.2)
    profile = model.sample_profile(zero_decoder, shots_per_weight=20_000, seed=13)
    np.testing.assert_array_equal(profile.sampled_weights, [0])
    result = profile.evaluate(0.2)
    assert abs(result.ler - 0.2) < 7 * result.standard_error
    for op in ["M(.1) 0", "MPAD(.1) 0"]:
        model = LinearNoiseModel(f"R 0\n{op}\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1)
        assert model.max_weight == 0
        assert model.enumerate_histories(zero_decoder, 0.3)["ler"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "prepare,measure",
    [
        ("R 0 1", "M(.1) !0 1"),
        ("RX 0 1", "MX(.1) 0 1"),
        ("RY 0 1", "MY(.1) 0 1"),
        ("R 0 1", "MR(.1) 0 1"),
        ("RX 0 1", "MRX(.1) 0 1"),
        ("RY 0 1", "MRY(.1) 0 1"),
        ("RX 0 1", "MXX(.1) 0 1"),
        ("RY 0 1", "MYY(.1) 0 1"),
        ("R 0 1", "MZZ(.1) 0 1"),
        ("R 0 1", "MPP(.1) Z0*Z1 Z0"),
    ],
)
def test_measurement_syntax_and_record_positions(prepare, measure):
    model = LinearNoiseModel(
        f"{prepare}\n{measure}\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    assert model.enumerate_histories(zero_decoder, 0.2)["ler"] == pytest.approx(0.2)
    assert_mc(model, zero_decoder, 0.2, 0.2)


def test_feedback_repeats_coordinates_and_multiple_observables():
    circuit = stim.Circuit("""QUBIT_COORDS(1,2) 0
R 0 1
REPEAT 2 {
    X_ERROR(.1) 0
    M 0
    CX rec[-1] 1
    R 0
    SHIFT_COORDS(0,1)
}
M 1
OBSERVABLE_INCLUDE(0) rec[-1]
OBSERVABLE_INCLUDE(1) rec[-2]
""")
    model = LinearNoiseModel(circuit, 0.1)
    decoder = lambda d: np.zeros((len(d), 2), dtype=np.bool_)
    truth = model.enumerate_histories(decoder, 0.2)["ler"]
    assert truth == pytest.approx(1 - 0.8**2)
    assert_mc(model, decoder, 0.2, truth)


_UNITARIES = [name for name, gate in stim.gate_data().items() if gate.is_unitary]


@pytest.mark.parametrize("name", _UNITARIES)
def test_every_installed_stim_unitary_and_multifault_linearity(name):
    gate = stim.gate_data(name)
    targets = "0 1" if gate.is_two_qubit_gate else "0"
    if name in {"SPP", "SPP_DAG"}:
        targets = "X0*Y1"
    operation = stim.Circuit(f"{name} {targets}")
    circuit = stim.Circuit("R 0 1\nDEPOLARIZE1(.1) 0 1") + operation
    circuit += stim.Circuit("DEPOLARIZE2(.1) 0 1")
    circuit += operation.inverse()
    circuit += stim.Circuit("M 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    model = LinearNoiseModel(circuit, 0.1)
    converter = model.ideal_circuit.compile_m2d_converter()
    rng = np.random.default_rng(55)
    for _ in range(10):
        history = {
            j: int(rng.integers(len(f.weights))) for j, f in enumerate(model._factors)
        }
        bits = np.logical_xor.reduce(
            [model._responses[j][a] for j, a in history.items()]
        )
        samples = model._forced_circuit(history).compile_sampler(seed=7).sample(3)
        actual = converter.convert(measurements=samples, append_observables=True)
        np.testing.assert_array_equal(actual, np.tile(bits, (3, 1)))


def test_conditional_sampler_matches_exact_weight_probabilities():
    model = LinearNoiseModel(
        repetition("DEPOLARIZE2(.08) 0 1\nDEPOLARIZE1(.02) 2\nX_ERROR(.04) 0"), 0.1
    )
    profile = model.sample_profile(majority_decoder, shots_per_weight=10_000, seed=82)
    for p in [0.01, 0.1, 0.6]:
        ratio = profile._likelihood(p)
        target = model.weight_distribution(p)
        for w in profile.sampled_weights:
            values = ratio[profile.weights == w] * profile._reference_mass[w]
            sigma = values.std(ddof=1) / math.sqrt(len(values))
            assert abs(values.mean() - target[w]) < 7 * sigma + 1e-13


def test_truncation_bound_and_profile_roundtrip(tmp_path, monkeypatch):
    model = LinearNoiseModel(repetition("DEPOLARIZE1(.1) 0 1 2"), 0.1)
    profile = model.sample_profile(
        majority_decoder, shots_per_weight=1000, max_weight=1, seed=1
    )
    estimate = profile.evaluate(0.2)
    assert estimate.ler == 0
    assert estimate.missing_probability_mass == pytest.approx(3 * 0.2**2 * 0.8 + 0.2**3)
    path = tmp_path / "profile.npz"
    profile.save(path)
    monkeypatch.setattr(
        LinearNoiseModel,
        "compile_responses",
        lambda _: pytest.fail("Resampled a saved profile"),
    )
    loaded = GeneralNoiseProfile.load(path)
    assert loaded.evaluate(0.2) == estimate
    assert loaded.model._responses is None


@pytest.mark.parametrize("p", [float("nan"), float("inf"), -0.1, 1.01])
def test_invalid_target_parameters(p):
    model = LinearNoiseModel(
        "R 0\nX_ERROR(.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    with pytest.raises(ValueError):
        model.weight_distribution(p)


def test_nondeterministic_ideal_observable_is_rejected():
    with pytest.raises(ValueError, match="non-deterministic"):
        LinearNoiseModel("RX 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1)


def test_zero_noise_and_identity_noise_annotations():
    model = LinearNoiseModel(
        "R 0\nI_ERROR(.2) 0\nII_ERROR(.2) 0 1\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    profile = model.sample_profile(zero_decoder, shots_per_weight=2)
    assert profile.evaluate(0.2).ler == 0
    assert model.max_p == 0.5


@pytest.mark.parametrize("shots", [0, 1, -1, 2.5, True])
def test_invalid_sampling_budget(shots):
    model = LinearNoiseModel("R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1)
    with pytest.raises(ValueError):
        model.sample_profile(zero_decoder, shots_per_weight=shots)


def test_tiny_truncation_tail_does_not_round_down_to_zero():
    model = LinearNoiseModel(repetition("DEPOLARIZE1(.1) 0 1 2"), 0.1)
    profile = model.sample_profile(
        majority_decoder, shots_per_weight=2, max_weight=2, seed=1
    )
    assert profile.evaluate(1e-10).missing_probability_mass == pytest.approx(
        1e-30, rel=1e-12, abs=0
    )


def test_stabir_frontend_uses_existing_noise_locations():
    from scalerqec.QEC.noisemodel import SI1000NoiseModel
    from scalerqec.QEC.surface import RepetitionCode

    code = RepetitionCode(distance=3, rounds=2)
    code.noisemodel = SI1000NoiseModel(0.01)
    model = LinearNoiseModel.from_stabcode(code, 0.01)
    assert code.is_IR_compiled()
    assert model.circuit == code.stimcirc.flattened()
    assert model.circuit_at(0.01) == code.stimcirc.flattened()
    assert any(op.name == "DEPOLARIZE2" for op in model.circuit)


def test_chain_skipped_branches_and_new_chain_reset():
    circuit = repetition(
        "E(.1) X0 X1\nELSE_CORRELATED_ERROR(.2) X1 X2\nE(.3) X0 X2\nELSE_CORRELATED_ERROR(.4) Z0"
    )
    model = LinearNoiseModel(circuit, 0.1)
    profile = model.sample_profile(majority_decoder, shots_per_weight=5000, seed=87)
    for p in [0, 0.02, 0.2, 0.25]:
        result = profile.evaluate(p)
        exact = model.enumerate_histories(majority_decoder, p)["ler"]
        assert abs(result.ler - exact) <= 7 * result.standard_error + 1e-13


def test_profile_detaches_and_rejects_malformed_statistics(tmp_path):
    model = LinearNoiseModel(
        "R 0\nX_ERROR(.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    profile = model.sample_profile(zero_decoder, shots_per_weight=2)
    assert profile.model is not model
    for field, value in [
        ("weights", [0.5] * 4),
        ("failures", [2] * 4),
        ("active", [20] * 4),
        ("misses", [[20]] * 4),
    ]:
        args = {
            "weights": profile.weights,
            "failures": profile.failures,
            "active": profile.active,
            "misses": profile.misses,
        }
        args[field] = value
        with pytest.raises(ValueError):
            GeneralNoiseProfile(model, **args)
    path = tmp_path / "bad.npz"
    np.savez(path, manifest=np.array('{"format":"wrong","version":1}'))
    with pytest.raises(ValueError, match="format"):
        GeneralNoiseProfile.load(path)


@pytest.mark.parametrize("reference", [0, -0.1, float("nan"), float("inf"), 1])
def test_reference_must_be_interior(reference):
    # At reference=1 the channel also has probability one, so support is lost.
    probability = 1 if reference == 1 else 0.1
    with pytest.raises(ValueError):
        LinearNoiseModel(f"R 0\nX_ERROR({probability}) 0\nM 0", reference)


def test_invalid_decoder_output_and_enumeration_budget():
    model = LinearNoiseModel(repetition("DEPOLARIZE1(.1) 0 1 2"), 0.1)
    with pytest.raises(ValueError, match="Decoder"):
        model.sample_profile(lambda d: np.full((len(d), 1), 2), shots_per_weight=2)
    with pytest.raises(ValueError, match="Enumeration"):
        model.enumerate_histories(majority_decoder, 0.1, max_histories=1)


def test_reweighting_combines_tiny_stratum_mass_before_exponentiating():
    # Raw P_p/P_p0 overflows; Z_w(p0)*P_p/P_p0 is just p here.
    p0 = 1e-310
    model = LinearNoiseModel(
        f"R 0\nX_ERROR({p0}) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", p0
    )
    profile = model.sample_profile(zero_decoder, shots_per_weight=2, seed=1)
    for p in [0.1, 0.5, 1]:
        result = profile.evaluate(p)
        assert result.ler == pytest.approx(p, rel=1e-12)
        assert result.standard_error == 0


def test_two_qubit_noise_requires_reweighting_inside_the_stratum():
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.2) 0 1\nX_ERROR(.4) 0\nX_ERROR(.6) 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]",
        0.2,
    )
    reference = model.enumerate_histories(zero_decoder, 0.2)
    target = model.enumerate_histories(zero_decoder, 0.001)
    naive = target["weight_mass"] @ reference["conditional_ler"]
    assert abs(naive / target["ler"] - 1) > 0.1
    assert abs(reference["conditional_ler"][2] - target["conditional_ler"][2]) > 0.1
    profile = model.sample_profile(zero_decoder, shots_per_weight=10_000, seed=58)
    result = profile.evaluate(0.001)
    exact = 38 * 0.001 / 15 - 32 * 0.001**2 / 15
    assert target["ler"] == pytest.approx(exact)
    assert abs(result.ler - exact) < 7 * result.standard_error
