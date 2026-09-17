"""Scientific and serialization contracts for reusable SID profiles."""

import json
import math
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import binom

from scalerqec.Stratified import LERProfile, Scaler


def test_exact_repetition_polynomial_over_full_probability_range():
    # Exhaustive enumeration: 12 of 27 weight-2 and 20 of 27 weight-3
    # Pauli patterns defeat a three-bit majority decoder.
    profile = LERProfile([0, 0, 12 / 27, 20 / 27])
    p = np.r_[0, np.geomspace(1e-100, 0.9, 40), 1]
    q = 2 * p / 3
    np.testing.assert_allclose(
        profile.evaluate(p), 3 * q**2 - 2 * q**3, rtol=2e-13, atol=0
    )
    polynomial = profile.to_polynomial()
    np.testing.assert_allclose(polynomial(p), profile.evaluate(p), rtol=3e-13, atol=0)
    coefficients = polynomial.power_coefficients()
    np.testing.assert_allclose(
        [float(c) for c in coefficients],
        [0, 0, 4 / 3, -16 / 27],
        rtol=1e-13,
        atol=1e-15,
    )


def test_sid_polynomial_preserves_extrapolation_provenance_and_zero_spectra():
    profile = LERProfile(
        [0, 0.25, 0.5],
        modeled_weights=[False, True, True],
        metadata={"decoder": "fixed"},
    )
    polynomial = profile.to_polynomial()
    assert polynomial.metadata["modeled_weights"] == [1, 2]
    assert polynomial.metadata["profile_metadata"] == {"decoder": "fixed"}
    assert polynomial(0.2) == pytest.approx(profile.evaluate(0.2))
    assert LERProfile([0, 0]).to_polynomial()(0.2) == 0


def test_full_support_does_not_discard_rare_logical_failures():
    # The old integer-rounded five-sigma interval ended at w=5 here.
    rates = np.zeros(101)
    rates[6:] = 0.5
    profile = LERProfile(rates)
    expected = 0.5 * binom.sf(5, 100, 0.001)
    assert profile.evaluate(0.001) == pytest.approx(expected, rel=1e-12, abs=0)
    assert expected > 0


@pytest.mark.parametrize("p", [0, 1, 1e-20, 0.5, 0.999])
def test_large_n_normalization(p):
    profile = LERProfile(np.full(50_001, 0.3))
    assert profile.evaluate(p, max_working_elements=997) == pytest.approx(
        0.3, rel=1e-11
    )


def test_shapes_empty_arrays_and_chunking():
    profile = LERProfile([0.1, 0.4, 0.8])
    p = np.array([[0, 0.1], [0.5, 1]])
    expected = 0.1 * (1 - p) ** 2 + 0.8 * p * (1 - p) + 0.8 * p**2
    assert isinstance(profile.evaluate(0.3), float)
    np.testing.assert_allclose(profile.evaluate(p, max_working_elements=1), expected)
    assert profile.evaluate([]).shape == (0,)
    assert LERProfile([0.2]).evaluate(0.7) == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [-0.1, 1.01, float("nan"), float("inf"), [0.1, -0.1]])
def test_invalid_p(bad):
    with pytest.raises(ValueError):
        LERProfile([0, 1]).evaluate(bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_chunk_size(bad):
    with pytest.raises(ValueError):
        LERProfile([0, 1]).evaluate(0.2, max_working_elements=bad)


@pytest.mark.parametrize(
    "rates,kwargs",
    [
        ([], {}),
        ([[0, 1]], {}),
        ([0, math.nan], {}),
        ([0, 1.1], {}),
        ([0, 1], {"modeled_weights": [0, 1]}),
        ([0, 1], {"modeled_weights": [True]}),
        ([0, 1], {"sample_counts": [0, -1]}),
        ([0, 1], {"sample_counts": [0.0, 1.0]}),
        ([0, 1], {"failure_counts": [0, 1]}),
        ([0, 1], {"metadata": []}),
    ],
)
def test_invalid_spectrum(rates, kwargs):
    with pytest.raises(ValueError):
        LERProfile(rates, **kwargs)


def test_diagnostics_and_snapshot_isolation(tmp_path):
    rates = np.array([0, 0.2, 0.8])
    metadata = {"decoder_reference_p": 0.001, "model": {"alpha": 2}}
    profile = LERProfile(
        rates,
        modeled_weights=[False, True, False],
        metadata=metadata,
        sample_counts=[0, 0, 10],
        failure_counts=[0, 0, 8],
    )
    rates[:] = 1
    metadata["model"]["alpha"] = 4
    assert profile.metadata["model"]["alpha"] == 2
    with pytest.raises(ValueError):
        profile.conditional_ler.setflags(write=True)
    curve = profile.curve([0.1, 0.9])
    np.testing.assert_allclose(curve.modeled_ler, [0.036, 0.036])
    np.testing.assert_allclose(curve.modeled_probability_mass, [0.18, 0.18])
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = LERProfile.load(path)
    np.testing.assert_array_equal(
        loaded.evaluate([0, 0.2, 1]), profile.evaluate([0, 0.2, 1])
    )
    assert loaded.metadata == profile.metadata
    np.testing.assert_array_equal(loaded.sample_counts, [0, 0, 10])
    np.testing.assert_array_equal(loaded.failure_counts, [0, 0, 8])


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("format", "other"),
        ("num_noise", 100),
    ],
)
def test_reject_corrupted_profile(tmp_path, key, value):
    path = tmp_path / "profile.json"
    LERProfile([0, 0.5]).save(path)
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        LERProfile.load(path)


def test_reject_missing_fields(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        LERProfile.load(path)


def test_plot_returns_axes():
    import matplotlib.pyplot as plt

    profile = LERProfile([0, 0.5])
    ax = profile.plot([0.01, 0.1], label="fixed decoder")
    assert ax.get_xscale() == "log"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.005, 0.05])
    plt.close(ax.figure)
    ax = profile.plot([0, 1])
    assert ax.get_xscale() == "linear"
    plt.close(ax.figure)
    with pytest.raises(ValueError):
        profile.plot([])


def test_no_profile_before_successful_fit():
    with pytest.raises(RuntimeError, match="successful fit"):
        Scaler().get_profile()


def test_reweighting_never_calls_sampler_or_decoder():
    from importlib import import_module

    scaler_module = import_module("scalerqec.Stratified.Scaler")
    profile = LERProfile([0, 0, 12 / 27, 20 / 27])
    with (
        patch.object(
            scaler_module,
            "return_samples_with_fixed_QEPG_numpy",
            side_effect=AssertionError,
        ),
        patch.object(scaler_module, "compile_QEPG", side_effect=AssertionError),
    ):
        assert profile.evaluate(0.1) > 0


@pytest.mark.parametrize(
    "circuit,message",
    [
        ("R 0\nDEPOLARIZE1(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", "noiseless"),
        ("R 0\nM(0.1) 0\nOBSERVABLE_INCLUDE(0) rec[-1]", "noiseless"),
        ("R 0\nM 0", "exactly one"),
        ("R 0\nM 0\nOBSERVABLE_INCLUDE(1) rec[-1]", "exactly one"),
        ("R 0 1\nSWAP 0 1\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", "Unsupported"),
    ],
)
def test_scaler_rejects_unsupported_inputs(tmp_path, circuit, message):
    path = tmp_path / "circuit.stim"
    path.write_text(circuit)
    with pytest.raises(ValueError, match=message):
        Scaler(error_rate=0.01).parse_from_file(path)


def test_scaler_normalizes_repeat_and_multitarget(tmp_path):
    path = tmp_path / "circuit.stim"
    path.write_text(
        "R 0 1 2\nREPEAT 2 {\n H 0 1\n}\nM 0 1 2\n"
        "DETECTOR rec[-3] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]"
    )
    scaler = Scaler(error_rate=0.01)
    scaler.parse_from_file(path)
    assert scaler._num_noise == 7
    assert scaler.calc_logical_error_rate_with_fixed_w(1, 0) == 0


def test_convenience_method_profiles_only_once():
    scaler = Scaler()
    profile = LERProfile([0, 0.5])
    with patch.object(scaler, "profile_from_file", return_value=profile) as sampling:
        curve = scaler.calculate_LER_curve_from_file("unused", [0.1, 0.2], 3)
    sampling.assert_called_once()
    np.testing.assert_allclose(curve.ler, [0.05, 0.1])


def test_profile_workflow_on_small_circuit_without_plotting(tmp_path):
    circuit = tmp_path / "repetition.stim"
    circuit.write_text(
        "R 0 1 2\nM 0 1 2\nDETECTOR rec[-3] rec[-2]\n"
        "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]"
    )
    scaler = Scaler(time_budget=2)
    profile = scaler.profile_from_file(circuit, 3)
    assert profile.num_noise == 3
    assert not profile.modeled_weights.any()  # all nonzero weights measured
    assert profile.metadata["decoder_policy"] == "fixed"
    assert profile.evaluate(0.1) == pytest.approx(0.01274074074, rel=0.04)
    counts = profile.sample_counts.copy()
    with patch.object(
        scaler, "_sampling_step", side_effect=AssertionError("resampled")
    ):
        profile.evaluate(np.geomspace(1e-5, 0.3, 100))
    np.testing.assert_array_equal(profile.sample_counts, counts)


def test_physical_model_parameters_update_transformed_curve():
    from scalerqec.Stratified.models import OurScurveModel

    model = OurScurveModel(t=1)
    model.set_params(alpha=7, mu=20, beta=4)
    weights = np.array([2, 5, 10, 20])
    np.testing.assert_allclose(
        model.transform(model.predict(weights)), model.linear_prediction(weights)
    )


def test_failed_circuit_replacement_invalidates_previous_fit(tmp_path):
    from scalerqec.Stratified.models import OurScurveModel

    scaler = Scaler(error_rate=0.01)
    scaler._model = OurScurveModel()
    scaler._model._is_fitted = True
    scaler._estimated_subspaceLER = {1: 0.2}
    with pytest.raises(FileNotFoundError):
        scaler.parse_from_file(tmp_path / "missing.stim")
    with pytest.raises(RuntimeError):
        scaler.get_profile()
    assert scaler._estimated_subspaceLER == {}
