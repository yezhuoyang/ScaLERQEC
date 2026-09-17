"""Polynomial export and lossless compression, checked against raw histories."""

import json
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
import sympy as sp

from scalerqec.Stratified.general_noise import LinearNoiseModel
from scalerqec.Stratified.noise_polynomial import LERPolynomial


def zero(det):
    return np.zeros((len(det), 1), dtype=np.bool_)


@pytest.fixture
def counter_profile():
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.2) 0 1\nX_ERROR(.4) 0\nX_ERROR(.6) 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]",
        0.2,
    )
    return model.sample_profile(zero, shots_per_weight=1000, seed=58)


def raw_moments(profile, p):
    """Reference calculation, deliberately using every individual sample."""
    ratios = profile._likelihood(p)
    ler = variance = 0.0
    ess = math.inf
    mass = profile.model.weight_distribution(p)
    for w, n in zip(profile.sampled_weights, profile.counts):
        selected = profile.weights == w
        r = ratios[selected]
        values = r * profile.failures[selected]
        z = profile._reference_mass[w]
        ler += z * values.mean()
        variance += z * z * values.var(ddof=1) / n
        if mass[w] > 0:
            ess = min(ess, r.sum() ** 2 / (r @ r) if np.any(r) else 0)
    missing = mass[np.setdiff1d(np.arange(len(mass)), profile.sampled_weights)].sum()
    return ler, math.sqrt(variance), ess, missing


@pytest.mark.parametrize(
    "noise",
    [
        "DEPOLARIZE2(.2) 0 1\nX_ERROR(.4) 0\nX_ERROR(.6) 1",
        "E(.2) X0\nELSE_CORRELATED_ERROR(.4) X0 X1\nELSE_CORRELATED_ERROR(.6) Z1",
        "HERALDED_PAULI_CHANNEL_1(.1,.2,.1,.05) 0\nX_ERROR(.1) 1",
        "DEPOLARIZE1(.2) 0 1\nDEPOLARIZE2(.3) 0 1",
    ],
)
def test_compressed_mean_variance_ess_and_tail_equal_raw_histories(noise):
    model = LinearNoiseModel(
        f"R 0 1\n{noise}\nM(.1) 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]", 0.2
    )
    profile = model.sample_profile(zero, shots_per_weight=700, max_weight=2, seed=81)
    ps = np.r_[0, np.geomspace(1e-7, 0.15, 270), model.max_p]
    curve = profile.curve(ps)
    for index in [0, 1, 7, 200, 256, 271]:
        row = curve[index]
        expected = raw_moments(profile, ps[index])
        np.testing.assert_allclose(
            [
                row.ler,
                row.standard_error,
                row.minimum_ess,
                row.missing_probability_mass,
            ],
            expected,
            rtol=2e-12,
            atol=2e-15,
        )
    assert profile.num_likelihood_records < len(profile.weights)


def test_factored_and_power_polynomials_agree_on_whole_interval(counter_profile):
    profile = counter_profile
    polynomial = profile.to_polynomial()
    assert polynomial.degree == 3
    assert (
        profile.model.max_weight == 4
    )  # Degree counts probability factors, not Pauli weight.
    ps = np.linspace(0, profile.model.max_p, 301)
    direct = np.array([e.ler for e in profile.curve(ps)])
    np.testing.assert_allclose(polynomial(ps), direct, rtol=2e-13, atol=1e-15)
    coefficients = polynomial.power_coefficients()
    assert len(coefficients) == 4 and all(isinstance(c, Decimal) for c in coefficients)
    with localcontext() as context:
        context.prec = 50
        expanded = [
            float(
                sum(
                    c * Decimal(str(p)) ** i
                    for i, c in enumerate(coefficients)
                    if i or c
                )
            )
            for p in ps
        ]
    np.testing.assert_allclose(expanded, direct, rtol=2e-13, atol=1e-15)
    symbol = sp.Symbol("noise")
    expression = polynomial.to_sympy(symbol, expanded=True)
    np.testing.assert_allclose(
        sp.lambdify(symbol, expression, "numpy")(ps), direct, rtol=2e-13, atol=1e-15
    )
    factored = polynomial.to_sympy()
    assert factored.free_symbols == {sp.Symbol("p")}
    assert float(factored.subs("p", 0.01)) == pytest.approx(polynomial(0.01))


def test_polynomial_roundtrip_needs_no_circuit_or_decoder(
    counter_profile, tmp_path, monkeypatch
):
    polynomial = counter_profile.to_polynomial()
    path = tmp_path / "polynomial.npz"
    polynomial.save(path)
    monkeypatch.setattr(
        LinearNoiseModel,
        "__init__",
        lambda *args, **kwargs: pytest.fail("Reconstructed a circuit"),
    )
    loaded = LERPolynomial.load(path)
    ps = np.array([[0, 0.01], [0.1, 0.3]])
    np.testing.assert_allclose(loaded(ps), polynomial(ps), rtol=1e-14)
    assert loaded.degree == polynomial.degree
    assert loaded.metadata == polynomial.metadata
    data = loaded.metadata
    data["meaning"] = "changed"
    assert loaded.metadata != data
    with pytest.raises(ValueError):
        loaded.rates.flags.writeable = True


def test_zero_polynomial_and_unbounded_no_noise_domain(tmp_path):
    model = LinearNoiseModel("R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1)
    polynomial = model.sample_profile(zero, shots_per_weight=2).to_polynomial()
    assert polynomial.num_terms == 0 and polynomial.degree == 0
    assert polynomial(100) == 0
    assert polynomial([]).shape == (0,)
    assert polynomial.to_sympy() == 0
    assert polynomial.power_coefficients() == (Decimal(0),)
    polynomial.save(tmp_path / "zero.npz")
    assert LERPolynomial.load(tmp_path / "zero.npz")(1) == 0


def test_extreme_reference_and_boundary_values():
    model = LinearNoiseModel(
        "R 0\nX_ERROR(1e-310) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 1e-310
    )
    polynomial = model.sample_profile(zero, shots_per_weight=2).to_polynomial()
    np.testing.assert_allclose(
        polynomial([0, 0.01, 0.5, 1]), [0, 0.01, 0.5, 1], rtol=1e-12
    )
    assert float(polynomial.power_coefficients()[1]) == pytest.approx(1, rel=1e-12)


def test_duplicate_terms_combine_and_zero_rates_do_not_inflate_degree():
    polynomial = LERPolynomial(
        0.1, [0, 1], [1, 1], [[99, 1], [99, 1]], np.log([0.02, 0.03]), max_p=1
    )
    assert polynomial.num_terms == 1 and polynomial.degree == 2
    assert polynomial(0.1) == pytest.approx(0.05)
    assert polynomial(0) == 0 and polynomial(1) == 0
    coefficients = polynomial.power_coefficients()
    assert float(coefficients[1]) == pytest.approx(0.05 / (0.1 * 0.9))
    assert float(coefficients[2]) == pytest.approx(-0.05 / (0.1 * 0.9))


@pytest.mark.parametrize("p", [-1, np.nan, np.inf, 0.34])
def test_invalid_polynomial_probabilities(counter_profile, p):
    with pytest.raises(ValueError):
        counter_profile.to_polynomial()(p)
    with pytest.raises(ValueError):
        counter_profile.curve([p])


def test_expansion_guards_and_invalid_shapes(counter_profile, tmp_path):
    polynomial = counter_profile.to_polynomial()
    for kwargs in [
        {"max_degree": 2},
        {"precision": 3},
        {"precision": True},
        {"max_degree": -1},
    ]:
        with pytest.raises(ValueError):
            polynomial.power_coefficients(**kwargs)
    with pytest.raises(TypeError):
        polynomial.to_sympy(3)
    with pytest.raises(ValueError):
        counter_profile.curve([[0.1]])
    assert counter_profile.curve([]) == []
    path = tmp_path / "bad.npz"
    np.savez(path, manifest=np.array(json.dumps({"format": "wrong", "version": 1})))
    with pytest.raises(ValueError, match="format"):
        LERPolynomial.load(path)


@pytest.mark.parametrize(
    "args",
    [
        (0, [1], [1], [[0]], [0], 1),
        (0.1, [-1], [1], [[0]], [0], 1),
        (0.1, [1], [0.5], [[0]], [0], 1),
        (0.1, [1], [1], [[-1]], [0], 1),
        (0.1, [1], [1], [[0]], [np.nan], 1),
        (0.1, [1], [1], [[]], [0], 1),
        (0.1, [1], [1], [[0]], [0], 2),
    ],
)
def test_invalid_polynomial_storage(args):
    p0, rates, active, misses, logs, bound = args
    with pytest.raises(ValueError):
        LERPolynomial(p0, rates, active, misses, logs, max_p=bound)
