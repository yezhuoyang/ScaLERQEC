"""Independent probability laws and non-asymptotic confidence regressions."""

import itertools
import math

import numpy as np
import pytest
import stim
from scipy.stats import binom

from benchmark.general_noise_oracles import (
    FAMILIES,
    circuit_at,
    exact_joint,
)
from scalerqec.Stratified import LERPolynomial, LinearNoiseModel, UniformizedSampler
from scalerqec.Stratified.confidence import bounded_empirical_interval
from scalerqec.Stratified.uniformized import bounded_kl_interval


def zero_decoder(d):
    return np.zeros((len(d), 1), dtype=bool)


def parity_model(n=10):
    c = stim.Circuit("R 0")
    c.append("X_ERROR", [0] * n, 0.2)
    c += stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    return LinearNoiseModel(c, 0.2)


@pytest.mark.parametrize("t", [0, 1, 4, 10])
def test_conditional_uniform_subsets_have_hypergeometric_law(t):
    model = parity_model()
    sampler = UniformizedSampler(model)
    history = np.empty((6000, 10), dtype=np.uint8)
    bits, weights = sampler.sample(
        np.full(len(history), t), np.random.default_rng(2916), outcome_buffer=history
    )
    assert np.all(history.sum(axis=1) == t)
    assert np.all(weights == t)
    assert np.all(bits == t % 2)
    values = history[:, :5].sum(axis=1)
    variance = t * 0.25 * (10 - t) / 9
    assert abs(values.mean() - t / 2) <= 7 * math.sqrt(variance / len(history)) + 1e-12


@pytest.mark.parametrize("p", [0, 0.01, 0.1, 0.5])
def test_full_joint_history_law_thinning_else_and_zero_weight(p):
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.1) 0 1\nE(.1) X0 X1\n"
        "ELSE_CORRELATED_ERROR(.2) Z0\nHERALDED_PAULI_CHANNEL_1(.03,.02,0,.01) 0\n"
        "M(.1) 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
        0.1,
    )
    sampler = UniformizedSampler(model)
    rng = np.random.default_rng(916)
    histories = np.empty((18000, len(model._factors)), dtype=np.int64)
    counts = rng.binomial(sampler.num_trials, sampler.rate * p, len(histories))
    bits, weights = sampler.sample(counts, rng, outcome_buffer=histories)
    unique, seen_counts = np.unique(histories, axis=0, return_counts=True)
    seen = {tuple(h): n for h, n in zip(unique, seen_counts)}
    for h in itertools.product(*(range(len(f.weights)) for f in model._factors)):
        probability = math.prod(
            f.probabilities(p)[a] for f, a in zip(model._factors, h)
        )
        observed = seen.get(h, 0) / len(histories)
        assert abs(probability - observed) <= 7 * math.sqrt(
            probability * (1 - probability) / len(histories)
        ) + 7 / len(histories)
    expected_bits = np.zeros_like(bits)
    expected_weight = np.zeros(len(histories), dtype=int)
    for j, f in enumerate(model._factors):
        expected_bits ^= model._responses[j][histories[:, j]]
        expected_weight += f.weights[histories[:, j]]
    np.testing.assert_array_equal(bits, expected_bits)
    np.testing.assert_array_equal(weights, expected_weight)
    if p:
        assert np.any(weights != counts)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", ["five_qubit", "steane", "repetition_5"])
def test_against_independent_pauli_algebra_across_codes_and_noise(name, family):
    p = 0.08
    model = LinearNoiseModel(circuit_at(name, family, p), p)
    sampler = UniformizedSampler(model)
    rng = np.random.default_rng(173)
    n = 12000
    bits, weights = sampler.sample(
        rng.binomial(sampler.num_trials, sampler.rate * p, n), rng
    )
    states = bits.astype(int) @ (1 << np.arange(bits.shape[1]))
    expected = exact_joint(name, family, p)
    observed = np.zeros_like(expected)
    np.add.at(observed, (weights, states), 1 / n)
    assert np.all(
        abs(observed - expected) <= 7 * np.sqrt(expected * (1 - expected) / n) + 7 / n
    )


def test_independent_original_instruction_replay():
    from benchmark.fault_replay_oracle import ReplayOracle

    c = stim.Circuit(
        "R 0 1\nH 0\nDEPOLARIZE2(.1) 0 1\nCX 0 1\n"
        "E(.1) X0 X1\nELSE_CORRELATED_ERROR(.2) Y0\n"
        "CX 0 1\nH 0\nM(.03) 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    model = LinearNoiseModel(c, 0.1)
    sampler, oracle = UniformizedSampler(model), ReplayOracle(c, 0.1)
    history = np.empty((30, len(model._factors)), dtype=int)
    bits, weights = sampler.sample(
        np.arange(30) % (sampler.num_trials + 1),
        np.random.default_rng(72),
        outcome_buffer=history,
    )
    for h, b, w in zip(history, bits, weights):
        np.testing.assert_array_equal(b, oracle.bits(h))
        assert w == sum(e.weights[a] for e, a in zip(oracle.events, h))


def test_bernstein_identity_has_p_independent_coefficients_and_joint_weights():
    # One DEPOLARIZE2: b_1=8/15 for Z logical on qubit 1, and of these
    # 2/15 have W=1, 6/15 have W=2. b_0=0. C=1, M=1.
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.1) 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    sampler = UniformizedSampler(model)
    bits, weights = sampler.sample(
        np.ones(30000, dtype=int), np.random.default_rng(123)
    )
    assert bits.mean() == pytest.approx(8 / 15, abs=0.015)
    assert np.mean(bits[:, 0] & (weights == 1)) == pytest.approx(2 / 15, abs=0.015)
    assert np.mean(bits[:, 0] & (weights == 2)) == pytest.approx(6 / 15, abs=0.015)


def test_full_polynomial_joint_decomposition_endpoints_and_save(tmp_path):
    model = parity_model(8)
    result = model.sample_bernstein_profile(
        zero_decoder,
        [0, 0.05, 0.25, 1],
        relative_error=0.2,
        absolute_error=0.005,
        confidence=0.99,
        max_shots=100000,
        max_seconds=120,
        batch_size=1024,
        seed=17,
    )
    assert result.converged
    poly = result.to_polynomial()
    assert poly.degree == 8
    grid = np.array([0, 0.03, 0.05, 0.13, 0.25, 0.4, 1])
    truth = (1 - (1 - 2 * grid) ** 8) / 2
    assert np.max(abs(poly(grid) - truth)) < 0.035
    for e in result.estimates:
        exact = (1 - (1 - 2 * e.p) ** 8) / 2
        assert e.lower <= exact <= e.upper
        assert e.error_bound <= 0.005 + 0.2 * e.lower
        assert poly(e.p) == pytest.approx(e.ler, rel=2e-12, abs=1e-15)
    assert sum(n for _, _, n, _ in result.joint_counts) == result.shots
    joint = sum(result.weight_polynomial(w)(grid) for w in range(9))
    np.testing.assert_allclose(joint, poly(grid), rtol=1e-12, atol=1e-15)
    assert not result.weight_polynomial(1).metadata["accuracy_certified"]
    path = tmp_path / "polynomial.npz"
    poly.save(path)
    np.testing.assert_array_equal(LERPolynomial.load(path)(grid), poly(grid))


def test_zero_failures_never_implies_relative_accuracy_and_underflow_not_skipped():
    model = parity_model(2)
    result = model.sample_bernstein_profile(
        zero_decoder,
        [1e-100],
        relative_error=0.01,
        max_shots=16,
        max_seconds=120,
        seed=1,
    )
    e = result.estimates[0]
    assert not result.converged
    assert e.upper > 0 and e.lower == 0
    with pytest.raises(RuntimeError, match="not met"):
        result.to_polynomial()
    assert not result.to_polynomial(allow_unconverged=True).metadata[
        "accuracy_certified"
    ]


@pytest.mark.parametrize("n", [1, 8, 100])
@pytest.mark.parametrize("p", [0.001, 0.1, 0.5, 0.99])
def test_exact_binomial_coverage_of_kl_intervals(n, p):
    delta = 0.05
    outside = 0.0
    for k in range(n + 1):
        lo, hi = bounded_kl_interval(k / n, n, delta)
        if not lo <= p <= hi:
            outside += binom.pmf(k, n, p)
    assert outside <= delta + 1e-14


def test_kl_nonbernoulli_mgf_coverage_by_exact_enumeration():
    # Distribution on {0, .25, 1}; its mean is .425. This is an importance
    # observation, not a Bernoulli count. Enumerate all 3^8 sample sequences.
    n, delta, mu = 8, 0.1, 0.425
    outside = 0
    intervals = {
        k: bounded_kl_interval(k / (4 * n), n, delta) for k in range(4 * n + 1)
    }
    for seq in itertools.product(range(3), repeat=n):
        score = sum([0, 1, 4][a] for a in seq)
        lo, hi = intervals[score]
        if not lo <= mu <= hi:
            outside += math.prod([0.2, 0.5, 0.3][a] for a in seq)
    assert outside <= delta


def test_joint_kl_and_variance_bounds_cover_nonbernoulli_law():
    n, delta, mu = 10, 0.1, 0.055
    # Enumerate all count triples instead of sequences. Low-variance observations
    # permit tighter empirical Bernstein bounds at large n, with delta split.
    missed = 0.0
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            mean = (0.1 * b + c) / n
            var = max(0, (0.01 * b + c - n * mean**2) / (n - 1))
            lo, hi = bounded_kl_interval(mean, n, delta / 2)
            elo, ehi = bounded_empirical_interval(mean, var, n, delta / 2)
            if not max(lo, elo) <= mu <= min(hi, ehi):
                missed += (
                    math.comb(n, a) * math.comb(n - a, b) * 0.9**a * 0.05**b * 0.05**c
                )
    assert missed <= delta
    lo, hi = bounded_empirical_interval(0.005, 1e-5, 100000, 0.005)
    klo, khi = bounded_kl_interval(0.005, 100000, 0.005)
    assert hi - lo < khi - klo


def test_conditional_bernstein_coefficients_for_nonuniform_else_and_depolarize2():
    # Independent algebra: f_D=8p/15, f_E=.5p(1-2p), and independent
    # logical flips combine by f_D+f_E-2*f_D*f_E.
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.1) 0 1\nE(.2) Z0\n"
        "ELSE_CORRELATED_ERROR(.05) X1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
        0.1,
    )
    sampler = UniformizedSampler(model)
    assert sampler.num_trials == 3 and sampler.rate == 2
    power = [0, 8 / 15 + 0.5, -1 - 8 / 15, 16 / 15]
    rng = np.random.default_rng(1916)
    for t in range(4):
        expected = sum(
            power[k] / 2**k * math.comb(t, k) / math.comb(3, k) for k in range(t + 1)
        )
        bits, _ = sampler.sample(np.full(16000, t), rng)
        assert (
            abs(bits.mean() - expected)
            <= 7 * math.sqrt(expected * (1 - expected) / len(bits)) + 1e-12
        )


def test_noise_free_zero_rates_and_no_suffix_table(monkeypatch):
    for circuit in [
        "R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "R 0\nX_ERROR(0) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ]:
        model = LinearNoiseModel(circuit, 0.1)
        monkeypatch.setattr(
            model, "_suffix_table", lambda *a, **kw: pytest.fail("Built a suffix table")
        )
        result = model.sample_bernstein_profile(
            zero_decoder,
            [0.1],
            absolute_error=0.05,
            max_shots=4096,
            max_seconds=120,
            seed=1,
        )
        assert result.converged
        assert result.num_trials == 0
        assert result.to_polynomial()(0.2) == 0
        assert all(t == w == f == 0 for t, w, n, f in result.joint_counts)


@pytest.mark.parametrize("counts", [[-1], [11], [1.5], [[1]]])
def test_bad_latent_counts(counts):
    with pytest.raises(ValueError, match="trial_counts"):
        UniformizedSampler(parity_model()).sample(counts, np.random.default_rng(0))


@pytest.mark.parametrize(
    "options",
    [
        {"confidence": 1},
        {"relative_error": -1},
        {"absolute_error": float("nan")},
        {"max_shots": True},
        {"max_seconds": 0},
        {"batch_size": 0},
        {"relative_error": 0, "absolute_error": 0},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        parity_model().sample_bernstein_profile(zero_decoder, [0.1], **options)


def test_small_initial_decode_batches_have_predeclared_sizes():
    sizes = []

    def decoder(d):
        sizes.append(len(d))
        return zero_decoder(d)

    result = parity_model().sample_bernstein_profile(
        decoder,
        [0.1],
        max_shots=15,
        max_seconds=120,
        relative_error=1e-12,
        seed=8,
    )
    assert sizes == [1, 2, 4, 8]
    assert result.shots == 15 and not result.converged


def test_time_budget_stops_after_one_slow_decoder_call(monkeypatch):
    from scalerqec.Stratified import uniformized

    clock = [0.0]
    monkeypatch.setattr(uniformized, "perf_counter", lambda: clock[0])

    def slow_decoder(d):
        clock[0] += 10
        return zero_decoder(d)

    result = parity_model().sample_bernstein_profile(
        slow_decoder,
        [0.1],
        max_seconds=3,
        seed=9,
    )
    assert result.shots == 1 and not result.converged
    assert result.seconds == 10
    assert "time budget" in result.reason


def test_large_trial_count_without_suffix_table_and_bounded_event_chunks(monkeypatch):
    model = parity_model(3000)
    monkeypatch.setattr(
        model, "_suffix_table", lambda *a, **kw: pytest.fail("Built a suffix table")
    )
    sampler = UniformizedSampler(model)
    bits, weights = sampler.sample(np.full(300, 1500), np.random.default_rng(41))
    assert not bits.any()
    assert np.all(weights == 1500)
    assert (
        sum(a.nbytes for a in vars(sampler).values() if isinstance(a, np.ndarray))
        < 512 * 3000
    )


def test_dense_fault_signatures_cancel_across_sparse_gather_boundaries():
    c = stim.Circuit("R 0\nX_ERROR(.1) 0 0\nM 0")
    for _ in range(1024):
        c.append("DETECTOR", [stim.target_rec(-1)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    sampler = UniformizedSampler(LinearNoiseModel(c, 0.1))
    bits, weights = sampler.sample(np.full(1024, 2), np.random.default_rng(12))
    assert not bits.any() and np.all(weights == 2)


def test_categorical_rounding_cannot_silently_remove_positive_support():
    model = LinearNoiseModel(
        "R 0\nPAULI_CHANNEL_1(.1,1e-21,0) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    with pytest.raises(FloatingPointError, match="sampling support"):
        UniformizedSampler(model)


@pytest.mark.parametrize(
    "buffer",
    [
        np.zeros((2, 10), dtype=float),
        np.zeros((2, 9), dtype=int),
        np.zeros((2, 10), dtype=bool),
    ],
)
def test_invalid_outcome_buffers(buffer):
    with pytest.raises(ValueError, match="outcome_buffer"):
        UniformizedSampler(parity_model()).sample(
            np.array([1, 2]), np.random.default_rng(7), outcome_buffer=buffer
        )
