"""Independent laws for the sparse, log-space conditional sampler."""

import itertools
import math

import numpy as np
import pytest
import stim
from scipy.special import gammaln

from scalerqec.Stratified import LinearNoiseModel


def parity_model(n, p):
    c = stim.Circuit("R 0")
    c.append("X_ERROR", [0] * n, p)
    c += stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    return LinearNoiseModel(c, p)


@pytest.mark.parametrize("n,w", [(1070, 1), (3000, 3), (3000, 0)])
def test_underflowed_normalizers_match_binomial_and_uniform_subsets(n, w):
    model = parity_model(n, 0.5)
    table = model._suffix_table(w)
    truth = gammaln(n + 1) - gammaln(w + 1) - gammaln(n - w + 1) - n * math.log(2)
    assert table.logs[0, w] == pytest.approx(truth, abs=2e-10)
    histories = np.empty((4096, n), dtype=np.uint8)
    bits, active, misses = model._sample_stratum(
        w,
        len(histories),
        np.random.default_rng(19),
        table,
        None,
        outcome_buffer=histories,
    )
    assert np.all(histories.sum(axis=1) == w)
    assert np.all(bits == (w % 2))
    assert np.all(active == w)
    assert np.all(misses == n - w)
    # Exchangeability gives the exact hypergeometric law in the first half.
    first = histories[:, : n // 2].sum(axis=1)
    if w:
        variance = w * 0.25 * (n - w) / (n - 1)
        assert abs(first.mean() - w / 2) < 7 * math.sqrt(variance / len(first))


@pytest.mark.parametrize("p", [0, 0.03, 0.2, 0.5])
def test_full_joint_conditional_law_including_zero_weight_and_else(p):
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.1) 0 1\nE(.1) X0 X1\n"
        "ELSE_CORRELATED_ERROR(.2) Z0\nHERALDED_PAULI_CHANNEL_1(.03,.02,0,.01) 0\n"
        "M(.1) 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
        0.1,
    )
    options = list(itertools.product(*(range(len(f.weights)) for f in model._factors)))
    probabilities = np.array(
        [
            math.prod(f.probabilities(p)[a] for f, a in zip(model._factors, h))
            for h in options
        ]
    )
    weights = np.array(
        [sum(f.weights[a] for f, a in zip(model._factors, h)) for h in options]
    )
    table = model._suffix_table(model.max_weight, p)
    rng = np.random.default_rng(200)
    for w in np.unique(weights[probabilities > 0]):
        truth = probabilities[weights == w]
        truth /= truth.sum()
        histories = np.empty((12000, len(model._factors)), dtype=np.int64)
        bits, active, misses = model._sample_stratum(
            w, len(histories), rng, table, None, outcome_buffer=histories
        )
        unique, counts = np.unique(histories, axis=0, return_counts=True)
        seen = {tuple(h): c for h, c in zip(unique, counts)}
        observed = np.array(
            [seen.get(h, 0) for h, ww in zip(options, weights) if ww == w]
        ) / len(histories)
        assert np.all(
            abs(observed - truth)
            <= 7 * np.sqrt(truth * (1 - truth) / len(histories)) + 7 / len(histories)
        )
        # Check every recorded K and miss count, not just output syndrome averages.
        assert np.array_equal(active, np.count_nonzero(histories, axis=1))
        expected = np.zeros_like(misses)
        replay = np.zeros_like(bits)
        for j, f in enumerate(model._factors):
            replay ^= model._responses[j][histories[:, j]]
            for r, rate in enumerate(f.rates):
                expected[:, np.searchsorted(model.rates, rate)] += (
                    ((histories[:, j] == 0) | (histories[:, j] > r + 1))
                    if f.chain
                    else histories[:, j] == 0
                )
        np.testing.assert_array_equal(misses, expected)
        np.testing.assert_array_equal(bits, replay)


def test_fixed_profile_preserves_underflowed_polynomial_and_reload(tmp_path):
    model = parity_model(1070, 0.5)
    profile = model.sample_profile(
        lambda d: np.zeros((len(d), 1), dtype=bool),
        shots_per_weight=4,
        max_weight=1,
        seed=3,
    )
    p = 0.001
    truth = 1070 * p * (1 - p) ** 1069
    assert profile.to_polynomial()(p) == pytest.approx(truth, rel=2e-10)
    assert profile.evaluate(p).ler == pytest.approx(truth, rel=2e-10)
    path = tmp_path / "tiny.npz"
    profile.save(path)
    assert type(profile).load(path).to_polynomial()(p) == pytest.approx(
        truth, rel=2e-10
    )
    interval = profile.confidence_bounds(p)
    assert interval.lower <= truth <= interval.upper


def test_impossible_weight_is_rejected():
    model = parity_model(2, 0.5)
    with pytest.raises(ValueError, match="impossible"):
        model._sample_stratum(
            1, 2, np.random.default_rng(0), model._suffix_table(1, 0), None
        )


@pytest.mark.parametrize("weight", [0, 1])
def test_zero_uniform_endpoint_skips_impossible_fault_locations(weight):
    class ZeroRng:
        def random(self, size):
            return np.zeros(size)

    model = LinearNoiseModel(
        "R 0\nX_ERROR(0) 0\nX_ERROR(.2) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.2
    )
    histories = np.empty((3, 2), dtype=int)
    bits, active, _ = model._sample_stratum(
        weight, 3, ZeroRng(), model._suffix_table(1), None, outcome_buffer=histories
    )
    assert not histories[:, 0].any()
    assert np.all(histories[:, 1] == weight)
    assert np.all(bits == weight)
    assert np.all(active == weight)


def test_independent_replay_audit_does_not_skip_underflowed_weights():
    from benchmark.fault_replay_oracle import ReplayOracle, audit_sampled_histories

    model = parity_model(1200, 0.5)
    assert model._suffix_table(2)[0, 2] == 0
    audit = audit_sampled_histories(model, ReplayOracle(model.circuit, 0.5))
    assert audit["histories_per_weight"] == {2: 4, 600: 4}


def test_separated_p_targets_both_get_sampling_work():
    model = parity_model(100, 0.5)
    result = model.sample_until_accuracy(
        lambda d: np.zeros((len(d), 1), dtype=bool),
        [0.01, 0.49],
        max_shots=512,
        exact_budget=0,
        max_seconds=None,
        seed=17,
    )
    assert min(result.sample_counts) <= 2
    assert max(result.sample_counts) >= 40
    assert all(e.ler > 0 for e in result.estimates)
    assert not result.converged


def test_pilot_covers_small_strata_before_refining_only_the_peak():
    model = parity_model(100, 0.5)
    result = model.sample_until_accuracy(
        lambda d: np.zeros((len(d), 1), dtype=bool),
        [0.5],
        max_shots=8192,
        exact_budget=0,
        max_seconds=None,
        relative_error=0.01,
        seed=19,
    )
    mass = model.weight_distribution(0.5)
    omitted = 1 - sum(mass[w] for w in result.sample_counts)
    assert omitted < 0.005
    assert not result.converged


@pytest.mark.parametrize("method", ["forced", "auto"])
def test_underflowed_else_outcome_keeps_its_response_and_polynomial(method):
    c = stim.Circuit("R 0 1")
    c.append("E", [stim.target_z(0)], 0.99999)
    for _ in range(68):
        c.append("ELSE_CORRELATED_ERROR", [stim.target_z(0)], 0.99999)
    c.append("ELSE_CORRELATED_ERROR", [stim.target_x(0), stim.target_x(1)], 0.99999)
    c += stim.Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]")
    model = LinearNoiseModel(c, 0.99999, compile_responses=False)
    assert model._factors[0].probabilities(model.reference_p)[-1] == 0
    model.compile_responses(method=method)
    assert model._responses[0][-1, 0]
    profile = model.sample_profile(
        lambda d: np.zeros((len(d), 1), dtype=bool), shots_per_weight=2, seed=23
    )
    assert profile.to_polynomial()(0.1) == pytest.approx(0.1 * 0.9**69, rel=2e-11)
