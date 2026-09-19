"""Independent accuracy oracles and sequential-stopping contract regressions."""

import itertools
import math

import numpy as np
import pytest
import stim
from scipy.special import logsumexp

from benchmark.general_noise_oracles import (
    FAMILIES,
    LookupDecoder,
    circuit_at,
    exact_joint,
)
from scalerqec.Stratified import (
    AccuracyControlledProfile,
    LERPolynomial,
    LinearNoiseModel,
    adaptive,
)


def zero_decoder(detectors):
    return np.zeros((len(detectors), 1), dtype=bool)


def readout_model(p=0.2):
    return LinearNoiseModel(f"R 0\nM({p}) 0\nOBSERVABLE_INCLUDE(0) rec[-1]", p)


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_enumeration_against_independent_pauli_oracle(family):
    name, p0 = "repetition_3", 0.05
    decoder = LookupDecoder(exact_joint(name, family, p0))
    model = LinearNoiseModel(circuit_at(name, family, p0), p0, compile_responses=False)
    ps = [0, 0.001, 0.05, 0.2, model.max_p]
    result = model.sample_until_accuracy(
        decoder,
        ps,
        relative_error=1e-10,
        max_shots=0,
        exact_budget=100_000,
        max_seconds=None,
        seed=123,
    )
    assert isinstance(result, AccuracyControlledProfile)
    assert result.converged
    assert result.shots == 0
    assert result.exact_histories > 0
    polynomial = result.to_polynomial()
    for p in ps + [0.007, 0.13]:
        truth = decoder.failure_mass(exact_joint(name, family, p)).sum()
        assert polynomial(p) == pytest.approx(truth, rel=1e-10, abs=2e-14)
    assert polynomial.metadata["accuracy_control"]["certified_p_values"] == sorted(
        set(ps)
    )


def test_depolarize2_previous_bias_has_correct_polynomial(tmp_path):
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.2) 0 1\nX_ERROR(.4) 0\nX_ERROR(.6) 1\n"
        "M 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]",
        0.2,
    )
    result = model.sample_until_accuracy(zero_decoder, [0.001, 0.1, 1 / 3])
    assert result.converged
    poly = result.to_polynomial()
    assert poly(0.001) == pytest.approx(0.0025312, rel=1e-12)
    power = list(map(float, poly.power_coefficients()))
    np.testing.assert_allclose(power[:3], [0, 38 / 15, -32 / 15], atol=3e-14)
    np.testing.assert_allclose(power[3:], 0, atol=3e-14)
    path = tmp_path / "adaptive_polynomial.json"
    poly.save(path)
    loaded = LERPolynomial.load(path)
    assert loaded.metadata == poly.metadata
    assert loaded(0.001) == poly(0.001)


@pytest.mark.parametrize("family", ["nonuniform_single", "heralded_pauli"])
def test_previous_high_ess_rare_failure_cases_now_meet_actual_accuracy(family):
    name, p0 = "repetition_7", 0.05
    decoder = LookupDecoder(exact_joint(name, family, p0))
    model = LinearNoiseModel(circuit_at(name, family, p0), p0)
    result = model.sample_until_accuracy(
        decoder,
        [0.001, 0.01, 0.05],
        seed=130927,
        max_seconds=None,
    )
    assert result.converged
    for estimate in result.estimates:
        truth = decoder.failure_mass(exact_joint(name, family, estimate.p)).sum()
        assert estimate.lower <= truth <= estimate.upper
        assert abs(estimate.ler - truth) <= 0.1 * truth
    assert result.exact_histories <= 100_000
    assert 3 in result.exact_weights  # rare failure histories are actually checked


@pytest.mark.parametrize("family", FAMILIES)
def test_mixture_density_and_bounds_against_all_histories(family):
    model = LinearNoiseModel(circuit_at("repetition_3", family, 0.05), 0.05)
    decoder = LookupDecoder(exact_joint("repetition_3", family, 0.05))
    anchors = np.array([0, 0.01, 0.05, model.max_p])
    ps = [0, 0.003, 0.02, model.max_p]
    table = adaptive._support_table(model, model.max_weight, 100_000)
    anchor_tables = [model._suffix_table(model.max_weight, p) for p in anchors]
    exact = np.zeros(len(ps))
    for w in range(model.max_weight + 1):
        count = int(table[0, w])
        if not count:
            continue
        indices = [i for i, t in enumerate(anchor_tables) if t[0, w] > 0]
        log_norms = [
            -math.log(len(indices)) - math.log(anchor_tables[i][0, w]) for i in indices
        ]
        total_q = 0
        for fail, active, misses, log_p0 in adaptive._enumerate_stratum(
            model,
            decoder,
            w,
            count,
            table,
            batch_size=7,
        ):
            log_q_ratio = logsumexp(
                adaptive._log_ratios(active, misses, anchors[indices], model)
                + log_norms,
                axis=1,
            )
            q = np.exp(log_p0 + log_q_ratio)
            total_q += q.sum()
            target_ratios = adaptive._log_ratios(active, misses, ps, model)
            importance = np.exp(target_ratios - log_q_ratio[:, None])
            exact += ((q * fail)[:, None] * importance).sum(axis=0)
            for j, p in enumerate(ps):
                maximum = adaptive._maximum_log_likelihood(model, p, w, 0.05)[w]
                bound = anchor_tables[2][0, w] * len(indices) * np.exp(maximum)
                assert np.all(importance[:, j] <= bound * (1 + 1e-12))
        assert total_q == pytest.approx(1, abs=3e-13)
    np.testing.assert_allclose(
        exact,
        [
            decoder.failure_mass(exact_joint("repetition_3", family, p)).sum()
            for p in ps
        ],
        atol=2e-13,
        rtol=2e-12,
    )


def test_unranking_visits_each_supported_history_with_correct_multiplicity():
    # Independent direct Cartesian enumeration of a chain and a zero-weight herald.
    model = LinearNoiseModel(
        "R 0 1\nE(.03) X0 X1\nELSE_CORRELATED_ERROR(.04) Z0\n"
        "HERALDED_PAULI_CHANNEL_1(.01,.02,.03,.04) 0\nM 0 1\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]",
        0.1,
    )
    brute = {}
    for outcomes in itertools.product(*(range(len(f.weights)) for f in model._factors)):
        w = sum(f.weights[a] for f, a in zip(model._factors, outcomes))
        p = math.prod(f.probabilities(0.1)[a] for f, a in zip(model._factors, outcomes))
        if p:
            brute.setdefault(w, []).append(p)
    table = adaptive._support_table(model, model.max_weight, 100)
    for w, probs in brute.items():
        assert table[0, w] == len(probs)
        enumerated = []
        for _, _, _, log_p0 in adaptive._enumerate_stratum(
            model, zero_decoder, w, len(probs), table, batch_size=2
        ):
            enumerated.extend(np.exp(log_p0))
        np.testing.assert_allclose(sorted(enumerated), sorted(probs), rtol=3e-14)


def test_unseen_failures_and_exhausted_budget_do_not_certify_zero():
    result = readout_model(1e-8).sample_until_accuracy(
        zero_decoder,
        [1e-8],
        exact_budget=0,
        max_shots=256,
        seed=3,
        max_seconds=None,
    )
    assert not result.converged
    (estimate,) = result.estimates
    assert estimate.ler == 0
    assert estimate.lower == 0
    assert estimate.upper > 1e-8
    assert not estimate.accuracy_met
    with pytest.raises(RuntimeError, match="Accuracy target was not met"):
        result.to_polynomial()
    poly = result.to_polynomial(allow_unconverged=True)
    assert poly.metadata["accuracy_control"]["certified_p_values"] == []


def test_sequential_sampling_converges_and_covers_independent_truth_across_seeds():
    model = readout_model()
    ps = np.linspace(
        0.02, 0.3, 11
    )  # exercises bounded anchor selection and non-anchor bounds
    shots = set()
    for seed in range(12):
        result = model.sample_until_accuracy(
            zero_decoder,
            ps,
            relative_error=0.15,
            absolute_error=0.005,
            confidence=0.99,
            exact_budget=0,
            max_shots=200_000,
            max_seconds=None,
            seed=seed,
        )
        assert result.converged
        assert result.exact_histories == 0
        assert len(result.proposal_probabilities) < len(ps)
        shots.add(result.shots)
        for e in result.estimates:
            assert e.lower <= e.p <= e.upper
            assert abs(e.ler - e.p) <= 0.005 + 0.15 * e.p
            assert e.error_bound <= 0.005 + 0.15 * e.lower
        assert result.shots >= 256 and result.shots & (result.shots - 1) == 0
    assert shots  # exact shot counts can coincide at coarse doubling checkpoints


def test_accuracy_request_controls_work_and_seed_reproducibility():
    model = readout_model()

    def run(tol):
        return model.sample_until_accuracy(
            zero_decoder,
            [0.2],
            relative_error=tol,
            exact_budget=0,
            seed=4,
            max_seconds=None,
        )

    loose, strict = run(0.3), run(0.05)
    assert loose.converged and strict.converged
    assert strict.shots > loose.shots
    again = run(0.3)
    assert again.estimates == loose.estimates
    assert again.sample_counts == loose.sample_counts


def test_zero_weight_must_not_be_assumed_failure_free():
    model = LinearNoiseModel("R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1)
    result = model.sample_until_accuracy(lambda d: ~zero_decoder(d), [0, 0.1])
    assert result.converged and result.exact_histories == 1
    assert all(e.ler == 1 for e in result.estimates)


def test_omitted_tail_is_expanded_when_it_dominates_uncertainty():
    circuit = stim.Circuit("R 0")
    for _ in range(20):
        circuit.append("X_ERROR", [0], 0.9)
    circuit += stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    model = LinearNoiseModel(circuit, 0.9)
    result = model.sample_until_accuracy(
        zero_decoder,
        [0.9],
        relative_error=0.2,
        exact_budget=0,
        max_seconds=None,
        seed=4,
    )
    assert result.converged
    assert max(result.sample_counts) > 16
    (e,) = result.estimates
    truth = (1 - (1 - 2 * 0.9) ** 20) / 2
    assert e.lower <= truth <= e.upper
    assert abs(e.ler - truth) <= 0.2 * truth


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"relative_error": -1}, "tolerances"),
        ({"relative_error": 0}, "tolerances"),
        ({"absolute_error": np.nan}, "tolerances"),
        ({"relative_error": np.inf}, "tolerances"),
        ({"confidence": 1}, "confidence"),
        ({"confidence": np.nan}, "confidence"),
        ({"max_shots": True}, "max_shots"),
        ({"max_shots": -1}, "max_shots"),
        ({"exact_budget": 1.5}, "exact_budget"),
        ({"exact_budget": 10**10}, "counter range"),
        ({"max_seconds": 0}, "max_seconds"),
        ({"max_seconds": np.inf}, "max_seconds"),
    ],
)
def test_invalid_controls(kwargs, match):
    with pytest.raises(ValueError, match=match):
        readout_model().sample_until_accuracy(zero_decoder, [0.1], **kwargs)


@pytest.mark.parametrize("ps", [[], [[0.1]], [0.1] * 1025, [-0.1], [np.nan], [1.1]])
def test_invalid_grid(ps):
    with pytest.raises(ValueError):
        readout_model().sample_until_accuracy(zero_decoder, ps)


def test_time_and_memory_limits_return_unresolved(monkeypatch):
    model = readout_model()
    timed = model.sample_until_accuracy(zero_decoder, [0.1], max_seconds=1e-12)
    assert not timed.converged and "Time budget" in timed.reason
    monkeypatch.setattr(adaptive, "_MAX_TABLE_CELLS", 1)
    memory = model.sample_until_accuracy(zero_decoder, [0.1])
    assert not memory.converged and "memory limit" in memory.reason
    assert memory.estimates[0].lower == 0 and memory.estimates[0].upper == 1


def test_endpoint_sampling_and_tiny_reference_are_finite():
    model = readout_model(1e-310)
    result = model.sample_until_accuracy(
        zero_decoder,
        [0, 0.5, 1],
        relative_error=0.2,
        absolute_error=0.02,
        exact_budget=0,
        seed=4,
        max_seconds=None,
    )
    assert result.converged
    for e in result.estimates:
        assert e.lower <= e.p <= e.upper


def test_sequential_error_spending_is_summable():
    # Union budget over all weights AND looks, including never-visited states.
    spending = sum(1 / ((j + 1) * (j + 2)) for j in range(100_000))
    assert spending == pytest.approx(1 - 1 / 100_001, abs=1e-13)
    assert spending**2 < 1


@pytest.mark.parametrize("at_reference", [True, False])
def test_chain_support_underflow_is_rejected_before_sampling(at_reference):
    probability = 0.99999 if at_reference else 0.01
    circuit = stim.Circuit("R 0")
    circuit.append("E", [stim.target_x(0)], probability)
    for _ in range(69):
        circuit.append("ELSE_CORRELATED_ERROR", [stim.target_x(0)], probability)
    circuit += stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    model = LinearNoiseModel(circuit, probability, compile_responses=False)
    with pytest.raises(FloatingPointError, match="underflowed"):
        model.sample_until_accuracy(zero_decoder, [0.99999])


def test_tail_extension_memory_limit_preserves_unresolved_tail(monkeypatch):
    circuit = stim.Circuit("R 0")
    for _ in range(20):
        circuit.append("X_ERROR", [0], 0.9)
    circuit += stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    model = LinearNoiseModel(circuit, 0.9)
    # Initial 21*17*2 table cells fit, but extension to all weights does not.
    monkeypatch.setattr(adaptive, "_MAX_TABLE_CELLS", 750)
    result = model.sample_until_accuracy(zero_decoder, [0.9], seed=4)
    assert not result.converged
    assert "Additional weight strata" in result.reason
    assert result.estimates[0].upper == pytest.approx(1)


def test_impossible_floating_point_precision_is_not_success():
    model = LinearNoiseModel(
        "R 0 1\nDEPOLARIZE2(.2) 0 1\nX_ERROR(.4) 0\nX_ERROR(.6) 1\n"
        "M 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]",
        0.2,
    )
    result = model.sample_until_accuracy(zero_decoder, [0.1], relative_error=1e-30)
    assert not result.converged
    assert "Floating-point resolution" in result.reason


def test_joint_probability_underflow_cannot_certify_zero_relative_error():
    model = LinearNoiseModel(circuit_at("repetition_3", "uniform_single", 0.05), 0.05)
    decoder = LookupDecoder(exact_joint("repetition_3", "uniform_single", 0.05))
    # Exact LER is 4*p^2/3 - 16*p^3/27: positive, but below float64 range.
    result = model.sample_until_accuracy(decoder, [1e-200])
    assert not result.converged
    assert result.status == "numerical_limit"
    assert result.estimates[0].upper == 1
    assert not result.estimates[0].accuracy_met
    with pytest.raises(RuntimeError):
        result.to_polynomial()


def test_exhaustive_zero_proof_still_allows_zero_relative_error():
    # Phase noise cannot change a Z measurement; checking all histories proves it.
    model = LinearNoiseModel(
        "R 0\nZ_ERROR(.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    result = model.sample_until_accuracy(zero_decoder, [0.001, 0.1])
    assert result.converged
    assert result.exact_histories == 2
    assert all(e.ler == e.lower == e.upper == 0 for e in result.estimates)


def test_enumerated_positive_polynomial_underflow_is_unresolved():
    model = LinearNoiseModel(
        "R 0 1 2\nM(.1) 0 1\nM 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]",
        0.1,
    )
    # This fixed nonlinear decoder fails exactly when both record bits flip.
    result = model.sample_until_accuracy(lambda d: d.all(axis=1)[:, None], [1e-200])
    assert result.exact_histories == 4
    assert result.status == "numerical_limit"
    poly = result.to_polynomial(allow_unconverged=True)
    assert poly(0.01) == pytest.approx(0.0001)
    assert result.estimates[0].upper == 1
