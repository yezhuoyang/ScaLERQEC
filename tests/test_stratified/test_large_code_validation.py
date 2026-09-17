"""Independent replay, CSS algebra, and safeguards for the larger benchmark."""

import numpy as np
import pytest
import stim

from benchmark.conditional_moment_audit import conditional_moments, labels
from benchmark.fault_replay_oracle import (
    ReplayOracle,
    audit_sampled_histories,
    hgp_all_columns,
)
from benchmark.hgp_bounded_oracle import SingleFaultRepair, bounded_oracle
from benchmark.large_code_cases import (
    P0,
    ROOT,
    cases,
    gf2_rref,
    hgp_matrices,
    make_circuit,
)
from benchmark.native_large_fault_audit import single_observable
from scalerqec.Stratified import LinearNoiseModel


def test_declared_matrix_contains_full_round_distance_13_and_two_qldpc_families():
    matrix = cases()
    assert len(matrix) == len({s["name"] for s in matrix}) == 56
    assert {s["distance"] for s in matrix if s["family"] == "surface"} == {
        3,
        5,
        7,
        9,
        11,
        13,
    }
    assert all(s["rounds"] == 13 for s in matrix if s.get("distance") == 13)
    assert {s["n"] for s in matrix if s["family"] == "bicycle"} == {
        72,
        90,
        108,
        144,
        288,
    }
    assert {s["n"] for s in matrix if s["family"] == "hgp"} == {58, 180, 245}
    assert all(
        s["data_only"] == (s["n"] == 288) for s in matrix if s["family"] == "bicycle"
    )


@pytest.mark.parametrize("n", [72, 90, 108, 144, 288])
def test_bicycle_fixture_scope_is_one_defined_logical_not_vector_width(n):
    spec = next(s for s in cases() if s["family"] == "bicycle" and s["n"] == n)
    filename = "bbcode_{n}_{k}_{distance}_rounds{rounds}".format(**spec)
    circuit = stim.Circuit.from_file(ROOT / "stimprograms" / "ldpc" / filename)
    ids = {
        int(op.gate_args_copy()[0])
        for op in circuit.flattened()
        if op.name == "OBSERVABLE_INCLUDE"
    }
    assert ids == set(spec["defined_logical_ids"]) == {spec["k"] - 1}
    assert circuit.num_observables == spec["k"]
    assert spec["memory_basis"] == "x"
    with pytest.raises(ValueError, match="does not define"):
        single_observable(circuit, 0)


@pytest.mark.parametrize("n,k", [(58, 16), (180, 36), (245, 49)])
def test_hypergraph_product_commutation_ranks_and_logicals(n, k):
    h, hx, hz, lz = hgp_matrices(n)
    assert not (hx @ hz.T % 2).any()
    assert not (hx @ lz.T % 2).any()
    assert n - len(gf2_rref(hx)[1]) - len(gf2_rref(hz)[1]) == k
    assert len(lz) == k
    assert len(gf2_rref(np.vstack([hz, lz]))[1]) == len(gf2_rref(hz)[1]) + k
    assert max(hx.sum(axis=1)) <= 9 and max(hx.sum(axis=0)) <= 9
    assert h.shape[0] < h.shape[1]


@pytest.mark.parametrize(
    "noise",
    ["uniform_single", "nonuniform_depolarizing", "biased_pauli", "correlated_else"],
)
def test_hgp_all_outcomes_and_actual_weighted_draws_match_independent_oracles(noise):
    circuit = make_circuit({"family": "hgp", "n": 58, "noise": noise}, P0)
    model = LinearNoiseModel(circuit, P0)
    replay = ReplayOracle(circuit, P0)
    _, hx, hz, lz = hgp_matrices(58)
    assert hgp_all_columns(model, replay, hx, hz, lz) > 58
    assert audit_sampled_histories(model, replay)["histories_replayed"] == 8


@pytest.mark.parametrize(
    "noise",
    [
        "uniform_single",
        "nonuniform_depolarizing",
        "biased_pauli",
        "correlated_else",
        "phenomenological",
    ],
)
def test_original_instruction_replay_includes_temporal_gate_and_record_noise(noise):
    circuit = make_circuit(
        {"family": "surface", "distance": 3, "rounds": 3, "basis": "x", "noise": noise},
        P0,
    )
    model = LinearNoiseModel(circuit, P0)
    audit = audit_sampled_histories(model, ReplayOracle(circuit, P0))
    assert audit["status"] == "passed"
    assert audit["histories_replayed"] == 4 * len(audit["weights"])


def test_replay_rejects_noise_it_cannot_independently_interpret():
    import stim

    with pytest.raises(NotImplementedError, match="HERALDED"):
        ReplayOracle(stim.Circuit("HERALDED_ERASE(.1) 0"), 0.1)


def test_outcome_audit_buffer_validation_and_replayed_sample():
    circuit = make_circuit({"family": "hgp", "n": 58, "noise": "uniform_single"}, P0)
    model = LinearNoiseModel(circuit, P0)
    table = model._suffix_table(2)
    plans = model._sampling_plan()
    rng = np.random.default_rng(3)
    for buffer in [np.empty((3, 58)), np.empty((2, 58)), np.empty((3, 57), dtype=int)]:
        with pytest.raises(ValueError, match="outcome_buffer"):
            model._sample_stratum(2, 3, rng, table, plans, outcome_buffer=buffer)


def test_bounded_oracle_normalization_with_a_decoder_that_always_fails():
    spec = {"family": "hgp", "n": 58, "noise": "uniform_single"}

    class Decoder:
        def decode_batch(self, det):
            # A constant prediction of all logical ones always fails for W<=2
            # here; all zero/one/two faults cannot flip this full logical vector.
            return np.ones((len(det), 16), dtype=bool)

    result = bounded_oracle(spec, Decoder(), [0.0001, 0.01])
    assert result["histories"] == 1 + 58 * 3 + (58 * 57 // 2) * 9
    for p in result["points"]:
        assert p["upper"] == pytest.approx(1, abs=2e-13)
        assert p["lower"] + p["omitted_mass"] == pytest.approx(1, abs=2e-13)


def test_conditional_moment_oracle_against_exhaustive_joint_distribution():
    import itertools

    import stim

    replay = ReplayOracle(
        stim.Circuit(
            "R 0 1\nDEPOLARIZE2(.1) 0 1\nX_ERROR(.03) 0\nM(.02) 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]"
        ),
        0.1,
    )
    mass = np.zeros(4)
    first = np.zeros((4, 4))
    second = first.copy()
    for outcomes in itertools.product(*(range(len(e.weights)) for e in replay.events)):
        w = sum(e.weights[a] for e, a in zip(replay.events, outcomes))
        probability = np.prod(
            [e.probabilities(0.03, 0.1)[a] for e, a in zip(replay.events, outcomes)]
        )
        values = sum(
            labels(e, j, len(replay.events))[:, a]
            for j, (e, a) in enumerate(zip(replay.events, outcomes))
        )
        mass[w] += probability
        first[w] += probability * values
        second[w] += probability * values**2
    for w in range(4):
        means, variances = conditional_moments(replay.events, 0.03, 0.1, w)
        np.testing.assert_allclose(means, first[w] / mass[w], atol=1e-13)
        np.testing.assert_allclose(
            variances, second[w] / mass[w] - means**2, atol=1e-13
        )


def test_repaired_qldpc_decoder_corrects_every_single_pauli_and_is_order_independent():
    pytest.importorskip("stimbposd")
    spec = {"family": "hgp", "n": 58, "noise": "uniform_single"}
    model = LinearNoiseModel(make_circuit(spec, P0), P0)
    bits = np.concatenate([r[1:] for r in model._responses])
    decoder = SingleFaultRepair(spec)
    det = bits[:, : model.num_detectors]
    obs = bits[:, model.num_detectors :]
    np.testing.assert_array_equal(decoder.decode_batch(det), obs)
    np.testing.assert_array_equal(decoder.decode_batch(det[::-1])[::-1], obs)


def test_subnormal_conditional_mass_is_rejected_before_returning_biased_draws():
    # Positive is not enough: 1070 Bernoulli sites at p=.5 have a subnormal
    # W=1 mass. Rounded DP normalizers cannot support a reliable conditional law.
    c = stim.Circuit("R 0")
    c.append("X_ERROR", [0] * 1070, 0.5)
    c += stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    model = LinearNoiseModel(c, 0.5)
    table = model._suffix_table(1)
    assert 0 < table[0, 1] < np.finfo(float).tiny
    with pytest.raises(FloatingPointError, match="subnormal"):
        model._sample_stratum(
            1, 4, np.random.default_rng(11), table, model._sampling_plan()
        )
