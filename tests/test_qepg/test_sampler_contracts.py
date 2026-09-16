"""Native input safety and exact sampling distribution regression checks."""

import numpy as np
import pytest

from scalerqec import qepg


@pytest.fixture
def graph():
    # Each fault location has its own output bit; locations cannot cancel.
    return qepg.compile_QEPG(
        "R 0\nR 1\nR 2\nM 0\nM 1\nM 2\n"
        "DETECTOR rec[-3]\nDETECTOR rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]"
    )


def nonuniform(graph, probs, shots=20, a=None, b=None, cp=None):
    return qepg.return_samples_nonuniform_to_numpy(
        graph,
        np.asarray(probs, dtype=float),
        np.asarray([] if a is None else a, dtype=np.uint64),
        np.asarray([] if b is None else b, dtype=np.uint64),
        np.asarray([] if cp is None else cp, dtype=float),
        shots,
    )


def test_fixed_weight_rejects_out_of_range(graph):
    with pytest.raises(ValueError, match="weight"):
        qepg.return_samples_with_fixed_QEPG_numpy(graph, 4, 1)


@pytest.mark.parametrize("weights,shots", [([1], []), ([], [1]), ([4], [1])])
def test_multiweight_rejects_invalid_batches(graph, weights, shots):
    with pytest.raises(ValueError):
        qepg.return_samples_many_weights_separate_obs_with_QEPG(graph, weights, shots)


def test_empty_batches_keep_shapes(graph):
    det, obs = qepg.return_samples_many_weights_separate_obs_with_QEPG(graph, [], [])
    assert det.shape == (0, 2)
    assert obs.shape == (0,)
    det, obs = qepg.return_samples_with_fixed_QEPG_numpy(graph, 0, 0)
    assert det.shape == (0, 2)


@pytest.mark.parametrize("p", [float("nan"), float("inf"), -0.01, 1.01])
def test_uniform_rejects_invalid_p(graph, p):
    with pytest.raises(ValueError):
        qepg.return_samples_Monte_separate_obs_with_QEPG(graph, p, 1)


@pytest.mark.parametrize("p", [0, 0.2, 0.8, 1])
def test_uniform_is_binomial_not_poisson(graph, p):
    shots = 100_000
    det, obs = qepg.return_samples_Monte_separate_obs_with_QEPG(graph, p, shots)
    bits = np.column_stack((det, obs))
    expected = 2 * p / 3
    tolerance = 7 * np.sqrt(expected * (1 - expected) / shots) + 1 / shots
    np.testing.assert_allclose(bits.mean(axis=0), expected, atol=tolerance, rtol=0)
    both = expected**2
    assert (
        abs(np.mean(bits[:, 0] * bits[:, 1]) - both)
        < 7 * np.sqrt(both * (1 - both) / shots) + 1 / shots
    )


def test_nonuniform_probability_one_and_strided_input(graph):
    data = np.zeros((3, 6))
    data[:, 0] = 1
    det, obs = nonuniform(graph, data[:, ::2])
    assert np.all(det == 1) and np.all(obs == 1)
    det, obs = nonuniform(graph, np.zeros((3, 3)))
    assert not det.any() and not obs.any()


@pytest.mark.parametrize(
    "probs",
    [
        np.zeros((2, 3)),
        np.zeros((3, 2)),
        np.full((3, 3), np.nan),
        np.full((3, 3), -0.1),
        np.full((3, 3), 0.4),
    ],
)
def test_nonuniform_rejects_invalid_probabilities(graph, probs):
    with pytest.raises(ValueError):
        nonuniform(graph, probs)


@pytest.mark.parametrize(
    "a,b,cp",
    [
        ([0], [], [0.1]),
        ([0], [3], [0.1]),
        ([0], [0], [0.1]),
        ([0], [1], [float("nan")]),
    ],
)
def test_nonuniform_rejects_invalid_pairs(graph, a, b, cp):
    with pytest.raises(ValueError):
        nonuniform(graph, np.zeros((3, 3)), a=a, b=b, cp=cp)


def test_sampler_calls_do_not_repeat_random_stream(graph):
    first = qepg.return_samples_with_fixed_QEPG_numpy(graph, 1, 500)
    second = qepg.return_samples_with_fixed_QEPG_numpy(graph, 1, 500)
    assert not (
        np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    )


@pytest.mark.parametrize(
    "instruction", ["DETECTOR rec[-2]", "OBSERVABLE_INCLUDE(0) rec[0]"]
)
def test_bad_record_reference_raises(instruction):
    with pytest.raises(ValueError):
        qepg.compile_QEPG("R 0\nM 0\n" + instruction)


def test_blank_crlf_lines_and_empty_graph():
    graph = qepg.compile_QEPG("\r\nR 0\r\n\r\nM 0\r\nOBSERVABLE_INCLUDE(0) rec[-1]\r\n")
    det, obs = qepg.return_samples_with_fixed_QEPG_numpy(graph, 0, 2)
    assert det.shape == (2, 0) and not obs.any()
    assert qepg.return_detector_matrix("") == []
