"""Legacy single-observable backends must never silently drop logical bits."""

import numpy as np
import pytest

from scalerqec import qepg
from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.Clifford.QEPGpython import QEPGpython
from scalerqec.Stratified import LinearNoiseModel


@pytest.mark.parametrize(
    "observable",
    [
        "OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "OBSERVABLE_INCLUDE(0) rec[-2] rec[-2] rec[-1]",
        "OBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ],
)
def test_native_and_python_match_stim_observable_xor_semantics(observable):
    base = "R 0\nR 1\nM 0\nM 1\nDETECTOR rec[-1] rec[-1]\n" + observable
    noisy = base.replace("M 0", "DEPOLARIZE1(.1) 0\nM 0").replace(
        "M 1", "DEPOLARIZE1(.1) 1\nM 1"
    )
    reference = LinearNoiseModel(noisy, 0.1)
    expected = np.concatenate(
        [np.array([r[a] for r in reference._responses]) for a in [1, 2, 3]]
    )
    np.testing.assert_array_equal(qepg.return_detector_matrix(base), expected)
    python = CliffordCircuit(2)
    python.compile_from_stim_circuit_str(base)
    graph = QEPGpython(python)
    graph.backword_graph_construction()
    np.testing.assert_array_equal(graph._propMatrix, expected)
    direct = CliffordCircuit(2)
    direct.compile_from_noisy_stim_circuit_str(noisy)
    graph = QEPGpython(direct)
    graph.backword_graph_construction()
    np.testing.assert_array_equal(graph._propMatrix, expected)


@pytest.mark.parametrize(
    "suffix",
    [
        "OBSERVABLE_INCLUDE(1) rec[-1]",
        "OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-1]",
        "OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(11) rec[-1]",
    ],
)
def test_multiple_logical_observables_rejected_by_legacy_backends(suffix):
    program = "R 0\nR 1\nM 0\nM 1\n" + suffix
    with pytest.raises(ValueError, match="observable index 0"):
        qepg.compile_QEPG(program)
    for method in [
        "compile_from_stim_circuit_str",
        "compile_from_noisy_stim_circuit_str",
    ]:
        with pytest.raises(ValueError, match="observable index 0"):
            getattr(CliffordCircuit(2), method)(program)
    # The multi-observable general-noise path must preserve every output.
    model = LinearNoiseModel(program.replace("M 0", "X_ERROR(.1) 0\nM 0"), 0.1)
    assert model.num_observables > 1
    assert model._responses[0].shape[1] == model.num_observables
