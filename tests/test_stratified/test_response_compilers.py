"""Fast fault signatures must agree bit-for-bit with independent forced Stim runs."""

import numpy as np
import pytest
import stim

from benchmark.general_noise_oracles import FAMILIES, circuit_at
from scalerqec.Stratified import LinearNoiseModel


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", ["repetition_3", "five_qubit", "steane"])
def test_all_columns_match_forced_simulator(name, family):
    model = LinearNoiseModel(circuit_at(name, family, 0.05), 0.05)
    fast = [x.copy() for x in model._responses]
    model.compile_responses(method="forced")
    for actual, expected in zip(fast, model._responses):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "task",
    [
        "surface_code:rotated_memory_x",
        "surface_code:rotated_memory_z",
        "repetition_code:memory",
        "color_code:memory_xyz",
    ],
)
def test_circuit_level_columns_match_forced_simulator(task):
    circuit = stim.Circuit.generated(
        task,
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.02,
        after_reset_flip_probability=0.005,
    )
    model = LinearNoiseModel(circuit, 0.01)
    fast = model._responses
    model.compile_responses(method="forced")
    for actual, expected in zip(fast, model._responses):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "body",
    [
        "R 0\nX 0\nX_ERROR[tag](.1) 0\nM(.02) !0\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "R 0 1\nM(.1) 0\nCX rec[-1] 1\nM 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "R 0 1\nE[original](.1) X0 X1\nELSE_CORRELATED_ERROR(.2) X1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "R 0\nZ_ERROR(.1) 0\nX_ERROR(0) 0\nMPAD(.1) 0 1\nM 0\nDETECTOR rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "R 0 1\nH 0\nCX 0 1\nPAULI_CHANNEL_2(.01,.02,.03,.04,0,0,0,0,0,0,0,0,0,0,0) 0 1\nMPP(.1) X0*X1 Z0*Z1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ],
)
def test_tagged_probe_semantics_include_records_and_silent_faults(body):
    model = LinearNoiseModel(body, 0.1)
    fast = model._responses
    model.compile_responses(method="forced")
    for actual, expected in zip(fast, model._responses):
        np.testing.assert_array_equal(actual, expected)


def test_explicit_method_validation_and_heralded_fallback():
    model = LinearNoiseModel(
        "R 0\nHERALDED_ERASE(.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", 0.1
    )
    with pytest.raises(ValueError, match="method"):
        model.compile_responses(method="unknown")
    with pytest.raises(NotImplementedError, match="forced"):
        model.compile_responses(method="explained")
    model.compile_responses(method="auto")
