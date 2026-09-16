"""Compare every native/Python propagation row with a forced Stim fault.

These checks test individual subspace samples, not just aggregate Monte Carlo
LER agreement. The Stim reference converter handles nonzero ideal records.
"""

import numpy as np
import pytest
import stim

from scalerqec import qepg
from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.Clifford.QEPGpython import QEPGpython
from scalerqec.Clifford.stimparser import rewrite_stim_code
from scalerqec.Stratified.general_noise import LinearNoiseModel


@pytest.mark.parametrize("seed", range(12))
def test_all_fault_columns_and_returned_native_weight_samples(seed):
    rng = np.random.default_rng(seed)
    unitary = stim.Circuit()
    for _ in range(12):
        gate = rng.choice(["H", "S", "X", "Y", "Z", "CX"])
        targets = rng.choice(3, 2 if gate == "CX" else 1, replace=False).tolist()
        unitary.append(gate, targets)
    circuit = stim.Circuit("R 0 1 2") + unitary + unitary.inverse()
    circuit += stim.Circuit(
        "M 0 1 2\nDETECTOR rec[-3] rec[-2]\nR 1\nM 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )
    program = rewrite_stim_code(str(circuit))
    noisy = stim.Circuit()
    for line in program.splitlines():
        op = stim.Circuit(line)[0]
        if op.name in {"H", "S", "X", "Y", "Z", "CX", "M"}:
            noisy.append("DEPOLARIZE1", op.targets_copy(), 0.01)
        noisy.append(op)
    oracle = LinearNoiseModel(noisy, 0.01)
    expected = np.concatenate(
        [np.array([response[a] for response in oracle._responses]) for a in [1, 2, 3]]
    )
    native = np.asarray(qepg.return_detector_matrix(program), dtype=np.bool_)
    np.testing.assert_array_equal(native, expected)
    clifford = CliffordCircuit(3)
    clifford.compile_from_stim_circuit_str(program)
    python_graph = QEPGpython(clifford)
    python_graph.backword_graph_construction()
    np.testing.assert_array_equal(python_graph._propMatrix, expected)
    n = len(oracle._factors)
    converter = oracle.ideal_circuit.compile_m2d_converter()
    for weight in [0, 1, 2, n // 2, n]:
        vectors, actual = qepg.return_samples_with_noise_vector(program, weight, 16)
        for faults, bits in zip(vectors, actual):
            assert len(faults) == weight
            assert len({location for location, _ in faults}) == weight
            history = dict(faults)  # Native pauli codes: 1=X, 2=Y, 3=Z.
            forced = oracle._forced_circuit(history)
            measurements = forced.compile_sampler(seed=123).sample(1)
            predicted = converter.convert(
                measurements=measurements, append_observables=True
            )[0]
            np.testing.assert_array_equal(bits, predicted)


def test_native_fixed_weight_distribution_is_uniform_within_stratum():
    program = "R 0\nR 1\nR 2\nM 0\nM 1\nM 2\nOBSERVABLE_INCLUDE(0) rec[-1]"
    vectors, _ = qepg.return_samples_with_noise_vector(program, 2, 60_000)
    counts = {}
    for faults in vectors:
        key = tuple(sorted(faults))
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 27  # choose(3,2)*3**2; distinct sites, uniform axes.
    expected = 60_000 / 27
    tolerance = 7 * np.sqrt(expected * (1 - 1 / 27))
    assert all(abs(value - expected) < tolerance for value in counts.values())
