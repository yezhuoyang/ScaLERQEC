"""Semantic checks of extraction, teleportation, fault records and profiling.

Flow checks use Stim's stabilizer-flow checker (signed checks have false-positive
probability at most 2^-256), not average LER agreement as a correctness oracle.
"""

import numpy as np
import pytest
import stim

from benchmark.fault_replay_oracle import ReplayOracle
from scalerqec.QEC import NoiseModel, RepetitionCode, StabCode, SurfaceCode
from scalerqec.QEC.extraction import (
    append_knill_round,
    append_parity_gadget,
    bell_record_parity,
    code_tableau,
    encoded_bell_circuit,
)
from scalerqec.QEC.small import ShorCode, fivequbitCode, steaneCode
from scalerqec.Stratified import LinearNoiseModel, UniformizedSampler

SCHEMES = ["Standard", "Flag", "Shor", "Knill"]


def test_raw_scheme_ir_entry_points_compile_the_selected_scheme():
    for scheme, build, compile_method in [
        ("Shor", "construct_IR_shor_scheme", "compile_stim_circuit_from_shor"),
        ("Flag", "construct_IR_flag_scheme", "compile_stim_circuit_from_flag"),
        ("Knill", "construct_IR_knill_scheme", "compile_stim_circuit_from_knill"),
    ]:
        code = make_code("repetition", "Standard")
        with pytest.raises(RuntimeError, match="IR not compiled"):
            getattr(code, compile_method)()
        getattr(code, build)()
        getattr(code, compile_method)()
        assert code.scheme.name == scheme.upper()
        before = code.stimcirc.copy()
        code.construct_circuit()
        assert code.stimcirc == before
        assert str(code._IRList[0])


def test_invalid_gadget_and_code_inputs_are_rejected():
    for s, scheme in [("II", "Flag"), ("X", "Unknown")]:
        with pytest.raises(ValueError):
            append_parity_gadget(stim.Circuit(), s, list(range(len(s))), len(s), scheme)
    with pytest.raises(ValueError, match="rank"):
        code_tableau(["ZZI"], ["ZZZ"], 3)
    with pytest.raises(ValueError):
        code_tableau(["ZZI", "IZZ"], ["ZZI"], 3)


def make_code(name, scheme, rounds=2):
    if name == "repetition":
        code = RepetitionCode(3)
    elif name == "surface":
        code = SurfaceCode(3)
    elif name == "surface13":
        code = SurfaceCode(13)
    elif name == "odd_y":
        code = StabCode(2, 1, 1)
        code.add_stab("YI")
        code.set_logical_Z(0, "IZ")
    elif name == "two_logicals":
        code = StabCode(4, 2, 2)
        code.add_stab("XXXX")
        code.add_stab("ZZZZ")
        code.set_logical_Z(0, "ZZII")
        code.set_logical_Z(1, "ZIZI")
    else:
        code = {"five": fivequbitCode, "steane": steaneCode, "shor": ShorCode}[name](
            scheme
        )
    code.scheme = scheme
    code.rounds = rounds
    return code


@pytest.mark.parametrize("scheme", ["Flag", "Shor"])
@pytest.mark.parametrize("pauli", ["X", "Y", "Z", "XYZ", "YYYY", "XZZXI", "IXXIIXX"])
def test_parity_gadget_measures_exact_operator_and_preserves_commutant(scheme, pauli):
    n = len(pauli)
    c = stim.Circuit()
    records, flags = append_parity_gadget(c, pauli, list(range(n)), n, scheme)
    p = stim.PauliString(pauli)
    assert c.has_flow(stim.Flow(input=p, measurements=records))
    assert c.has_flow(stim.Flow(input=p, output=p))
    for flag in flags:
        assert c.has_flow(stim.Flow(measurements=[flag]))
    # A generator set for the Pauli centralizer: single-site matching Paulis
    # and pairs of anticommuting Paulis. Preserving these prevents an unwanted
    # extra measurement of individual data qubits.
    anti = []
    for q, letter in enumerate(pauli):
        for axis in "XYZ":
            r = stim.PauliString(n)
            r[q] = axis
            if r.commutes(p):
                assert c.has_flow(stim.Flow(input=r, output=r))
            else:
                anti.append(r)
    for r in anti[1:]:
        op = anti[0] * r
        op.sign = 1
        assert c.has_flow(stim.Flow(input=op, output=op))


@pytest.mark.parametrize(
    "name", ["repetition", "five", "steane", "shor", "odd_y", "two_logicals"]
)
def test_knill_preserves_both_logical_axes_and_reports_input_syndrome(name):
    code = make_code(name, "Knill")
    zs = [code._logicalZ[j] for j in range(code.k)]
    tableau, rank = code_tableau(code._stabs, zs, code.n)
    prep = encoded_bell_circuit(tableau, rank)
    n = code.n
    circuit = stim.Circuit()
    xr, zr = append_knill_round(
        circuit, list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n)), prep
    )
    for j in range(rank, n):
        for p in (tableau.x_output(j), tableau.z_output(j)):
            text = str(p)[1:].replace("_", "I")
            refs = bell_record_parity(text, xr, zr)
            out = stim.PauliString(2 * n) + p
            assert circuit.has_flow(stim.Flow(input=p, output=out, measurements=refs))
    for s in code._stabs:
        assert circuit.has_flow(
            stim.Flow(
                input=stim.PauliString(s), measurements=bell_record_parity(s, xr, zr)
            )
        )
        assert circuit.has_flow(
            stim.Flow(output=stim.PauliString(2 * n) + stim.PauliString(s))
        )


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize(
    "name", ["repetition", "surface", "five", "steane", "shor", "odd_y", "two_logicals"]
)
def test_whole_memory_is_deterministic_and_compilation_is_idempotent(scheme, name):
    code = make_code(name, scheme)
    code.construct_circuit()
    ideal = code.stimcirc.copy()
    assert ideal.detector_error_model().num_errors == 0
    assert (
        not ideal.compile_detector_sampler(seed=91)
        .sample(64, append_observables=True)
        .any()
    )
    code.construct_circuit()
    assert code.stimcirc == ideal
    noise = NoiseModel(
        0.01, p_1q=0.002, p_2q=0.01, p_meas=0.05, p_reset=0.02, p_idle=0.001
    )
    code.noisemodel = noise
    code.construct_circuit()
    assert code.stimcirc == noise.apply(ideal)
    first = code.stimcirc.copy()
    code.construct_circuit()
    assert code.stimcirc == first
    # Stim rejects gauge/nondeterministic detectors and logicals here.
    assert code.stimcirc.detector_error_model().num_errors > 0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_surface_distance_13_three_rounds_has_valid_detectors_and_observable(scheme):
    code = make_code("surface13", scheme, rounds=3)
    code.construct_circuit()
    ideal = code.stimcirc
    assert ideal.num_observables == 1
    assert ideal.num_detectors >= 3 * 168
    assert (
        not ideal.compile_detector_sampler(seed=37)
        .sample(8, append_observables=True)
        .any()
    )
    noisy = NoiseModel(0.001, p_1q=0.0002, p_meas=0.005, p_reset=0).apply(ideal)
    assert noisy.detector_error_model().num_errors > 0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_every_single_fault_outcome_matches_independent_instruction_replay(scheme):
    code = make_code("repetition", scheme, rounds=1)
    code.noisemodel = NoiseModel(0.01, p_1q=0.002, p_2q=0.01, p_meas=0.05, p_reset=0.01)
    model = LinearNoiseModel.from_stabcode(code, 0.01)
    oracle = ReplayOracle(model.circuit, 0.01)
    for j, factor in enumerate(model._factors):
        h = np.zeros(len(model._factors), dtype=int)
        for a in range(len(factor.weights)):
            h[j] = a
            np.testing.assert_array_equal(model._responses[j][a], oracle.bits(h))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("name", ["five", "surface", "two_logicals"])
def test_uniformized_joint_sample_matches_stim_with_nonuniform_gate_noise(scheme, name):
    code = make_code(name, scheme, rounds=1)
    code.noisemodel = NoiseModel(
        0.01, p_1q=0.002, p_2q=0.01, p_meas=0.05, p_reset=0.003, p_idle=0.001
    )
    model = LinearNoiseModel.from_stabcode(code, 0.01)
    sampler = UniformizedSampler(model)
    rng = np.random.default_rng(925)
    n = 20000
    for p in [0.003, 0.02]:
        bits, _ = sampler.sample(
            rng.binomial(sampler.num_trials, sampler.rate * p, n), rng
        )
        direct = (
            model.circuit_at(p)
            .compile_detector_sampler(seed=1403)
            .sample(n, append_observables=True)
        )
        # Test all margins and random parities to expose correlation errors.
        masks = rng.integers(0, 2, size=(16, bits.shape[1]), dtype=np.uint8)
        a = np.concatenate((bits, (bits.astype(np.uint8) @ masks.T) % 2), axis=1).mean(
            axis=0
        )
        b = np.concatenate(
            (direct, (direct.astype(np.uint8) @ masks.T) % 2), axis=1
        ).mean(axis=0)
        pooled = (a + b) / 2
        assert np.all(abs(a - b) <= 8 * np.sqrt(2 * pooled * (1 - pooled) / n) + 8 / n)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_mutating_rounds_scheme_and_noise_rebuilds_circuit(scheme):
    code = make_code("repetition", scheme, 1)
    code.construct_circuit()
    old_measurements = code.stimcirc.num_measurements
    code.rounds = 2
    assert code.stimcirc is None
    code.construct_circuit()
    assert code.stimcirc.num_measurements > old_measurements
    code.scheme = "Standard"
    assert code.stimcirc is None
    code.construct_circuit()
    expected = make_code("repetition", "Standard", 2)
    expected.construct_circuit()
    assert code.stimcirc == expected.stimcirc


def test_standard_y_measurement_is_repeatable():
    code = make_code("odd_y", "Standard", rounds=3)
    code.construct_circuit()
    # Old H-CX-H-CX implementation measured the wrong operator and disturbed
    # its ancilla. Direct record-repeatability and a signed input flow catch it.
    assert code.stimcirc.has_flow(stim.Flow(measurements=[0, 1]))
    assert code.stimcirc.has_flow(stim.Flow(measurements=[1, 2]))
    gadget = code.stimcirc[2:]
    assert gadget.has_flow(stim.Flow(input=stim.PauliString("YI"), measurements=[0]))


def test_invalid_code_definitions_fail_instead_of_silently_changing_observable():
    code = StabCode(2, 1, 1)
    code.add_stab("YI")
    code.set_logical_Z(0, "ZZ")
    with pytest.raises(ValueError):
        code.construct_circuit()
    with pytest.raises(ValueError):
        code.construct_IR_standard_scheme()
    code.set_logical_Z(0, "IX")
    with pytest.raises(ValueError, match="Z-type"):
        code.construct_circuit()
    for rounds in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="positive integer"):
            code.rounds = rounds


def test_single_flag_detects_all_ancilla_x_hooks_within_the_data_sequence():
    circuit = stim.Circuit()
    _, flags = append_parity_gadget(circuit, "XXXX", list(range(4)), 4, "Flag")
    # Insert an X fault on syndrome ancilla after each data interaction.
    for index, op in enumerate(circuit):
        if op.name == "CX" and op.targets_copy()[-1].value in range(4):
            faulty = circuit[: index + 1]
            faulty.append("X", [4])
            faulty += circuit[index + 1 :]
            measurements = faulty.compile_sampler(seed=4).sample(32)
            assert measurements[:, flags[0]].all()
