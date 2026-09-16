"""
Test non-uniform noise model support for QEPG Monte Carlo sampling.

Validates that:
1. noise_model_parser correctly extracts per-source (px, py, pz) from Stim circuits
2. QEPGpython.sample_nonuniform_batch produces correct detector/observable outcomes
3. LER from non-uniform sampling matches Stim's native detector sampler
"""
import os
import pytest
import numpy as np
import stim
import pymatching

from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.Clifford.QEPGpython import QEPGpython
from scalerqec.Clifford.stimparser import rewrite_stim_code
from scalerqec.Monte.noise_model_parser import (
    extract_noise_model,
    NonuniformNoiseModel,
)


STIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "stimprograms",
    "surface",
)


class TestNoiseModelParser:
    """Test extraction of per-noise-source probabilities."""

    def test_uniform_depolarize1(self):
        """DEPOLARIZE1(p) should give (p/3, p/3, p/3) per source."""
        # Each DEPOLARIZE1 is followed by a gate on every affected qubit,
        # so no noise accumulates across directives.
        circuit_str = """
R 0
DEPOLARIZE1(0.03) 0
H 0
TICK
DEPOLARIZE1(0.03) 0
M 0
DETECTOR(0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        model = extract_noise_model(circuit_str)
        assert model.num_noise > 0
        # Check that all noise sources have (0.01, 0.01, 0.01)
        for i in range(model.num_noise):
            row = model.noise_probs[i]
            if np.sum(row) > 0:  # skip zero-noise sources
                np.testing.assert_allclose(row, [0.01, 0.01, 0.01], atol=1e-10)

    def test_x_error_only(self):
        """X_ERROR(p) should give (p, 0, 0)."""
        circuit_str = """
R 0
X_ERROR(0.05) 0
H 0
M 0
DETECTOR(0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        model = extract_noise_model(circuit_str)
        # Find the noise source with non-zero probability
        nonzero = model.noise_probs[np.sum(model.noise_probs, axis=1) > 0]
        assert len(nonzero) > 0
        np.testing.assert_allclose(nonzero[0], [0.05, 0.0, 0.0], atol=1e-10)

    def test_z_error_only(self):
        """Z_ERROR(p) should give (0, 0, p)."""
        circuit_str = """
R 0
Z_ERROR(0.02) 0
H 0
M 0
DETECTOR(0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        model = extract_noise_model(circuit_str)
        nonzero = model.noise_probs[np.sum(model.noise_probs, axis=1) > 0]
        assert len(nonzero) > 0
        np.testing.assert_allclose(nonzero[0], [0.0, 0.0, 0.02], atol=1e-10)

    def test_mixed_noise_rates(self):
        """Different rates on different qubits should be preserved."""
        circuit_str = """
R 0
R 1
DEPOLARIZE1(0.03) 0
DEPOLARIZE1(0.06) 1
H 0
H 1
M 0
M 1
DETECTOR(0, 0) rec[-2]
DETECTOR(1, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        model = extract_noise_model(circuit_str)
        assert model.num_noise >= 4  # at least H(0), H(1), M(0), M(1)
        # First H gate on qubit 0 should have rate 0.03/3
        # First H gate on qubit 1 should have rate 0.06/3
        probs_q0 = model.noise_probs[0]  # first noise source
        probs_q1 = model.noise_probs[1]  # second noise source
        np.testing.assert_allclose(probs_q0, [0.01, 0.01, 0.01], atol=1e-10)
        np.testing.assert_allclose(probs_q1, [0.02, 0.02, 0.02], atol=1e-10)

    def test_depolarize2_creates_correlated_pair(self):
        """DEPOLARIZE2 should create a CorrelatedNoisePair."""
        circuit_str = """
R 0
R 1
DEPOLARIZE2(0.015) 0 1
CX 0 1
M 0 1
DETECTOR(0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        model = extract_noise_model(circuit_str)
        assert len(model.correlated_pairs) == 1
        pair = model.correlated_pairs[0]
        assert pair.prob == pytest.approx(0.015)
        assert pair.source_a != pair.source_b

    def test_no_noise_gives_zero_probs(self):
        """A circuit with no noise directives should have all-zero probs."""
        circuit_str = """
R 0
H 0
M 0
DETECTOR(0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        model = extract_noise_model(circuit_str)
        assert model.num_noise > 0
        assert np.all(model.noise_probs == 0)


class TestNonuniformSampling:
    """Test that non-uniform sampling produces correct statistics."""

    def _build_qepg(self, circuit_str: str):
        """Helper: normalize, compile, and build QEPG."""
        normalized = rewrite_stim_code(circuit_str)
        circuit = CliffordCircuit(2)
        circuit.compile_from_stim_circuit_str(normalized)
        graph = QEPGpython(circuit)
        graph.backword_graph_construction()
        # Attach C++ QEPG for accelerated sampling
        try:
            from scalerqec import qepg as qepg_cpp
            graph._cpp_graph = qepg_cpp.compile_QEPG(normalized)
        except Exception:
            graph._cpp_graph = None
        return graph, circuit

    def test_zero_noise_produces_zero_outcomes(self):
        """With all-zero noise probs, all detector/observable outcomes should be 0."""
        circuit_str = """
R 0
R 1
H 0
CX 0 1
M 0 1
DETECTOR(0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph, circuit = self._build_qepg(circuit_str)
        N = circuit.totalnoise
        noise_probs = np.zeros((N, 3))

        det, obs = graph.sample_nonuniform_batch(noise_probs, 1000)
        assert np.all(det == 0)
        assert np.all(obs == 0)

    def test_uniform_noise_nonzero_detections(self):
        """With uniform noise, should see some non-zero detections."""
        filepath = os.path.join(STIM_DIR, "surface3")
        if not os.path.exists(filepath):
            pytest.skip("surface3 stim program not found")

        with open(filepath) as f:
            stim_str = f.read()

        graph, circuit = self._build_qepg(stim_str)
        N = circuit.totalnoise
        p = 0.01
        noise_probs = np.full((N, 3), p / 3)

        det, obs = graph.sample_nonuniform_batch(noise_probs, 10000)
        # With p=0.01, we should see some detector events
        assert np.any(det != 0), "Expected some non-zero detector events"


class TestLERNonuniformVsStim:
    """Compare LER from non-uniform QEPG sampling vs Stim native sampling.

    For each noise model, both Stim and QEPG sample until at least
    MIN_ERRORS logical error events are observed. The two LER estimates
    must agree within TOLERANCE relative difference.
    """

    # For two independent estimates, 30k failures each make a 5% difference
    # roughly six sampling standard deviations even in the rare-event limit.
    # The old 500-event target allowed ordinary sampling noise to fail CI.
    MIN_ERRORS = 30_000
    TOLERANCE = 0.05  # 5% relative difference
    BATCH_SIZE = 100000
    MAX_SHOTS = 10_000_000

    # -- Helpers --

    @staticmethod
    def _inject_noise(bare_str, noise_fn):
        """Insert noise directives before gates in a bare circuit.

        Args:
            bare_str: Bare Stim circuit string (no noise directives).
            noise_fn: Callable(gate_name, qubits_list) -> list of noise
                directive strings to insert before this gate.
        """
        gate_names = {"H", "S", "M", "X", "Y", "Z", "CX", "CZ", "MR", "MX", "MY"}
        lines = bare_str.strip().split("\n")
        output = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            gate = tokens[0]
            if gate in gate_names:
                qubits = tokens[1:]
                directives = noise_fn(gate, qubits)
                output.extend(directives)
            output.append(stripped)
        return "\n".join(output)

    def _sample_ler(self, sampler_fn, decoder):
        """Sample in batches until MIN_ERRORS logical errors are found.

        Args:
            sampler_fn: Callable(batch_size) -> (det, obs) arrays.
            decoder: pymatching.Matching decoder.

        Returns:
            (ler, total_shots, total_errors)
        """
        total_shots = 0
        total_errors = 0
        while total_errors < self.MIN_ERRORS and total_shots < self.MAX_SHOTS:
            det, obs = sampler_fn(self.BATCH_SIZE)
            pred = decoder.decode_batch(det)
            total_errors += np.count_nonzero(
                obs.ravel() != pred.ravel()
            )
            total_shots += self.BATCH_SIZE
        ler = total_errors / total_shots if total_shots > 0 else 0.0
        return ler, total_shots, total_errors

    def _run_comparison(self, bare_str, noise_fn):
        """Run Stim vs QEPG LER comparison for a given noise model.

        1. Inject noise into bare circuit.
        2. Build Stim sampler + decoder from noisy circuit.
        3. Build QEPG sampler from bare circuit + extracted noise model.
        4. Sample both until MIN_ERRORS, assert LERs match within TOLERANCE.
        """
        noisy_str = self._inject_noise(bare_str, noise_fn)
        noisy_circuit = stim.Circuit(noisy_str)

        # Decoder from noisy circuit's DEM
        dem = noisy_circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        decoder = pymatching.Matching.from_detector_error_model(dem)

        # --- Stim native sampler ---
        stim_sampler = noisy_circuit.compile_detector_sampler()

        def stim_sample(n):
            return stim_sampler.sample(n, separate_observables=True)

        ler_stim, shots_stim, errs_stim = self._sample_ler(stim_sample, decoder)

        # --- QEPG non-uniform sampler ---
        normalized = rewrite_stim_code(bare_str)
        clifford = CliffordCircuit(2)
        clifford.error_rate = 0.001  # dummy; overridden by noise_probs
        clifford.compile_from_stim_circuit_str(normalized)

        graph = QEPGpython(clifford)
        graph.backword_graph_construction()

        # Attach C++ QEPG for accelerated sampling
        try:
            from scalerqec import qepg as qepg_cpp
            graph._cpp_graph = qepg_cpp.compile_QEPG(normalized)
        except Exception:
            graph._cpp_graph = None

        noise_model = extract_noise_model(noisy_str)
        assert noise_model.num_noise == clifford.totalnoise, (
            f"Noise source count mismatch: parser={noise_model.num_noise} "
            f"vs QEPG={clifford.totalnoise}"
        )
        correlated = noise_model.correlated_pairs or None

        def qepg_sample(n):
            return graph.sample_nonuniform_batch(
                noise_model.noise_probs, n, correlated_pairs=correlated
            )

        ler_qepg, shots_qepg, errs_qepg = self._sample_ler(qepg_sample, decoder)

        # --- Assertions ---
        assert errs_stim >= self.MIN_ERRORS, (
            f"Stim only found {errs_stim} errors in {shots_stim} shots"
        )
        assert errs_qepg >= self.MIN_ERRORS, (
            f"QEPG only found {errs_qepg} errors in {shots_qepg} shots"
        )
        rel_diff = abs(ler_qepg - ler_stim) / max(ler_stim, ler_qepg)
        assert rel_diff < self.TOLERANCE, (
            f"LER mismatch: QEPG={ler_qepg:.6f} ({errs_qepg}/{shots_qepg}) "
            f"vs Stim={ler_stim:.6f} ({errs_stim}/{shots_stim}), "
            f"rel_diff={rel_diff:.4f}"
        )

    # -- Noise model definitions --

    @staticmethod
    def _uniform_depolarize1(p):
        """DEPOLARIZE1(p) before every non-R gate."""
        def fn(gate, qubits):
            if gate == "R":
                return []
            return [f"DEPOLARIZE1({p}) {' '.join(qubits)}"]
        return fn

    @staticmethod
    def _x_error_only(p):
        """X_ERROR(p) before every non-R gate."""
        def fn(gate, qubits):
            if gate == "R":
                return []
            return [f"X_ERROR({p}) {' '.join(qubits)}"]
        return fn

    @staticmethod
    def _y_error_only(p):
        """Y_ERROR(p) before every non-R gate."""
        def fn(gate, qubits):
            if gate == "R":
                return []
            return [f"Y_ERROR({p}) {' '.join(qubits)}"]
        return fn

    @staticmethod
    def _biased_xy(px, py):
        """X_ERROR(px) + Y_ERROR(py) before every non-R gate."""
        def fn(gate, qubits):
            if gate == "R":
                return []
            q_str = " ".join(qubits)
            return [f"X_ERROR({px}) {q_str}", f"Y_ERROR({py}) {q_str}"]
        return fn

    @staticmethod
    def _depolarize_mixed_rates(p_1q, p_2q):
        """DEPOLARIZE1(p_1q) on 1q gates, DEPOLARIZE1(p_2q) on 2q gates."""
        def fn(gate, qubits):
            if gate == "R":
                return []
            if gate in ("CX", "CZ"):
                return [f"DEPOLARIZE1({p_2q}) {' '.join(qubits)}"]
            return [f"DEPOLARIZE1({p_1q}) {' '.join(qubits)}"]
        return fn

    @staticmethod
    def _per_qubit_varying_depolarize(base_p, increment):
        """DEPOLARIZE1 rate varies by qubit index: base_p + increment*(q%5).

        Each qubit gets a distinct error rate, testing that per-source
        probabilities are correctly tracked through the entire pipeline.
        """
        def fn(gate, qubits):
            if gate == "R":
                return []
            directives = []
            for q in qubits:
                p = base_p + increment * (int(q) % 5)
                directives.append(f"DEPOLARIZE1({p}) {q}")
            return directives
        return fn

    @staticmethod
    def _mixed_channels_per_gate():
        """Different noise channels for different gate types.

        H  -> DEPOLARIZE1(0.012): symmetric depolarizing on Hadamards
        CX -> X_ERROR(0.008) + Z_ERROR(0.004): biased noise on entangling gates
        M  -> X_ERROR(0.015): bit-flip measurement errors
        Other 1q gates -> DEPOLARIZE1(0.005)
        """
        def fn(gate, qubits):
            if gate == "R":
                return []
            q_str = " ".join(qubits)
            if gate == "H":
                return [f"DEPOLARIZE1(0.012) {q_str}"]
            if gate == "CX":
                return [f"X_ERROR(0.008) {q_str}", f"Z_ERROR(0.004) {q_str}"]
            if gate == "M":
                return [f"X_ERROR(0.015) {q_str}"]
            return [f"DEPOLARIZE1(0.005) {q_str}"]
        return fn

    @staticmethod
    def _measurement_noise_only(p):
        """X_ERROR(p) only before M gates; all other gates noiseless.

        Tests sparse noise models where most noise sources have zero
        probability — only measurement noise sources are active.
        """
        def fn(gate, qubits):
            if gate == "M":
                return [f"X_ERROR({p}) {' '.join(qubits)}"]
            return []
        return fn

    @staticmethod
    def _three_axis_asymmetric(px, py, pz):
        """X_ERROR(px) + Y_ERROR(py) + Z_ERROR(pz) with all three distinct.

        Tests that three independent noise channels accumulate correctly
        in the parser and that the sampler handles the full (px, py, pz)
        triple with all components nonzero.
        """
        def fn(gate, qubits):
            if gate == "R":
                return []
            q_str = " ".join(qubits)
            return [
                f"X_ERROR({px}) {q_str}",
                f"Y_ERROR({py}) {q_str}",
                f"Z_ERROR({pz}) {q_str}",
            ]
        return fn

    @staticmethod
    def _layered_depolarize_plus_bias(p_depol, p_x_extra):
        """DEPOLARIZE1(p_depol) + X_ERROR(p_x_extra) on every gate.

        Tests noise accumulation: the parser should combine both channels
        into (p_depol/3 + p_x_extra, p_depol/3, p_depol/3) per source.
        """
        def fn(gate, qubits):
            if gate == "R":
                return []
            q_str = " ".join(qubits)
            return [
                f"DEPOLARIZE1({p_depol}) {q_str}",
                f"X_ERROR({p_x_extra}) {q_str}",
            ]
        return fn

    @staticmethod
    def _depolarize2_only(p):
        """DEPOLARIZE2(p) only on CX gates; all other gates noiseless.

        Tests the correlated pair sampling path in isolation — all noise
        comes from two-qubit depolarization, with zero independent noise.
        """
        def fn(gate, qubits):
            if gate == "CX":
                return [f"DEPOLARIZE2({p}) {' '.join(qubits)}"]
            return []
        return fn

    @staticmethod
    def _depolarize2_plus_depolarize1(p_2q, p_1q):
        """DEPOLARIZE2(p_2q) + DEPOLARIZE1(p_1q) on CX, DEPOLARIZE1(p_1q) on 1q.

        Tests that independent single-qubit noise and correlated two-qubit
        noise compose correctly via XOR in the sampler.
        """
        def fn(gate, qubits):
            if gate == "R":
                return []
            q_str = " ".join(qubits)
            if gate == "CX":
                return [
                    f"DEPOLARIZE1({p_1q}) {q_str}",
                    f"DEPOLARIZE2({p_2q}) {q_str}",
                ]
            return [f"DEPOLARIZE1({p_1q}) {q_str}"]
        return fn

    # -- Parametrized tests --

    @pytest.fixture
    def bare_surface3(self):
        filepath = os.path.join(STIM_DIR, "surface3")
        if not os.path.exists(filepath):
            pytest.skip("surface3 stim program not found")
        with open(filepath) as f:
            return f.read()

    def test_uniform_depolarize1(self, bare_surface3):
        """Uniform DEPOLARIZE1(0.01) on all gates."""
        self._run_comparison(bare_surface3, self._uniform_depolarize1(0.01))

    def test_x_error_only(self, bare_surface3):
        """X_ERROR(0.03) on all gates — only X errors, no Y or Z."""
        self._run_comparison(bare_surface3, self._x_error_only(0.03))

    def test_y_error_only(self, bare_surface3):
        """Y_ERROR(0.03) on all gates — only Y errors, no X or Z."""
        self._run_comparison(bare_surface3, self._y_error_only(0.03))

    def test_biased_xy(self, bare_surface3):
        """Asymmetric X/Y bias: X_ERROR(0.005) + Y_ERROR(0.01)."""
        self._run_comparison(bare_surface3, self._biased_xy(0.005, 0.01))

    def test_mixed_1q_2q_rates(self, bare_surface3):
        """Different DEPOLARIZE1 rates on 1q vs 2q gates."""
        self._run_comparison(bare_surface3, self._depolarize_mixed_rates(0.005, 0.02))

    def test_per_qubit_varying_rates(self, bare_surface3):
        """DEPOLARIZE1 rate varies by qubit index: 0.004 to 0.012."""
        self._run_comparison(
            bare_surface3, self._per_qubit_varying_depolarize(0.004, 0.002)
        )

    def test_mixed_channels_per_gate(self, bare_surface3):
        """H->depol, CX->biased X+Z, M->bit-flip, other->weak depol."""
        self._run_comparison(bare_surface3, self._mixed_channels_per_gate())

    def test_measurement_noise_only(self, bare_surface3):
        """X_ERROR(0.05) only on M gates, all other gates noiseless."""
        self._run_comparison(bare_surface3, self._measurement_noise_only(0.05))

    def test_three_axis_asymmetric(self, bare_surface3):
        """X_ERROR(0.008) + Y_ERROR(0.004) + Z_ERROR(0.002) on all gates."""
        self._run_comparison(
            bare_surface3, self._three_axis_asymmetric(0.008, 0.004, 0.002)
        )

    def test_layered_depolarize_plus_bias(self, bare_surface3):
        """DEPOLARIZE1(0.006) + X_ERROR(0.003) — tests noise accumulation."""
        self._run_comparison(
            bare_surface3, self._layered_depolarize_plus_bias(0.006, 0.003)
        )

    def test_depolarize2_only(self, bare_surface3):
        """DEPOLARIZE2(0.01) only on CX — correlated pairs, zero independent noise."""
        self._run_comparison(bare_surface3, self._depolarize2_only(0.01))

    def test_depolarize2_plus_depolarize1(self, bare_surface3):
        """DEPOLARIZE2(0.01) + DEPOLARIZE1(0.005) on CX, DEPOLARIZE1(0.005) on 1q."""
        self._run_comparison(
            bare_surface3, self._depolarize2_plus_depolarize1(0.01, 0.005)
        )
