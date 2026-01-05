"""
Tests for SIMD-accelerated sampling functions.

These tests verify that the SIMD implementations produce statistically
equivalent results to the non-SIMD implementations.
"""
import pytest
import numpy as np

import scalerqec.qepg as qepg


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_circuit():
    """Simple single-qubit circuit."""
    return """R 0
H 0
M 0
DETECTOR(0, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.fixture
def two_qubit_circuit():
    """Two-qubit circuit with CNOT."""
    return """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 0, 0) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.fixture
def repetition_code_circuit():
    """Distance-3 repetition code circuit."""
    return """R 0
R 1
R 2
R 3
R 4
CX 0 3
CX 1 3
CX 1 4
CX 2 4
M 3
M 4
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-5] rec[-4]
DETECTOR(1, 0, 0) rec[-4] rec[-3]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-2] rec[-1]
"""


# ============================================================================
# Test Class: SIMD Function Availability
# ============================================================================

class TestSIMDFunctionAvailability:
    """Verify SIMD functions are available in the module."""

    def test_simd_weight_function_exists(self):
        """Check that SIMD weight-based sampling function exists."""
        assert hasattr(qepg, 'return_samples_many_weights_separate_obs_with_QEPG_simd')

    def test_simd_monte_function_exists(self):
        """Check that SIMD Monte Carlo sampling function exists."""
        assert hasattr(qepg, 'return_samples_Monte_separate_obs_with_QEPG_simd')


# ============================================================================
# Test Class: SIMD Correctness - Weight-based Sampling
# ============================================================================

class TestSIMDCorrectnessWeightBased:
    """
    Verify SIMD weight-based sampling produces correct results.

    We test correctness by:
    1. Verifying output shapes match
    2. Verifying zero-weight produces all zeros
    3. Statistical comparison of flip rates between SIMD and non-SIMD
    """

    def test_simd_output_shape_simple(self, simple_circuit):
        """SIMD sampling returns correct output shape for simple circuit."""
        graph = qepg.compile_QEPG(simple_circuit)
        weights = [1, 2]
        shots = [50, 50]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        # Total samples = 50 + 50 = 100
        assert det_samples.shape[0] == 100
        assert obs_samples.shape[0] == 100
        # 1 detector
        assert det_samples.shape[1] == 1
        # Observable may be 1D or 2D depending on implementation
        assert len(obs_samples.shape) >= 1

    def test_simd_output_shape_repetition_code(self, repetition_code_circuit):
        """SIMD sampling returns correct shape for repetition code."""
        graph = qepg.compile_QEPG(repetition_code_circuit)
        weights = [1, 2, 3]
        shots = [30, 40, 30]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        # Total samples = 30 + 40 + 30 = 100
        assert det_samples.shape[0] == 100
        assert obs_samples.shape[0] == 100
        # 2 detectors
        assert det_samples.shape[1] == 2
        # Observable may be 1D or 2D depending on implementation
        assert len(obs_samples.shape) >= 1

    def test_simd_zero_weight_no_flips(self, two_qubit_circuit):
        """SIMD sampling with weight 0 produces no flips."""
        graph = qepg.compile_QEPG(two_qubit_circuit)
        weights = [0]
        shots = [100]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        # All zeros expected
        assert np.all(det_samples == False)
        assert np.all(obs_samples == False)

    def test_simd_vs_nonsimd_shape_match(self, repetition_code_circuit):
        """SIMD and non-SIMD produce same output shapes."""
        graph = qepg.compile_QEPG(repetition_code_circuit)
        weights = [1, 2]
        shots = [50, 50]

        det_simd, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )
        det_nonsimd, obs_nonsimd = qepg.return_samples_many_weights_separate_obs_with_QEPG(
            graph, weights, shots
        )

        assert det_simd.shape == det_nonsimd.shape
        assert obs_simd.shape == obs_nonsimd.shape

    def test_simd_vs_nonsimd_statistical_equivalence(self, repetition_code_circuit):
        """
        SIMD and non-SIMD produce statistically similar results.

        We compare the mean flip rates - they should be close for large samples.
        """
        graph = qepg.compile_QEPG(repetition_code_circuit)
        weights = [2]
        shots = [5000]

        det_simd, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )
        det_nonsimd, obs_nonsimd = qepg.return_samples_many_weights_separate_obs_with_QEPG(
            graph, weights, shots
        )

        # Compare mean flip rates (should be close for large sample size)
        simd_det_rate = np.mean(det_simd)
        nonsimd_det_rate = np.mean(det_nonsimd)

        simd_obs_rate = np.mean(obs_simd)
        nonsimd_obs_rate = np.mean(obs_nonsimd)

        # Allow 10% relative tolerance due to randomness
        assert abs(simd_det_rate - nonsimd_det_rate) < 0.1, \
            f"Detector flip rates differ: SIMD={simd_det_rate:.3f}, non-SIMD={nonsimd_det_rate:.3f}"
        assert abs(simd_obs_rate - nonsimd_obs_rate) < 0.1, \
            f"Observable flip rates differ: SIMD={simd_obs_rate:.3f}, non-SIMD={nonsimd_obs_rate:.3f}"


# ============================================================================
# Test Class: SIMD Correctness - Monte Carlo Sampling
# ============================================================================

class TestSIMDCorrectnessMonteCarlo:
    """
    Verify SIMD Monte Carlo sampling produces correct results.
    """

    def test_simd_monte_output_shape(self, two_qubit_circuit):
        """SIMD Monte Carlo returns correct output shape."""
        graph = qepg.compile_QEPG(two_qubit_circuit)
        error_rate = 0.01
        shots = 100

        det_samples, obs_samples = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, error_rate, shots
        )

        assert det_samples.shape[0] == shots
        assert obs_samples.shape[0] == shots
        # 2 detectors
        assert det_samples.shape[1] == 2
        # Observable may be 1D or 2D depending on implementation
        assert len(obs_samples.shape) >= 1

    def test_simd_monte_zero_error_rate(self, simple_circuit):
        """SIMD Monte Carlo with zero error rate produces no flips."""
        graph = qepg.compile_QEPG(simple_circuit)
        error_rate = 0.0
        shots = 100

        det_samples, obs_samples = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, error_rate, shots
        )

        # All zeros expected with zero error rate
        assert np.all(det_samples == False)
        assert np.all(obs_samples == False)

    def test_simd_monte_vs_nonsimd_shape_match(self, repetition_code_circuit):
        """SIMD and non-SIMD Monte Carlo produce same shapes."""
        graph = qepg.compile_QEPG(repetition_code_circuit)
        error_rate = 0.05
        shots = 100

        det_simd, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, error_rate, shots
        )
        det_nonsimd, obs_nonsimd = qepg.return_samples_Monte_separate_obs_with_QEPG(
            graph, error_rate, shots
        )

        assert det_simd.shape == det_nonsimd.shape
        assert obs_simd.shape == obs_nonsimd.shape

    def test_simd_monte_vs_nonsimd_statistical_equivalence(self, repetition_code_circuit):
        """
        SIMD and non-SIMD Monte Carlo produce statistically similar results.
        """
        graph = qepg.compile_QEPG(repetition_code_circuit)
        error_rate = 0.05
        shots = 5000

        det_simd, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, error_rate, shots
        )
        det_nonsimd, obs_nonsimd = qepg.return_samples_Monte_separate_obs_with_QEPG(
            graph, error_rate, shots
        )

        # Compare mean flip rates
        simd_det_rate = np.mean(det_simd)
        nonsimd_det_rate = np.mean(det_nonsimd)

        simd_obs_rate = np.mean(obs_simd)
        nonsimd_obs_rate = np.mean(obs_nonsimd)

        # Allow 10% relative tolerance
        assert abs(simd_det_rate - nonsimd_det_rate) < 0.1, \
            f"Detector flip rates differ: SIMD={simd_det_rate:.3f}, non-SIMD={nonsimd_det_rate:.3f}"
        assert abs(simd_obs_rate - nonsimd_obs_rate) < 0.1, \
            f"Observable flip rates differ: SIMD={simd_obs_rate:.3f}, non-SIMD={nonsimd_obs_rate:.3f}"


# ============================================================================
# Test Class: SIMD Edge Cases
# ============================================================================

class TestSIMDEdgeCases:
    """Test edge cases for SIMD sampling."""

    def test_simd_single_shot(self, simple_circuit):
        """SIMD sampling works with single shot."""
        graph = qepg.compile_QEPG(simple_circuit)
        weights = [1]
        shots = [1]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        assert det_samples.shape[0] == 1
        assert obs_samples.shape[0] == 1

    def test_simd_large_shots(self, two_qubit_circuit):
        """SIMD sampling works with large number of shots."""
        graph = qepg.compile_QEPG(two_qubit_circuit)
        weights = [1]
        shots = [10000]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        assert det_samples.shape[0] == 10000
        assert obs_samples.shape[0] == 10000

    def test_simd_multiple_weights(self, repetition_code_circuit):
        """SIMD sampling works with multiple weight values."""
        graph = qepg.compile_QEPG(repetition_code_circuit)
        weights = [1, 2, 3, 4, 5]
        shots = [100, 100, 100, 100, 100]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        assert det_samples.shape[0] == 500
        assert obs_samples.shape[0] == 500

    def test_simd_large_circuit(self):
        """SIMD sampling works with larger circuits."""
        # Create a 10-qubit circuit with many gates
        lines = ["R " + str(i) for i in range(10)]
        # Add some Hadamards
        lines += ["H " + str(i) for i in range(5)]
        # Add CNOT chain
        for i in range(9):
            lines.append(f"CX {i} {i+1}")
        # Add measurements
        lines += ["M " + str(i) for i in range(10)]
        # Add detectors
        lines += [f"DETECTOR({i}, 0, 0) rec[-{10-i}]" for i in range(10)]
        lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
        stim_str = "\n".join(lines)

        graph = qepg.compile_QEPG(stim_str)
        weights = [2]
        shots = [1000]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        assert det_samples.shape[0] == 1000
        assert det_samples.shape[1] == 10
        assert obs_samples.shape[0] == 1000


# ============================================================================
# Test Class: SIMD Data Type Verification
# ============================================================================

class TestSIMDDataTypes:
    """Verify SIMD functions return correct data types."""

    def test_simd_returns_bool_array(self, simple_circuit):
        """SIMD sampling returns boolean numpy arrays."""
        graph = qepg.compile_QEPG(simple_circuit)
        weights = [1]
        shots = [10]

        det_samples, obs_samples = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        assert det_samples.dtype == np.bool_
        assert obs_samples.dtype == np.bool_

    def test_simd_monte_returns_bool_array(self, simple_circuit):
        """SIMD Monte Carlo returns boolean numpy arrays."""
        graph = qepg.compile_QEPG(simple_circuit)

        det_samples, obs_samples = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, 0.01, 10
        )

        assert det_samples.dtype == np.bool_
        assert obs_samples.dtype == np.bool_
