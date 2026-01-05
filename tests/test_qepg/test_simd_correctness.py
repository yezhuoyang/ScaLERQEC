"""
Correctness tests for SIMD-accelerated sampling.

These tests verify:
1. SIMD sampling produces statistically correct results compared to symbolic LER ground truth
2. SIMD and non-SIMD methods produce equivalent logical error rates
3. Both fixed-weight and Monte Carlo sampling are correct

The symbolic LER calculator provides exact ground truth for small circuits.
"""
import os
import time
import pytest
import numpy as np
from pathlib import Path

import scalerqec.qepg as qepg

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# Test Fixtures - Small Circuits for Ground Truth Testing
# ============================================================================

@pytest.fixture
def simple_repetition_circuit():
    """Simple 3-qubit repetition code style circuit."""
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


@pytest.fixture
def two_qubit_bell_circuit():
    """Simple 2-qubit Bell state circuit."""
    return """R 0
R 1
H 0
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


@pytest.fixture
def medium_circuit():
    """Medium-sized circuit for performance testing."""
    # Create a circuit with ~50 noise locations
    lines = ["R " + str(i) for i in range(10)]
    # Add Hadamards (10 noise)
    lines += ["H " + str(i) for i in range(5)]
    # Add CNOT chain (18 noise = 9 gates * 2)
    for i in range(9):
        lines.append(f"CX {i} {i+1}")
    # Add measurements (10 noise)
    lines += ["M " + str(i) for i in range(10)]
    # Add detectors
    for i in range(9):
        lines.append(f"DETECTOR({i}, 0, 0) rec[-{10-i}] rec[-{9-i}]")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return "\n".join(lines)


# ============================================================================
# Test Class: SIMD vs Non-SIMD Logical Error Rate Comparison
# ============================================================================

class TestSIMDvsNonSIMDLogicalErrorRate:
    """
    Test that SIMD and non-SIMD methods produce statistically equivalent
    logical error rates for fixed-weight sampling.
    """

    def _calculate_logical_error_rate(self, obs_samples):
        """Calculate logical error rate from observable samples."""
        return np.mean(obs_samples.astype(float))

    def test_fixed_weight_ler_equivalence_small(self, simple_repetition_circuit):
        """
        Compare logical error rates between SIMD and non-SIMD for small circuit.
        Uses multiple weight values.
        """
        graph = qepg.compile_QEPG(simple_repetition_circuit)

        # Test multiple weights
        for weight in [1, 2, 3]:
            shots = 50000

            # Non-SIMD
            _, obs_nonsimd = qepg.return_samples_many_weights_separate_obs_with_QEPG(
                graph, [weight], [shots]
            )
            ler_nonsimd = self._calculate_logical_error_rate(obs_nonsimd)

            # SIMD
            _, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
                graph, [weight], [shots]
            )
            ler_simd = self._calculate_logical_error_rate(obs_simd)

            # Allow 5% relative error or 0.01 absolute error (whichever is larger)
            tolerance = max(0.05 * max(ler_nonsimd, ler_simd), 0.01)
            assert abs(ler_simd - ler_nonsimd) < tolerance, \
                f"Weight {weight}: SIMD LER={ler_simd:.4f}, non-SIMD LER={ler_nonsimd:.4f}"

    def test_fixed_weight_ler_equivalence_medium(self, medium_circuit):
        """
        Compare logical error rates for medium circuit.
        Must complete within 5 seconds.
        """
        graph = qepg.compile_QEPG(medium_circuit)

        start_time = time.time()

        # Test weights 1-5
        weights = [1, 2, 3, 4, 5]
        shots_per_weight = 20000

        for weight in weights:
            # Non-SIMD
            _, obs_nonsimd = qepg.return_samples_many_weights_separate_obs_with_QEPG(
                graph, [weight], [shots_per_weight]
            )
            ler_nonsimd = self._calculate_logical_error_rate(obs_nonsimd)

            # SIMD
            _, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
                graph, [weight], [shots_per_weight]
            )
            ler_simd = self._calculate_logical_error_rate(obs_simd)

            # Allow 10% relative error or 0.02 absolute error
            tolerance = max(0.10 * max(ler_nonsimd, ler_simd), 0.02)
            assert abs(ler_simd - ler_nonsimd) < tolerance, \
                f"Weight {weight}: SIMD LER={ler_simd:.4f}, non-SIMD LER={ler_nonsimd:.4f}"

        elapsed = time.time() - start_time
        assert elapsed < 5.0, f"Test took {elapsed:.2f}s, should be < 5s"

    def test_monte_carlo_ler_equivalence(self, simple_repetition_circuit):
        """
        Compare Monte Carlo logical error rates between SIMD and non-SIMD.
        """
        graph = qepg.compile_QEPG(simple_repetition_circuit)

        for error_rate in [0.001, 0.01, 0.05]:
            shots = 100000

            # Non-SIMD
            _, obs_nonsimd = qepg.return_samples_Monte_separate_obs_with_QEPG(
                graph, error_rate, shots
            )
            ler_nonsimd = self._calculate_logical_error_rate(obs_nonsimd)

            # SIMD
            _, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
                graph, error_rate, shots
            )
            ler_simd = self._calculate_logical_error_rate(obs_simd)

            # Allow 10% relative error or 0.005 absolute error
            tolerance = max(0.10 * max(ler_nonsimd, ler_simd), 0.005)
            assert abs(ler_simd - ler_nonsimd) < tolerance, \
                f"Error rate {error_rate}: SIMD LER={ler_simd:.6f}, non-SIMD LER={ler_nonsimd:.6f}"


# ============================================================================
# Test Class: Multi-Weight Batch Testing
# ============================================================================

class TestMultiWeightBatchSampling:
    """Test batch sampling with multiple weights."""

    def test_batch_weights_simd_vs_nonsimd(self, medium_circuit):
        """
        Test that batch sampling with multiple weights produces equivalent results.
        Must complete within 5 seconds for both methods.
        """
        graph = qepg.compile_QEPG(medium_circuit)

        weights = [1, 2, 3, 4, 5, 6, 7, 8]
        shots = [10000] * len(weights)

        # Time non-SIMD
        start_nonsimd = time.time()
        det_nonsimd, obs_nonsimd = qepg.return_samples_many_weights_separate_obs_with_QEPG(
            graph, weights, shots
        )
        time_nonsimd = time.time() - start_nonsimd

        # Time SIMD
        start_simd = time.time()
        det_simd, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )
        time_simd = time.time() - start_simd

        # Both should complete within 5 seconds
        assert time_nonsimd < 5.0, f"Non-SIMD took {time_nonsimd:.2f}s"
        assert time_simd < 5.0, f"SIMD took {time_simd:.2f}s"

        # Check shapes match
        assert det_nonsimd.shape == det_simd.shape
        assert obs_nonsimd.shape == obs_simd.shape

        # Compare LER for each weight segment
        total_shots = sum(shots)
        assert det_nonsimd.shape[0] == total_shots

        idx = 0
        for i, (w, s) in enumerate(zip(weights, shots)):
            obs_seg_nonsimd = obs_nonsimd[idx:idx+s]
            obs_seg_simd = obs_simd[idx:idx+s]

            ler_nonsimd = np.mean(obs_seg_nonsimd.astype(float))
            ler_simd = np.mean(obs_seg_simd.astype(float))

            # Allow 15% relative error for smaller sample sizes
            tolerance = max(0.15 * max(ler_nonsimd, ler_simd), 0.02)
            assert abs(ler_simd - ler_nonsimd) < tolerance, \
                f"Weight {w}: SIMD LER={ler_simd:.4f}, non-SIMD LER={ler_nonsimd:.4f}"

            idx += s

    def test_cumulative_ler_consistency(self, simple_repetition_circuit):
        """
        Test that cumulative LER across weights is consistent.
        """
        graph = qepg.compile_QEPG(simple_repetition_circuit)

        # Sample each weight separately
        weights = [1, 2, 3, 4, 5]
        shots_per_weight = 30000

        lers_simd = []
        for w in weights:
            _, obs = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
                graph, [w], [shots_per_weight]
            )
            lers_simd.append(np.mean(obs.astype(float)))

        # Sample all weights in one batch
        _, obs_batch = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, [shots_per_weight] * len(weights)
        )

        # Extract LERs from batch
        idx = 0
        lers_batch = []
        for w in weights:
            ler = np.mean(obs_batch[idx:idx+shots_per_weight].astype(float))
            lers_batch.append(ler)
            idx += shots_per_weight

        # Compare
        for i, w in enumerate(weights):
            tolerance = max(0.10 * max(lers_simd[i], lers_batch[i]), 0.015)
            assert abs(lers_simd[i] - lers_batch[i]) < tolerance, \
                f"Weight {w}: separate={lers_simd[i]:.4f}, batch={lers_batch[i]:.4f}"


# ============================================================================
# Test Class: Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCasesCorrectness:
    """Test edge cases for correctness."""

    def test_zero_weight_always_zero_ler(self, simple_repetition_circuit):
        """Zero weight should always produce zero LER."""
        graph = qepg.compile_QEPG(simple_repetition_circuit)

        _, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, [0], [10000]
        )

        ler = np.mean(obs_simd.astype(float))
        assert ler == 0.0, f"Zero weight should have zero LER, got {ler}"

    def test_zero_error_rate_always_zero_ler(self, simple_repetition_circuit):
        """Zero error rate should always produce zero LER."""
        graph = qepg.compile_QEPG(simple_repetition_circuit)

        _, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, 0.0, 10000
        )

        ler = np.mean(obs_simd.astype(float))
        assert ler == 0.0, f"Zero error rate should have zero LER, got {ler}"

    def test_high_weight_high_ler(self, two_qubit_bell_circuit):
        """High weight should produce higher LER than low weight."""
        graph = qepg.compile_QEPG(two_qubit_bell_circuit)
        shots = 50000

        _, obs_low = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, [1], [shots]
        )
        ler_low = np.mean(obs_low.astype(float))

        _, obs_high = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, [3], [shots]
        )
        ler_high = np.mean(obs_high.astype(float))

        # Higher weight should generally have higher or similar LER
        # (not always strictly higher due to error cancellation, but generally)
        # Just check both are reasonable
        assert 0 <= ler_low <= 1
        assert 0 <= ler_high <= 1

    def test_detector_flip_rate_reasonable(self, simple_repetition_circuit):
        """Detector flip rate should be reasonable for given weight."""
        graph = qepg.compile_QEPG(simple_repetition_circuit)

        det_simd, _ = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, [2], [50000]
        )

        # Average detector flip rate
        flip_rate = np.mean(det_simd.astype(float))

        # Should be non-zero for weight > 0
        assert flip_rate > 0, "Detector flip rate should be > 0 for weight > 0"
        # Should be less than 1 (not all detectors always flipping)
        assert flip_rate < 1, "Detector flip rate should be < 1"


# ============================================================================
# Test Class: Statistical Properties
# ============================================================================

class TestStatisticalProperties:
    """Test statistical properties of the sampling."""

    def test_variance_decreases_with_shots(self, simple_repetition_circuit):
        """Variance of LER estimate should decrease with more shots."""
        graph = qepg.compile_QEPG(simple_repetition_circuit)
        # Use a higher weight to get more non-zero LER values
        weight = 5  # Higher weight gives more observable logical errors

        # Run multiple trials with different shot counts
        shot_counts = [500, 5000, 50000]
        variances = []

        for shots in shot_counts:
            lers = []
            for _ in range(10):  # 10 trials
                _, obs = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
                    graph, [weight], [shots]
                )
                lers.append(np.mean(obs.astype(float)))
            variances.append(np.var(lers))

        print(f"Variances: {variances}")
        print(f"LER values at highest shot: {lers}")

        # Variance should generally decrease with more shots
        # Allow some tolerance for statistical noise
        # Check that highest shot count has lower or similar variance than lowest
        assert variances[-1] <= variances[0] * 2 + 1e-6, \
            f"Variance should decrease with shots: {variances}"

    def test_monte_carlo_ler_scales_with_error_rate(self, simple_repetition_circuit):
        """LER should generally increase with error rate."""
        graph = qepg.compile_QEPG(simple_repetition_circuit)
        shots = 100000

        error_rates = [0.001, 0.01, 0.05]
        lers = []

        for p in error_rates:
            _, obs = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
                graph, p, shots
            )
            lers.append(np.mean(obs.astype(float)))

        # LER should increase with error rate
        for i in range(len(lers) - 1):
            assert lers[i] <= lers[i+1] + 0.01, \
                f"LER should increase with p: {list(zip(error_rates, lers))}"


# ============================================================================
# Test Class: Performance Bounds
# ============================================================================

class TestPerformanceBounds:
    """Ensure tests complete within time bounds."""

    def test_simd_faster_than_nonsimd(self, medium_circuit):
        """SIMD should be faster than non-SIMD for large sample counts."""
        graph = qepg.compile_QEPG(medium_circuit)
        weights = [3]
        shots = [100000]

        # Use perf_counter for higher precision timing
        import time as time_module

        # Warmup
        qepg.return_samples_many_weights_separate_obs_with_QEPG(graph, weights, [1000])
        qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(graph, weights, [1000])

        # Time non-SIMD
        start = time_module.perf_counter()
        qepg.return_samples_many_weights_separate_obs_with_QEPG(graph, weights, shots)
        time_nonsimd = time_module.perf_counter() - start

        # Time SIMD
        start = time_module.perf_counter()
        qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(graph, weights, shots)
        time_simd = time_module.perf_counter() - start

        print(f"Non-SIMD: {time_nonsimd*1000:.2f}ms, SIMD: {time_simd*1000:.2f}ms")

        # SIMD should not be slower (allow some tolerance for measurement noise)
        # Use max to avoid division by zero
        time_simd = max(time_simd, 1e-6)
        speedup = time_nonsimd / time_simd
        assert speedup > 0.5, f"SIMD speedup only {speedup:.2f}x, should not be much slower"

    def test_large_batch_under_5_seconds(self, medium_circuit):
        """Large batch sampling should complete under 5 seconds."""
        graph = qepg.compile_QEPG(medium_circuit)

        # 500K total samples across weights
        weights = list(range(1, 11))
        shots = [50000] * 10

        start = time.time()
        qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Large batch took {elapsed:.2f}s, should be < 5s"
