"""
Tests that verify SIMD sampling correctness.

The primary validation is comparing SIMD vs non-SIMD implementations - they use
the exact same QEPG graph and should produce statistically equivalent results.

NOTE: Symbolic ground truth tests are marked as slow since they require matching
two different implementations (Python vs C++) of noise injection.
"""
import pytest
import numpy as np
from pathlib import Path
import time

import scalerqec.qepg as qepg

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_subspace_ler_from_samples(obs_samples, n_trials=1):
    """
    Estimate subspace LER (P(logical error | weight)) from samples.

    For fixed-weight sampling, this is simply the fraction of samples
    with observable=True.
    """
    return np.mean(obs_samples.astype(float))


def calculate_confidence_interval(ler_estimate, n_samples, confidence=0.99):
    """
    Calculate confidence interval for LER estimate using normal approximation.

    For large n, the sample proportion follows approximately:
    p_hat ~ N(p, p*(1-p)/n)

    Returns the margin of error for the given confidence level.
    """
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)
    # Use conservative estimate (p=0.5 gives maximum variance)
    p_estimate = max(min(ler_estimate, 0.5), 0.001)  # Clamp to avoid edge cases
    std_error = np.sqrt(p_estimate * (1 - p_estimate) / n_samples)
    return z * std_error


# ============================================================================
# Test Class: SIMD vs Non-SIMD Consistency (Primary Correctness Tests)
# ============================================================================

class TestSIMDNonSIMDConsistency:
    """
    Test that SIMD and non-SIMD produce statistically equivalent results.

    These are the primary correctness tests for the SIMD implementation.
    Both methods use the exact same compiled QEPG graph, so they should
    produce identical distributions (up to statistical noise).
    """

    def test_weight_sampling_consistency_small(self):
        """
        Test SIMD vs non-SIMD for small repetition code with moderate shots.
        Should complete in < 2 seconds.
        """
        stim_str = """R 0
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
        graph = qepg.compile_QEPG(stim_str)

        start_time = time.perf_counter()

        for weight in [1, 2, 3, 4]:
            shots = 50000  # Reduced for faster testing

            # Non-SIMD
            _, obs_nonsimd = qepg.return_samples_many_weights_separate_obs_with_QEPG(
                graph, [weight], [shots]
            )
            ler_nonsimd = np.mean(obs_nonsimd.astype(float))

            # SIMD
            _, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
                graph, [weight], [shots]
            )
            ler_simd = np.mean(obs_simd.astype(float))

            # Allow 5% relative error or 0.02 absolute (wider for fewer shots)
            tolerance = max(0.05 * max(ler_nonsimd, ler_simd), 0.02)

            print(f"Weight {weight}: non-SIMD={ler_nonsimd:.4f}, SIMD={ler_simd:.4f}")

            assert abs(ler_simd - ler_nonsimd) < tolerance, \
                f"Weight {weight}: SIMD={ler_simd:.4f} vs non-SIMD={ler_nonsimd:.4f}"

        elapsed = time.perf_counter() - start_time
        print(f"Total time: {elapsed:.2f}s")
        assert elapsed < 5.0, f"Test took {elapsed:.2f}s, should be < 5s"

    def test_monte_carlo_consistency(self):
        """
        Test that SIMD and non-SIMD Monte Carlo produce the same LER.
        Should complete in < 2 seconds.
        """
        stim_str = """R 0
R 1
H 0
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        start_time = time.perf_counter()

        for error_rate in [0.001, 0.01, 0.05]:
            shots = 100000  # Reduced for faster testing

            # Non-SIMD
            _, obs_nonsimd = qepg.return_samples_Monte_separate_obs_with_QEPG(
                graph, error_rate, shots
            )
            ler_nonsimd = np.mean(obs_nonsimd.astype(float))

            # SIMD
            _, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
                graph, error_rate, shots
            )
            ler_simd = np.mean(obs_simd.astype(float))

            # Allow 15% relative error or 0.01 absolute
            tolerance = max(0.15 * max(ler_nonsimd, ler_simd), 0.01)

            print(f"p={error_rate}: non-SIMD={ler_nonsimd:.6f}, SIMD={ler_simd:.6f}")

            assert abs(ler_simd - ler_nonsimd) < tolerance, \
                f"p={error_rate}: SIMD={ler_simd:.6f} vs non-SIMD={ler_nonsimd:.6f}"

        elapsed = time.perf_counter() - start_time
        print(f"Total time: {elapsed:.2f}s")
        assert elapsed < 5.0, f"Test took {elapsed:.2f}s, should be < 5s"

    def test_detector_pattern_consistency(self):
        """
        Test that detector flip patterns are consistent between SIMD and non-SIMD.
        Should complete in < 1 second.
        """
        stim_str = """R 0
R 1
R 2
CX 0 1
CX 1 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3] rec[-2]
DETECTOR(1, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)
        weight = 2
        shots = 50000

        start_time = time.perf_counter()

        # Non-SIMD
        det_nonsimd, _ = qepg.return_samples_many_weights_separate_obs_with_QEPG(
            graph, [weight], [shots]
        )

        # SIMD
        det_simd, _ = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, [weight], [shots]
        )

        # Compare detector flip rates for each detector
        n_det = det_nonsimd.shape[1]
        for d in range(n_det):
            rate_nonsimd = np.mean(det_nonsimd[:, d].astype(float))
            rate_simd = np.mean(det_simd[:, d].astype(float))

            tolerance = max(0.08 * max(rate_nonsimd, rate_simd), 0.02)

            print(f"Detector {d}: non-SIMD={rate_nonsimd:.4f}, SIMD={rate_simd:.4f}")

            assert abs(rate_simd - rate_nonsimd) < tolerance, \
                f"Detector {d}: SIMD={rate_simd:.4f} vs non-SIMD={rate_nonsimd:.4f}"

        elapsed = time.perf_counter() - start_time
        print(f"Total time: {elapsed:.2f}s")
        assert elapsed < 5.0, f"Test took {elapsed:.2f}s, should be < 5s"


# ============================================================================
# Test Class: Multiple Weight Batch Consistency
# ============================================================================

class TestMultiWeightBatchConsistency:
    """
    Test batch sampling with multiple weights simultaneously.
    """

    def test_batch_vs_individual_weights(self):
        """
        Test that batch sampling produces same results as individual weight sampling.
        Should complete in < 2 seconds.
        """
        stim_str = """R 0
R 1
R 2
CX 0 1
CX 1 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3] rec[-2]
DETECTOR(1, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        weights = [1, 2, 3]
        shots_per_weight = [20000, 20000, 20000]

        start_time = time.perf_counter()

        # Batch SIMD
        _, obs_batch_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots_per_weight
        )

        # Individual SIMD
        total_shots = sum(shots_per_weight)
        individual_lers = []
        offset = 0
        for w, s in zip(weights, shots_per_weight):
            _, obs_single = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
                graph, [w], [s]
            )
            individual_lers.append(np.mean(obs_single.astype(float)))

        # Compare batch results at different offsets
        offset = 0
        for i, (w, s) in enumerate(zip(weights, shots_per_weight)):
            batch_ler = np.mean(obs_batch_simd[offset:offset+s].astype(float))
            individual_ler = individual_lers[i]

            # Statistical tolerance
            tolerance = max(0.1 * max(batch_ler, individual_ler), 0.02)

            print(f"Weight {w}: batch={batch_ler:.4f}, individual={individual_ler:.4f}")

            assert abs(batch_ler - individual_ler) < tolerance, \
                f"Weight {w}: batch={batch_ler:.4f} vs individual={individual_ler:.4f}"

            offset += s

        elapsed = time.perf_counter() - start_time
        print(f"Total time: {elapsed:.2f}s")
        assert elapsed < 5.0, f"Test took {elapsed:.2f}s, should be < 5s"


# ============================================================================
# Test Class: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_weight_gives_zero_errors(self):
        """Test that weight=0 sampling returns no logical errors."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Weight 0 means no errors, so observable should always be 0
        # (assuming ideal circuit produces 0)
        _, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, [0], [1000]
        )

        # All samples should be False (no logical error when no physical errors)
        assert np.mean(obs_simd) == 0.0, \
            f"Weight=0 should give LER=0, got {np.mean(obs_simd)}"

    def test_very_low_error_rate_monte_carlo(self):
        """Test Monte Carlo at very low error rate (uses geometric skipping)."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Very low error rate tests the geometric skipping path
        _, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, 0.0001, 50000
        )
        _, obs_nonsimd = qepg.return_samples_Monte_separate_obs_with_QEPG(
            graph, 0.0001, 50000
        )

        ler_simd = np.mean(obs_simd.astype(float))
        ler_nonsimd = np.mean(obs_nonsimd.astype(float))

        print(f"p=0.0001: SIMD={ler_simd:.6f}, non-SIMD={ler_nonsimd:.6f}")

        # At very low error rates, both should be very close to 0
        # Allow larger relative tolerance due to sparse sampling
        assert ler_simd < 0.01, f"LER at p=0.0001 should be very low, got {ler_simd}"
        assert ler_nonsimd < 0.01, f"LER at p=0.0001 should be very low, got {ler_nonsimd}"

    def test_high_error_rate_monte_carlo(self):
        """Test Monte Carlo at high error rate."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # High error rate
        _, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, 0.2, 30000
        )
        _, obs_nonsimd = qepg.return_samples_Monte_separate_obs_with_QEPG(
            graph, 0.2, 30000
        )

        ler_simd = np.mean(obs_simd.astype(float))
        ler_nonsimd = np.mean(obs_nonsimd.astype(float))

        print(f"p=0.2: SIMD={ler_simd:.4f}, non-SIMD={ler_nonsimd:.4f}")

        # Should be similar
        tolerance = max(0.1 * max(ler_simd, ler_nonsimd), 0.02)
        assert abs(ler_simd - ler_nonsimd) < tolerance, \
            f"p=0.2: SIMD={ler_simd:.4f} vs non-SIMD={ler_nonsimd:.4f}"


# ============================================================================
# Test Class: Quick Validation (for CI)
# ============================================================================

class TestQuickValidation:
    """Quick tests that run fast for CI."""

    def test_simd_nonsimd_quick_comparison(self):
        """Quick comparison that runs in < 1 second."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        # Quick test with fewer shots
        _, obs_nonsimd = qepg.return_samples_many_weights_separate_obs_with_QEPG(
            graph, [1, 2], [10000, 10000]
        )
        _, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, [1, 2], [10000, 10000]
        )

        # Just check shapes match and values are reasonable
        assert obs_nonsimd.shape == obs_simd.shape
        assert 0 <= np.mean(obs_nonsimd) <= 1
        assert 0 <= np.mean(obs_simd) <= 1

    def test_monte_carlo_quick_comparison(self):
        """Quick Monte Carlo comparison."""
        stim_str = """R 0
R 1
CX 0 1
M 0
M 1
DETECTOR(0, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        _, obs_nonsimd = qepg.return_samples_Monte_separate_obs_with_QEPG(
            graph, 0.01, 10000
        )
        _, obs_simd = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
            graph, 0.01, 10000
        )

        assert obs_nonsimd.shape == obs_simd.shape
        assert 0 <= np.mean(obs_nonsimd) <= 1
        assert 0 <= np.mean(obs_simd) <= 1

    def test_output_shapes_correct(self):
        """Verify output shapes are correct."""
        stim_str = """R 0
R 1
R 2
CX 0 1
CX 1 2
M 0
M 1
M 2
DETECTOR(0, 0, 0) rec[-3] rec[-2]
DETECTOR(1, 0, 0) rec[-2] rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""
        graph = qepg.compile_QEPG(stim_str)

        weights = [1, 2, 3]
        shots = [100, 200, 300]

        det_simd, obs_simd = qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(
            graph, weights, shots
        )

        total_shots = sum(shots)
        n_det = 2  # 2 detectors in circuit

        assert det_simd.shape == (total_shots, n_det), \
            f"Expected det shape ({total_shots}, {n_det}), got {det_simd.shape}"
        assert obs_simd.shape == (total_shots,), \
            f"Expected obs shape ({total_shots},), got {obs_simd.shape}"
