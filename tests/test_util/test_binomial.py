"""
Tests for scalerqec.util.binomial module.

Tests the binomial_weight() and subspace_size() functions.
"""
import pytest
import math
from scalerqec.util import binomial_weight, subspace_size


class TestSubspaceSize:
    """Tests for the subspace_size() function."""

    def test_subspace_size_basic(self):
        """Basic combination calculations."""
        # C(5, 0) = 1
        assert subspace_size(5, 0) == 1
        # C(5, 1) = 5
        assert subspace_size(5, 1) == 5
        # C(5, 2) = 10
        assert subspace_size(5, 2) == 10
        # C(5, 5) = 1
        assert subspace_size(5, 5) == 1

    def test_subspace_size_symmetry(self):
        """C(n, k) = C(n, n-k)."""
        n = 10
        for k in range(n + 1):
            assert subspace_size(n, k) == subspace_size(n, n - k)

    def test_subspace_size_pascals_triangle(self):
        """C(n, k) = C(n-1, k-1) + C(n-1, k) for 0 < k < n."""
        n = 8
        for k in range(1, n):
            assert subspace_size(n, k) == subspace_size(n - 1, k - 1) + subspace_size(n - 1, k)

    def test_subspace_size_large_n(self):
        """Test with larger N values."""
        # C(100, 50) is a large number
        assert subspace_size(100, 50) == math.comb(100, 50)
        # C(1000, 3) = 1000 * 999 * 998 / 6
        assert subspace_size(1000, 3) == 166167000

    def test_subspace_size_edge_cases(self):
        """Edge cases for subspace_size."""
        # C(0, 0) = 1
        assert subspace_size(0, 0) == 1
        # C(n, 0) = 1 for any n
        assert subspace_size(100, 0) == 1
        # C(n, n) = 1 for any n
        assert subspace_size(100, 100) == 1


class TestBinomialWeight:
    """Tests for the binomial_weight() function."""

    def test_binomial_weight_small_n_exact(self):
        """For small N, verify against exact calculation."""
        test_cases = [
            (10, 3, 0.2),
            (10, 0, 0.5),
            (10, 10, 0.9),
            (5, 2, 0.3),
            (7, 4, 0.5),
        ]
        for N, W, p in test_cases:
            expected = math.comb(N, W) * (p ** W) * ((1 - p) ** (N - W))
            result = binomial_weight(N, W, p)
            assert abs(result - expected) < 1e-10, \
                f"binomial_weight({N}, {W}, {p}) = {result}, expected {expected}"

    def test_binomial_weight_probabilities_sum_to_one(self):
        """Sum of P(X=k) for k=0 to N should equal 1."""
        test_cases = [
            (10, 0.3),
            (20, 0.5),
            (15, 0.1),
            (50, 0.02),
        ]
        for N, p in test_cases:
            total = sum(binomial_weight(N, k, p) for k in range(N + 1))
            assert abs(total - 1.0) < 1e-10, \
                f"Sum for N={N}, p={p} is {total}, expected 1.0"

    def test_binomial_weight_p_zero(self):
        """When p=0, only W=0 has non-zero probability."""
        N = 10
        p = 0.0
        assert binomial_weight(N, 0, p) == 1.0
        for W in range(1, N + 1):
            assert binomial_weight(N, W, p) == 0.0

    def test_binomial_weight_p_one(self):
        """When p=1, only W=N has non-zero probability."""
        N = 10
        p = 1.0
        assert binomial_weight(N, N, p) == 1.0
        for W in range(0, N):
            assert binomial_weight(N, W, p) == 0.0

    def test_binomial_weight_symmetric_around_half(self):
        """When p=0.5, P(W=k) = P(W=N-k)."""
        N = 10
        p = 0.5
        for k in range(N // 2 + 1):
            left = binomial_weight(N, k, p)
            right = binomial_weight(N, N - k, p)
            assert abs(left - right) < 1e-10, \
                f"P(W={k}) = {left}, P(W={N-k}) = {right}"

    def test_binomial_weight_large_n(self):
        """Test with larger N values (uses scipy internally)."""
        test_cases = [
            (100, 10, 0.1),
            (500, 5, 0.01),
            (1000, 10, 0.01),
        ]
        for N, W, p in test_cases:
            result = binomial_weight(N, W, p)
            # Just verify it returns a valid probability
            assert 0.0 <= result <= 1.0, \
                f"binomial_weight({N}, {W}, {p}) = {result} is not a valid probability"

    def test_binomial_weight_peak_near_expectation(self):
        """The peak of binomial distribution is near N*p."""
        N = 100
        p = 0.3
        expected_peak = int(N * p)  # 30

        # Check that peak is near expected value
        peak_prob = binomial_weight(N, expected_peak, p)
        for offset in [-10, -5, 5, 10]:
            nearby_prob = binomial_weight(N, expected_peak + offset, p)
            # Peak should be higher or close to nearby values
            # (allowing for discrete distribution behavior)
            assert peak_prob >= nearby_prob * 0.5, \
                f"Peak at {expected_peak} should be near maximum"

    def test_binomial_weight_monotonicity_tails(self):
        """Probabilities should decrease in the tails."""
        N = 50
        p = 0.3
        expectation = N * p  # 15

        # In the left tail (below expectation), should be increasing
        for k in range(1, int(expectation) - 5):
            assert binomial_weight(N, k, p) <= binomial_weight(N, k + 1, p) * 1.5

        # In the right tail (above expectation), should be decreasing
        for k in range(int(expectation) + 5, N - 1):
            assert binomial_weight(N, k, p) >= binomial_weight(N, k + 1, p) * 0.5


class TestBinomialWeightEdgeCases:
    """Edge case tests for binomial_weight."""

    def test_binomial_weight_zero_n(self):
        """N=0 case: only W=0 is valid."""
        assert binomial_weight(0, 0, 0.5) == 1.0

    def test_binomial_weight_w_equals_n(self):
        """W=N case: P(W=N) = p^N."""
        N = 5
        p = 0.4
        expected = p ** N
        result = binomial_weight(N, N, p)
        assert abs(result - expected) < 1e-10

    def test_binomial_weight_w_equals_zero(self):
        """W=0 case: P(W=0) = (1-p)^N."""
        N = 5
        p = 0.4
        expected = (1 - p) ** N
        result = binomial_weight(N, 0, p)
        assert abs(result - expected) < 1e-10

    def test_binomial_weight_very_small_p(self):
        """Very small p values."""
        N = 1000
        p = 0.0001
        # P(W=0) should be approximately e^(-N*p) for small p
        result = binomial_weight(N, 0, p)
        poisson_approx = math.exp(-N * p)
        # Should be reasonably close to Poisson approximation
        assert abs(result - poisson_approx) < 0.01

    def test_binomial_weight_consistency_with_subspace_size(self):
        """binomial_weight uses subspace_size (comb) internally."""
        N = 10
        W = 4
        p = 0.3
        # binomial_weight(N, W, p) = C(N, W) * p^W * (1-p)^(N-W)
        expected = subspace_size(N, W) * (p ** W) * ((1 - p) ** (N - W))
        result = binomial_weight(N, W, p)
        assert abs(result - expected) < 1e-10
