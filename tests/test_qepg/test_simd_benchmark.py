"""
Benchmark tests for SIMD-accelerated sampling functions.

These tests compare performance between SIMD and non-SIMD implementations.
Run with: pytest tests/test_qepg/test_simd_benchmark.py -v -s
"""
import time
import pytest
import numpy as np

import scalerqec.qepg as qepg


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def small_circuit():
    """Small circuit for quick benchmarks."""
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
def medium_circuit():
    """Medium-sized repetition code circuit."""
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
def large_circuit():
    """Large circuit for meaningful benchmarks."""
    # Create a 20-qubit circuit with many gates
    lines = ["R " + str(i) for i in range(20)]
    # Add Hadamards
    lines += ["H " + str(i) for i in range(10)]
    # Add CNOT chain
    for i in range(19):
        lines.append(f"CX {i} {i+1}")
    # Add more CNOTs for complexity
    for i in range(0, 18, 2):
        lines.append(f"CX {i} {i+2}")
    # Add measurements
    lines += ["M " + str(i) for i in range(20)]
    # Add detectors
    lines += [f"DETECTOR({i}, 0, 0) rec[-{20-i}]" for i in range(20)]
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return "\n".join(lines)


# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_function(func, *args, warmup=1, iterations=3, **kwargs):
    """
    Benchmark a function with warmup and multiple iterations.
    Returns (mean_time, std_time, min_time).
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times), np.min(times)


# ============================================================================
# Benchmark Tests: Weight-based Sampling
# ============================================================================

class TestWeightBasedBenchmark:
    """Benchmark tests for weight-based sampling."""

    @pytest.mark.benchmark
    def test_benchmark_small_circuit(self, small_circuit):
        """Benchmark SIMD vs non-SIMD on small circuit."""
        graph = qepg.compile_QEPG(small_circuit)
        weights = [2]
        shots = [10000]

        # Benchmark non-SIMD
        mean_nonsimd, std_nonsimd, min_nonsimd = benchmark_function(
            qepg.return_samples_many_weights_separate_obs_with_QEPG,
            graph, weights, shots
        )

        # Benchmark SIMD
        mean_simd, std_simd, min_simd = benchmark_function(
            qepg.return_samples_many_weights_separate_obs_with_QEPG_simd,
            graph, weights, shots
        )

        speedup = mean_nonsimd / mean_simd if mean_simd > 0 else float('inf')

        print(f"\n=== Small Circuit Benchmark (10000 shots) ===")
        print(f"Non-SIMD: {mean_nonsimd*1000:.2f}ms (+/- {std_nonsimd*1000:.2f}ms)")
        print(f"SIMD:     {mean_simd*1000:.2f}ms (+/- {std_simd*1000:.2f}ms)")
        print(f"Speedup:  {speedup:.2f}x")

        # SIMD should be at least as fast (allow 20% slower due to overhead on small circuits)
        assert mean_simd < mean_nonsimd * 1.2, "SIMD should not be significantly slower"

    @pytest.mark.benchmark
    def test_benchmark_medium_circuit(self, medium_circuit):
        """Benchmark SIMD vs non-SIMD on medium circuit."""
        graph = qepg.compile_QEPG(medium_circuit)
        weights = [3]
        shots = [50000]

        # Benchmark non-SIMD
        mean_nonsimd, std_nonsimd, min_nonsimd = benchmark_function(
            qepg.return_samples_many_weights_separate_obs_with_QEPG,
            graph, weights, shots
        )

        # Benchmark SIMD
        mean_simd, std_simd, min_simd = benchmark_function(
            qepg.return_samples_many_weights_separate_obs_with_QEPG_simd,
            graph, weights, shots
        )

        speedup = mean_nonsimd / mean_simd if mean_simd > 0 else float('inf')

        print(f"\n=== Medium Circuit Benchmark (50000 shots) ===")
        print(f"Non-SIMD: {mean_nonsimd*1000:.2f}ms (+/- {std_nonsimd*1000:.2f}ms)")
        print(f"SIMD:     {mean_simd*1000:.2f}ms (+/- {std_simd*1000:.2f}ms)")
        print(f"Speedup:  {speedup:.2f}x")

    @pytest.mark.benchmark
    def test_benchmark_large_circuit(self, large_circuit):
        """Benchmark SIMD vs non-SIMD on large circuit."""
        graph = qepg.compile_QEPG(large_circuit)
        weights = [5]
        shots = [100000]

        # Benchmark non-SIMD
        mean_nonsimd, std_nonsimd, min_nonsimd = benchmark_function(
            qepg.return_samples_many_weights_separate_obs_with_QEPG,
            graph, weights, shots
        )

        # Benchmark SIMD
        mean_simd, std_simd, min_simd = benchmark_function(
            qepg.return_samples_many_weights_separate_obs_with_QEPG_simd,
            graph, weights, shots
        )

        speedup = mean_nonsimd / mean_simd if mean_simd > 0 else float('inf')

        print(f"\n=== Large Circuit Benchmark (100000 shots) ===")
        print(f"Non-SIMD: {mean_nonsimd*1000:.2f}ms (+/- {std_nonsimd*1000:.2f}ms)")
        print(f"SIMD:     {mean_simd*1000:.2f}ms (+/- {std_simd*1000:.2f}ms)")
        print(f"Speedup:  {speedup:.2f}x")

    @pytest.mark.benchmark
    def test_benchmark_scaling(self, large_circuit):
        """Benchmark how performance scales with shot count."""
        graph = qepg.compile_QEPG(large_circuit)
        weight = 3

        print(f"\n=== Scaling Benchmark (large circuit, weight={weight}) ===")
        print(f"{'Shots':>10} | {'Non-SIMD (ms)':>15} | {'SIMD (ms)':>15} | {'Speedup':>10}")
        print("-" * 60)

        for shot_count in [1000, 10000, 50000, 100000]:
            weights = [weight]
            shots = [shot_count]

            mean_nonsimd, _, _ = benchmark_function(
                qepg.return_samples_many_weights_separate_obs_with_QEPG,
                graph, weights, shots,
                warmup=1, iterations=2
            )

            mean_simd, _, _ = benchmark_function(
                qepg.return_samples_many_weights_separate_obs_with_QEPG_simd,
                graph, weights, shots,
                warmup=1, iterations=2
            )

            speedup = mean_nonsimd / mean_simd if mean_simd > 0 else float('inf')
            print(f"{shot_count:>10} | {mean_nonsimd*1000:>15.2f} | {mean_simd*1000:>15.2f} | {speedup:>10.2f}x")


# ============================================================================
# Benchmark Tests: Monte Carlo Sampling
# ============================================================================

class TestMonteCarloBenchmark:
    """Benchmark tests for Monte Carlo sampling."""

    @pytest.mark.benchmark
    def test_benchmark_monte_carlo(self, large_circuit):
        """Benchmark Monte Carlo SIMD vs non-SIMD."""
        graph = qepg.compile_QEPG(large_circuit)
        error_rate = 0.01
        shots = 100000

        # Benchmark non-SIMD
        mean_nonsimd, std_nonsimd, _ = benchmark_function(
            qepg.return_samples_Monte_separate_obs_with_QEPG,
            graph, error_rate, shots
        )

        # Benchmark SIMD
        mean_simd, std_simd, _ = benchmark_function(
            qepg.return_samples_Monte_separate_obs_with_QEPG_simd,
            graph, error_rate, shots
        )

        speedup = mean_nonsimd / mean_simd if mean_simd > 0 else float('inf')

        print(f"\n=== Monte Carlo Benchmark (100000 shots, p=0.01) ===")
        print(f"Non-SIMD: {mean_nonsimd*1000:.2f}ms (+/- {std_nonsimd*1000:.2f}ms)")
        print(f"SIMD:     {mean_simd*1000:.2f}ms (+/- {std_simd*1000:.2f}ms)")
        print(f"Speedup:  {speedup:.2f}x")

    @pytest.mark.benchmark
    def test_benchmark_monte_carlo_error_rates(self, large_circuit):
        """Benchmark Monte Carlo at different error rates."""
        graph = qepg.compile_QEPG(large_circuit)
        shots = 50000

        print(f"\n=== Monte Carlo Error Rate Scaling (50000 shots) ===")
        print(f"{'Error Rate':>12} | {'Non-SIMD (ms)':>15} | {'SIMD (ms)':>15} | {'Speedup':>10}")
        print("-" * 65)

        for error_rate in [0.001, 0.01, 0.05, 0.1]:
            mean_nonsimd, _, _ = benchmark_function(
                qepg.return_samples_Monte_separate_obs_with_QEPG,
                graph, error_rate, shots,
                warmup=1, iterations=2
            )

            mean_simd, _, _ = benchmark_function(
                qepg.return_samples_Monte_separate_obs_with_QEPG_simd,
                graph, error_rate, shots,
                warmup=1, iterations=2
            )

            speedup = mean_nonsimd / mean_simd if mean_simd > 0 else float('inf')
            print(f"{error_rate:>12.3f} | {mean_nonsimd*1000:>15.2f} | {mean_simd*1000:>15.2f} | {speedup:>10.2f}x")


# ============================================================================
# Throughput Tests
# ============================================================================

class TestThroughput:
    """Test sampling throughput (samples per second)."""

    @pytest.mark.benchmark
    def test_throughput(self, large_circuit):
        """Measure throughput in samples per second."""
        graph = qepg.compile_QEPG(large_circuit)
        weights = [3]
        shots = [1000000]  # 1 million samples

        print(f"\n=== Throughput Test (1M samples) ===")

        # Measure non-SIMD throughput
        start = time.perf_counter()
        qepg.return_samples_many_weights_separate_obs_with_QEPG(graph, weights, shots)
        nonsimd_time = time.perf_counter() - start
        nonsimd_throughput = shots[0] / nonsimd_time

        # Measure SIMD throughput
        start = time.perf_counter()
        qepg.return_samples_many_weights_separate_obs_with_QEPG_simd(graph, weights, shots)
        simd_time = time.perf_counter() - start
        simd_throughput = shots[0] / simd_time

        print(f"Non-SIMD: {nonsimd_throughput/1e6:.2f} M samples/sec ({nonsimd_time:.2f}s)")
        print(f"SIMD:     {simd_throughput/1e6:.2f} M samples/sec ({simd_time:.2f}s)")
        print(f"Speedup:  {simd_throughput/nonsimd_throughput:.2f}x")


if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([__file__, "-v", "-s", "-m", "benchmark"])
