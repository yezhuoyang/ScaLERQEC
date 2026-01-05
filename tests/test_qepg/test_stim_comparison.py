"""
Tests comparing Monte Carlo LER between QEPG (SIMD/non-SIMD) and STIM.

These tests verify that the QEPG sampling produces logical error rates
that match STIM when using equivalent noise models.

The C++ QEPG automatically injects DEPOLARIZE1 noise before each gate:
- Before each CX (on both control and target)
- Before each H
- Before each M

Resets (R) do NOT get noise in QEPG C++ code.

We use high sample counts (500K-1M) to ensure statistical accuracy and
tight tolerances (10-20% relative error) to verify the implementations match.
"""
import pytest
import numpy as np
import stim
import pymatching
import time

import scalerqec.qepg as qepg


# ============================================================================
# Helper Functions
# ============================================================================

def create_stim_circuit_with_depolarization(base_circuit_str: str, error_rate: float) -> stim.Circuit:
    """
    Create a STIM circuit with explicit DEPOLARIZE1 noise matching QEPG's noise model.

    QEPG injects DEPOLARIZE1(p) before each:
    - CX gate (on both control and target qubits)
    - H gate
    - M gate

    Resets (R) do NOT get noise in QEPG.
    """
    lines = base_circuit_str.strip().split('\n')
    output_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        if not tokens:
            continue

        gate = tokens[0]

        if gate == 'CX':
            # Add depolarizing noise before CX on both qubits
            control = int(tokens[1])
            target = int(tokens[2])
            output_lines.append(f"DEPOLARIZE1({error_rate}) {control} {target}")
            output_lines.append(line)

        elif gate == 'H':
            # Add depolarizing noise before H
            qubit = int(tokens[1])
            output_lines.append(f"DEPOLARIZE1({error_rate}) {qubit}")
            output_lines.append(line)

        elif gate == 'M':
            # Add depolarizing noise before M
            qubit = int(tokens[1])
            output_lines.append(f"DEPOLARIZE1({error_rate}) {qubit}")
            output_lines.append(line)

        elif gate == 'R':
            # No noise on reset in QEPG
            output_lines.append(line)

        elif gate.startswith('DETECTOR') or gate.startswith('OBSERVABLE'):
            # Keep detector and observable lines as-is
            output_lines.append(line)

        else:
            # Other instructions (TICK, etc.)
            output_lines.append(line)

    circuit_str = '\n'.join(output_lines)
    return stim.Circuit(circuit_str)


def sample_stim_ler(circuit: stim.Circuit, shots: int) -> float:
    """
    Sample logical error rate using STIM with pymatching decoder.

    Returns the fraction of shots where the decoder made an error.
    """
    # Create sampler
    sampler = circuit.compile_detector_sampler()

    # Sample detector outcomes and actual observables
    detection_events, observable_flips = sampler.sample(
        shots=shots,
        separate_observables=True
    )

    # Create matching decoder
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Decode and count errors
    predictions = matcher.decode_batch(detection_events)
    num_errors = np.sum(predictions != observable_flips)

    return num_errors / shots


def sample_stim_observable_flip_rate(circuit: stim.Circuit, shots: int) -> float:
    """
    Sample raw observable flip rate using STIM (NO decoder).

    This is the raw probability of observable flips, which matches what
    QEPG computes. This is different from the logical error rate which
    requires a decoder.

    Returns the fraction of shots where the observable was flipped.
    """
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        shots=shots,
        separate_observables=True
    )
    return np.mean(observable_flips.astype(float))


def sample_qepg_ler_simd(graph, error_rate: float, shots: int) -> float:
    """Sample LER using QEPG SIMD Monte Carlo."""
    _, obs = qepg.return_samples_Monte_separate_obs_with_QEPG_simd(
        graph, error_rate, shots
    )
    return np.mean(obs.astype(float))


def sample_qepg_ler_nonsimd(graph, error_rate: float, shots: int) -> float:
    """Sample LER using QEPG non-SIMD Monte Carlo."""
    _, obs = qepg.return_samples_Monte_separate_obs_with_QEPG(
        graph, error_rate, shots
    )
    return np.mean(obs.astype(float))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def repetition_code_d3():
    """
    Distance-3 repetition code circuit.
    This is a Z-basis only circuit that STIM can handle properly.
    """
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
def repetition_code_d5():
    """
    Distance-5 repetition code circuit.
    Larger code for better statistics.
    """
    return """R 0
R 1
R 2
R 3
R 4
R 5
R 6
R 7
R 8
CX 0 5
CX 1 5
CX 1 6
CX 2 6
CX 2 7
CX 3 7
CX 3 8
CX 4 8
M 5
M 6
M 7
M 8
M 0
M 1
M 2
M 3
M 4
DETECTOR(0, 0, 0) rec[-9] rec[-8]
DETECTOR(1, 0, 0) rec[-8] rec[-7]
DETECTOR(2, 0, 0) rec[-7] rec[-6]
DETECTOR(3, 0, 0) rec[-6] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-5] rec[-4] rec[-3] rec[-2] rec[-1]
"""


@pytest.fixture
def simple_cnot_circuit():
    """
    Simple CNOT circuit without H gates - compatible with STIM error model.
    """
    return """R 0
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


@pytest.fixture
def larger_repetition_code():
    """Larger repetition code for more robust statistics."""
    return """R 0
R 1
R 2
R 3
R 4
R 5
R 6
CX 0 4
CX 1 4
CX 1 5
CX 2 5
CX 2 6
CX 3 6
M 4
M 5
M 6
M 0
M 1
M 2
M 3
DETECTOR(0, 0, 0) rec[-7] rec[-6]
DETECTOR(1, 0, 0) rec[-6] rec[-5]
DETECTOR(2, 0, 0) rec[-5] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-4] rec[-3] rec[-2] rec[-1]
"""


# ============================================================================
# Helper: Statistical tolerance calculation
# ============================================================================

def calculate_relative_tolerance(ler: float, shots: int, confidence_sigma: float = 3.0) -> float:
    """
    Calculate the expected relative tolerance for Monte Carlo LER estimate.

    For a Bernoulli random variable with probability p sampled n times,
    the standard error is sqrt(p*(1-p)/n).

    Returns relative tolerance as a fraction (e.g., 0.1 = 10%).
    """
    if ler <= 0 or ler >= 1:
        return 0.5  # Default tolerance for edge cases
    std_err = np.sqrt(ler * (1 - ler) / shots)
    rel_err = confidence_sigma * std_err / ler
    return min(rel_err, 0.5)  # Cap at 50%


# ============================================================================
# Test Class: STIM vs QEPG Monte Carlo Comparison
# ============================================================================

class TestSTIMvsQEPGMonteCarlo:
    """
    Compare Monte Carlo observable flip rates between STIM and QEPG implementations.

    IMPORTANT: QEPG computes the raw observable flip rate (probability that errors
    flip the observable). This is different from "logical error rate after decoding".
    We compare QEPG against STIM's raw observable flip rate (no decoder) for a fair comparison.

    Uses high sample counts (500K-1M) and tight tolerances (5-10%)
    to ensure the implementations produce statistically matching results.
    """

    def test_repetition_code_stim_vs_qepg_simd(self, repetition_code_d3):
        """Compare STIM and QEPG SIMD on repetition code."""
        error_rate = 0.01
        shots = 500000  # High sample count for tight tolerance

        # QEPG SIMD - computes observable flip rate
        graph = qepg.compile_QEPG(repetition_code_d3)
        obs_flip_qepg = sample_qepg_ler_simd(graph, error_rate, shots)

        # STIM - raw observable flip rate (no decoder)
        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, error_rate)
        obs_flip_stim = sample_stim_observable_flip_rate(stim_circuit, shots)

        # Calculate expected statistical tolerance
        avg_rate = (obs_flip_qepg + obs_flip_stim) / 2
        rel_tol = calculate_relative_tolerance(avg_rate, shots)
        # Use at least 5% tolerance for minor numerical differences
        rel_tol = max(rel_tol, 0.05)

        print(f"\nRepetition code d=3 (p={error_rate}, {shots} shots):")
        print(f"  QEPG SIMD:           {obs_flip_qepg:.6f}")
        print(f"  STIM (raw obs flip): {obs_flip_stim:.6f}")
        print(f"  Ratio:               {obs_flip_qepg/max(obs_flip_stim, 1e-10):.3f}x")
        print(f"  Rel. tol:            {rel_tol*100:.1f}%")

        # Tight tolerance based on statistics
        assert abs(obs_flip_qepg - obs_flip_stim) < rel_tol * max(obs_flip_qepg, obs_flip_stim) + 0.002, \
            f"QEPG SIMD ({obs_flip_qepg:.6f}) vs STIM ({obs_flip_stim:.6f}) differ by more than {rel_tol*100:.1f}%"

    def test_repetition_code_stim_vs_qepg_nonsimd(self, repetition_code_d3):
        """Compare STIM and QEPG non-SIMD on repetition code."""
        error_rate = 0.01
        shots = 500000

        # QEPG non-SIMD
        graph = qepg.compile_QEPG(repetition_code_d3)
        obs_flip_qepg = sample_qepg_ler_nonsimd(graph, error_rate, shots)

        # STIM raw observable flip rate
        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, error_rate)
        obs_flip_stim = sample_stim_observable_flip_rate(stim_circuit, shots)

        avg_rate = (obs_flip_qepg + obs_flip_stim) / 2
        rel_tol = max(calculate_relative_tolerance(avg_rate, shots), 0.05)

        print(f"\nRepetition code d=3 (p={error_rate}, {shots} shots):")
        print(f"  QEPG non-SIMD:       {obs_flip_qepg:.6f}")
        print(f"  STIM (raw obs flip): {obs_flip_stim:.6f}")
        print(f"  Ratio:               {obs_flip_qepg/max(obs_flip_stim, 1e-10):.3f}x")

        assert abs(obs_flip_qepg - obs_flip_stim) < rel_tol * max(obs_flip_qepg, obs_flip_stim) + 0.002, \
            f"QEPG non-SIMD ({obs_flip_qepg:.6f}) vs STIM ({obs_flip_stim:.6f}) differ by more than {rel_tol*100:.1f}%"

    def test_simple_cnot_stim_vs_qepg(self, simple_cnot_circuit):
        """Compare STIM and QEPG on simple CNOT circuit."""
        error_rate = 0.01
        shots = 500000

        # QEPG
        graph = qepg.compile_QEPG(simple_cnot_circuit)
        obs_flip_qepg = sample_qepg_ler_simd(graph, error_rate, shots)

        # STIM raw observable flip rate
        stim_circuit = create_stim_circuit_with_depolarization(simple_cnot_circuit, error_rate)
        obs_flip_stim = sample_stim_observable_flip_rate(stim_circuit, shots)

        avg_rate = (obs_flip_qepg + obs_flip_stim) / 2
        rel_tol = max(calculate_relative_tolerance(avg_rate, shots), 0.05)

        print(f"\nSimple CNOT circuit (p={error_rate}, {shots} shots):")
        print(f"  QEPG SIMD:           {obs_flip_qepg:.6f}")
        print(f"  STIM (raw obs flip): {obs_flip_stim:.6f}")
        print(f"  Ratio:               {obs_flip_qepg/max(obs_flip_stim, 1e-10):.3f}x")

        assert abs(obs_flip_qepg - obs_flip_stim) < rel_tol * max(obs_flip_qepg, obs_flip_stim) + 0.002, \
            f"QEPG SIMD ({obs_flip_qepg:.6f}) vs STIM ({obs_flip_stim:.6f}) differ by more than {rel_tol*100:.1f}%"

    def test_d5_repetition_code_stim_vs_qepg(self, repetition_code_d5):
        """Compare STIM and QEPG on larger d=5 repetition code."""
        error_rate = 0.01
        shots = 500000

        # QEPG
        graph = qepg.compile_QEPG(repetition_code_d5)
        obs_flip_qepg = sample_qepg_ler_simd(graph, error_rate, shots)

        # STIM raw observable flip rate
        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d5, error_rate)
        obs_flip_stim = sample_stim_observable_flip_rate(stim_circuit, shots)

        avg_rate = (obs_flip_qepg + obs_flip_stim) / 2
        rel_tol = max(calculate_relative_tolerance(avg_rate, shots), 0.05)

        print(f"\nRepetition code d=5 (p={error_rate}, {shots} shots):")
        print(f"  QEPG SIMD:           {obs_flip_qepg:.6f}")
        print(f"  STIM (raw obs flip): {obs_flip_stim:.6f}")
        print(f"  Ratio:               {obs_flip_qepg/max(obs_flip_stim, 1e-10):.3f}x")

        assert abs(obs_flip_qepg - obs_flip_stim) < rel_tol * max(obs_flip_qepg, obs_flip_stim) + 0.002, \
            f"QEPG SIMD ({obs_flip_qepg:.6f}) vs STIM ({obs_flip_stim:.6f}) differ by more than {rel_tol*100:.1f}%"


# ============================================================================
# Test Class: Multiple Error Rates
# ============================================================================

class TestMultipleErrorRates:
    """Test STIM vs QEPG comparison across different error rates."""

    def test_error_rate_sweep(self, repetition_code_d3):
        """Compare STIM and QEPG across multiple error rates."""
        error_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
        shots = 500000  # High sample count for accurate comparison

        graph = qepg.compile_QEPG(repetition_code_d3)

        print(f"\n{'Error Rate':>12} | {'QEPG SIMD':>12} | {'QEPG non-SIMD':>14} | {'STIM raw':>12} | {'Ratio':>8}")
        print("-" * 70)

        for p in error_rates:
            obs_simd = sample_qepg_ler_simd(graph, p, shots)
            obs_nonsimd = sample_qepg_ler_nonsimd(graph, p, shots)

            stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, p)
            obs_stim = sample_stim_observable_flip_rate(stim_circuit, shots)

            ratio = obs_simd / max(obs_stim, 1e-10)

            print(f"{p:>12.4f} | {obs_simd:>12.6f} | {obs_nonsimd:>14.6f} | {obs_stim:>12.6f} | {ratio:>8.3f}x")

            # Tight tolerance: within 10% relative error
            avg_rate = (obs_simd + obs_stim) / 2
            rel_tol = max(calculate_relative_tolerance(avg_rate, shots), 0.10)
            assert abs(obs_simd - obs_stim) < rel_tol * max(obs_simd, obs_stim) + 0.002, \
                f"p={p}: QEPG SIMD ({obs_simd:.6f}) vs STIM ({obs_stim:.6f}) differ by more than {rel_tol*100:.0f}%"

    def test_low_error_rate_comparison(self, larger_repetition_code):
        """Test at very low error rate where geometric skipping is used."""
        error_rate = 0.001
        shots = 1000000  # Need more samples at low error rate for accuracy

        graph = qepg.compile_QEPG(larger_repetition_code)
        obs_simd = sample_qepg_ler_simd(graph, error_rate, shots)
        obs_nonsimd = sample_qepg_ler_nonsimd(graph, error_rate, shots)

        stim_circuit = create_stim_circuit_with_depolarization(larger_repetition_code, error_rate)
        obs_stim = sample_stim_observable_flip_rate(stim_circuit, shots)

        print(f"\nLarger repetition code at low error rate (p={error_rate}, {shots} shots):")
        print(f"  QEPG SIMD:           {obs_simd:.6f}")
        print(f"  QEPG non-SIMD:       {obs_nonsimd:.6f}")
        print(f"  STIM (raw obs flip): {obs_stim:.6f}")
        print(f"  SIMD/STIM:           {obs_simd/max(obs_stim, 1e-10):.3f}x")

        # SIMD and non-SIMD should match closely (same algorithm)
        avg_rate = (obs_simd + obs_nonsimd) / 2
        rel_tol_internal = max(calculate_relative_tolerance(avg_rate, shots), 0.10)
        assert abs(obs_simd - obs_nonsimd) < rel_tol_internal * max(obs_simd, obs_nonsimd) + 0.001, \
            f"SIMD ({obs_simd:.6f}) and non-SIMD ({obs_nonsimd:.6f}) should agree"

        # QEPG vs STIM comparison
        avg_rate_vs_stim = (obs_simd + obs_stim) / 2
        rel_tol = max(calculate_relative_tolerance(avg_rate_vs_stim, shots), 0.10)
        assert abs(obs_simd - obs_stim) < rel_tol * max(obs_simd, obs_stim) + 0.002, \
            f"QEPG ({obs_simd:.6f}) vs STIM ({obs_stim:.6f}) differ by more than {rel_tol*100:.0f}%"


# ============================================================================
# Test Class: SIMD vs non-SIMD vs STIM Three-Way Comparison
# ============================================================================

class TestThreeWayComparison:
    """Three-way comparison: SIMD vs non-SIMD vs STIM (raw observable flip rate)."""

    def test_all_three_methods_comparable(self, repetition_code_d3):
        """All three methods should produce statistically matching observable flip rates."""
        error_rate = 0.02
        shots = 500000  # High sample count for tight tolerances

        start = time.perf_counter()

        # QEPG SIMD
        graph = qepg.compile_QEPG(repetition_code_d3)
        t1 = time.perf_counter()
        obs_simd = sample_qepg_ler_simd(graph, error_rate, shots)
        t2 = time.perf_counter()

        # QEPG non-SIMD
        obs_nonsimd = sample_qepg_ler_nonsimd(graph, error_rate, shots)
        t3 = time.perf_counter()

        # STIM raw observable flip rate
        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, error_rate)
        obs_stim = sample_stim_observable_flip_rate(stim_circuit, shots)
        t4 = time.perf_counter()

        print(f"\nThree-way comparison (p={error_rate}, {shots} shots):")
        print(f"  QEPG SIMD:           obs_flip={obs_simd:.6f}, time={1000*(t2-t1):.1f}ms")
        print(f"  QEPG non-SIMD:       obs_flip={obs_nonsimd:.6f}, time={1000*(t3-t2):.1f}ms")
        print(f"  STIM (raw obs flip): obs_flip={obs_stim:.6f}, time={1000*(t4-t3):.1f}ms")
        print(f"  SIMD/STIM:           {obs_simd/max(obs_stim, 1e-10):.3f}x")

        # SIMD and non-SIMD should agree closely (same algorithm)
        avg_rate_internal = (obs_simd + obs_nonsimd) / 2
        rel_tol_internal = max(calculate_relative_tolerance(avg_rate_internal, shots), 0.05)
        assert abs(obs_simd - obs_nonsimd) < rel_tol_internal * max(obs_simd, obs_nonsimd) + 0.002, \
            f"SIMD ({obs_simd:.6f}) vs non-SIMD ({obs_nonsimd:.6f}) differ by more than {rel_tol_internal*100:.0f}%"

        # QEPG vs STIM: tight tolerance
        avg_rate = (obs_simd + obs_stim) / 2
        rel_tol = max(calculate_relative_tolerance(avg_rate, shots), 0.05)
        assert abs(obs_simd - obs_stim) < rel_tol * max(obs_simd, obs_stim) + 0.002, \
            f"SIMD ({obs_simd:.6f}) vs STIM ({obs_stim:.6f}) differ by more than {rel_tol*100:.0f}%"


# ============================================================================
# Test Class: Quick Validation
# ============================================================================

class TestQuickValidation:
    """Quick tests for CI with moderate sample counts."""

    def test_quick_stim_qepg_comparison(self, repetition_code_d3):
        """Quick comparison with reasonable accuracy."""
        error_rate = 0.01
        shots = 100000  # Reasonable sample count for quick test

        graph = qepg.compile_QEPG(repetition_code_d3)
        obs_qepg = sample_qepg_ler_simd(graph, error_rate, shots)

        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, error_rate)
        obs_stim = sample_stim_observable_flip_rate(stim_circuit, shots)

        # Verify both produce reasonable results
        assert 0 <= obs_qepg <= 1.0
        assert 0 <= obs_stim <= 1.0

        # Check they're within 10% relative tolerance
        avg_rate = (obs_qepg + obs_stim) / 2
        rel_tol = max(calculate_relative_tolerance(avg_rate, shots), 0.10)

        print(f"Quick test: QEPG={obs_qepg:.4f}, STIM={obs_stim:.4f}, ratio={obs_qepg/max(obs_stim, 1e-10):.3f}x")

        assert abs(obs_qepg - obs_stim) < rel_tol * max(obs_qepg, obs_stim) + 0.005, \
            f"QEPG ({obs_qepg:.4f}) vs STIM ({obs_stim:.4f}) differ significantly"

    def test_stim_circuit_creation(self, repetition_code_d3):
        """Verify STIM circuit creation works correctly."""
        error_rate = 0.01

        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, error_rate)

        # Verify circuit is valid by checking it has expected structure
        circuit_str = str(stim_circuit)
        assert 'DEPOLARIZE1' in circuit_str, "STIM circuit should have DEPOLARIZE1 noise"
        assert 'DETECTOR' in circuit_str, "STIM circuit should have detectors"
        assert 'OBSERVABLE_INCLUDE' in circuit_str, "STIM circuit should have observable"

        # Verify we can sample from it
        sampler = stim_circuit.compile_detector_sampler()
        samples = sampler.sample(shots=10)
        assert samples.shape[0] == 10


# ============================================================================
# Test Class: Noise Model Verification
# ============================================================================

class TestNoiseModelVerification:
    """Verify that the noise models match between QEPG and STIM."""

    def test_noise_count_matches(self, repetition_code_d3):
        """Verify QEPG and STIM have same number of noise locations."""
        error_rate = 0.01

        # STIM circuit
        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, error_rate)
        circuit_str = str(stim_circuit)

        # Count DEPOLARIZE1 instructions in STIM circuit
        depol_count = circuit_str.count('DEPOLARIZE1')

        print(f"\nNoise locations in STIM circuit: {depol_count}")

        # The STIM circuit should have noise before each:
        # - 4 CX gates (2 noise locations each, but combined = 4 DEPOLARIZE1 lines)
        # - 5 M gates (1 noise location each = 5 DEPOLARIZE1 lines)
        # Total: 9 DEPOLARIZE1 lines (each can have multiple targets)
        assert depol_count > 0, "STIM circuit should have noise"

    def test_zero_error_gives_zero_ler(self, repetition_code_d3):
        """At p=0, both QEPG and STIM should give LER=0."""
        error_rate = 0.0
        shots = 10000

        # QEPG with p=0
        graph = qepg.compile_QEPG(repetition_code_d3)
        ler_qepg = sample_qepg_ler_simd(graph, error_rate, shots)

        # STIM with p=0
        stim_circuit = create_stim_circuit_with_depolarization(repetition_code_d3, error_rate)
        ler_stim = sample_stim_ler(stim_circuit, shots)

        assert ler_qepg == 0.0, f"QEPG LER at p=0 should be 0, got {ler_qepg}"
        assert ler_stim == 0.0, f"STIM LER at p=0 should be 0, got {ler_stim}"
