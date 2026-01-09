"""
Test the simplified progressive sampling Scaler on repetition code.
Compare with Monte Carlo ground truth.
"""

import os
import sys
import time
import numpy as np
import pymatching

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from scalerqec.Stratified.Scaler import Scaler
from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.qepg import compile_QEPG, return_samples_Monte_separate_obs_with_QEPG


def monte_carlo_ler(circuit_path: str, error_rate: float, shots: int = 100000) -> float:
    """Compute Monte Carlo ground truth LER."""
    with open(circuit_path, 'r', encoding='utf-8') as f:
        circuit_str = f.read()

    # Create noisy circuit
    cc = CliffordCircuit(4)
    cc.error_rate = error_rate
    cc.compile_from_stim_circuit_str(circuit_str)

    # Create decoder
    detector_error_model = cc.stimcircuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Sample using QEPG Monte Carlo
    graph = compile_QEPG(circuit_str)
    det_result, obs_result = return_samples_Monte_separate_obs_with_QEPG(
        graph, error_rate, shots
    )

    # Decode
    predictions = matcher.decode_batch(det_result)
    predictions = predictions.ravel()
    obs_flat = obs_result.ravel()

    num_errors = np.sum(predictions != obs_flat)
    return num_errors / shots


def test_circuit(circuit_path: str, code_type: str, code_distance: int, error_rate: float, time_budget: int = 60, mc_shots: int = 500000):
    """Test Scaler on a specific circuit."""
    if not os.path.exists(circuit_path):
        print(f"Circuit file not found: {circuit_path}")
        return None

    print("=" * 70)
    print(f"Testing Scaler on {code_type}-{code_distance} at p={error_rate}")
    print("=" * 70)

    # 1. Monte Carlo ground truth
    print("\n1. Computing Monte Carlo ground truth...")
    start_mc = time.perf_counter()
    mc_ler = monte_carlo_ler(circuit_path, error_rate, mc_shots)
    mc_time = time.perf_counter() - start_mc
    print(f"   Monte Carlo LER: {mc_ler:.6e} ({mc_shots:,} shots, {mc_time:.1f}s)")

    # 2. Scaler
    print("\n2. Running Scaler...")
    scaler = Scaler(error_rate=error_rate, time_budget=time_budget)

    start_scaler = time.perf_counter()
    scaler_ler = scaler.calculate_LER_from_file(
        filepath=circuit_path,
        pvalue=error_rate,
        codedistance=code_distance,
        figname=f"{code_type}{code_distance}_p{error_rate}_",
        titlename=f"{code_type}-{code_distance}",
    )
    scaler_time = time.perf_counter() - start_scaler

    # 3. Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Monte Carlo LER: {mc_ler:.6e} ({mc_shots:,} shots, {mc_time:.1f}s)")
    print(f"  Scaler LER:      {scaler_ler:.6e} ({sum(scaler._subspace_sample_used.values()):,} shots, {scaler_time:.1f}s)")

    if mc_ler > 0:
        relative_error = abs(scaler_ler - mc_ler) / mc_ler
        print(f"  Relative error: {relative_error*100:.1f}%")
    else:
        print(f"  Absolute error: {abs(scaler_ler - mc_ler):.6e}")

    print(f"\n  R^2: {scaler._R_square_score:.4f}")
    print(f"  Sweet spot: {scaler._sweet_spot}")
    print(f"  Sampled weights: {sorted(scaler._subspace_sample_used.keys())}")

    return {
        'mc_ler': mc_ler,
        'scaler_ler': scaler_ler,
        'mc_time': mc_time,
        'scaler_time': scaler_time,
        'r_squared': scaler._R_square_score,
        'sweet_spot': scaler._sweet_spot,
        'total_samples': sum(scaler._subspace_sample_used.values()),
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Test 1: Repetition code d=5, p=0.01
    print("\n" + "#" * 80)
    print("# TEST 1: Repetition-5, p=0.01")
    print("#" * 80)
    rep_path = os.path.join(base_dir, 'stimprograms', 'repetition', 'repetition5')
    test_circuit(rep_path, "Repetition", 5, 0.01, time_budget=30, mc_shots=500000)

    # Test 2: Repetition code d=9, p=0.0005
    print("\n" + "#" * 80)
    print("# TEST 2: Repetition-9, p=0.0005")
    print("#" * 80)
    rep9_path = os.path.join(base_dir, 'stimprograms', 'repetition', 'repetition9')
    test_circuit(rep9_path, "Repetition", 9, 0.0005, time_budget=60, mc_shots=1000000)


if __name__ == '__main__':
    main()
