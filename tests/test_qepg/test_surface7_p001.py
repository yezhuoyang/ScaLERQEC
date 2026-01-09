"""
Test the progressive sampling Scaler on Surface code distance 7.
Compare with Stim's ground truth.
"""

import os
import sys
import time
import numpy as np
import stim
import pymatching

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from scalerqec.Stratified.Scaler import Scaler
from scalerqec.Clifford.clifford import CliffordCircuit


def stim_ground_truth_ler(circuit_path: str, error_rate: float, shots: int = 100000) -> float:
    """Compute ground truth LER using Stim's sampler."""
    with open(circuit_path, 'r', encoding='utf-8') as f:
        circuit_str = f.read()

    # Create noisy circuit using CliffordCircuit (which builds Stim circuit)
    cc = CliffordCircuit(4)
    cc.error_rate = error_rate
    cc.compile_from_stim_circuit_str(circuit_str)

    # Use Stim's native sampler
    sampler = cc.stimcircuit.compile_detector_sampler()
    det_result, obs_result = sampler.sample(shots, separate_observables=True)

    # Create decoder from detector error model
    detector_error_model = cc.stimcircuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Decode
    predictions = matcher.decode_batch(det_result)
    predictions = predictions.ravel()
    obs_flat = obs_result.ravel()

    num_errors = np.sum(predictions != obs_flat)
    return num_errors / shots


def test_surface7():
    """Test Scaler on Surface-7 at p=0.001."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    circuit_path = os.path.join(base_dir, 'stimprograms', 'surface', 'surface7')

    code_type = "Surface"
    code_distance = 7
    error_rate = 0.001  # p=0.001 as requested
    time_budget = 120  # 2 minutes
    mc_shots = 1000000  # 1M shots for ground truth

    if not os.path.exists(circuit_path):
        print(f"Circuit file not found: {circuit_path}")
        return None

    print("=" * 70)
    print(f"Testing Scaler on {code_type}-{code_distance} at p={error_rate}")
    print("=" * 70)

    # 1. Stim ground truth
    print("\n1. Computing Stim ground truth...")
    start_mc = time.perf_counter()
    stim_ler = stim_ground_truth_ler(circuit_path, error_rate, mc_shots)
    mc_time = time.perf_counter() - start_mc
    print(f"   Stim LER: {stim_ler:.6e} ({mc_shots:,} shots, {mc_time:.1f}s)")

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
    print(f"  Stim LER:   {stim_ler:.6e} ({mc_shots:,} shots, {mc_time:.1f}s)")
    print(f"  Scaler LER: {scaler_ler:.6e} ({sum(scaler._subspace_sample_used.values()):,} shots, {scaler_time:.1f}s)")

    if stim_ler > 0:
        relative_error = abs(scaler_ler - stim_ler) / stim_ler
        print(f"  Relative error: {relative_error*100:.1f}%")
    else:
        print(f"  Absolute error: {abs(scaler_ler - stim_ler):.6e}")

    print(f"\n  R^2: {scaler._R_square_score:.4f}")
    print(f"  Sweet spot: {scaler._sweet_spot}")
    print(f"  Sampled weights: {sorted(scaler._subspace_sample_used.keys())}")

    return {
        'stim_ler': stim_ler,
        'scaler_ler': scaler_ler,
        'stim_time': mc_time,
        'scaler_time': scaler_time,
        'r_squared': scaler._R_square_score,
        'sweet_spot': scaler._sweet_spot,
        'total_samples': sum(scaler._subspace_sample_used.values()),
    }


if __name__ == '__main__':
    test_surface7()
