"""
Test the original stratifiedScurveLER on Surface code distance 7.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from scalerqec.Stratified.stratifiedScurveLER import StratifiedScurveLERcalc

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    circuit_path = os.path.join(base_dir, 'stimprograms', 'surface', 'surface7')

    p = 0.001
    d = 7
    t = (d - 1) // 2

    print("=" * 70)
    print(f"Testing Original StratifiedScurveLERcalc on Surface-{d} at p={p}")
    print("=" * 70)

    calc = StratifiedScurveLERcalc(p, sampleBudget=1000000, k_range=5, num_subspace=6, beta=4)
    calc.set_t(t)
    calc.set_sample_bound(
        MIN_NUM_LE_EVENT=100,
        SAMPLE_GAP=100,
        MAX_SAMPLE_GAP=5000,
        MAX_SUBSPACE_SAMPLE=50000
    )

    calc.calculate_LER_from_file(circuit_path, p, d, "OrigSurface7_", "Surface-7", repeat=1)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Estimated LER: {calc._LER:.4e}")
    print(f"  R^2: {calc._R_square_score:.4f}")
    print(f"  Sweet spot: {calc._sweet_spot}")
    print(f"  Sampled weights: {sorted(calc._subspace_sample_used.keys())}")
    print(f"  Total samples: {sum(calc._subspace_sample_used.values()):,}")


if __name__ == '__main__':
    main()
