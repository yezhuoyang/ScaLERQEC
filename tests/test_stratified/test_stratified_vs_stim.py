"""
Test stratified sampling by comparing with stim Monte Carlo simulation.

This test validates that the stratified sampling method (sampling ALL subspaces)
produces results that match stim Monte Carlo within sampling error.
"""
import os
import pytest
from scalerqec.Stratified.stratifiedLER import StratifiedLERcalc
from scalerqec.Monte.monteLER import MonteLERcalc


@pytest.fixture
def stratified_tolerance():
    """Relative tolerance for stratified vs stim."""
    return 0.20  # 20% tolerance accounting for sampling variance


# Global results storage for summary
test_results = []


class TestStratifiedVsStim:
    """Test stratified sampling method against stim Monte Carlo."""

    @pytest.mark.parametrize("circuit_name,error_rate", [
        # Test ALL circuits in small/ folder at 4 different error rates
        ("1cnot", 0.01),
        ("1cnot", 0.005),
        ("1cnot", 0.001),
        ("1cnot", 0.0005),
        ("1cnot1R", 0.01),
        ("1cnot1R", 0.005),
        ("1cnot1R", 0.001),
        ("1cnot1R", 0.0005),
        ("1cnoth", 0.01),
        ("1cnoth", 0.005),
        ("1cnoth", 0.001),
        ("1cnoth", 0.0005),
        ("2cnot", 0.01),
        ("2cnot", 0.005),
        ("2cnot", 0.001),
        ("2cnot", 0.0005),
        ("2cnot2", 0.01),
        ("2cnot2", 0.005),
        ("2cnot2", 0.001),
        ("2cnot2", 0.0005),
        ("2cnot2R", 0.01),
        ("2cnot2R", 0.005),
        ("2cnot2R", 0.001),
        ("2cnot2R", 0.0005),
        ("cnot0", 0.01),
        ("cnot0", 0.005),
        ("cnot0", 0.001),
        ("cnot0", 0.0005),
        ("cnot01", 0.01),
        ("cnot01", 0.005),
        ("cnot01", 0.001),
        ("cnot01", 0.0005),
        ("cnot01h01", 0.01),
        ("cnot01h01", 0.005),
        ("cnot01h01", 0.001),
        ("cnot01h01", 0.0005),
        ("cnot1", 0.01),
        ("cnot1", 0.005),
        ("cnot1", 0.001),
        ("cnot1", 0.0005),
        ("cnoth0", 0.01),
        ("cnoth0", 0.005),
        ("cnoth0", 0.001),
        ("cnoth0", 0.0005),
        ("cnoth01", 0.01),
        ("cnoth01", 0.005),
        ("cnoth01", 0.001),
        ("cnoth01", 0.0005),
        ("repetition3r2", 0.01),
        ("repetition3r2", 0.005),
        ("repetition3r2", 0.001),
        ("repetition3r2", 0.0005),
        ("repetition3r3", 0.01),
        ("repetition3r3", 0.005),
        ("repetition3r3", 0.001),
        ("repetition3r3", 0.0005),
        ("repetition3r4", 0.01),
        ("repetition3r4", 0.005),
        ("repetition3r4", 0.001),
        ("repetition3r4", 0.0005),
        ("simple", 0.01),
        ("simple", 0.005),
        ("simple", 0.001),
        ("simple", 0.0005),
        ("simpleh", 0.01),
        ("simpleh", 0.005),
        ("simpleh", 0.001),
        ("simpleh", 0.0005),
        ("simpleMultiObs", 0.01),
        ("simpleMultiObs", 0.005),
        ("simpleMultiObs", 0.001),
        ("simpleMultiObs", 0.0005),
        ("surface3r1", 0.01),
        ("surface3r1", 0.005),
        ("surface3r1", 0.001),
        ("surface3r1", 0.0005),
        ("surface3r2", 0.01),
        ("surface3r2", 0.005),
        ("surface3r2", 0.001),
        ("surface3r2", 0.0005),
        ("surface3r3", 0.01),
        ("surface3r3", 0.005),
        ("surface3r3", 0.001),
        ("surface3r3", 0.0005),
    ])
    def test_stratified_vs_stim_all_error_rates(
        self, circuit_base_path, circuit_name, error_rate, stratified_tolerance
    ):
        """
        Test stratified sampling (all subspaces) against STIM at multiple error rates.

        For each circuit and error rate combination:
        1. Run stratified sampling with sample_all_subspace (every weight from 0 to num_noise)
        2. Run STIM Monte Carlo with large sample budget
        3. Compare the two LER results
        """
        # Adjust sample sizes based on error rate
        if error_rate >= 0.01:
            sample_size = 1000000  # 1M per subspace
            stim_samples = 50000000  # 50M total
        elif error_rate >= 0.005:
            sample_size = 2000000  # 2M per subspace
            stim_samples = 80000000  # 80M total
        elif error_rate >= 0.001:
            sample_size = 1000000  # 1M per subspace
            stim_samples = 100000000  # 100M total
        else:  # 0.0005
            sample_size = 2000000  # 2M per subspace
            stim_samples = 100000000  # 100M total

        filepath = os.path.join(circuit_base_path, circuit_name)

        print(f"\n{'='*70}")
        print(f"Circuit: {circuit_name}, Error rate: {error_rate}")
        print(f"{'='*70}")

        # Step 1: Run stratified sampling (all subspaces)
        print(f"\n[Step 1] Running Stratified Sampling (all subspaces)...")
        print(f"  Samples per subspace: {sample_size:,}")

        stratified_calc = StratifiedLERcalc(error_rate)
        stratified_calc.parse_from_file(filepath)

        num_noise = stratified_calc._num_noise
        print(f"  Num noise sources: {num_noise}")
        print(f"  Sampling all weights from 0 to {num_noise}...")

        # Sample all subspaces
        stratified_calc.sample_all_subspace(shots_each_subspace=sample_size)

        # Calculate LER from stratified samples
        stratified_calc.calculate_LER()
        stratified_ler = stratified_calc._ler
        print(f"  Stratified LER: {stratified_ler:.6e}")

        # Step 2: Run STIM Monte Carlo
        print(f"\n[Step 2] Running STIM Monte Carlo...")
        print(f"  Total samples: {stim_samples:,}")

        stim_calc = MonteLERcalc()
        stim_ler = stim_calc.calculate_LER_from_file(
            stim_samples, filepath, error_rate
        )
        print(f"  STIM LER:       {stim_ler:.6e}")

        # Step 3: Compare results
        print(f"\n{'='*70}")
        print(f"COMPARISON")
        print(f"{'='*70}")

        result = {
            'circuit': circuit_name,
            'error_rate': error_rate,
            'stratified': stratified_ler,
            'stim': stim_ler,
            'num_noise': num_noise,
            'sample_size': sample_size,
            'stim_samples': stim_samples,
            'passed': False,
            'rel_error': 0.0
        }

        if stim_ler > 0 and stratified_ler > 0:
            rel_error = abs(stratified_ler - stim_ler) / stim_ler
            result['rel_error'] = rel_error

            print(f"  Stratified:   {stratified_ler:.6e}")
            print(f"  STIM:         {stim_ler:.6e}")
            print(f"  Rel Error:    {rel_error:.2%}")

            if rel_error < stratified_tolerance:
                print(f"  Status:       [PASS] (within {stratified_tolerance:.0%} tolerance)")
                result['passed'] = True
            else:
                print(f"  Status:       [FAIL] (exceeds {stratified_tolerance:.0%} tolerance)")

            print(f"{'='*70}")

            # Store result for summary
            test_results.append(result)

            assert rel_error < stratified_tolerance, \
                f"Stratified vs STIM error {rel_error:.2%} exceeds tolerance {stratified_tolerance:.0%}"

        elif stim_ler == 0 and stratified_ler == 0:
            print(f"  Both methods: 0.0")
            print(f"  Status:       [PASS] (both zero)")
            print(f"{'='*70}")
            result['passed'] = True
            test_results.append(result)

        else:
            print(f"  Stratified:   {stratified_ler:.6e}")
            print(f"  STIM:         {stim_ler:.6e}")
            print(f"  Status:       [FAIL] (one is zero, other is not)")
            print(f"{'='*70}")
            result['rel_error'] = float('inf')
            test_results.append(result)
            pytest.fail(f"STIM={stim_ler:.6e} but Stratified={stratified_ler:.6e}")


def pytest_sessionfinish(session, exitstatus):
    """Print summary tables after all tests complete."""
    if not test_results:
        return

    print(f"\n\n{'='*100}")
    print(f"TEST SUMMARY - Stratified vs STIM Comparison")
    print(f"{'='*100}")

    # Separate passed and failed tests
    passed_tests = [r for r in test_results if r['passed']]
    failed_tests = [r for r in test_results if not r['passed']]

    # Print PASSED table
    print(f"\n✅ PASSED TESTS ({len(passed_tests)}/{len(test_results)}):")
    print(f"{'-'*100}")
    print(f"{'Circuit':<15} {'Error Rate':<12} {'Stratified LER':<18} {'STIM LER':<18} {'Rel Error':<12} {'Noise':<8}")
    print(f"{'-'*100}")
    for r in passed_tests:
        print(f"{r['circuit']:<15} {r['error_rate']:<12.4f} {r['stratified']:<18.6e} {r['stim']:<18.6e} {r['rel_error']:<12.2%} {r['num_noise']:<8}")
    print(f"{'-'*100}")

    # Print FAILED table
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}/{len(test_results)}):")
        print(f"{'-'*100}")
        print(f"{'Circuit':<15} {'Error Rate':<12} {'Stratified LER':<18} {'STIM LER':<18} {'Rel Error':<12} {'Noise':<8}")
        print(f"{'-'*100}")
        for r in failed_tests:
            if r['rel_error'] == float('inf'):
                print(f"{r['circuit']:<15} {r['error_rate']:<12.4f} {r['stratified']:<18.6e} {r['stim']:<18.6e} {'N/A':<12} {r['num_noise']:<8}")
            else:
                print(f"{r['circuit']:<15} {r['error_rate']:<12.4f} {r['stratified']:<18.6e} {r['stim']:<18.6e} {r['rel_error']:<12.2%} {r['num_noise']:<8}")
        print(f"{'-'*100}")

    # Print overall statistics
    print(f"\n{'='*100}")
    print(f"OVERALL STATISTICS:")
    print(f"  Total Tests:  {len(test_results)}")
    print(f"  Passed:       {len(passed_tests)} ({100*len(passed_tests)/len(test_results):.1f}%)")
    print(f"  Failed:       {len(failed_tests)} ({100*len(failed_tests)/len(test_results):.1f}%)")
    print(f"{'='*100}\n")
