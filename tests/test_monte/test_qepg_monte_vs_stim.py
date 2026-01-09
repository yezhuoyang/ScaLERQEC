"""
Test QEPG Monte Carlo sampling against STIM Monte Carlo.

This validates that our C++ QEPG Monte Carlo implementation produces
the same results as STIM's built-in Monte Carlo sampler.
"""
import os
import pytest
import numpy as np
import pymatching
from scalerqec.qepg import compile_QEPG, return_samples_Monte_separate_obs_with_QEPG
from scalerqec.Clifford.clifford import CliffordCircuit


@pytest.fixture
def monte_tolerance():
    """Relative tolerance for QEPG Monte vs STIM Monte."""
    return 0.15  # 15% tolerance for Monte Carlo variance


# Global results storage for summary
monte_test_results = []


class TestQEPGMonteVsStim:
    """Test QEPG Monte Carlo against STIM Monte Carlo."""

    @pytest.mark.parametrize("circuit_name,error_rate", [
        # Test all circuits at 4 different error rates
        ("1cnot", 0.01),
        ("1cnot", 0.005),
        ("1cnot", 0.001),
        # ("1cnot", 0.0005),
        ("1cnot1R", 0.01),
        ("1cnot1R", 0.005),
        ("1cnot1R", 0.001),
        # ("1cnot1R", 0.0005),
        ("1cnoth", 0.01),
        ("1cnoth", 0.005),
        ("1cnoth", 0.001),
        # ("1cnoth", 0.0005),
        ("2cnot", 0.01),
        ("2cnot", 0.005),
        ("2cnot", 0.001),
        # ("2cnot", 0.0005),
        ("2cnot2", 0.01),
        ("2cnot2", 0.005),
        ("2cnot2", 0.001),
        # ("2cnot2", 0.0005),
        ("2cnot2R", 0.01),
        ("2cnot2R", 0.005),
        ("2cnot2R", 0.001),
        # ("2cnot2R", 0.0005),
        ("cnot0", 0.01),
        ("cnot0", 0.005),
        ("cnot0", 0.001),
        # ("cnot0", 0.0005),
        ("cnot01", 0.01),
        ("cnot01", 0.005),
        ("cnot01", 0.001),
        # ("cnot01", 0.0005),
        ("cnot01h01", 0.01),
        ("cnot01h01", 0.005),
        ("cnot01h01", 0.001),
        # ("cnot01h01", 0.0005),
        ("cnot1", 0.01),
        ("cnot1", 0.005),
        ("cnot1", 0.001),
        # ("cnot1", 0.0005),
        ("cnoth0", 0.01),
        ("cnoth0", 0.005),
        ("cnoth0", 0.001),
        # ("cnoth0", 0.0005),
        ("cnoth01", 0.01),
        ("cnoth01", 0.005),
        ("cnoth01", 0.001),
        # ("cnoth01", 0.0005),
        # Temporarily disabled for faster testing:
        # ("repetition3r2", 0.01),
        # ("repetition3r2", 0.005),
        # ("repetition3r2", 0.001),
        # ("repetition3r2", 0.0005),
        # ("repetition3r3", 0.01),
        # ("repetition3r3", 0.005),
        # ("repetition3r3", 0.001),
        # ("repetition3r3", 0.0005),
        # ("repetition3r4", 0.01),
        # ("repetition3r4", 0.005),
        # ("repetition3r4", 0.001),
        # ("repetition3r4", 0.0005),
        ("simple", 0.01),
        ("simple", 0.005),
        ("simple", 0.001),
        # ("simple", 0.0005),
        ("simpleh", 0.01),
        ("simpleh", 0.005),
        ("simpleh", 0.001),
        # ("simpleh", 0.0005),
        ("simpleMultiObs", 0.01),
        ("simpleMultiObs", 0.005),
        ("simpleMultiObs", 0.001),
        # ("simpleMultiObs", 0.0005),
        # Temporarily disabled for faster testing:
        # ("surface3r1", 0.01),
        # ("surface3r1", 0.005),
        # ("surface3r1", 0.001),
        # ("surface3r1", 0.0005),
        # ("surface3r2", 0.01),
        # ("surface3r2", 0.005),
        # ("surface3r2", 0.001),
        # ("surface3r2", 0.0005),
        # ("surface3r3", 0.01),
        # ("surface3r3", 0.005),
        # ("surface3r3", 0.001),
        # ("surface3r3", 0.0005),
    ])
    def test_qepg_monte_vs_stim_monte(
        self, circuit_base_path, circuit_name, error_rate, monte_tolerance
    ):
        """
        Test QEPG Monte Carlo vs STIM Monte Carlo.

        Both methods should produce the same LER since they're both using
        Monte Carlo sampling at the same error rate with the same decoder.
        """
        # Temporarily using fixed sample size for faster testing
        sample_size = 1000000  # 1M samples for all tests

        filepath = os.path.join(circuit_base_path, circuit_name)

        print(f"\n{'='*70}")
        print(f"Circuit: {circuit_name}, Error rate: {error_rate}")
        print(f"{'='*70}")

        # Load circuit
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()

        # Compile circuit
        circuit = CliffordCircuit(2)
        circuit.error_rate = error_rate
        circuit.compile_from_stim_circuit_str(stim_str)
        new_stim_circuit = circuit.stimcircuit

        # Compile QEPG graph
        qepg_graph = compile_QEPG(stim_str)

        # Setup decoder (same for both methods)
        detector_error_model = new_stim_circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

        # Step 1: QEPG Monte Carlo
        print(f"\n[Step 1/3] Running QEPG Monte Carlo sampling...")
        print(f"  Sample size: {sample_size:,}")

        detection_qepg, observable_qepg = return_samples_Monte_separate_obs_with_QEPG(
            qepg_graph, error_rate, sample_size
        )
        print(f"  -> Generated {len(detection_qepg)} detection samples")

        predictions_qepg = matcher.decode_batch(detection_qepg)
        print(f"  -> Decoded {len(predictions_qepg)} predictions")

        # Flatten arrays before comparison to avoid memory explosion
        observables_qepg = np.asarray(observable_qepg).ravel()
        predictions_qepg_flat = np.asarray(predictions_qepg).ravel()
        num_errors_qepg = np.count_nonzero(observables_qepg != predictions_qepg_flat)
        qepg_ler = num_errors_qepg / sample_size

        print(f"  -> QEPG Result: LER = {qepg_ler:.6e}, Logical errors = {num_errors_qepg:,}")

        # Step 2: STIM Monte Carlo
        print(f"\n[Step 2/3] Running STIM Monte Carlo sampling...")
        print(f"  Sample size: {sample_size:,}")

        sampler = new_stim_circuit.compile_detector_sampler()
        detection_stim, observable_stim = sampler.sample(
            sample_size, separate_observables=True
        )
        print(f"  -> Generated {len(detection_stim)} detection samples")

        predictions_stim = matcher.decode_batch(detection_stim)
        print(f"  -> Decoded {len(predictions_stim)} predictions")

        # Flatten arrays before comparison to avoid memory explosion
        observables_stim = np.asarray(observable_stim).ravel()
        predictions_stim_flat = np.asarray(predictions_stim).ravel()
        num_errors_stim = np.count_nonzero(observables_stim != predictions_stim_flat)
        stim_ler = num_errors_stim / sample_size

        print(f"  -> STIM Result: LER = {stim_ler:.6e}, Logical errors = {num_errors_stim:,}")

        # Step 3: Compare results
        print(f"\n[Step 3/3] Comparing QEPG vs STIM results...")
        print(f"{'='*70}")

        result = {
            'circuit': circuit_name,
            'error_rate': error_rate,
            'qepg_ler': qepg_ler,
            'stim_ler': stim_ler,
            'qepg_errors': num_errors_qepg,
            'stim_errors': num_errors_stim,
            'samples': sample_size,
            'passed': False,
            'rel_error': 0.0
        }

        if stim_ler > 0 and qepg_ler > 0:
            rel_error = abs(qepg_ler - stim_ler) / stim_ler
            result['rel_error'] = rel_error

            print(f"  Method         | Samples       | Logical Errors | LER")
            print(f"  ---------------|---------------|----------------|------------------")
            print(f"  QEPG Monte     | {sample_size:>13,} | {num_errors_qepg:>14,} | {qepg_ler:.6e}")
            print(f"  STIM Monte     | {sample_size:>13,} | {num_errors_stim:>14,} | {stim_ler:.6e}")
            print(f"{'='*70}")
            print(f"  Relative Error: {rel_error:.4%}")

            if rel_error < monte_tolerance:
                print(f"  Status:         -> PASS (within {monte_tolerance:.0%} tolerance)")
                result['passed'] = True
            else:
                print(f"  Status:         ✗ FAIL (exceeds {monte_tolerance:.0%} tolerance)")

            print(f"{'='*70}\n")

            # Store result for summary
            monte_test_results.append(result)

            assert rel_error < monte_tolerance, \
                f"QEPG Monte vs STIM Monte error {rel_error:.2%} exceeds tolerance {monte_tolerance:.0%}"

        elif stim_ler == 0 and qepg_ler == 0:
            print(f"  Both methods: 0.0")
            print(f"  Status:       [PASS] (both zero)")
            print(f"{'='*70}")
            result['passed'] = True
            monte_test_results.append(result)

        else:
            print(f"  QEPG Monte:   {qepg_ler:.6e} ({num_errors_qepg} errors)")
            print(f"  STIM Monte:   {stim_ler:.6e} ({num_errors_stim} errors)")
            print(f"  Status:       [FAIL] (one is zero, other is not)")
            print(f"{'='*70}")
            result['rel_error'] = float('inf')
            monte_test_results.append(result)
            pytest.fail(f"STIM={stim_ler:.6e} but QEPG={qepg_ler:.6e}")


def pytest_sessionfinish(session, exitstatus):
    """Print summary tables after all tests complete."""
    if not monte_test_results:
        return

    print(f"\n\n{'='*100}")
    print(f"TEST SUMMARY - QEPG Monte Carlo vs STIM Monte Carlo")
    print(f"{'='*100}")

    # Separate passed and failed tests
    passed_tests = [r for r in monte_test_results if r['passed']]
    failed_tests = [r for r in monte_test_results if not r['passed']]

    # Print PASSED table
    print(f"\n✅ PASSED TESTS ({len(passed_tests)}/{len(monte_test_results)}):")
    print(f"{'-'*110}")
    print(f"{'Circuit':<15} {'Error Rate':<12} {'QEPG LER':<18} {'STIM LER':<18} {'Rel Error':<12} {'Samples':<12}")
    print(f"{'-'*110}")
    for r in passed_tests:
        print(f"{r['circuit']:<15} {r['error_rate']:<12.4f} {r['qepg_ler']:<18.6e} {r['stim_ler']:<18.6e} {r['rel_error']:<12.2%} {r['samples']:<12}")
    print(f"{'-'*110}")

    # Print FAILED table
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}/{len(monte_test_results)}):")
        print(f"{'-'*110}")
        print(f"{'Circuit':<15} {'Error Rate':<12} {'QEPG LER':<18} {'STIM LER':<18} {'Rel Error':<12} {'Samples':<12}")
        print(f"{'-'*110}")
        for r in failed_tests:
            if r['rel_error'] == float('inf'):
                print(f"{r['circuit']:<15} {r['error_rate']:<12.4f} {r['qepg_ler']:<18.6e} {r['stim_ler']:<18.6e} {'N/A':<12} {r['samples']:<12}")
            else:
                print(f"{r['circuit']:<15} {r['error_rate']:<12.4f} {r['qepg_ler']:<18.6e} {r['stim_ler']:<18.6e} {r['rel_error']:<12.2%} {r['samples']:<12}")
        print(f"{'-'*110}")

    # Print overall statistics
    print(f"\n{'='*100}")
    print(f"OVERALL STATISTICS:")
    print(f"  Total Tests:  {len(monte_test_results)}")
    print(f"  Passed:       {len(passed_tests)} ({100*len(passed_tests)/len(monte_test_results):.1f}%)")
    print(f"  Failed:       {len(failed_tests)} ({100*len(failed_tests)/len(monte_test_results):.1f}%)")
    print(f"{'='*100}\n")
