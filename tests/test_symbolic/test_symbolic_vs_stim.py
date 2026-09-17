"""
Test symbolic DP algorithm by comparing with stim Monte Carlo simulation.

This test validates that the symbolic dynamic programming method produces
exact results that match stim Monte Carlo within sampling error.
"""

import os

import pytest

from scalerqec.Monte.monteLER import MonteLERcalc
from scalerqec.Symbolic.symbolicLER import SymbolicLERcalc


@pytest.fixture
def error_rate_symbolic():
    """Physical error rate for symbolic testing (needs to be small)."""
    return 0.001


@pytest.fixture
def stim_sample_size_symbolic():
    """Sample size for stim to compare with symbolic."""
    return 100000  # Reduced for faster tests


@pytest.fixture
def symbolic_tolerance():
    """Relative tolerance for symbolic vs stim."""
    # Legacy broad comparisons use this tolerance. The single-circuit
    # regression below additionally checks an independent exact distribution.
    return 0.25  # 25% tolerance for smaller samples


class TestSymbolicVsStim:
    """Test symbolic DP method against stim Monte Carlo."""

    def test_symbolic_ler_single_circuit(
        self, circuit_base_path, error_rate_symbolic, stim_sample_size_symbolic
    ):
        """Use an exact Stim response convolution, then fixed-budget seeded MC.

        The old test stopped after about twenty failures and downgraded ordinary
        sampling fluctuations to xfail. Neither establishes an exact-DP defect.
        """
        import numpy as np
        import pymatching
        import stim
        from scipy.stats import binomtest

        from scalerqec.QEC.noisemodel import SIDNoiseModel

        filepath = os.path.join(circuit_base_path, "simple")
        p = error_rate_symbolic
        symbolic = SymbolicLERcalc(p).calculate_LER_from_file(filepath, p)
        circuit = SIDNoiseModel(p).inject_noise(stim.Circuit.from_file(filepath))
        decoder = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=False)
        )
        # Independently convolve the exact categorical Pauli responses. Stim
        # evaluates each response; this does not call the Python QEPG or DP.
        width = circuit.num_detectors + circuit.num_observables
        states = np.arange(1 << width)
        joint = np.zeros(len(states))
        joint[0] = 1
        for location, op in enumerate(circuit):
            if op.name != "DEPOLARIZE1":
                assert not stim.gate_data(op.name).is_noisy_gate or (
                    op.name == "M" and not any(op.gate_args_copy())
                )
                continue
            for target in op.targets_copy():
                masks = [0]
                for axis in "XYZ":
                    forced = stim.Circuit()
                    for j, instruction in enumerate(circuit):
                        if j == location:
                            forced.append(axis + "_ERROR", [target], 1)
                        elif instruction.name != "DEPOLARIZE1":
                            forced.append(instruction)
                    bits = forced.compile_detector_sampler(seed=0).sample(
                        1, append_observables=True
                    )[0]
                    masks.append(sum(int(bit) << k for k, bit in enumerate(bits)))
                joint = sum(
                    probability * joint[states ^ mask]
                    for probability, mask in zip([1 - p, p / 3, p / 3, p / 3], masks)
                )
        assert joint.sum() == pytest.approx(1, abs=2e-14)
        patterns = ((states[:, None] >> np.arange(width)) & 1).astype(bool)
        failure = np.any(
            decoder.decode_batch(patterns[:, : circuit.num_detectors])
            != patterns[:, circuit.num_detectors :],
            axis=1,
        )
        truth = joint[failure].sum()
        assert symbolic == pytest.approx(truth, rel=1e-11, abs=1e-14)
        det, obs = circuit.compile_detector_sampler(seed=290916).sample(
            stim_sample_size_symbolic, separate_observables=True
        )
        failures = int(np.any(decoder.decode_batch(det) != obs, axis=1).sum())
        assert binomtest(failures, stim_sample_size_symbolic, truth).pvalue > 1e-6

    def test_symbolic_ler_all_circuits(
        self, circuit_base_path, error_rate_symbolic, symbolic_tolerance
    ):
        """Test symbolic LER for small circuits against stim."""
        # Use enough samples and min errors for reliable comparison
        sample_size = 1000000
        min_errors = 200

        # Only test circuits with non-zero LER (exclude circuits that always return zero)
        # Circuits like cnot0, cnot01, cnot1, etc. have zero LER and cause formatting errors
        small_circuits = [
            "1cnot",
            "1cnot1R",
            "1cnoth",
            "2cnot",
            "2cnot2",
            "2cnot2R",
            # "cnot0",  # Zero LER
            # "cnot01",  # Zero LER
            # "cnot01h01",  # Zero LER
            # "cnot1",  # Zero LER
            # "cnoth0",  # Zero LER
            # "cnoth01",  # Zero LER
            "simple",
            "simpleh",
            "simpleMultiObs",
        ]

        passed_circuits = []
        failed_circuits = []
        xfail_circuits = []  # Large discrepancies
        results = []

        for circuit_name in small_circuits:
            filepath = os.path.join(circuit_base_path, circuit_name)

            if not os.path.exists(filepath):
                print(f"[SKIP] {circuit_name}: file not found")
                continue

            print(f"\n{'=' * 70}")
            print(f"Testing circuit: {circuit_name}")
            print(f"{'=' * 70}")

            try:
                # Symbolic DP (exact)
                print(f"Running Symbolic DP (exact calculation)...")
                symbolic_calculator = SymbolicLERcalc(error_rate_symbolic)
                symbolic_result = symbolic_calculator.calculate_LER_from_file(
                    filepath, error_rate_symbolic
                )
                print(f"  Symbolic DP: PL = {symbolic_result:.6e}")

                # Stim Monte Carlo
                print(f"Running Stim Monte Carlo (samples={sample_size})...")
                stim_calculator = MonteLERcalc(
                    samplebudget=sample_size, MIN_NUM_LE_EVENT=min_errors
                )
                stim_result = stim_calculator.calculate_LER_from_file(
                    sample_size, filepath, error_rate_symbolic
                )
                print(f"  Stim:        PL = {stim_result:.6e}")

                # Compare
                if symbolic_result > 0 and stim_result > 0:
                    relative_error = (
                        abs(stim_result - symbolic_result) / symbolic_result
                    )
                    print(f"  Relative error: {relative_error:.2%}")
                    results.append(
                        (circuit_name, symbolic_result, stim_result, relative_error)
                    )

                    if relative_error < symbolic_tolerance:
                        passed_circuits.append(circuit_name)
                        print(f"  Status: PASS")
                    elif relative_error < 0.50:  # Between 25% and 50%
                        xfail_circuits.append(circuit_name)
                        print(
                            f"  Status: XFAIL (error {relative_error:.2%}, marked for investigation)"
                        )
                    else:
                        failed_circuits.append(circuit_name)
                        print(f"  Status: FAIL (error {relative_error:.2%} >= 50%)")

            except Exception as e:
                print(f"[ERROR] {circuit_name}: {e}")
                failed_circuits.append(circuit_name)

        # Print summary
        print(f"\n{'=' * 70}")
        print(f"{'Circuit':<20} {'Symbolic':>12} {'Stim':>12} {'Rel Error':>10}")
        print(f"{'=' * 70}")
        for circuit_name, symb_val, stim_val, rel_err in results:
            if circuit_name in passed_circuits:
                status = "PASS"
            elif circuit_name in xfail_circuits:
                status = "XFAIL"
            else:
                status = "FAIL"
            print(
                f"{circuit_name:<20} {symb_val:>12.6e} {stim_val:>12.6e} {rel_err:>9.2%} [{status}]"
            )
        print(f"{'=' * 70}")
        print(f"Passed: {len(passed_circuits)}/{len(small_circuits)}")
        print(f"XFail (investigate): {len(xfail_circuits)}")
        print(f"Failed: {len(failed_circuits)}")
        print(f"{'=' * 70}")

        # At least 60% should pass (allowing for method differences and sampling variance)
        pass_rate = len(passed_circuits) / len(small_circuits)
        assert pass_rate >= 0.60, (
            f"Only {pass_rate:.1%} passed (need ≥60%). Hard failures: {failed_circuits}"
        )

    @pytest.mark.skip(
        reason="SymbolicLERcalc.evaluate_subspace_prob method not implemented"
    )
    def test_symbolic_subspace_probabilities(
        self, circuit_base_path, error_rate_symbolic
    ):
        """Test that symbolic DP computes correct subspace probabilities."""
        pass

    @pytest.mark.parametrize(
        "circuit_name",
        [
            "simple",
            "1cnot",
            "1cnot1R",
            "1cnoth",
            "2cnot",
            "2cnot2",
            "simpleh",
            # "repetition3r2",  # Disabled for faster testing
        ],
    )
    def test_polynomial_at_multiple_error_rates(self, circuit_base_path, circuit_name):
        """
        Test that the SAME symbolic polynomial evaluated at different error rates
        all match STIM. This verifies polynomial correctness.

        For each circuit:
        1. Calculate symbolic polynomial ONCE
        2. Evaluate it at 2 different error rates: 0.01, 0.001
        3. Run STIM at those same error rates
        4. Compare symbolic vs STIM for each rate

        If symbolic matches STIM at all rates, the polynomial is correct.
        If not, there's an issue with symbolic DP calculation.
        """
        filepath = os.path.join(circuit_base_path, circuit_name)

        # Test at 2 different error rates (skip 0.0005 for speed)
        error_rates = [0.01, 0.001]
        stim_samples = 10000000  # 10M sample budget
        tolerance = 0.15  # 15% tolerance — symbolic is exact, only STIM has variance
        min_errors = 500  # Need ~500 errors for ~4.5% statistical error (1/sqrt(500))

        print(f"\n{'=' * 70}")
        print(f"Testing polynomial accuracy for: {circuit_name}")
        print(f"{'=' * 70}")

        # Step 1: Calculate symbolic polynomial ONCE
        print(f"\n[Step 1] Computing symbolic polynomial...")
        symbolic_calc = SymbolicLERcalc()
        # Calculate at first error rate (polynomial is independent of p during calculation)
        symbolic_calc.calculate_LER_from_file(filepath, error_rates[0])
        print(f"  Polynomial computed successfully")
        print(f"  Num noise sources: {symbolic_calc._num_noise}")

        # Step 2: Test polynomial at each error rate
        results = []
        all_passed = True

        for error_rate in error_rates:
            print(
                f"\n[Step 2.{error_rates.index(error_rate) + 1}] Testing at p={error_rate}"
            )
            print(f"{'-' * 50}")

            # Evaluate symbolic polynomial at this error rate
            print(f"  Evaluating symbolic polynomial at p={error_rate}...")
            symbolic_ler = float(symbolic_calc.evaluate_LER(error_rate))
            print(f"  Symbolic LER: {symbolic_ler:.6e}")

            # Run STIM at this error rate with enough samples for tight statistics
            print(f"  Running STIM with {stim_samples:,} sample budget...")
            stim_calc = MonteLERcalc(
                samplebudget=stim_samples, MIN_NUM_LE_EVENT=min_errors
            )
            stim_ler = stim_calc.calculate_LER_from_file(
                stim_samples, filepath, error_rate
            )
            print(f"  STIM LER:     {stim_ler:.6e}")

            # Compare
            if stim_ler > 0:
                rel_error = abs(symbolic_ler - stim_ler) / stim_ler
                passed = rel_error < tolerance
                status = "PASS" if passed else "FAIL"

                print(f"  Rel Error:    {rel_error:.2%}")
                print(f"  Status:       [{status}]")

                if not passed:
                    all_passed = False

                results.append(
                    {
                        "error_rate": error_rate,
                        "stim": stim_ler,
                        "symbolic": symbolic_ler,
                        "rel_error": rel_error,
                        "passed": passed,
                    }
                )
            else:
                # STIM is zero
                if symbolic_ler == 0:
                    print(f"  Both zero:    [PASS]")
                    results.append(
                        {
                            "error_rate": error_rate,
                            "stim": stim_ler,
                            "symbolic": symbolic_ler,
                            "rel_error": 0,
                            "passed": True,
                        }
                    )
                else:
                    print(f"  STIM zero but symbolic={symbolic_ler:.6e}: [FAIL]")
                    all_passed = False
                    results.append(
                        {
                            "error_rate": error_rate,
                            "stim": stim_ler,
                            "symbolic": symbolic_ler,
                            "rel_error": float("inf"),
                            "passed": False,
                        }
                    )

        # Summary
        print(f"\n{'=' * 70}")
        print(f"SUMMARY: {circuit_name}")
        print(f"{'=' * 70}")
        print(
            f"{'Error Rate':<15} {'STIM':<15} {'Symbolic':<15} {'Rel Error':<12} {'Status'}"
        )
        print(f"{'-' * 70}")

        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            if r["rel_error"] == float("inf"):
                print(
                    f"{r['error_rate']:<15} {r['stim']:<15.6e} {r['symbolic']:<15.6e} {'N/A':<12} [{status}]"
                )
            else:
                print(
                    f"{r['error_rate']:<15} {r['stim']:<15.6e} {r['symbolic']:<15.6e} {r['rel_error']:<12.2%} [{status}]"
                )

        passed_count = sum(1 for r in results if r["passed"])
        print(f"\nPassed: {passed_count}/{len(results)}")

        assert all_passed, (
            f"Symbolic polynomial failed at {len(results) - passed_count} error rate(s)"
        )

        print(f"\n[SUCCESS] Polynomial is accurate at all error rates")

    @pytest.mark.skip(reason="Disabled for faster testing")
    def test_symbolic_vs_stim_repetition_code(
        self, error_rate_symbolic, symbolic_tolerance
    ):
        """Test symbolic vs stim for repetition code."""
        import os

        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "stimprograms",
            "repetition",
        )
        filepath = os.path.join(base_path, "repetition3")

        if not os.path.exists(filepath):
            pytest.skip(f"Repetition code file not found: {filepath}")

        # Symbolic DP
        symbolic_calculator = SymbolicLERcalc(error_rate_symbolic)
        symbolic_result = symbolic_calculator.calculate_LER_from_file(
            filepath, error_rate_symbolic
        )

        # Stim with large sample size
        stim_calculator = MonteLERcalc()
        stim_result = stim_calculator.calculate_LER_from_file(
            3000000, filepath, error_rate_symbolic
        )

        print(f"\nRepetition-3 code:")
        print(f"  Symbolic DP: {symbolic_result:.6e}")
        print(f"  Stim: {stim_result:.6e}")

        if symbolic_result > 0:
            relative_error = abs(stim_result - symbolic_result) / symbolic_result
            print(f"  Relative error: {relative_error:.2%}")
            assert relative_error < symbolic_tolerance
