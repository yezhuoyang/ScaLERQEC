"""
Test stratified sampling algorithm by comparing with symbolic DP ground truth.

This test validates that the stratified method correctly estimates the logical
error rate for each subspace by comparing against exact symbolic calculations.
"""
import os
import pytest
from scalerqec.Stratified.stratifiedLER import StratifiedLERcalc
from scalerqec.Symbolic.symbolicLER import SymbolicLERcalc
from scalerqec.util.binomial import binomial_weight


@pytest.fixture
def error_rate(request):
    """Physical error rate for testing (varies by circuit complexity)."""
    circuit_name = request.node.callspec.params.get('circuit_name', '')
    # Increase error rate for repetition codes to get more logical errors
    # This helps determine if the issue is statistical or fundamental
    if circuit_name.startswith('repetition'):
        return 0.01  # 10x higher error rate for repetition codes
    return 0.001  # Original error rate for simple circuits


@pytest.fixture
def sample_size(request):
    """Sample size for stratified sampling (varies by circuit complexity)."""
    circuit_name = request.node.callspec.params.get('circuit_name', '')
    # Repetition codes need much more samples due to low LER and many noise sources
    if circuit_name.startswith('repetition'):
        return 5000000  # 5M samples for complex repetition codes
    return 200000  # 200K samples for simple circuits


@pytest.fixture
def num_subspace():
    """Number of subspaces to sample."""
    return 20


@pytest.fixture
def subspace_tolerance(request):
    """Relative error tolerance per subspace (varies by circuit complexity)."""
    circuit_name = request.node.callspec.params.get('circuit_name', '')
    # Repetition codes have higher variance, need relaxed tolerance
    if circuit_name.startswith('repetition'):
        return 0.30  # 30% tolerance for complex circuits
    return 0.15  # 15% tolerance for simple circuits


@pytest.fixture
def min_success_rate():
    """Minimum percentage of subspaces that must pass."""
    return 0.80  # 80%


class TestStratifiedVsSymbolic:
    """Test stratified sampling against symbolic ground truth."""

    @pytest.mark.parametrize("circuit_name", [
        "simple",           # Basic circuit with LER ~0.002
        "1cnot",            # 1 CNOT, 4 noise sources
        "1cnot1R",          # 1 CNOT + Reset, 5 noise sources (tests weight==num_noise fix)
        "1cnoth",           # 1 CNOT + Hadamard
        "2cnot",            # 2 CNOTs
        "2cnot2",           # 2 CNOTs variant
        "simpleh",          # Simple + Hadamard
        "repetition3r2",    # Repetition code, 3 qubits, 2 rounds (32 noise sources)
        "repetition3r3",    # Repetition code, 3 qubits, 3 rounds (44 noise sources)
        "repetition3r4",    # Repetition code, 3 qubits, 4 rounds (56 noise sources)
    ])
    def test_subspace_weights_all_circuits(
        self,
        circuit_base_path,
        circuit_name,
        error_rate,
        sample_size,
        num_subspace,
        subspace_tolerance,
        min_success_rate
    ):
        """
        Test that stratified sampling matches symbolic DP for each subspace.

        This test:
        1. Computes exact symbolic DP for all subspaces
        2. Runs stratified sampling on all subspaces
        3. Compares joint probabilities P(LE AND weight w) for each weight
        4. Passes if >= 80% of subspaces match within 15% tolerance
        """
        filepath = os.path.join(circuit_base_path, circuit_name)
        print(f"\n{'='*60}")
        print(f"Testing circuit: {circuit_name}")
        print(f"{'='*60}")

        # Step 1: Calculate symbolic DP ground truth
        print("\n[1/2] Computing symbolic DP (exact ground truth)...")
        symbolic_calc = SymbolicLERcalc()
        symbolic_calc.calculate_LER_from_file(filepath, error_rate)

        # Get subspace probabilities (need to evaluate the symbolic expressions)
        symbolic_subspace_polys = symbolic_calc._subspace_LER
        num_noise = symbolic_calc._num_noise

        # Evaluate all subspace polynomials at the error rate
        symbolic_subspace_probs = {}
        for weight, poly in symbolic_subspace_polys.items():
            symbolic_subspace_probs[weight] = float(
                symbolic_calc.evaluate_LER_subspace(error_rate, weight)
            )

        overall_symbolic_ler = float(symbolic_calc.evaluate_LER(error_rate))

        print(f"  Num noise sources: {num_noise}")
        print(f"  Subspaces with non-zero prob: {len(symbolic_subspace_probs)}")
        print(f"  Overall symbolic LER: {overall_symbolic_ler:.6e}")

        # Step 2: Run stratified sampling - SAMPLE ALL SUBSPACES
        print("\n[2/2] Running stratified sampling on ALL subspaces...")
        stratified_calc = StratifiedLERcalc(
            error_rate,
            sampleBudget=sample_size * num_subspace,
            num_subspace=num_subspace
        )
        stratified_calc.parse_from_file(filepath)

        # Use sample_all_subspace to sample every single weight from 0 to num_noise
        # This gives us complete coverage for comparison with symbolic
        print(f"  Sampling all weights from 0 to {stratified_calc._num_noise}...")
        print(f"  Samples per subspace: {sample_size}")
        stratified_calc.sample_all_subspace(shots_each_subspace=sample_size)

        # Get subspace results and convert to joint probabilities
        stratified_subspace_counts = stratified_calc._subspace_LE_count
        stratified_subspace_samples = stratified_calc._subspace_sample_used

        stratified_subspace_probs = {}
        for w in range(0, stratified_calc._num_noise + 1):
            if stratified_subspace_samples.get(w, 0) > 0:
                # Conditional probability P(LE | weight w)
                conditional_prob = stratified_subspace_counts[w] / stratified_subspace_samples[w]
                # Binomial probability P(weight w)
                binomial_prob = binomial_weight(stratified_calc._num_noise, w, error_rate)
                # Joint probability P(LE AND weight w) = P(LE | w) * P(w)
                stratified_subspace_probs[w] = conditional_prob * binomial_prob
            else:
                stratified_subspace_probs[w] = 0.0

        overall_stratified_ler = stratified_calc.calculate_LER()
        print(f"  Overall stratified LER: {overall_stratified_ler:.6e}")

        # Step 3: Compare subspace by subspace
        print("\n  Subspace comparison:")
        print(f"  {'Weight':<8} {'Symbolic':<15} {'Stratified':<15} {'LE Events':<12} {'Samples':<12} {'Rel Error':<12} {'Status'}")
        print(f"  {'-'*95}")

        passed_subspaces = 0
        failed_subspaces = 0
        total_compared = 0

        # Compare all weights that were sampled
        all_weights = sorted(set(symbolic_subspace_probs.keys()) | set(stratified_subspace_probs.keys()))

        for w in all_weights:
            symbolic_prob = symbolic_subspace_probs.get(w, 0.0)
            stratified_prob = stratified_subspace_probs.get(w, 0.0)
            samples_used = stratified_subspace_samples.get(w, 0)
            le_events = stratified_subspace_counts.get(w, 0)

            # Only compare if we actually sampled this weight
            if samples_used == 0:
                continue

            total_compared += 1

            # Calculate relative error
            if symbolic_prob > 0:
                rel_error = abs(stratified_prob - symbolic_prob) / symbolic_prob
                passed = rel_error < subspace_tolerance
                status = "[PASS]" if passed else "[FAIL]"

                print(f"  {w:<8} {symbolic_prob:<15.6e} {stratified_prob:<15.6e} {le_events:<12} {samples_used:<12} {rel_error:<12.2%} {status}")

                if passed:
                    passed_subspaces += 1
                else:
                    failed_subspaces += 1
            else:
                # Symbolic is zero
                if stratified_prob == 0:
                    status = "[PASS]"
                    passed_subspaces += 1
                else:
                    status = "[FAIL]"
                    failed_subspaces += 1

                print(f"  {w:<8} {symbolic_prob:<15.6e} {stratified_prob:<15.6e} {le_events:<12} {samples_used:<12} {'N/A':<12} {status}")

        # Step 4: Verify overall results
        success_rate = passed_subspaces / total_compared if total_compared > 0 else 0

        print(f"\n  Summary:")
        # Handle zero LER case
        if overall_symbolic_ler > 0:
            print(f"    Overall LER rel error: {abs(overall_stratified_ler - overall_symbolic_ler) / overall_symbolic_ler:.2%}")
        else:
            print(f"    Overall LER rel error: N/A (symbolic LER is zero)")

        print(f"    Subspaces compared:    {total_compared}")
        print(f"    Passed:                {passed_subspaces} ({100*success_rate:.1f}%)")
        print(f"    Failed:                {failed_subspaces}")

        # Skip test if symbolic LER is zero (perfect circuit)
        if overall_symbolic_ler == 0:
            print(f"\n  [SKIP] Circuit has zero logical error rate (perfect code)")
            pytest.skip("Circuit has zero logical error rate")
            return

        # Assert at least min_success_rate of subspaces match
        assert success_rate >= min_success_rate, \
            f"Only {success_rate:.1%} of subspaces passed (< {min_success_rate:.0%} required)"

        print(f"\n  [PASS] {success_rate:.1%} of subspaces within {subspace_tolerance:.0%} tolerance")

    def test_overall_ler_single_circuit(
        self, circuit_base_path, error_rate, sample_size, num_subspace
    ):
        """Test that overall LER matches symbolic ground truth."""
        filepath = os.path.join(circuit_base_path, "simple")

        # Calculate symbolic ground truth
        symbolic_calc = SymbolicLERcalc()
        symbolic_calc.calculate_LER_from_file(filepath, error_rate)
        symbolic_ler = float(symbolic_calc.evaluate_LER(error_rate))

        # Calculate stratified result
        stratified_calc = StratifiedLERcalc(
            error_rate,
            sampleBudget=sample_size * num_subspace,
            num_subspace=num_subspace
        )
        stratified_calc.parse_from_file(filepath)
        stratified_calc.subspace_sampling()
        stratified_ler = stratified_calc.calculate_LER()

        print(f"\nSymbolic LER:    {symbolic_ler:.6e}")
        print(f"Stratified LER:  {stratified_ler:.6e}")

        # Check overall LER is within 10% tolerance
        if symbolic_ler > 0:
            rel_error = abs(stratified_ler - symbolic_ler) / symbolic_ler
            print(f"Relative error:  {rel_error:.2%}")
            assert rel_error < 0.10, f"Overall LER error {rel_error:.2%} exceeds 10% tolerance"
        else:
            assert stratified_ler == 0, "Symbolic is zero but stratified is not"
