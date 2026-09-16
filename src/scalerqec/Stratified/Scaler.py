from __future__ import annotations

"""Primary ScaLER algorithm for time-budgeted logical error rate estimation.

This module implements the :class:`Scaler` class, which is the main
entry point for the ScaLER (Scalable Logical Error Rate) algorithm.
It uses a three-phase, time-budgeted approach:

1. **Phase 1 (Initialisation):** Binary-search to bracket the S-curve
   (``w_err``, ``w_sat``), then sample 5 uniformly-spaced weights.
2. **Phase 2 (Sweet-spot exploration):** Fit an S-curve model, compute
   the sweet spot, and sample between the sweet spot and ``w_err``.
3. **Phase 3 (Adaptive refinement):** Iteratively add samples at
   weights with insufficient logical error events until the time budget
   is exhausted or all weights are converged.

Multiple S-curve model backends are supported via the
:mod:`~scalerqec.Stratified.models` abstraction layer.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from scalerqec.Stratified.stratifiedScurveLER import format_with_uncertainty
from scalerqec.qepg import (
    compile_QEPG,
    return_samples_many_weights_separate_obs_with_QEPG,
    return_samples_with_fixed_QEPG_numpy,
    QEPGGraph,
)
import pymatching
from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.Stratified.models import (
    ScurveModelBase,
    OurScurveModel,
    ModelType,
    ModelFactory,
)
from scalerqec.Stratified.ScurveModel import evenly_spaced_ints
from scalerqec.Stratified.fitting import r_squared
from scalerqec.Stratified.profile import LERProfile, _probabilities
import matplotlib.pyplot as plt


class Scaler:
    """Time-budgeted stratified LER estimator (the primary ScaLER algorithm).

    Estimates the logical error rate of a quantum error correction code
    by stratifying over fault weight *w*, fitting an S-curve model to
    P_L(w), and integrating against the binomial weight distribution::

        LER = sum_w  P_L(w) * Binom(N, w) * p^w * (1-p)^(N-w)

    The algorithm proceeds in three phases:

    1. **Phase 1 -- Initialisation:** Binary-search for the onset
       weight ``w_err`` and saturation weight ``w_sat``, then sample 5
       uniformly-spaced weights between them.
    2. **Phase 2 -- Sweet-spot exploration:** Fit the S-curve model,
       compute the sweet spot, and sample between the sweet spot and
       ``w_err``.
    3. **Phase 3 -- Adaptive refinement:** Iteratively add samples at
       weights that have not yet accumulated enough logical error events,
       until the wall-clock time budget is exhausted.

    Multiple S-curve model backends are supported via the
    :mod:`~scalerqec.Stratified.models` abstraction layer:

    * ``OUR_MODEL`` (default): Definition 1 logistic model with pole term.
    * ``IBM_MODEL``: Definition 2 min-fail enclosure model.

    Args:
        error_rate: Physical error rate per noise location.
        time_budget: Wall-clock time budget in seconds.
        model_type: Which S-curve model to use.
        gamma: Sweet-spot tuning parameter for the condition
            ``d^2 y / dw^2 = gamma * |dy/dw|``.
        num_subspaces_phase2: Number of uniformly-spaced subspaces to
            sample between ``w_sweet`` and ``w_err`` in Phase 2.
    """

    def __init__(
        self,
        error_rate: float = 0.0,
        time_budget: int = 30,
        model_type: ModelType = ModelType.OUR_MODEL,
        gamma: float = 1,
        num_subspaces_phase2: int = 12,
        decoder=None,
    ):
        """
        Initialize the Scaler.

        Args:
            error_rate: Physical error rate (probability of error per gate)
            time_budget: Time budget in seconds
            model_type: Which S-curve model to use
            gamma: Sweet spot tuning parameter for d²y/dw² = γ * dy/dw
            num_subspaces_phase2: Number of uniform subspaces to sample between
                                  w_sweet and w_err in Phase 2 (default: 12)
            decoder: Any object with a ``decode_batch`` method. If ``None``,
                a pymatching decoder is built automatically from the circuit's
                detector error model.
        """
        if not np.isfinite(time_budget) or time_budget <= 0:
            raise ValueError("time_budget must be finite and positive.")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive.")
        if (isinstance(num_subspaces_phase2, bool) or not isinstance(num_subspaces_phase2, int)
                or num_subspaces_phase2 < 1):
            raise ValueError("num_subspaces_phase2 must be a positive integer.")
        if np.ndim(error_rate) != 0:
            raise ValueError("error_rate must be a scalar.")
        _probabilities(error_rate)
        self._error_rate: float = error_rate
        self._time_budget: float = float(time_budget)
        self._remaining_time_budget: float = float(time_budget)

        # Model configuration
        self._model_type: ModelType = model_type
        self._gamma: float = gamma
        self._num_subspaces_phase2: int = num_subspaces_phase2
        self._model: Optional[ScurveModelBase] = None

        # For model comparison
        self._models: Dict[ModelType, ScurveModelBase] = {}
        self._model_scores: Dict[ModelType, float] = {}

        # Measured samples per second (shots/second)
        self._sampling_rate: float = 0.0

        # Hard cap on total shots per sampling call (to control memory).
        self._MAX_SHOTS_PER_STEP: int = 50_000

        # Circuit-related
        self._cliffordcircuit: CliffordCircuit = CliffordCircuit(4)
        self._num_noise: int = 0
        self._num_detector: int = 0
        self._stim_str_after_rewrite: str = ""
        self._detector_error_model = None
        self._decoder = decoder
        self._matcher = None
        self._QEPG_graph: Optional[QEPGGraph] = None

        # Subspace statistics
        self._subspace_LE_count: Dict[int, int] = {}
        self._subspace_sample_used: Dict[int, int] = {}
        self._estimated_subspaceLER: Dict[int, float] = {}

        # Legacy S-curve model parameters (kept for backward compatibility)
        self._a: float = 0.0
        self._b: float = 0.0
        self._c: float = 0.0
        self._R_square_score: float = 0.0
        self._sweet_spot: Optional[int] = None

        # QEC parameters
        self._circuit_level_code_distance: int = 1
        self._t: int = 0  # (d-1)/2
        self._k_range: int = 5

        # Weight bracketing
        self._has_logical_errorw: int = 1
        self._saturatew: int = 1
        self._minw: int = 1
        self._maxw: int = 1
        self._max_PL: float = 0.15  # plateau threshold

        # Sampling policy constants
        self._min_num_ke_event: int = 30  # min LE events to trust subspace LER

        # Tighter requirements in the band around sweet spot
        self._BAND_HALF_WIDTH: int = 4
        self._TARGET_EVENTS_BAND: int = 160

        # Progressive sampling state
        self._current_frontier: int = 0
        self._SHOTS_PER_SUBSPACE: int = 5000

        # System hyperparameter: maximum TOTAL samples per sampling step (across all weights)
        # Larger batches may cause memory issues; smaller batches allow better budget control
        self._MAX_BATCH_SIZE: int = 50000

        # Final LER
        self._ler: float = 0.0

        # Iteration log for paper/debugging
        self._iteration_log: List[Dict] = []

    # ------------------------------------------------------------------
    #  Model management
    # ------------------------------------------------------------------

    def _initialize_model(self) -> None:
        """Create the S-curve model instance based on ``_model_type``."""
        self._model = ModelFactory.create(
            self._model_type, t=self._t, gamma=self._gamma
        )

    def _initialize_all_models(self) -> None:
        """Create one instance of every registered model type for comparison."""
        self._models = ModelFactory.create_all(t=self._t, gamma=self._gamma)

    def get_model(self) -> Optional[ScurveModelBase]:
        """Return the currently active S-curve model, or ``None``."""
        return self._model

    def set_model_type(self, model_type: ModelType) -> None:
        """
        Change the S-curve model type.

        Args:
            model_type: The new model type to use
        """
        self._model_type = model_type
        if self._model is not None:
            self._initialize_model()

    def set_gamma(self, gamma: float) -> None:
        """
        Set the sweet spot tuning parameter.

        Args:
            gamma: New gamma value
        """
        self._gamma = gamma
        if self._model is not None:
            self._model.gamma = gamma

    def compare_models(self) -> Dict[ModelType, Dict[str, float]]:
        """
        Fit all available models and return their scores.

        Returns:
            Dictionary mapping ModelType to metrics (R², LER estimate)
        """
        if not self._models:
            self._initialize_all_models()

        results: Dict[ModelType, Dict[str, float]] = {}

        # Prepare data for fitting
        weights = list(self._estimated_subspaceLER.keys())
        p_values = [self._estimated_subspaceLER[w] for w in weights]
        sample_counts = [self._subspace_sample_used.get(w, 0) for w in weights]
        le_counts = [self._subspace_LE_count.get(w, 0) for w in weights]

        for model_type, model in self._models.items():
            model.fit(weights, p_values, sample_counts, le_counts)
            ler = self._calc_LER_with_model(model)

            results[model_type] = {
                "r_squared": model.r_squared,
                "ler": ler,
                "sweet_spot": model.calculate_sweet_spot(),
            }
            self._model_scores[model_type] = model.r_squared

        return results

    def select_best_model(self) -> ModelType:
        """
        Select the model with the best R² score.

        Returns:
            The ModelType with highest R² score
        """
        if not self._model_scores:
            self.compare_models()

        best_type = max(self._model_scores, key=lambda x: self._model_scores[x])
        self._model_type = best_type
        self._model = self._models[best_type]
        return best_type

    # ------------------------------------------------------------------
    #  Circuit / QEPG setup and basic helpers
    # ------------------------------------------------------------------

    def parse_from_file(self, filepath: str) -> None:
        """Load a STIM circuit from *filepath* and prepare for sampling.

        Compiles the circuit into a Clifford representation, builds a
        QEPG graph for efficient weight-stratified sampling, and
        initialises a PyMatching decoder from the detector error model.

        Args:
            filepath: Path to a STIM circuit file.
        """
        # A failed replacement load must not expose the previous circuit's fit
        # with partially updated circuit metadata.
        self._model = None
        self._models.clear()
        self._model_scores.clear()
        self._subspace_LE_count.clear()
        self._subspace_sample_used.clear()
        self._estimated_subspaceLER.clear()
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()

        import stim
        from scalerqec.Clifford.stimparser import rewrite_stim_code

        # Flatten REPEAT and canonicalize aliases/spacing using Stim first.
        circuit = stim.Circuit(stim_str).flattened()
        if circuit.num_observables != 1:
            raise ValueError("Scaler currently requires exactly one logical observable (index 0).")
        measurements_names = {"M", "MX", "MY", "MR", "MRX", "MRY"}
        for instruction in circuit:
            name = instruction.name
            if stim.gate_data(name).is_noisy_gate:
                # Measurements are classed as noisy gates even without noise.
                if name not in measurements_names or any(instruction.gate_args_copy()):
                    raise ValueError("Scaler expects a noiseless circuit and applies uniform SID noise; explicit noise is unsupported.")
            if any(t.is_measurement_record_target or t.is_sweep_bit_target or t.is_inverted_result_target
                   for t in instruction.targets_copy()) and name not in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
                raise ValueError("Classical controls and inverted measurements are unsupported by Scaler.")
        # Multiple OBSERVABLE_INCLUDE(0) instructions accumulate by parity.
        measurements = 0
        observable = set()
        body = stim.Circuit()
        for instruction in circuit:
            if instruction.name == "OBSERVABLE_INCLUDE":
                for target in instruction.targets_copy():
                    if not target.is_measurement_record_target:
                        raise ValueError("Only measurement-record logical observables are supported.")
                    index = measurements + target.value
                    observable.symmetric_difference_update({index})
            else:
                body.append(instruction)
            measurements += stim.Circuit(str(instruction)).num_measurements
        if not observable:
            raise ValueError("Scaler requires a nonempty measurement-record logical observable.")
        body.append("OBSERVABLE_INCLUDE", [stim.target_rec(i - measurements) for i in sorted(observable)], 0)
        stim_str = rewrite_stim_code(str(body))
        supported = {"H", "S", "X", "Y", "Z", "CX", "R", "M", "TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}
        for line in stim_str.splitlines():
            name = line.split()[0].split("(")[0]
            if name not in supported:
                raise ValueError(f"Unsupported Scaler instruction: {name}")

        self._cliffordcircuit.compile_from_stim_circuit_str(stim_str)
        self._num_noise = self._cliffordcircuit.totalnoise
        if self._num_noise == 0:
            raise ValueError("Scaler requires at least one SID fault location.")
        self._num_detector = len(self._cliffordcircuit.parityMatchGroup)
        self._stim_str_after_rewrite = stim_str

        # Inject noise for decoder (DEM needs noisy circuit)
        from scalerqec.QEC.noisemodel import SIDNoiseModel

        noisy_stim = SIDNoiseModel(self._error_rate).inject_noise(
            self._cliffordcircuit.stimcircuit
        )

        # Configure a decoder using the noisy circuit.
        self._detector_error_model = noisy_stim.detector_error_model(
            decompose_errors=self._decoder is None
        )
        if self._decoder is not None:
            self._matcher = self._decoder
        else:
            self._matcher = pymatching.Matching.from_detector_error_model(
                self._detector_error_model
            )

        # Compile QEPG graph once.
        self._QEPG_graph = compile_QEPG(stim_str)

    def calc_logical_error_rate_with_fixed_w(self, shots: int, w: int) -> float:
        """Estimate P_L(w) by sampling *shots* fault patterns of weight *w*.

        Generates random fault configurations with exactly *w* faults,
        decodes each one via PyMatching, and returns the fraction that
        produce a logical error.

        Args:
            shots: Number of samples to draw.
            w: Exact Pauli fault weight.

        Returns:
            Estimated conditional logical error probability P_L(w).
        """
        assert self._QEPG_graph is not None, (
            "QEPG graph must be initialized before sampling"
        )
        assert self._matcher is not None, "Matcher must be initialized before decoding"
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots <= 0:
            raise ValueError("shots must be a positive integer.")
        states, observables = return_samples_with_fixed_QEPG_numpy(self._QEPG_graph, w, shots)
        observables = np.asarray(observables).ravel()
        predictions = self._decode_single_observable(states)
        num_errors = np.count_nonzero(observables != predictions)
        return num_errors / shots

    def _decode_single_observable(self, states):
        predictions = np.asarray(self._matcher.decode_batch(states))
        if predictions.shape not in {(len(states),), (len(states), 1)}:
            raise ValueError("Decoder must return one prediction per shot for observable 0.")
        return predictions.reshape(-1)

    # ------------------------------------------------------------------
    #  Binary search helpers to bracket the S-curve
    # ------------------------------------------------------------------

    def binary_search_upper(self, low: int, high: int, shots: int) -> int:
        """Find the smallest weight in [low, high] where P_L exceeds ``_max_PL``.

        Args:
            low: Lower bound of the search range (inclusive).
            high: Upper bound of the search range (inclusive).
            shots: Number of samples per probe.

        Returns:
            The smallest weight at which P_L > ``_max_PL``.
        """
        left = low
        right = high
        epsilon = self._max_PL
        while left < right:
            mid = (left + right) // 2
            er = self.calc_logical_error_rate_with_fixed_w(shots, mid)
            if er > epsilon:
                right = mid
            else:
                left = mid + 1
        return left

    def binary_search_lower(
        self, low: int, high: int, shots: int = 2000, epsilon: float = 0.002
    ) -> int:
        """Find the smallest weight in [low, high] where P_L > *epsilon*.

        Args:
            low: Lower bound of the search range (inclusive).
            high: Upper bound of the search range (inclusive).
            shots: Number of samples per probe.
            epsilon: Detection threshold for P_L.

        Returns:
            The smallest weight at which P_L is detectable.
        """
        left = low
        right = high
        while left < right:
            mid = (left + right) // 2
            er = self.calc_logical_error_rate_with_fixed_w(shots, mid)
            if er > epsilon:
                right = mid
            else:
                left = mid + 1
        return left

    def determine_lower_w(self) -> None:
        """Determine the first weight with detectable logical errors.

        Sets ``_has_logical_errorw`` via binary search.  For very small
        circuits (N <= 8), defaults to ``t + 1``.
        """
        if self._num_noise <= 8:
            self._has_logical_errorw = self._t + 1
        else:
            self._has_logical_errorw = self.binary_search_lower(
                self._t + 1, self._num_noise
            )

    def determine_saturated_w(self) -> None:
        """Determine the weight at which P_L reaches its plateau.

        Sets ``_saturatew`` via binary search.  Ensures a minimum span
        of 8 weights between ``w_err`` and ``w_sat``.
        """
        if self._num_noise <= 8:
            self._saturatew = self._num_noise
        else:
            self._saturatew = self.binary_search_upper(
                self._has_logical_errorw, self._num_noise, shots=2000
            )
            if self._saturatew < self._has_logical_errorw + 8:
                self._saturatew = min(self._num_noise, self._has_logical_errorw + 8)

    # ------------------------------------------------------------------
    #  Sampling rate measurement
    # ------------------------------------------------------------------

    def measure_sample_rates(self) -> None:
        """Calibrate the sampling throughput (shots/second).

        Runs a small 1000-shot benchmark at a representative weight
        and stores the result in ``_sampling_rate`` for subsequent
        time-budget calculations.  Also deducts the calibration time
        from ``_remaining_time_budget``.
        """
        assert self._QEPG_graph is not None, (
            "QEPG graph must be initialized before sampling"
        )
        wlist = [max(1, self._num_noise // 2)]
        slist = [1000]

        start_time = time.perf_counter()
        print("Start time for sampling rate measurement:", start_time)
        _det, _obs = return_samples_many_weights_separate_obs_with_QEPG(
            self._QEPG_graph, wlist, slist
        )
        self._decode_single_observable(_det)
        end_time = time.perf_counter()
        print("End time for sampling rate measurement:", end_time)
        elapsed = end_time - start_time
        if elapsed <= 0:
            elapsed = 1e-6
        self._sampling_rate = 1000.0 / elapsed
        self._remaining_time_budget -= elapsed

        print(
            "Elapsed time for sampling rate measurement: {:.6f} seconds".format(elapsed)
        )
        print(f"Measured sampling rate: {self._sampling_rate:.2f} shots/second")

    def profile_optimal_batch_size(
        self,
        test_weight: int | None = None,
        batch_sizes: List[int] | None = None,
        save_plot: str | None = None,
    ) -> tuple[int, dict]:
        """
        Profile sampling+decoding throughput at different batch sizes to find optimal.

        Args:
            test_weight: Weight to test at (default: num_noise // 2)
            batch_sizes: List of batch sizes to test (default: exponential range)
            save_plot: If provided, save throughput plot to this filename

        Returns:
            (optimal_batch_size, results_dict)
            results_dict contains: {batch_size: {'throughput', 'sample_time', 'decode_time', 'total_time'}}
        """
        assert self._QEPG_graph is not None, "QEPG graph must be initialized"
        assert self._matcher is not None, "Matcher must be initialized"

        if test_weight is None:
            test_weight = max(1, self._num_noise // 2)

        if batch_sizes is None:
            batch_sizes = [
                1000,
                2000,
                5000,
                10000,
                20000,
                50000,
                100000,
                200000,
                500000,
            ]

        results = {}
        best_throughput = 0
        optimal_batch = batch_sizes[0]

        print(f"\nProfiling batch sizes at weight={test_weight}...")
        print("=" * 70)
        print(
            f"{'Batch Size':>12} | {'Sample Time':>12} | {'Decode Time':>12} | {'Total Time':>12} | {'Throughput':>15}"
        )
        print("-" * 70)

        for batch in batch_sizes:
            wlist = [test_weight]
            slist = [batch]

            # Measure sampling time
            sample_start = time.perf_counter()
            detector_result, obsresult = (
                return_samples_many_weights_separate_obs_with_QEPG(
                    self._QEPG_graph, wlist, slist
                )
            )
            sample_end = time.perf_counter()
            sample_time = sample_end - sample_start

            # Measure decoding time
            decode_start = time.perf_counter()
            _predictions = self._matcher.decode_batch(detector_result)
            decode_end = time.perf_counter()
            decode_time = decode_end - decode_start

            total_time = sample_time + decode_time
            throughput = batch / total_time if total_time > 0 else 0

            results[batch] = {
                "throughput": throughput,
                "sample_time": sample_time,
                "decode_time": decode_time,
                "total_time": total_time,
            }

            print(
                f"{batch:>12,} | {sample_time:>11.4f}s | {decode_time:>11.4f}s | {total_time:>11.4f}s | {throughput:>12,.0f}/s"
            )

            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch = batch

            # Early stopping: if throughput drops significantly, we've passed the peak
            if throughput < best_throughput * 0.7 and batch > optimal_batch:
                print(f"  (Stopping early: throughput dropped below 70% of peak)")
                break

        print("=" * 70)
        print(
            f"Optimal batch size: {optimal_batch:,} (throughput: {best_throughput:,.0f} shots/s)"
        )

        # Save plot if requested
        if save_plot:
            self._plot_batch_profile(results, optimal_batch, save_plot)

        return optimal_batch, results

    def _plot_batch_profile(
        self,
        results: dict,
        optimal_batch: int,
        filename: str,
    ) -> None:
        """Generate a three-panel plot of batch-size profiling results.

        Panels show throughput vs. batch size, time breakdown
        (sampling vs. decoding), and per-shot efficiency.

        Args:
            results: Dict mapping batch size to timing metrics (from
                :meth:`profile_optimal_batch_size`).
            optimal_batch: The batch size with peak throughput.
            filename: Output path for the PNG/PDF plot.
        """
        batch_sizes = sorted(results.keys())
        throughputs = [results[b]["throughput"] for b in batch_sizes]
        sample_times = [results[b]["sample_time"] for b in batch_sizes]
        decode_times = [results[b]["decode_time"] for b in batch_sizes]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Throughput vs Batch Size
        ax1 = axes[0]
        ax1.plot(batch_sizes, throughputs, "b-o", linewidth=2, markersize=8)
        ax1.axvline(
            optimal_batch,
            color="red",
            linestyle="--",
            label=f"Optimal: {optimal_batch:,}",
        )
        ax1.set_xscale("log")
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Throughput (shots/s)")
        ax1.set_title("Throughput vs Batch Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Time breakdown
        ax2 = axes[1]
        ax2.plot(batch_sizes, sample_times, "g-o", label="Sample Time", linewidth=2)
        ax2.plot(batch_sizes, decode_times, "r-o", label="Decode Time", linewidth=2)
        ax2.set_xscale("log")
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Time (s)")
        ax2.set_title("Time Breakdown")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Time per shot
        ax3 = axes[2]
        time_per_shot = [
            results[b]["total_time"] / b * 1000 for b in batch_sizes
        ]  # ms per shot
        ax3.plot(batch_sizes, time_per_shot, "m-o", linewidth=2, markersize=8)
        ax3.axvline(
            optimal_batch,
            color="red",
            linestyle="--",
            label=f"Optimal: {optimal_batch:,}",
        )
        ax3.set_xscale("log")
        ax3.set_xlabel("Batch Size")
        ax3.set_ylabel("Time per Shot (ms)")
        ax3.set_title("Efficiency: Time per Shot")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved batch profile plot to: {filename}")

    # ------------------------------------------------------------------
    #  One multi-weight sampling step
    # ------------------------------------------------------------------

    def _sampling_step(self, wlist: List[int], slist: List[int]) -> float:
        """Perform one multi-weight sampling call and update subspace statistics.

        Samples each weight in *wlist* with the corresponding number
        of shots from *slist*, decodes the results, and updates the
        internal ``_subspace_LE_count``, ``_subspace_sample_used``,
        and ``_estimated_subspaceLER`` dictionaries.  Also updates
        ``_sampling_rate`` via an exponential moving average.

        Args:
            wlist: List of fault weights to sample.
            slist: Number of shots for each weight (same length as
                *wlist*).

        Returns:
            Wall-clock time (seconds) consumed by this step.
        """
        assert self._QEPG_graph is not None, (
            "QEPG graph must be initialized before sampling"
        )
        assert self._matcher is not None, "Matcher must be initialized before decoding"
        if len(wlist) != len(slist):
            raise ValueError("weights and shots must have equal lengths.")
        if any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s <= 0 for s in slist):
            raise ValueError("Each shot count must be a positive integer.")
        if not wlist:
            return 0.0

        total_shots = int(sum(slist))
        print("Sampling weights and shots:", list(zip(wlist, slist)))
        print("  Total shots this step:", total_shots)

        start_time = time.perf_counter()
        detector_result, obsresult = return_samples_many_weights_separate_obs_with_QEPG(
            self._QEPG_graph, wlist, slist
        )
        predictions_result = self._decode_single_observable(detector_result)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        begin_index = 0
        for w, shots in zip(wlist, slist):
            end_index = begin_index + shots
            observables = np.asarray(obsresult[begin_index:end_index])
            predictions = np.asarray(predictions_result[begin_index:end_index]).ravel()

            num_errors = int(np.count_nonzero(observables != predictions))

            self._subspace_LE_count[w] = self._subspace_LE_count.get(w, 0) + num_errors
            self._subspace_sample_used[w] = self._subspace_sample_used.get(w, 0) + shots
            self._estimated_subspaceLER[w] = (
                self._subspace_LE_count[w] / self._subspace_sample_used[w]
            )

            begin_index = end_index

        print("  PL values:")
        for w in wlist:
            pl = self._estimated_subspaceLER.get(w, 0)
            le = self._subspace_LE_count.get(w, 0)
            samples = self._subspace_sample_used.get(w, 0)
            print(f"    w={w}: PL={pl:.4f}, LE_count={le}, samples={samples}")

        if elapsed > 0 and total_shots > 0:
            inst_rate = total_shots / elapsed
            if self._sampling_rate <= 0:
                self._sampling_rate = inst_rate
            else:
                alpha = 0.5
                self._sampling_rate = (
                    alpha * inst_rate + (1.0 - alpha) * self._sampling_rate
                )

        print(f"  Step elapsed: {elapsed:.6f} s")
        print(f"  Updated sampling rate: {self._sampling_rate:.2f} shots/second")

        return elapsed

    # ------------------------------------------------------------------
    #  Budget-aware sampling cost estimation
    # ------------------------------------------------------------------

    def _estimate_sampling_cost(self, w: int, target_le_events: int = 30) -> float:
        """Estimate the time (seconds) to collect *target_le_events* at weight *w*.

        Uses the measured or model-predicted P_L(w) and the calibrated
        ``_sampling_rate`` to estimate the required wall-clock time.

        Args:
            w: Fault weight.
            target_le_events: Number of logical error events to target.

        Returns:
            Estimated seconds, or ``inf`` if estimation is impossible.
        """
        # Get P_L(w) from measured data if available, otherwise from model
        if w in self._estimated_subspaceLER and self._estimated_subspaceLER[w] > 0:
            pl_w = self._estimated_subspaceLER[w]
        elif self._model is not None:
            pl_w = float(self._model.predict(w))
        else:
            return float("inf")  # Can't estimate without model

        if pl_w <= 0:
            return float("inf")  # Can't estimate

        # Expected samples needed = target_events / P_L(w)
        samples_needed = target_le_events / pl_w

        # Time = samples / sampling_rate
        if self._sampling_rate <= 0:
            return float("inf")

        return samples_needed / self._sampling_rate

    def _get_practical_sweet_spot(self, remaining_budget: float) -> int:
        """Determine a budget-feasible sweet-spot weight.

        If sampling at the theoretical sweet spot (computed by the
        model's :meth:`calculate_sweet_spot`) would exceed
        *remaining_budget*, this method shifts to a higher weight
        where P_L is larger and therefore fewer samples are needed.

        Args:
            remaining_budget: Remaining wall-clock time in seconds.

        Returns:
            A weight >= theoretical sweet spot that can be sampled
            within the given budget.
        """
        if self._sweet_spot is None:
            return self._t + 1

        theoretical_w = self._sweet_spot
        target_le = self._min_num_ke_event  # 30 by default

        # Estimate cost at theoretical sweet spot
        estimated_cost = self._estimate_sampling_cost(theoretical_w, target_le)

        # If within budget, use theoretical sweet spot (no adjustment)
        if estimated_cost <= remaining_budget:
            return theoretical_w

        # Only if budget is insufficient: find practical sweet spot
        # Move to higher weights where P_L is larger (cheaper to sample)
        max_w = self._saturatew

        for w in range(theoretical_w + 1, max_w + 1):
            cost = self._estimate_sampling_cost(w, target_le)
            if cost <= remaining_budget:
                print(
                    f"  Practical sweet_spot: {theoretical_w} -> {w} (budget constraint)"
                )
                return w

        # If no weight fits budget, use maximum available (highest P_L)
        return max_w

    def _get_weights_around(self, center_w: int) -> List[int]:
        """Return weights within ``BAND_HALF_WIDTH`` of *center_w*.

        Args:
            center_w: Centre weight of the band.

        Returns:
            Sorted list of valid weights in ``(t, w_sat]`` near *center_w*.
        """
        band = self._BAND_HALF_WIDTH  # 4 by default
        weights = []

        for delta in range(-band, band + 1):
            w = center_w + delta
            if self._t < w <= self._saturatew:
                weights.append(w)

        return sorted(weights)

    def _get_additional_sweet_spot_weights(self) -> List[int]:
        """Select extra weights to sample when the budget has not been exhausted.

        Prioritises un-sampled weights in a wide band around the sweet
        spot, then falls back to adding more samples to the closest
        already-sampled weights.

        Returns:
            Up to 5 weights to sample next.
        """
        if self._sweet_spot is None:
            return []

        # First priority: find weights not yet sampled in wide range
        new_weights = []
        for delta in range(-20, 21):  # Wide range
            w = self._sweet_spot + delta
            if self._t < w <= self._saturatew:
                if w not in self._subspace_sample_used:
                    new_weights.append(w)

        if new_weights:
            return sorted(new_weights)[:5]

        # Second priority: add more samples to existing weights (prioritize near sweet spot)
        weights_by_distance = []
        for w in self._subspace_sample_used.keys():
            if (
                self._subspace_sample_used[w] < self._MAX_BATCH_SIZE * 10
            ):  # Can sample more
                dist = abs(w - self._sweet_spot)
                weights_by_distance.append((dist, w))

        weights_by_distance.sort()  # Sort by distance to sweet spot
        return [w for _, w in weights_by_distance[:5]]

    # ------------------------------------------------------------------
    #  Fitting the log-S model (using model abstraction)
    # ------------------------------------------------------------------

    def fit_log_S_model(
        self,
        filename: str | None = None,
        savefigure: bool = False,
        time_val: float | None = None,
        practical_sweet_spot: int | None = None,
    ) -> None:
        """Fit the S-curve model to the currently collected data.

        Delegates to the active model's :meth:`fit` method in log-logit
        space, updates the R^2 score and sweet-spot estimate, and
        optionally generates diagnostic plots.

        Args:
            filename: Output path for the diagnostic PDF plot.
            savefigure: If ``True``, save the diagnostic plot.
            time_val: Elapsed wall-clock time (seconds) to annotate on
                the plot.
            practical_sweet_spot: If provided, annotate this weight on
                the plot as the budget-adjusted sweet spot.
        """
        # Initialize model if not done yet
        if self._model is None:
            self._initialize_model()

        assert self._model is not None

        # Prepare data for fitting
        weights = sorted(self._estimated_subspaceLER.keys())
        p_values = [self._estimated_subspaceLER[w] for w in weights]
        sample_counts = [self._subspace_sample_used.get(w, 0) for w in weights]
        le_counts = [self._subspace_LE_count.get(w, 0) for w in weights]

        # Fit the model
        self._model.fit(weights, p_values, sample_counts, le_counts)

        # Update legacy parameters for backward compatibility
        self._R_square_score = self._model.r_squared

        if isinstance(self._model, OurScurveModel):
            internal = self._model.get_internal_params()
            self._a = internal["a"]
            self._b = internal["b"]
            self._c = internal["c"]

        # Determine minw / maxw region
        sigma = int(
            np.sqrt(self._error_rate * (1.0 - self._error_rate) * self._num_noise)
        )
        if sigma == 0:
            sigma = 1
        ep = int(self._error_rate * self._num_noise)
        self._minw = max(self._t + 1, ep - self._k_range * sigma)
        self._maxw = min(self._num_noise, ep + self._k_range * sigma)

        # Update sweet spot using the model
        w_sweet = self._model.calculate_sweet_spot()
        if w_sweet <= self._t:
            w_sweet = self._t + 1
        w_sweet = min(self._saturatew, max(self._t + 1, w_sweet))
        self._sweet_spot = w_sweet

        # Plot for debugging
        if savefigure:
            self._plot_fit(filename, time_val, practical_sweet_spot)

    def _plot_fit(
        self,
        filename: str | None,
        time_val: float | None,
        practical_sweet_spot: int | None = None,
    ) -> None:
        """Generate a diagnostic plot of the fitted S-curve in log-logit space.

        Plots measured data points as bars with error bars, the fitted
        curve, annotated key weights (``w_err``, ``w_sat``,
        ``w_sweet``), the fault-tolerant region, and the critical
        region ``[w_min, w_max]``.  Also produces a companion S-curve
        plot in probability space via :meth:`_plot_scurve`.

        Args:
            filename: Output path for the PDF. If ``None``, a default
                name is generated.
            time_val: Elapsed time (seconds) to annotate.
            practical_sweet_spot: Budget-adjusted sweet spot to annotate.
        """
        assert self._model is not None

        # Build data lists
        x_list = [
            x
            for x in self._estimated_subspaceLER.keys()
            if 0.0 < self._estimated_subspaceLER[x] < 0.5
            and self._subspace_LE_count.get(x, 0) > 0
        ]

        if not x_list:
            return

        x_list = sorted(x_list)

        # Get y values in transformed space and calculate error bars
        y_list = []
        y_err_list = []
        for x in x_list:
            p_w = self._estimated_subspaceLER[x]
            n_samples = self._subspace_sample_used.get(x, 1)
            y_val = float(self._model.transform(p_w))
            y_list.append(y_val)

            # Calculate standard error for P_w using binomial variance
            # SE(P_w) = sqrt(P_w * (1 - P_w) / n)
            se_p = np.sqrt(p_w * (1 - p_w) / n_samples) if n_samples > 0 else 0

            # Propagate error to transformed space using delta method
            # y = log(0.5/p - 1), dy/dp = -0.5 / (p * (0.5 - p))
            if 0 < p_w < 0.5 and se_p > 0:
                dy_dp = -0.5 / (p_w * (0.5 - p_w))
                y_err = abs(dy_dp) * se_p
            else:
                y_err = 0
            y_err_list.append(y_err)

        # x-range for fitted curve
        x_fit = np.linspace(self._t + 1, max(x_list), 1000)
        y_fit = [float(self._model.linear_prediction(x)) for x in x_fit]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate bar width based on data spacing
        if len(x_list) > 1:
            bar_width = min(np.diff(sorted(x_list))) * 0.6
        else:
            bar_width = 1.0

        # Plot data points as bars with error bars
        ax.bar(
            x_list,
            y_list,
            width=bar_width,
            color="orange",
            alpha=0.7,
            edgecolor="darkorange",
            linewidth=1.2,
            label="Data points (log-S)",
            yerr=y_err_list,
            capsize=3,
            error_kw={"elinewidth": 1.5, "capthick": 1.2, "ecolor": "black"},
        )

        # Fitted curve
        ax.plot(
            x_fit,
            y_fit,
            label=f"Fitted ({self._model.name}), R²={self._R_square_score:.4f}",
            color="blue",
            linestyle="-",
            linewidth=2,
        )

        # w_err (first weight with logical error) annotation
        if self._has_logical_errorw is not None:
            ax.axvline(
                self._has_logical_errorw, color="brown", linestyle=":", linewidth=2
            )
            ax.annotate(
                r"$w_{\mathrm{err}}$" + f"={self._has_logical_errorw}",
                xy=(self._has_logical_errorw, ax.get_ylim()[1] * 0.95),
                xytext=(self._has_logical_errorw + 3, ax.get_ylim()[1] * 0.95),
                fontsize=11,
                color="brown",
                arrowprops=dict(arrowstyle="->", color="brown"),
            )

        # w_sat annotation
        ax.axvline(self._saturatew, color="darkgreen", linestyle=":", linewidth=2)
        ax.annotate(
            r"$w_{\mathrm{sat}}$" + f"={self._saturatew}",
            xy=(self._saturatew, ax.get_ylim()[1] * 0.85),
            xytext=(self._saturatew - 8, ax.get_ylim()[1] * 0.85),
            fontsize=11,
            color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen"),
        )

        # Sweet spot marker (theoretical)
        if self._sweet_spot is not None:
            sweet_spot_y = float(self._model.linear_prediction(self._sweet_spot))
            ax.scatter(
                self._sweet_spot,
                sweet_spot_y,
                color="purple",
                marker="*",
                s=200,
                zorder=5,
                label=r"$w_{\mathrm{sweet}}$" + f"={self._sweet_spot}",
            )
            ax.annotate(
                r"$w_{\mathrm{sweet}}$" + f"={self._sweet_spot}",
                xy=(self._sweet_spot, sweet_spot_y),
                xytext=(self._sweet_spot + 5, sweet_spot_y + 0.5),
                fontsize=11,
                color="purple",
                arrowprops=dict(arrowstyle="->", color="purple"),
            )

        # Practical sweet spot marker (if different from theoretical)
        if (
            practical_sweet_spot is not None
            and practical_sweet_spot != self._sweet_spot
        ):
            practical_y = float(self._model.linear_prediction(practical_sweet_spot))
            ax.scatter(
                practical_sweet_spot,
                practical_y,
                color="red",
                marker="s",
                s=100,
                zorder=5,
            )
            ax.annotate(
                f"Practical={practical_sweet_spot}",
                xy=(practical_sweet_spot, practical_y),
                xytext=(practical_sweet_spot + 3, practical_y + 0.3),
                fontsize=10,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red"),
            )

        # Fault-tolerant region
        ax.axvspan(0, self._t, color="green", alpha=0.15)
        ax.text(
            self._t / 2,
            max(y_list) * 0.9 if y_list else 1.0,
            "Fault\ntolerant",
            ha="center",
            color="green",
            fontsize=10,
            fontweight="bold",
        )

        # Critical region annotation (w_min to w_max)
        ax.axvspan(self._minw, self._maxw, color="lightblue", alpha=0.3)
        ax.axvline(self._minw, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(self._maxw, color="red", linestyle="--", linewidth=1.5)

        # Add bracket annotation for critical region
        mid_critical = (self._minw + self._maxw) / 2
        ax.annotate(
            "",
            xy=(self._minw, ax.get_ylim()[0] + 0.3),
            xytext=(self._maxw, ax.get_ylim()[0] + 0.3),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )
        ax.text(
            mid_critical,
            ax.get_ylim()[0] + 0.6,
            f"Critical Region\n"
            + r"$[w_{\min}="
            + f"{self._minw}"
            + r", w_{\max}="
            + f"{self._maxw}]$",
            ha="center",
            color="red",
            fontsize=10,
            fontweight="bold",
        )

        # Side annotation box
        params = self._model.get_params()
        text_lines = [
            f"Model: {self._model.name}",
            f"γ (gamma): {self._gamma:.3f}",
            "",
            "Fitted Parameters:",
        ]
        for name, value in params.items():
            text_lines.append(f"  {name}: {value:.4f}")
        text_lines.extend(
            [
                "",
                "Key Weights:",
                f"  w_err: {self._has_logical_errorw}",
                f"  w_min: {self._minw}",
                f"  w_max: {self._maxw}",
                f"  w_sweet: {self._sweet_spot}",
                f"  w_sat: {self._saturatew}",
                "",
                "Circuit Info:",
                f"  #detector: {self._num_detector}",
                f"  #noise: {self._num_noise}",
            ]
        )
        if self._ler > 0:
            text_lines.append(f"  P_L: {self._ler:.2e}")
        if time_val is not None:
            text_lines.append(f"  Time: {time_val:.2f}s")

        fig.subplots_adjust(right=0.72)
        fig.text(
            0.74,
            0.5,
            "\n".join(text_lines),
            fontsize=8,
            va="center",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
        )

        ax.set_xlabel("Weight $w$", fontsize=12)
        ax.set_ylabel(r"$\log\left(\frac{0.5}{P_L(w)} - 1\right)$", fontsize=12)
        ax.set_title(
            f"Log-S Curve Fit (d={self._circuit_level_code_distance}, p={self._error_rate})",
            fontsize=14,
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if filename is None:
            filename = f"logS_fit_debug_p{self._error_rate:.3g}_d{self._circuit_level_code_distance}.pdf"

        print(f"Saving log-S fit figure to: {filename}")
        fig.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Also generate Y-curve (S-curve in original probability space)
        self._plot_scurve(
            filename.replace(".pdf", "_Scurve.pdf"), time_val, practical_sweet_spot
        )

    def _plot_scurve(
        self,
        filename: str,
        time_val: float | None,
        practical_sweet_spot: int | None = None,
    ) -> None:
        """Generate the S-curve plot in original probability space.

        Similar to :meth:`_plot_fit` but shows P_L(w) directly rather
        than the log-logit transform, with a saturation line at 0.5.

        Args:
            filename: Output path for the PDF.
            time_val: Elapsed time (seconds) to annotate.
            practical_sweet_spot: Budget-adjusted sweet spot to annotate.
        """
        assert self._model is not None

        # Build data lists
        x_list = [
            x
            for x in self._estimated_subspaceLER.keys()
            if 0.0 < self._estimated_subspaceLER[x] < 0.5
            and self._subspace_LE_count.get(x, 0) > 0
        ]

        if not x_list:
            return

        x_list = sorted(x_list)

        # Get y values (P_L(w)) and error bars
        y_list = []
        y_err_list = []
        for x in x_list:
            p_w = self._estimated_subspaceLER[x]
            n_samples = self._subspace_sample_used.get(x, 1)
            y_list.append(p_w)

            # Standard error for P_w
            se_p = np.sqrt(p_w * (1 - p_w) / n_samples) if n_samples > 0 else 0
            y_err_list.append(se_p)

        # x-range for fitted curve
        x_fit = np.linspace(self._t + 1, max(x_list), 1000)
        y_fit = [float(self._model.predict(x)) for x in x_fit]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate bar width based on data spacing
        if len(x_list) > 1:
            bar_width = min(np.diff(sorted(x_list))) * 0.6
        else:
            bar_width = 1.0

        # Plot data points as bars with error bars
        ax.bar(
            x_list,
            y_list,
            width=bar_width,
            color="orange",
            alpha=0.7,
            edgecolor="darkorange",
            linewidth=1.2,
            label="Measured $P_L(w)$",
            yerr=y_err_list,
            capsize=3,
            error_kw={"elinewidth": 1.5, "capthick": 1.2, "ecolor": "black"},
        )

        # Fitted curve
        ax.plot(
            x_fit,
            y_fit,
            label=f"Fitted S-curve, R²={self._R_square_score:.4f}",
            color="blue",
            linestyle="-",
            linewidth=2,
        )

        # w_err annotation
        if self._has_logical_errorw is not None:
            ax.axvline(
                self._has_logical_errorw, color="brown", linestyle=":", linewidth=2
            )
            ax.annotate(
                r"$w_{\mathrm{err}}$" + f"={self._has_logical_errorw}",
                xy=(self._has_logical_errorw, 0.45),
                xytext=(self._has_logical_errorw + 3, 0.45),
                fontsize=11,
                color="brown",
                arrowprops=dict(arrowstyle="->", color="brown"),
            )

        # w_sat annotation
        ax.axvline(self._saturatew, color="darkgreen", linestyle=":", linewidth=2)
        ax.annotate(
            r"$w_{\mathrm{sat}}$" + f"={self._saturatew}",
            xy=(self._saturatew, 0.35),
            xytext=(self._saturatew - 8, 0.35),
            fontsize=11,
            color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen"),
        )

        # Sweet spot marker
        if self._sweet_spot is not None:
            sweet_spot_y = float(self._model.predict(self._sweet_spot))
            ax.scatter(
                self._sweet_spot,
                sweet_spot_y,
                color="purple",
                marker="*",
                s=200,
                zorder=5,
                label=r"$w_{\mathrm{sweet}}$" + f"={self._sweet_spot}",
            )
            ax.annotate(
                r"$w_{\mathrm{sweet}}$" + f"={self._sweet_spot}",
                xy=(self._sweet_spot, sweet_spot_y),
                xytext=(self._sweet_spot + 5, sweet_spot_y + 0.05),
                fontsize=11,
                color="purple",
                arrowprops=dict(arrowstyle="->", color="purple"),
            )

        # Fault-tolerant region
        ax.axvspan(0, self._t, color="green", alpha=0.15)
        ax.text(
            self._t / 2,
            0.4,
            "Fault\ntolerant",
            ha="center",
            color="green",
            fontsize=10,
            fontweight="bold",
        )

        # Critical region
        ax.axvspan(self._minw, self._maxw, color="lightblue", alpha=0.3)
        ax.axvline(self._minw, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(self._maxw, color="red", linestyle="--", linewidth=1.5)

        mid_critical = (self._minw + self._maxw) / 2
        ax.text(
            mid_critical,
            0.02,
            f"Critical Region\n" + r"$[w_{\min}, w_{\max}]$",
            ha="center",
            color="red",
            fontsize=10,
            fontweight="bold",
        )

        # Saturation line at 0.5
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(
            max(x_list) + 1, 0.5, "Saturation", fontsize=9, color="gray", va="center"
        )

        # Side annotation box
        params = self._model.get_params()
        text_lines = [
            f"Model: {self._model.name}",
            "",
            "Fitted Parameters:",
        ]
        for name, value in params.items():
            text_lines.append(f"  {name}: {value:.4f}")
        text_lines.extend(
            [
                "",
                "Key Weights:",
                f"  w_sweet: {self._sweet_spot}",
                f"  w_sat: {self._saturatew}",
                "",
                f"Estimated P_L: {self._ler:.2e}",
            ]
        )
        if time_val is not None:
            text_lines.append(f"Time: {time_val:.2f}s")

        fig.subplots_adjust(right=0.75)
        fig.text(
            0.77,
            0.5,
            "\n".join(text_lines),
            fontsize=8,
            va="center",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
        )

        ax.set_xlabel("Weight $w$", fontsize=12)
        ax.set_ylabel(r"$P_L(w)$", fontsize=12)
        ax.set_title(
            f"S-Curve (d={self._circuit_level_code_distance}, p={self._error_rate})",
            fontsize=14,
        )
        ax.set_ylim(-0.02, 0.55)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        print(f"Saving S-curve figure to: {filename}")
        fig.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    #  Parameter / band / PL stability checks
    # ------------------------------------------------------------------

    def params_stable(
        self,
        theta_new: tuple[float, ...],
        theta_old: tuple[float, ...] | None,
        tol: float,
        r2: float,
        r2_target: float,
    ) -> bool:
        """Check whether fitted parameters have converged.

        Returns ``True`` when the relative change between *theta_new*
        and *theta_old* is below *tol* **and** the R^2 score meets
        *r2_target*.

        Args:
            theta_new: Current fitted parameter vector.
            theta_old: Previous fitted parameter vector, or ``None``
                on the first iteration.
            tol: Maximum allowed relative L2 change.
            r2: Current R^2 score.
            r2_target: Minimum acceptable R^2.

        Returns:
            ``True`` if parameters are stable and fit quality is sufficient.
        """
        if theta_old is None:
            return False
        num = sum((x - y) ** 2 for x, y in zip(theta_new, theta_old)) ** 0.5
        den = max(1e-12, sum(y**2 for y in theta_old) ** 0.5)
        rel = num / den
        return (rel < tol) and (r2 >= r2_target)

    def _band_weights(self) -> List[int]:
        """Return the list of weights in the refinement band around the sweet spot.

        Returns:
            Sorted list of weights within ``BAND_HALF_WIDTH`` of
            ``_sweet_spot``, clamped to ``[w_err, w_sat]``.
        """
        if self._sweet_spot is None:
            return []
        left = max(self._has_logical_errorw, self._sweet_spot - self._BAND_HALF_WIDTH)
        right = min(self._saturatew, self._sweet_spot + self._BAND_HALF_WIDTH)
        return list(range(left, right + 1))

    def _band_well_sampled(self) -> bool:
        """Check if all weights in the refinement band are sufficiently sampled.

        Returns:
            ``True`` if every band weight has at least
            ``_TARGET_EVENTS_BAND`` logical error events.
        """
        band = self._band_weights()
        if not band:
            return False
        for w in band:
            if self._subspace_LE_count.get(w, 0) < self._TARGET_EVENTS_BAND:
                return False
        return True

    def _pl_stable(
        self, pl_new: float, pl_old: Optional[float], rel_tol: float
    ) -> bool:
        """Check if the overall LER estimate has converged.

        Args:
            pl_new: Current LER estimate.
            pl_old: Previous LER estimate, or ``None`` on the first call.
            rel_tol: Maximum allowed relative change.

        Returns:
            ``True`` if the relative change is below *rel_tol*.
        """
        if pl_old is None:
            return False
        if pl_old == 0.0:
            return abs(pl_new) < rel_tol
        rel = abs(pl_new - pl_old) / abs(pl_old)
        return rel < rel_tol

    # ------------------------------------------------------------------
    #  Decide the next sampling step
    # ------------------------------------------------------------------

    def _choose_candidate_weights(self) -> List[int]:
        """Select candidate weights for the next sampling step.

        Combines weights near the sweet spot (+-2) with a 5-point
        evenly-spaced grid spanning ``[w_min, w_max]``, and always
        includes the boundary weights.

        Returns:
            Sorted list of candidate weights.
        """
        if self._sweet_spot is None:
            ep = int(self._error_rate * self._num_noise)
            center = max(self._t + 1, ep)
        else:
            center = int(self._sweet_spot)

        left_bracket = min(self._minw, center)
        right_bracket = max(self._maxw, center)

        left_bracket = max(self._t + 1, left_bracket)
        right_bracket = min(self._saturatew, right_bracket)

        if right_bracket < left_bracket:
            left_bracket = self._t + 1
            right_bracket = self._saturatew

        if right_bracket < left_bracket:
            return [center]

        if center < left_bracket:
            center = left_bracket
        if center > right_bracket:
            center = right_bracket

        W = set()

        for delta in range(-2, 3):
            w = center + delta
            if left_bracket <= w <= right_bracket:
                W.add(w)

        grid_points = evenly_spaced_ints(left_bracket, right_bracket, 5)
        for w in grid_points:
            if left_bracket <= w <= right_bracket:
                W.add(w)

        W.add(left_bracket)
        W.add(right_bracket)

        wlist = sorted(W)
        print("  Candidate weights (choose_candidate_weights):", wlist)
        return wlist

    def next_step(self) -> Tuple[List[int], List[int]]:
        """Plan the next sampling step within the time budget.

        Determines which weights to sample and how many shots to
        allocate to each, biasing towards the sweet spot and
        un-sampled weights.

        Returns:
            Tuple ``(wlist, slist)`` of weights and corresponding
            shot counts.  Both lists are empty if no budget remains.
        """
        if self._sampling_rate <= 0.0 or self._remaining_time_budget <= 0.0:
            return [], []

        remaining_shots = int(self._remaining_time_budget * self._sampling_rate)
        if remaining_shots <= 0:
            return [], []

        step_shots = min(remaining_shots, self._MAX_SHOTS_PER_STEP)

        wlist = self._choose_candidate_weights()
        if not wlist:
            return [], []

        factors = []
        center = self._sweet_spot

        for w in wlist:
            f = 1.0
            if center is not None:
                d = abs(w - center)
                if d <= 1:
                    f *= 6.0
                elif d <= 2:
                    f *= 3.0
            if self._subspace_sample_used.get(w, 0) == 0:
                f *= 4.0
            factors.append(f)

        total_factor = sum(factors)
        if total_factor <= 0.0:
            shots_per_w = max(1, step_shots // len(wlist))
            return wlist, [shots_per_w] * len(wlist)

        min_shots_per_w = max(500, int(step_shots * 0.02))

        slist: List[int] = []
        remaining = step_shots

        for i, (w, f) in enumerate(zip(wlist, factors)):
            if i == len(wlist) - 1:
                s = max(1, remaining)
            else:
                raw = step_shots * (f / total_factor)
                s = max(min_shots_per_w, int(round(raw)))
                remaining -= s
                if remaining < 0:
                    remaining = 0
            slist.append(s)

        print("  Candidate weights (choose_candidate_weights):", wlist)
        print("  Allocated shots:", slist, " (total =", sum(slist), ")")

        return wlist, slist

    # ------------------------------------------------------------------
    #  Final LER integration from fitted S-curve
    # ------------------------------------------------------------------

    def _calc_LER_from_fit(self) -> float:
        """Compute the total LER by integrating the fitted S-curve.

        Uses measured P_L(w) where available and the model's
        prediction for un-sampled weights in the critical region.

        Returns:
            The estimated total logical error rate.
        """
        if self._model is None:
            self._initialize_model()

        assert self._model is not None
        return self._calc_LER_with_model(self._model)

    def _calc_LER_with_model(self, model: ScurveModelBase) -> float:
        """Compute the total LER using a specific S-curve model.

        For each weight in the critical region, uses the measured
        P_L(w) if available, otherwise the model's prediction, and
        multiplies by the binomial weight.

        Args:
            model: The S-curve model to use for un-sampled weights.

        Returns:
            The estimated total logical error rate.
        """
        self._ler = self._make_profile(model).evaluate(self._error_rate)
        return self._ler

    def _make_profile(self, model):
        import hashlib
        from importlib.metadata import version

        weights = np.arange(self._num_noise + 1)
        rates = np.asarray(model.predict(weights), dtype=float).copy()
        modeled = weights > self._t
        rates[~modeled] = 0.0
        samples = np.zeros(len(weights), dtype=np.int64)
        failures = np.zeros(len(weights), dtype=np.int64)
        for w, count in self._subspace_sample_used.items():
            samples[w] = count
            failures[w] = self._subspace_LE_count.get(w, 0)
        for w, rate in self._estimated_subspaceLER.items():
            if w > self._t:
                rates[w] = rate
                modeled[w] = False
            elif rate > 0:
                raise ValueError("Observed failures contradict the supplied circuit-level distance.")
        return LERProfile(rates, modeled_weights=modeled, sample_counts=samples,
                          failure_counts=failures, metadata={
            "noise_model": "uniform-independent-SID-before-primitive-gates",
            "decoder_policy": "fixed", "decoder_reference_p": self._error_rate,
            "decoder_class": type(self._matcher).__module__ + "." + type(self._matcher).__qualname__,
            "circuit_sha256": hashlib.sha256(self._stim_str_after_rewrite.encode()).hexdigest(),
            "num_detectors": self._num_detector,
            "circuit_level_distance": self._circuit_level_code_distance,
            "assumed_zero_through_weight": self._t,
            "model": model.name, "model_parameters": model.get_params(),
            "model_fitted": model.is_fitted,
            "fit_r_squared": model.r_squared, "scalerqec_version": version("scalerqec"),
            "systematic_error": "Unquantified S-curve extrapolation error; diagnostics are not confidence intervals.",
        })

    def get_profile(self) -> LERProfile:
        """Snapshot the completed fit; subsequent evaluation never samples again."""
        complete = self._num_noise > 0 and all(
            w in self._estimated_subspaceLER for w in range(self._t + 1, self._num_noise + 1)
        )
        if self._model is None or (not self._model.is_fitted and not complete):
            raise RuntimeError("No successful fit is available; increase the profiling budget.")
        return self._make_profile(self._model)

    def profile_from_file(self, filepath: str, codedistance: int, *,
                          decoder_reference_p: float = 0.001) -> LERProfile:
        """Sample and fit once for a fixed decoder under uniform SID noise.

        The input must be noiseless; SID locations are inserted before each
        normalized primitive gate/measurement, excluding reset. The decoder
        is built at decoder_reference_p (or supplied to the constructor).
        codedistance must be the circuit-level distance for that decoder.
        """
        if not 0 < float(_probabilities(decoder_reference_p)) < 1:
            raise ValueError("decoder_reference_p must be strictly between 0 and 1.")
        result = self.calculate_LER_from_file(filepath, decoder_reference_p,
                                              codedistance, None, None)
        if result is None:
            raise RuntimeError("Profiling budget exhausted before a fit was available.")
        return self.get_profile()

    def calculate_LER_curve_from_file(self, filepath, p_values, codedistance, *,
                                     decoder_reference_p=0.001):
        """Convenience API: profile once, then return an LERCurve for a p grid."""
        _probabilities(p_values)
        return self.profile_from_file(filepath, codedistance,
                                      decoder_reference_p=decoder_reference_p).curve(p_values)

    # ------------------------------------------------------------------
    #  Main entry point: iterative, time-budgeted estimation
    # ------------------------------------------------------------------

    def calculate_LER_from_file(
        self,
        filepath: str,
        pvalue: float,
        codedistance: int,
        figname: str | None = None,
        titlename: str | None = None,
        repeat: int = 1,
    ) -> float | None:
        """
        Calculate LER using the ScaLER algorithm.

        Algorithm (from paper):
        1. Phase 1 (Initialization):
           - Binary search to find w_has_error and w_saturated
           - Uniformly sample 5 points between them
           - Fit initial S-curve and estimate w_sweet

        2. Phase 2 (Progressive refinement):
           - Sample between sweet_spot and w_has_error
           - Step size = (w_sweet - w_has_error) / 5

        3. Phase 3 (Iterative refinement):
           - Sample more at weights with insufficient LE events
           - Stop when convergence or timeout
        """
        if np.ndim(pvalue) != 0:
            raise ValueError("pvalue must be a scalar.")
        _probabilities(pvalue)
        if isinstance(codedistance, bool) or not isinstance(codedistance, (int, np.integer)) or codedistance < 1:
            raise ValueError("codedistance must be a positive integer.")
        if not np.isfinite(self._time_budget) or self._time_budget <= 0:
            raise ValueError("time_budget must be finite and positive.")
        self._error_rate = float(pvalue)
        self._circuit_level_code_distance = codedistance
        self._t = max(0, (codedistance - 1) // 2)

        self.parse_from_file(filepath)
        if self._t >= self._num_noise:
            raise ValueError("codedistance is incompatible with the number of fault locations.")

        # Initialize model
        self._initialize_model()

        # Reset stats
        self._subspace_LE_count.clear()
        self._subspace_sample_used.clear()
        self._estimated_subspaceLER.clear()
        self._iteration_log.clear()
        self._models.clear()
        self._model_scores.clear()
        self._sampling_rate = 0.0
        self._sweet_spot = None
        self._ler = 0.0
        self._a = 0.0
        self._b = 0.0
        self._c = 0.0
        self._remaining_time_budget = float(self._time_budget)

        start_time = time.perf_counter()

        print("=" * 60)
        print(f"ScaLER: Using {self._model_type.value} model (gamma={self._gamma})")
        print(f"  Time budget: {self._time_budget:.1f}s")
        print(f"  Error rate: {pvalue:.2e}")
        print(f"  Code distance: {codedistance} (t={self._t})")
        print("=" * 60)

        # ============================================================
        # PHASE 1: Binary search + uniform 5-point sampling
        # ============================================================
        print("\nPhase 1: Determining S-curve bounds...")
        self.determine_lower_w()
        print(f"  w_has_error = {self._has_logical_errorw}")
        self.determine_saturated_w()
        print(f"  w_saturated = {self._saturatew}")

        print("\nMeasuring sampling rate...")
        self.measure_sample_rates()
        if self._remaining_time_budget <= 0.0:
            print("Time budget exhausted during calibration.")
            return None

        print("\nPhase 1: Initial 5-point uniform sampling...")
        wlist_init = evenly_spaced_ints(self._has_logical_errorw, self._saturatew, 5)
        # Total shots limited by _MAX_BATCH_SIZE
        shots_per_w = self._MAX_BATCH_SIZE // len(wlist_init)
        shots_per_w = max(2000, shots_per_w)
        print(
            f"  Shots per weight: {shots_per_w} (total: {shots_per_w * len(wlist_init)})"
        )
        slist_init = [shots_per_w] * len(wlist_init)

        elapsed = self._sampling_step(wlist_init, slist_init)
        self._remaining_time_budget -= elapsed

        figure_prefix = figname or ""
        # Only fit model without saving intermediate plots (only final plot is saved)
        self.fit_log_S_model(
            filename=figure_prefix + "phase1.pdf",
            savefigure=False,
            time_val=time.perf_counter() - start_time,
        )

        print(f"  Initial sweet_spot = {self._sweet_spot}")
        print(f"  Initial R² = {self._R_square_score:.4f}")
        self._calc_LER_from_fit()
        print(f"  Initial LER = {self._ler:.3e}")

        # Log Phase 1
        self._iteration_log.append(
            {
                "iteration": 0,
                "phase": "phase1_initial",
                "weights_sampled": wlist_init,
                "shots_per_weight": shots_per_w,
                "total_shots": shots_per_w * len(wlist_init),
                "sweet_spot_after": self._sweet_spot,
                "r_squared": self._R_square_score,
                "ler": self._ler,
                "elapsed_time": elapsed,
                "weight_status_after": {
                    w: {
                        "samples": self._subspace_sample_used.get(w, 0),
                        "le_count": self._subspace_LE_count.get(w, 0),
                        "p_w": self._estimated_subspaceLER.get(w, 0),
                    }
                    for w in wlist_init
                },
            }
        )

        # ============================================================
        # PHASE 2: Sample between sweet_spot and w_has_error
        # ============================================================
        print("\nPhase 2: Sampling between sweet_spot and w_has_error...")

        assert self._sweet_spot is not None

        # Check if we can afford the theoretical sweet spot
        # Update remaining budget from wall-clock time
        elapsed_total = time.perf_counter() - start_time
        self._remaining_time_budget = self._time_budget - elapsed_total

        practical_sweet = self._get_practical_sweet_spot(self._remaining_time_budget)

        if practical_sweet != self._sweet_spot:
            print(f"  Theoretical sweet_spot = {self._sweet_spot}")
            print(f"  Practical sweet_spot = {practical_sweet} (budget constraint)")
            sampling_sweet_spot = practical_sweet
        else:
            print(f"  sweet_spot = {self._sweet_spot}")
            sampling_sweet_spot = self._sweet_spot

        print(f"  w_has_error = {self._has_logical_errorw}")

        # Sample uniform points between sampling_sweet_spot and w_has_error
        wlist_phase2 = evenly_spaced_ints(
            sampling_sweet_spot, self._has_logical_errorw, self._num_subspaces_phase2
        )
        wlist_phase2 = [w for w in wlist_phase2 if w not in self._subspace_sample_used]

        if wlist_phase2:
            print(f"  Phase 2 weights to sample: {wlist_phase2}")

            # Allocate shots based on remaining budget
            elapsed_total = time.perf_counter() - start_time
            self._remaining_time_budget = self._time_budget - elapsed_total

            # Total shots limited by _MAX_BATCH_SIZE, distribute evenly across weights
            shots_per_w = self._MAX_BATCH_SIZE // len(wlist_phase2)
            shots_per_w = max(self._SHOTS_PER_SUBSPACE, shots_per_w)

            print(
                f"  Shots per weight: {shots_per_w} (total: {shots_per_w * len(wlist_phase2)})"
            )

            slist_phase2 = [shots_per_w] * len(wlist_phase2)

            self._sampling_step(wlist_phase2, slist_phase2)

            # Recalculate practical sweet spot after sampling
            elapsed_total = time.perf_counter() - start_time
            self._remaining_time_budget = self._time_budget - elapsed_total
            practical_sweet = self._get_practical_sweet_spot(
                self._remaining_time_budget
            )

            # Only fit model without saving intermediate plots (only final plot is saved)
            self.fit_log_S_model(
                filename=figure_prefix + "phase2.pdf",
                savefigure=False,
                time_val=time.perf_counter() - start_time,
                practical_sweet_spot=practical_sweet,
            )
            self._calc_LER_from_fit()

            print(f"  Updated sweet_spot = {self._sweet_spot}")
            print(f"  R² = {self._R_square_score:.4f}")
            print(f"  LER = {self._ler:.3e}")

            # Log Phase 2
            self._iteration_log.append(
                {
                    "iteration": 0,
                    "phase": "phase2_sweet_spot",
                    "weights_sampled": wlist_phase2,
                    "shots_per_weight": shots_per_w,
                    "total_shots": shots_per_w * len(wlist_phase2),
                    "sweet_spot_after": self._sweet_spot,
                    "practical_sweet_spot": practical_sweet,
                    "r_squared": self._R_square_score,
                    "ler": self._ler,
                    "elapsed_time": elapsed_total,
                    "weight_status_after": {
                        w: {
                            "samples": self._subspace_sample_used.get(w, 0),
                            "le_count": self._subspace_LE_count.get(w, 0),
                            "p_w": self._estimated_subspaceLER.get(w, 0),
                        }
                        for w in wlist_phase2
                    },
                }
            )
        else:
            print("  No new weights to sample in Phase 2.")

        # ============================================================
        # PHASE 3: Adaptive refinement towards theoretical sweet spot
        # ============================================================
        print("\nPhase 3: Adaptive refinement...")

        MIN_BUDGET_THRESHOLD = 10.0
        iter_idx = 0

        while True:  # No max iteration limit - only limited by time budget
            # Calculate remaining budget from wall-clock time
            elapsed_total = time.perf_counter() - start_time
            self._remaining_time_budget = self._time_budget - elapsed_total

            if self._remaining_time_budget <= MIN_BUDGET_THRESHOLD:
                print(
                    f"\n  Time budget exhausted (used {elapsed_total:.1f}s of {self._time_budget}s)"
                )
                break

            # Get practical sweet spot based on current remaining budget
            practical_sweet = self._get_practical_sweet_spot(
                self._remaining_time_budget
            )
            theoretical_sweet = self._sweet_spot

            # Get the current leftmost sampled weight
            min_sampled = min(self._subspace_sample_used.keys())

            iter_idx += 1
            print(f"\n--- Refinement iteration {iter_idx} ---")
            print(f"  Remaining budget: {self._remaining_time_budget:.1f}s")
            print(f"  Theoretical sweet_spot: {theoretical_sweet}")
            print(f"  Practical sweet_spot: {practical_sweet}")
            print(f"  Current leftmost sampled: {min_sampled}")

            # SIMPLIFIED LOGIC: Just keep sampling weights that need more LE events
            # NEVER try to sample at lower weights (closer to sweet spot) - they're harder
            # Keep sampling existing weights until either:
            #   1. All weights have sufficient LE events (>= _min_num_ke_event), OR
            #   2. Time budget is exhausted

            wlist_refine = []
            for w in sorted(self._subspace_sample_used.keys()):
                le_count = self._subspace_LE_count.get(w, 0)
                if le_count < self._min_num_ke_event:
                    wlist_refine.append(w)

            if not wlist_refine:
                print("  All weights have sufficient LE events. Done.")
                break

            print(f"  Refining weights with insufficient LE: {wlist_refine}")

            # Show current weight distribution for weights needing refinement
            print("  Current status:")
            for w in wlist_refine:
                samples = self._subspace_sample_used.get(w, 0)
                le_count = self._subspace_LE_count.get(w, 0)
                pl = self._estimated_subspaceLER.get(w, 0)
                print(
                    f"    w={w}: {samples:,} samples, {le_count} LE events, P_L={pl:.4e}"
                )

            # Allocate shots - total limited by _MAX_BATCH_SIZE, distribute across weights
            shots_per_w = self._MAX_BATCH_SIZE // len(wlist_refine)
            shots_per_w = max(self._SHOTS_PER_SUBSPACE, shots_per_w)
            print(
                f"  Shots per weight: {shots_per_w} (total: {shots_per_w * len(wlist_refine)})"
            )
            slist_refine = [shots_per_w] * len(wlist_refine)

            # Log this iteration
            iter_log = {
                "iteration": iter_idx,
                "phase": "refinement",
                "remaining_budget": self._remaining_time_budget,
                "theoretical_sweet_spot": theoretical_sweet,
                "practical_sweet_spot": practical_sweet,
                "weights_sampled": wlist_refine.copy(),
                "shots_per_weight": shots_per_w,
                "total_shots": shots_per_w * len(wlist_refine),
                "weight_status_before": {
                    w: {
                        "samples": self._subspace_sample_used.get(w, 0),
                        "le_count": self._subspace_LE_count.get(w, 0),
                        "p_w": self._estimated_subspaceLER.get(w, 0),
                    }
                    for w in wlist_refine
                },
            }

            self._sampling_step(wlist_refine, slist_refine)

            # Recalculate practical sweet spot after sampling
            elapsed_total = time.perf_counter() - start_time
            self._remaining_time_budget = self._time_budget - elapsed_total
            practical_sweet = self._get_practical_sweet_spot(
                self._remaining_time_budget
            )

            # Only fit model without saving intermediate plots (only final plot is saved)
            self.fit_log_S_model(
                filename=figure_prefix + f"refine{iter_idx}.pdf",
                savefigure=False,
                time_val=time.perf_counter() - start_time,
                practical_sweet_spot=practical_sweet,
            )
            self._calc_LER_from_fit()

            # Update log with results after this iteration
            iter_log["weight_status_after"] = {
                w: {
                    "samples": self._subspace_sample_used.get(w, 0),
                    "le_count": self._subspace_LE_count.get(w, 0),
                    "p_w": self._estimated_subspaceLER.get(w, 0),
                }
                for w in wlist_refine
            }
            iter_log["sweet_spot_after"] = self._sweet_spot
            iter_log["r_squared"] = self._R_square_score
            iter_log["ler"] = self._ler
            iter_log["elapsed_time"] = elapsed_total
            self._iteration_log.append(iter_log)

            print(f"  Updated theoretical sweet_spot = {self._sweet_spot}")
            print(f"  Updated practical sweet_spot = {practical_sweet}")
            print(f"  R² = {self._R_square_score:.4f}")
            print(f"  LER = {self._ler:.3e}")

        # ============================================================
        # Final results
        # ============================================================
        total_time = time.perf_counter() - start_time

        # Get final practical sweet spot for the plot
        final_remaining_budget = self._time_budget - total_time
        final_practical_sweet = self._get_practical_sweet_spot(
            max(0, final_remaining_budget)
        )

        self.fit_log_S_model(
            filename=figure_prefix + "final.pdf",
            savefigure=bool(figname),
            time_val=total_time,
            practical_sweet_spot=final_practical_sweet,
        )
        # Default model parameters must never be presented as a successful fit.
        ler_est = self.get_profile().evaluate(self._error_rate)
        self._ler = ler_est

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Model: {self._model_type.value}")
        print(f"  Gamma: {self._gamma}")
        print(f"  Estimated LER: {ler_est:.4e}")
        print(f"  Final R²: {self._R_square_score:.4f}")
        print(f"  Theoretical sweet spot: w = {self._sweet_spot}")
        if final_practical_sweet != self._sweet_spot:
            print(f"  Practical sweet spot: w = {final_practical_sweet}")
        print(f"  Sampled weights: {sorted(self._subspace_sample_used.keys())}")
        print(f"  Total samples: {sum(self._subspace_sample_used.values()):,}")
        print(f"  Total time: {total_time:.1f}s")
        print("=" * 60)

        # Save iteration log to file
        if figname:
            self._save_iteration_log(figname + "iteration_log.json")

        return ler_est

    def _save_iteration_log(self, filename: str) -> None:
        """Serialise the iteration log to a JSON file for analysis.

        The log includes circuit metadata, model configuration, final
        results, per-weight statistics, and detailed per-iteration
        records of sampling decisions and outcomes.

        Args:
            filename: Output path for the JSON file.
        """
        import json

        log_data = {
            "circuit_info": {
                "code_distance": self._circuit_level_code_distance,
                "error_rate": self._error_rate,
                "num_noise": self._num_noise,
                "num_detector": self._num_detector,
                "t": self._t,
                "w_err": self._has_logical_errorw,
                "w_sat": self._saturatew,
                "w_min": self._minw,
                "w_max": self._maxw,
            },
            "model_info": {
                "model_type": self._model_type.value,
                "gamma": self._gamma,
            },
            "final_results": {
                "ler": self._ler,
                "r_squared": self._R_square_score,
                "sweet_spot": self._sweet_spot,
                "total_samples": sum(self._subspace_sample_used.values()),
            },
            "final_weight_distribution": {
                str(w): {
                    "samples": self._subspace_sample_used.get(w, 0),
                    "le_count": self._subspace_LE_count.get(w, 0),
                    "p_w": self._estimated_subspaceLER.get(w, 0),
                }
                for w in sorted(self._subspace_sample_used.keys())
            },
            "iterations": self._iteration_log,
        }

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        log_data = convert_to_serializable(log_data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        print(f"Saved iteration log to: {filename}")

    def get_iteration_log(self) -> List[Dict]:
        """Get the iteration log for analysis."""
        return self._iteration_log


if __name__ == "__main__":
    # Test on repetition code with different models
    filepath = "C:/Users/yezhu/Documents/ScaLER/stimprograms/repetition/repetition5"

    # Test with Our Model
    print("\n" + "=" * 60)
    print("Testing with OurModel")
    print("=" * 60)
    scaler = Scaler(
        error_rate=0.01,
        time_budget=60,
        model_type=ModelType.OUR_MODEL,
        gamma=0.05,
    )
    scaler.calculate_LER_from_file(
        filepath,
        pvalue=0.01,
        codedistance=5,
        figname="rep5_our_model_",
        titlename="Repetition-5 (Our Model)",
        repeat=1,
    )

    # Test with IBM Model
    print("\n" + "=" * 60)
    print("Testing with IBMModel")
    print("=" * 60)
    scaler_ibm = Scaler(
        error_rate=0.01,
        time_budget=60,
        model_type=ModelType.IBM_MODEL,
        gamma=0.05,
    )
    scaler_ibm.calculate_LER_from_file(
        filepath,
        pvalue=0.01,
        codedistance=5,
        figname="rep5_ibm_model_",
        titlename="Repetition-5 (IBM Model)",
        repeat=1,
    )
