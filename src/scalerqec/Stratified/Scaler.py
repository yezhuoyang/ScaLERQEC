# An updated version of the main method
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymatching
from scipy.optimize import curve_fit

from scalerqec.Clifford.clifford import CliffordCircuit
from scalerqec.qepg import (
    compile_QEPG,
    return_samples_many_weights_separate_obs_with_QEPG,
    return_samples_with_fixed_QEPG,
)
from scalerqec.Stratified.fitting import r_squared
from scalerqec.Stratified.ScurveModel import (
    bias_estimator,
    compute_mse_optimal_allocation,
    compute_turning_point,
    evenly_spaced_ints,
    fit_power_law_to_lowest_weights,
    hybrid_pl_exponential,
    modified_linear_function,
    modified_linear_function_with_d,
    modified_sigmoid_function,
    refined_sweet_spot,
    sigma_estimator,
)
from scalerqec.Stratified.visualization import plot_log_scurve
from scalerqec.util.binomial import binomial_weight


class Scaler:
    """
    Use stratified sampling to estimate the logical error rate of a quantum error
    correction code. The only user-facing hyperparameters are:
      - physical error rate (error_rate)
      - time budget (time_budget, in seconds)
    The algorithm internally manages which subspaces to sample and how many shots.
    """

    def __init__(self, error_rate: float = 0.0, time_budget: int = 30):
        self._error_rate: float = error_rate
        self._time_budget: float = float(time_budget)
        self._remaining_time_budget: float = float(time_budget)

        # Measured samples per second (shots/second)
        self._sampling_rate: float = 0.0

        # Hard cap on total shots per sampling call (to control memory).
        self._MAX_SHOTS_PER_STEP: int = 500_00

        # Circuit-related
        self._cliffordcircuit: CliffordCircuit = CliffordCircuit(4)
        self._num_noise: int = 0
        self._num_detector: int = 0
        self._stim_str_after_rewrite: str = ""
        self._detector_error_model = None
        self._matcher: Optional[pymatching.Matching] = None
        self._QEPG_graph = None

        # Subspace statistics
        self._subspace_LE_count: Dict[int, int] = {}
        self._subspace_sample_used: Dict[int, int] = {}
        self._estimated_subspaceLER: Dict[int, float] = {}

        # S-curve model parameters and meta
        self._a: float = 0.0
        self._b: float = 0.0
        self._c: float = 0.0
        self._R_square_score: float = 0.0
        self._sweet_spot: Optional[int] = None
        self._turning_point: Optional[float] = None  # MSE-optimal turning point

        # QEC parameters
        self._circuit_level_code_distance: int = 1
        self._t: int = 0  # (d-1)/2
        self._k_range: int = 5
        self._beta: float = 4.0
        self._ratio: float = 0.05  # used in refined_sweet_spot

        # Weight bracketing
        self._has_logical_errorw: int = 1
        self._saturatew: int = 1
        self._minw: int = 1
        self._maxw: int = 1
        self._max_PL: float = 0.005  # plateau threshold

        # Sampling policy constants (internal, not user-exposed)
        self._MIN_NUM_LE_EVENT: int = 30  # min LE events to trust subspace LER

        # NEW: tighter requirements in the band around sweet spot
        self._BAND_HALF_WIDTH: int = 4
        self._TARGET_EVENTS_BAND: int = 160  # target LE events per band weight

        # Final LER
        self._LER: float = 0.0

    # ------------------------------------------------------------------
    #  Circuit / QEPG setup and basic helpers
    # ------------------------------------------------------------------

    def parse_from_file(self, filepath: str):
        """
        Read the circuit, parse from the file, compile stim circuit and QEPG graph.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()

        self._cliffordcircuit.error_rate = self._error_rate
        self._cliffordcircuit.compile_from_stim_circuit_str(stim_str)
        self._num_noise = self._cliffordcircuit.totalnoise
        self._num_detector = len(self._cliffordcircuit.parityMatchGroup)
        self._stim_str_after_rewrite = stim_str

        # Configure a decoder using the circuit.
        self._detector_error_model = (
            self._cliffordcircuit.stimcircuit.detector_error_model(
                decompose_errors=True
            )
        )
        self._matcher = pymatching.Matching.from_detector_error_model(
            self._detector_error_model
        )

        # Compile QEPG graph once.
        self._QEPG_graph = compile_QEPG(stim_str)

    def calc_logical_error_rate_with_fixed_w(self, shots: int, w: int) -> float:
        """
        Calculate the logical error rate with fixed Pauli weight w.
        """
        result = return_samples_with_fixed_QEPG(self._QEPG_graph, w, shots)
        arr = np.asarray(result)
        states = arr[:, :-1]
        observables = arr[:, -1]
        predictions = np.squeeze(self._matcher.decode_batch(states))
        num_errors = np.count_nonzero(observables != predictions)
        return num_errors / shots

    # ------------------------------------------------------------------
    #  Binary search helpers to bracket the S-curve
    # ------------------------------------------------------------------

    def binary_search_upper(self, low: int, high: int, shots: int) -> int:
        """
        Find the smallest w in [low, high] such that PL(w) > _max_PL.
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
        self, low: int, high: int, shots: int = 2000, epsilon: float = 0.001
    ) -> int:
        """
        Find the smallest w in [low, high] such that PL(w) > epsilon.

        Note: Uses adaptive shot budget - at each step, we need enough shots
        to detect PL > epsilon with reasonable confidence. For epsilon=0.001,
        we need ~3000 shots to expect 3 events at the threshold.
        """
        # Adaptive shot budget: need at least 3/epsilon shots to detect threshold
        min_shots = max(shots, int(5.0 / epsilon))
        shots = min(min_shots, 10000)  # Cap at 10K to avoid too much time

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

    def determine_lower_w(self):
        """Determine the first weight where PL is noticeably non-zero."""
        if self._num_noise <= 8:
            self._has_logical_errorw = 1
        else:
            self._has_logical_errorw = self.binary_search_lower(1, self._num_noise)

    def determine_saturated_w(self):
        """Determine the weight where PL is essentially saturated (near plateau)."""
        if self._num_noise <= 8:
            self._saturatew = self._num_noise
        else:
            self._saturatew = self.binary_search_upper(
                self._has_logical_errorw, self._num_noise, shots=2000
            )
            # Ensure some separation from the lower bound
            if self._saturatew < self._has_logical_errorw + 8:
                self._saturatew = min(self._num_noise, self._has_logical_errorw + 8)

    # ------------------------------------------------------------------
    #  Sampling rate measurement
    # ------------------------------------------------------------------

    def measure_sample_rates(self):
        """
        Measure the sampling rate of the given circuit.
        This method is used to estimate how many samples can be done within
        the time budget (shots per second).
        """
        # Use a central weight as a proxy.
        wlist = [max(1, self._num_noise // 2)]
        slist = [1000]

        start_time = time.perf_counter()
        print("Start time for sampling rate measurement:", start_time)
        _det, _obs = return_samples_many_weights_separate_obs_with_QEPG(
            self._QEPG_graph, wlist, slist
        )
        end_time = time.perf_counter()
        print("End time for sampling rate measurement:", end_time)
        elapsed = end_time - start_time
        if elapsed <= 0:
            elapsed = 1e-6
        # Initial estimate
        self._sampling_rate = 1000.0 / elapsed

        # Deduct the cost of this calibration from time budget
        self._remaining_time_budget -= elapsed

        print(
            "Elapsed time for sampling rate measurement: {:.6f} seconds".format(elapsed)
        )
        print(f"Measured sampling rate: {self._sampling_rate:.2f} shots/second")

    # ------------------------------------------------------------------
    #  One multi-weight sampling step (updates subspace stats + rate)
    # ------------------------------------------------------------------

    def _sampling_step(self, wlist: List[int], slist: List[int]) -> float:
        """
        Perform one multi-weight sampling call and update subspace statistics.
        Also refine the sampling rate based on actual performance of this step.

        Returns the elapsed time for this step (seconds).
        """
        if not wlist:
            return 0.0

        total_shots = int(sum(slist))
        print("Sampling weights and shots:", list(zip(wlist, slist)))
        print("  Total shots this step:", total_shots)

        start_time = time.perf_counter()
        detector_result, obsresult = return_samples_many_weights_separate_obs_with_QEPG(
            self._QEPG_graph, wlist, slist
        )
        predictions_result = self._matcher.decode_batch(detector_result)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        begin_index = 0
        for w, shots in zip(wlist, slist):
            end_index = begin_index + shots
            observables = np.asarray(obsresult[begin_index:end_index])
            predictions = np.asarray(predictions_result[begin_index:end_index]).ravel()

            num_errors = int(np.count_nonzero(observables != predictions))

            # Update stats
            self._subspace_LE_count[w] = self._subspace_LE_count.get(w, 0) + num_errors
            self._subspace_sample_used[w] = self._subspace_sample_used.get(w, 0) + shots
            self._estimated_subspaceLER[w] = (
                self._subspace_LE_count[w] / self._subspace_sample_used[w]
            )

            begin_index = end_index

        # Refine sampling rate using this step (to capture memory / overhead effects).
        if elapsed > 0 and total_shots > 0:
            inst_rate = total_shots / elapsed
            if self._sampling_rate <= 0:
                self._sampling_rate = inst_rate
            else:
                alpha = 0.5  # EMA smoothing factor
                self._sampling_rate = (
                    alpha * inst_rate + (1.0 - alpha) * self._sampling_rate
                )

        print(f"  Step elapsed: {elapsed:.6f} s")
        print(f"  Updated sampling rate: {self._sampling_rate:.2f} shots/second")

        return elapsed

    # ------------------------------------------------------------------
    #  Fitting the log-S model
    # ------------------------------------------------------------------

    def fit_log_S_model(self, filename=None, savefigure: bool = False, time_val=None):
        """
        Fit log(0.5 / PL - 1) ≈ modified_linear_function(w; a, b, c, t)
        and update (a, b, c), R^2, and sweet_spot.

        If savefigure=True, also plot:
          - transformed data points with error bars (sigma_estimator)
          - fitted log-S curve
          - sweet-spot marker and region annotations
        """
        # -----------------------------
        # 1. Build x, y, sigma lists
        # -----------------------------
        x_list = [
            x
            for x in self._estimated_subspaceLER.keys()
            if (
                0.0 < self._estimated_subspaceLER[x] < 0.5
                and self._subspace_LE_count.get(x, 0) > 0
            )
        ]
        if not x_list:
            # Not enough data; keep defaults
            self._R_square_score = 0.0
            ep = int(self._error_rate * self._num_noise)
            self._sweet_spot = max(self._t + 1, ep)
            return

        x_list = sorted(x_list)

        sigma_list = [
            sigma_estimator(self._subspace_sample_used[x], self._subspace_LE_count[x])
            for x in x_list
        ]
        y_list = [
            np.log(0.5 / self._estimated_subspaceLER[x] - 1.0)
            - bias_estimator(self._subspace_sample_used[x], self._subspace_LE_count[x])
            for x in x_list
        ]

        # -----------------------------
        # 2. Determine minw / maxw region
        # -----------------------------
        sigma = int(
            np.sqrt(self._error_rate * (1.0 - self._error_rate) * self._num_noise)
        )
        if sigma == 0:
            sigma = 1
        ep = int(self._error_rate * self._num_noise)
        self._minw = max(self._t + 1, ep - self._k_range * sigma)
        self._maxw = min(self._num_noise, ep + self._k_range * sigma)

        # -----------------------------
        # 3. Initial guess for (a, b, c)
        # -----------------------------
        if self._a == 0.0:
            if len(x_list) >= 2:
                x0, x1 = x_list[0], x_list[-1]
                y0, y1 = y_list[0], y_list[-1]
                self._a = (y1 - y0) / max(1e-6, (x1 - x0))
                self._b = y0 - self._a * x0
            else:
                self._a = -0.1
                self._b = 0.0

        alpha = -1.0 / self._a
        beta = alpha

        initial_guess = (self._a, self._b, beta)

        lower = [
            min(self._a * 5.0, self._a * 0.2),
            min(self._b * 0.2, self._b * 5.0),
            min(beta * 0.2, beta * 5.0),
        ]
        upper = [
            max(self._a * 5.0, self._a * 0.2),
            max(self._b * 0.2, self._b * 5.0),
            max(beta * 0.2, beta * 5.0),
        ]

        # -----------------------------
        # 4. Non-linear fit in log-space
        # -----------------------------
        popt, _pcov = curve_fit(
            modified_linear_function(self._t),
            x_list,
            y_list,
            p0=initial_guess,
            bounds=(lower, upper),
            maxfev=50_000,
        )

        self._a, self._b, self._c = popt[0], popt[1], popt[2]

        # Rebuild y_list (for clarity) and compute predictions
        y_list = [
            np.log(0.5 / self._estimated_subspaceLER[x] - 1.0)
            - bias_estimator(self._subspace_sample_used[x], self._subspace_LE_count[x])
            for x in x_list
        ]
        y_predicted = [
            modified_linear_function_with_d(x, self._a, self._b, self._c, self._t)
            for x in x_list
        ]
        self._R_square_score = r_squared(y_list, y_predicted)

        # -----------------------------
        # 5. Update turning point (MSE-optimal) and sweet spot
        # -----------------------------
        alpha = -1.0 / self._a
        beta = self._c

        # Compute theoretical turning point where dy/dw = 0
        # This is where sampling minimizes parameter estimation bias
        self._turning_point = compute_turning_point(alpha, beta, self._t)

        # For backward compatibility, also set sweet_spot based on MSE considerations
        # Blend turning point and binomial peak based on R² quality
        ep_approx = int(self._error_rate * self._num_noise)
        r2 = self._R_square_score

        # At low R², favor turning point (need better curve fit)
        # At high R², favor binomial peak (need variance reduction)
        if self._turning_point is not None:
            w_sweet = int(r2 * ep_approx + (1 - r2) * self._turning_point)
        else:
            w_sweet = int(
                refined_sweet_spot(alpha, self._c, self._t, ratio=self._ratio)
            )

        # Ensure valid range
        w_sweet = min(self._saturatew, max(self._t + 1, w_sweet))
        self._sweet_spot = w_sweet

        sweet_spot_y = modified_linear_function_with_d(
            self._sweet_spot, self._a, self._b, self._c, self._t
        )

        sample_cost_list = [self._subspace_sample_used[x] for x in x_list]

        # -----------------------------
        # 6. Plot for debugging (flexible visualization)
        # -----------------------------
        if not savefigure:
            return

        fig = plot_log_scurve(
            x_list=x_list,
            y_list=y_list,
            sigma_list=sigma_list,
            sample_cost_list=sample_cost_list,
            a=self._a,
            b=self._b,
            c=self._c,
            t=self._t,
            minw=self._minw,
            maxw=self._maxw,
            saturatew=self._saturatew,
            sweet_spot=self._sweet_spot,
            has_logical_errorw=self._has_logical_errorw,
            num_noise=self._num_noise,
            num_detector=self._num_detector,
            error_rate=self._error_rate,
            code_distance=self._circuit_level_code_distance,
            r_squared=self._R_square_score,
            ler=self._LER,
            time_elapsed=time_val,
            total_samples=sum(self._subspace_sample_used.values()),
            filename=filename,
            k_range=self._k_range,
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    #  Parameter / band / PL stability checks
    # ------------------------------------------------------------------

    def params_stable(self, theta_new, theta_old, tol, r2, r2_target):
        """
        Decide if the parameters are stable enough throughout iterations.
        """
        if theta_old is None:
            return False
        num = sum((x - y) ** 2 for x, y in zip(theta_new, theta_old)) ** 0.5
        den = max(1e-12, sum(y**2 for y in theta_old) ** 0.5)
        rel = num / den
        return (rel < tol) and (r2 >= r2_target)

    def _band_weights(self) -> List[int]:
        """
        Return weights in the refinement band around sweet_spot.
        """
        if self._sweet_spot is None:
            return []
        left = max(self._has_logical_errorw, self._sweet_spot - self._BAND_HALF_WIDTH)
        right = min(self._saturatew, self._sweet_spot + self._BAND_HALF_WIDTH)
        return list(range(left, right + 1))

    def _band_well_sampled(self) -> bool:
        """
        Check if all band weights have enough logical error events.
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
        """
        Check if the overall PL estimate is stable in relative error.
        """
        if pl_old is None:
            return False
        if pl_old == 0.0:
            return abs(pl_new) < rel_tol
        rel = abs(pl_new - pl_old) / abs(pl_old)
        return rel < rel_tol

    # ------------------------------------------------------------------
    #  Decide the next sampling step (Progressive Left-to-Right Strategy)
    # ------------------------------------------------------------------

    def _get_exploration_frontier(self) -> int:
        """
        Get the current leftmost weight that has been adequately sampled.
        This defines the 'frontier' of our progressive exploration.
        """
        # Start from saturation and work left to find the frontier
        for w in range(self._saturatew, self._t, -1):
            if self._subspace_LE_count.get(w, 0) >= self._MIN_NUM_LE_EVENT:
                continue
            # Found a weight that needs more sampling
            return w
        return self._t + 1

    def _choose_candidate_weights_progressive(self) -> List[int]:
        """
        MSE-optimal weight selection strategy.

        Key insight: We want to balance two objectives:
        1. Bias reduction: Sample near the turning point where parameter
           estimation is most stable (dy/dw = 0)
        2. Variance reduction: Sample where binomial weights are high

        The turning point w* = t + (βα/2)^(2/3) is where the S-curve has
        an inflection and provides the most stable parameter estimates.
        """
        # Compute turning point if we have valid S-curve parameters
        alpha = -1.0 / self._a if self._a != 0 else 1.0
        beta = self._c if self._c > 0 else 1.0

        # Update turning point
        self._turning_point = compute_turning_point(alpha, beta, self._t)

        # Determine center based on MSE optimization
        # Use turning point for bias reduction, but also consider binomial peak
        ep = int(self._error_rate * self._num_noise)
        if self._turning_point is not None:
            # Adaptive center: blend turning point and binomial peak based on R²
            # Low R² → favor turning point (need better fit)
            # High R² → favor binomial peak (need variance reduction)
            r2 = self._R_square_score
            center = int(r2 * ep + (1 - r2) * self._turning_point)
        else:
            center = max(self._t + 1, ep)

        # Legacy: also update sweet_spot for backward compatibility
        self._sweet_spot = center

        # Define the working region
        left_limit = max(self._t + 1, self._has_logical_errorw)
        right_limit = self._saturatew

        if right_limit <= left_limit:
            return [center]

        W = set()

        # 1) Include region around the turning point (for bias reduction)
        if self._turning_point is not None:
            w_turn = int(self._turning_point)
            for delta in range(-3, 4):
                w = w_turn + delta
                if left_limit <= w <= right_limit:
                    W.add(w)

        # 2) Include region around binomial peak (for variance reduction)
        sigma = max(
            1, int(np.sqrt(self._error_rate * (1 - self._error_rate) * self._num_noise))
        )
        for delta in range(-2 * sigma, 2 * sigma + 1):
            w = ep + delta
            if left_limit <= w <= right_limit:
                W.add(w)

        # 3) Include the exploration frontier - weights that need more samples
        frontier_weights = []
        for w in range(left_limit, right_limit + 1):
            events = self._subspace_LE_count.get(w, 0)
            if events < self._MIN_NUM_LE_EVENT:
                frontier_weights.append(w)

        # Focus on frontier weights near the center
        frontier_weights.sort(key=lambda w: abs(w - center))
        for w in frontier_weights[:8]:  # Take up to 8 frontier weights
            W.add(w)

        # 4) Add anchor points for curve fitting stability
        if left_limit <= self._minw <= right_limit:
            W.add(self._minw)
        if left_limit <= self._maxw <= right_limit:
            W.add(self._maxw)

        # 5) Add evenly spaced grid points for global coverage
        span = right_limit - left_limit
        if span > 10:
            num_grid = min(5, span // 3)
            for i in range(num_grid):
                w = left_limit + int(i * span / max(1, num_grid - 1))
                if left_limit <= w <= right_limit:
                    W.add(w)

        # 6) Include saturation region for anchoring the high-PL end
        W.add(right_limit)
        if right_limit - 1 >= left_limit:
            W.add(right_limit - 1)

        wlist = sorted(W)
        return wlist

    def _choose_candidate_weights_low_p(self) -> List[int]:
        """
        Specialized weight selection for LOW error rates (p < 0.001).

        At low error rates, the key challenge is the GAP between:
        - Binomial peak (ep = p * N) where most weight is concentrated
        - First observable error (first_error_w) where we start seeing logical errors

        Strategy:
        1. PRIORITIZE sampling near first_error_w to get the best possible
           anchor points for S-curve extrapolation
        2. Sample the lowest weights where ANY errors can be observed
        3. Use aggressive shot allocation at these critical weights
        4. Still sample high weights for S-curve shape calibration

        This strategy aims to MINIMIZE the extrapolation distance by pushing
        data collection as low as possible.
        """
        ep = int(self._error_rate * self._num_noise)
        sigma = max(
            1, int(np.sqrt(self._error_rate * (1 - self._error_rate) * self._num_noise))
        )

        first_error_w = self._has_logical_errorw
        saturate_w = self._saturatew

        W = set()

        # 1) CRITICAL: Weights near first_error_w (lowest observable)
        #    These are the most valuable anchor points for extrapolation
        for delta in range(-3, 6):
            w = first_error_w + delta
            if self._t + 1 <= w <= saturate_w:
                W.add(w)

        # 2) Weights where we've seen SOME errors but need more confidence
        #    These help refine the S-curve near the critical region
        for w in sorted(self._estimated_subspaceLER.keys()):
            if w > self._t and self._subspace_LE_count.get(w, 0) < 50:
                W.add(w)
                if len(W) > 15:
                    break

        # 3) Critical region around binomial peak
        #    Even if we can't observe errors here, we need context
        minw_critical = max(self._t + 1, ep - 3 * sigma)
        maxw_critical = min(saturate_w, ep + 3 * sigma)

        # Only add if they're at or above first_error_w
        for w in range(
            max(minw_critical, first_error_w), min(maxw_critical, saturate_w) + 1
        ):
            W.add(w)

        # 4) Sparse grid across the observable range for shape calibration
        observable_span = saturate_w - first_error_w
        if observable_span > 10:
            for i in range(5):
                w = first_error_w + int(i * observable_span / 4)
                if self._t + 1 <= w <= saturate_w:
                    W.add(w)

        # 5) High-weight anchors for S-curve saturation behavior
        W.add(saturate_w)
        if saturate_w - 2 >= first_error_w:
            W.add(saturate_w - 2)

        wlist = sorted(W)
        return wlist

    def _compute_shot_allocation_low_p(
        self, wlist: List[int], step_shots: int
    ) -> List[int]:
        """
        Shot allocation optimized for low error rates.

        Key insight: At low p, most variance comes from:
        1. The lowest observable weights (low PL = high variance)
        2. Weights near first_error_w (critical for extrapolation)

        Strategy:
        - Allocate MORE shots to lower weights (lower PL = more samples needed)
        - BOOST allocation for weights near first_error_w
        - Minimum allocation for high weights (already saturated)
        """
        if not wlist:
            return []

        first_error_w = self._has_logical_errorw

        factors = []
        for w in wlist:
            f = 1.0

            # Distance from first_error_w: closer = MORE important
            dist_from_first = abs(w - first_error_w)
            if dist_from_first == 0:
                f *= 10.0
            elif dist_from_first <= 2:
                f *= 7.0
            elif dist_from_first <= 5:
                f *= 4.0
            elif dist_from_first <= 10:
                f *= 2.0

            # Under-sampled weights need more shots
            events = self._subspace_LE_count.get(w, 0)
            samples = self._subspace_sample_used.get(w, 0)

            if samples == 0:
                f *= 5.0  # Never sampled
            elif events == 0:
                f *= 3.0  # Sampled but no errors yet
            elif events < 10:
                f *= 2.5  # Few events, high variance
            elif events < 30:
                f *= 1.5  # Moderate events

            # Lower weights are MORE important for extrapolation
            # Scale factor inversely with weight relative to first_error_w
            if w <= first_error_w + 5:
                f *= 2.0

            # High weights (near saturation) need fewer samples
            if w > first_error_w + 20:
                f *= 0.5

            factors.append(f)

        total_factor = sum(factors)
        if total_factor <= 0:
            shots_per = max(1, step_shots // len(wlist))
            return [shots_per] * len(wlist)

        # Minimum shots per weight
        min_shots_per_w = max(500, int(step_shots * 0.03))

        slist = []
        remaining = step_shots

        for i, (w, f) in enumerate(zip(wlist, factors)):
            if i == len(wlist) - 1:
                s = max(1, remaining)
            else:
                raw = step_shots * (f / total_factor)
                s = max(min_shots_per_w, int(round(raw)))
                remaining -= s
                if remaining < min_shots_per_w:
                    remaining = min_shots_per_w

            slist.append(s)

        return slist

    def _compute_shot_allocation_mse_optimal(
        self, wlist: List[int], step_shots: int
    ) -> List[int]:
        """
        MSE-optimal sample allocation balancing bias and variance.

        This method minimizes MSE = Bias² + Variance by:
        1. Bias reduction: Focus samples near the turning point where
           parameter estimation is most stable
        2. Variance reduction: Focus samples where binomial weights are high

        The balance between bias and variance reduction adapts based on
        the current R² fit quality:
        - Low R²: Focus on bias reduction (need better curve fit)
        - High R²: Focus on variance reduction (refine LER estimate)
        """
        if not wlist:
            return []

        N = self._num_noise
        p = self._error_rate

        # Compute S-curve parameters for turning point
        alpha = -1.0 / self._a if self._a != 0 else 1.0
        beta = self._c if self._c > 0 else 1.0

        # Compute binomial weights for each candidate weight
        binomial_weights = {w: binomial_weight(N, w, p) for w in wlist}

        # Get current PL estimates (empirical or fitted)
        estimated_PL: Dict[int, float] = {}
        for w in wlist:
            if w in self._estimated_subspaceLER:
                estimated_PL[w] = self._estimated_subspaceLER[w]
            else:
                # Use S-curve extrapolation for weights without data
                estimated_PL[w] = min(
                    modified_sigmoid_function(w, self._a, self._b, self._c, self._t),
                    0.5,
                )
                # Ensure non-zero for variance calculation
                estimated_PL[w] = max(0.001, estimated_PL[w])

        # Minimum samples per weight
        min_samples = max(100, step_shots // (10 * max(1, len(wlist))))

        # Use MSE-optimal allocation from ScurveModel
        allocation = compute_mse_optimal_allocation(
            weights=wlist,
            estimated_PL=estimated_PL,
            binomial_weights=binomial_weights,
            total_budget=step_shots,
            alpha=alpha,
            beta=beta,
            t=self._t,
            r_squared=self._R_square_score,
            min_samples_per_weight=min_samples,
        )

        # Convert dict to list in same order as wlist
        return [allocation.get(w, min_samples) for w in wlist]

    def _compute_shot_allocation_progressive(
        self, wlist: List[int], step_shots: int
    ) -> List[int]:
        """
        Legacy progressive allocation - now delegates to MSE-optimal.

        This method is kept for backward compatibility but now uses
        the MSE-optimal allocation strategy internally.
        """
        return self._compute_shot_allocation_mse_optimal(wlist, step_shots)

    def next_step(self) -> Tuple[List[int], List[int]]:
        """
        Decide the next step using adaptive sampling strategy.

        Strategy selection:
          - For low error rates (p < 0.001): Use specialized low-p strategy
            that focuses on weights near first_error_w
          - For normal error rates: Use progressive strategy from saturation
            toward sweet spot
        """
        if self._sampling_rate <= 0.0 or self._remaining_time_budget <= 0.0:
            return [], []

        remaining_shots = int(self._remaining_time_budget * self._sampling_rate)
        if remaining_shots <= 0:
            return [], []

        step_shots = min(remaining_shots, self._MAX_SHOTS_PER_STEP)

        # Determine if we should use low-p strategy
        # Based on gap analysis between binomial peak and first_error_w
        ep = int(self._error_rate * self._num_noise)
        first_error_w = self._has_logical_errorw
        gap = first_error_w - ep
        sigma = max(
            1, int(np.sqrt(self._error_rate * (1 - self._error_rate) * self._num_noise))
        )

        use_low_p_strategy = (
            self._error_rate < 0.001  # Low physical error rate
            and gap > 2 * sigma  # Significant gap exists
        )

        if use_low_p_strategy:
            # Use specialized low-p strategy
            wlist = self._choose_candidate_weights_low_p()
            if not wlist:
                return [], []
            slist = self._compute_shot_allocation_low_p(wlist, step_shots)
            strategy_name = "Low-p"
        else:
            # Use standard progressive strategy
            wlist = self._choose_candidate_weights_progressive()
            if not wlist:
                return [], []
            slist = self._compute_shot_allocation_progressive(wlist, step_shots)
            strategy_name = "Progressive"

        print(f"  {strategy_name} weights: {wlist}")
        print(f"  Allocated shots: {slist} (total={sum(slist)})")

        return wlist, slist

    # ------------------------------------------------------------------
    #  Final LER integration with bias reduction
    # ------------------------------------------------------------------

    def _calc_LER_from_fit(self) -> float:
        """
        Integrate the fitted S-curve against the binomial distribution,
        with bias reduction through weighted combination of empirical
        and fitted estimates.
        """
        return self._calc_LER_bias_reduced()

    def _calc_LER_bias_reduced(self) -> float:
        """
        Calculate LER with bias reduction using weighted combination:
        - Use empirical PL(w) where we have enough samples (low bias)
        - Use S-curve extrapolation elsewhere (including below first_error_w)
        - Apply variance-weighted blending in transition regions

        CRITICAL insight: At low error rates, the binomial distribution is
        concentrated at low weights where we cannot observe logical errors
        directly. However, the S-curve model provides a reliable extrapolation
        to these weights. Setting PL=0 for w < first_error_w causes massive
        underestimation bias (up to -99% at p=0.0005).

        Instead, we extrapolate the fitted S-curve to low weights, which
        provides much more accurate LER estimates (bias typically <15%).

        Constraints on PL(w):
        - PL(w) = 0 for w <= t (code threshold - truly fault-tolerant)
        - S-curve extrapolation is capped at 0.5 for any weight
        - For weights far above sampled region, cap PL at boundary value
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = int(np.sqrt(p * (1.0 - p) * N))
        if sigma == 0:
            sigma = 1
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        first_error_w = self._has_logical_errorw
        saturate_w = self._saturatew

        # Extend maxw to include at least up to saturate_w if it's above current maxw
        if maxw < saturate_w:
            maxw = saturate_w

        # Define the "safe extrapolation" limit for high weights:
        # Beyond 2*range above the sampled region, S-curve extrapolation is unreliable
        extrap_limit = saturate_w + 2 * (saturate_w - first_error_w)
        boundary_PL = min(
            modified_sigmoid_function(extrap_limit, self._a, self._b, self._c, self._t),
            0.5,
        )

        # Threshold for "reliable" empirical estimate
        RELIABLE_EVENTS = self._MIN_NUM_LE_EVENT
        BLEND_EVENTS = RELIABLE_EVENTS // 3  # Start blending here

        for w in range(minw, maxw + 1):
            binom_w = binomial_weight(N, w, p)

            # For w <= t (code threshold), PL is truly 0 (fault-tolerant property)
            if w <= self._t:
                sub_PL = 0.0
            elif w > extrap_limit:
                # Beyond safe extrapolation limit: use boundary PL
                sub_PL = boundary_PL
            elif w in self._estimated_subspaceLER:
                empirical_PL = self._estimated_subspaceLER[w]
                events = self._subspace_LE_count.get(w, 0)

                if events >= RELIABLE_EVENTS:
                    # High confidence: use empirical directly
                    sub_PL = empirical_PL
                elif events >= BLEND_EVENTS:
                    # Medium confidence: blend empirical with fitted
                    fitted_PL = modified_sigmoid_function(
                        w, self._a, self._b, self._c, self._t
                    )
                    # Cap fitted PL at 0.5 (theoretical maximum)
                    fitted_PL = min(fitted_PL, 0.5)
                    # Blend weight: more events = more trust in empirical
                    alpha = (events - BLEND_EVENTS) / (RELIABLE_EVENTS - BLEND_EVENTS)
                    alpha = max(0.0, min(1.0, alpha))
                    sub_PL = alpha * empirical_PL + (1 - alpha) * fitted_PL
                else:
                    # Low confidence: use fitted curve (capped at 0.5)
                    sub_PL = min(
                        modified_sigmoid_function(
                            w, self._a, self._b, self._c, self._t
                        ),
                        0.5,
                    )
            else:
                # No empirical data: use S-curve extrapolation
                # This is the KEY change: we extrapolate to ALL weights > t
                sub_PL = min(
                    modified_sigmoid_function(w, self._a, self._b, self._c, self._t),
                    0.5,
                )

            LER += sub_PL * binom_w

        self._LER = LER
        return LER

    def _calc_LER_empirical_only(self) -> float:
        """
        Calculate LER using only empirical estimates (no curve fitting).
        Useful for comparison and bias assessment.
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = int(np.sqrt(p * (1.0 - p) * N))
        if sigma == 0:
            sigma = 1
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        for w in range(minw, maxw + 1):
            if w in self._estimated_subspaceLER:
                sub_PL = self._estimated_subspaceLER[w]
            else:
                # No data for this weight: skip or use zero
                sub_PL = 0.0
            LER += sub_PL * binomial_weight(N, w, p)

        return float(LER)

    def _calc_LER_fitted_only(self) -> float:
        """
        Calculate LER using only the fitted S-curve (no empirical data).
        Useful for comparison and understanding model fit.

        Extrapolates S-curve to all weights > t (code threshold).
        S-curve values are capped at 0.5 (theoretical maximum).
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = int(np.sqrt(p * (1.0 - p) * N))
        if sigma == 0:
            sigma = 1
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        saturate_w = self._saturatew
        first_error_w = self._has_logical_errorw

        # Extend integration range at low error rates
        if maxw < saturate_w:
            maxw = saturate_w

        # Define safe extrapolation limit for high weights
        extrap_limit = saturate_w + 2 * (saturate_w - first_error_w)
        boundary_PL = min(
            modified_sigmoid_function(extrap_limit, self._a, self._b, self._c, self._t),
            0.5,
        )

        for w in range(minw, maxw + 1):
            # PL = 0 only for w <= t (code threshold - fault-tolerant property)
            if w <= self._t:
                sub_PL = 0.0
            elif w > extrap_limit:
                # Beyond safe extrapolation limit
                sub_PL = boundary_PL
            else:
                # Use S-curve extrapolation for all weights > t
                sub_PL = min(
                    modified_sigmoid_function(w, self._a, self._b, self._c, self._t),
                    0.5,
                )
            LER += sub_PL * binomial_weight(N, w, p)

        return LER

    def _estimate_LER_uncertainty(self) -> Tuple[float, float]:
        """
        Estimate uncertainty bounds for the LER estimate.
        Returns (lower_bound, upper_bound) as rough 1-sigma confidence interval.

        Uses propagation of uncertainty from subspace estimates.
        """
        N = self._num_noise
        p = self._error_rate

        sigma = int(np.sqrt(p * (1.0 - p) * N))
        if sigma == 0:
            sigma = 1
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        # Accumulate variance contribution from each weight
        var_LER = 0.0

        for w in range(minw, maxw + 1):
            binom_w = binomial_weight(N, w, p)

            if w in self._estimated_subspaceLER:
                events = self._subspace_LE_count.get(w, 0)
                samples = self._subspace_sample_used.get(w, 1)

                if events > 0 and samples > 0:
                    # Variance of binomial proportion: p(1-p)/n
                    pl_w = self._estimated_subspaceLER[w]
                    var_pl = pl_w * (1 - pl_w) / samples
                    # Contribution to LER variance
                    var_LER += (binom_w**2) * var_pl

        std_LER = float(np.sqrt(var_LER))
        return (max(0.0, self._LER - std_LER), self._LER + std_LER)

    # ------------------------------------------------------------------
    #  Multi-Strategy LER Calculation (Bias Reduction at Low p)
    # ------------------------------------------------------------------

    def _calc_LER_multi_strategy(
        self, strategy: str = "auto"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate LER using multiple strategies and return results for comparison.

        At low error rates (p < 0.001), the gap between binomial peak and
        first observable logical error creates systematic bias. Different
        strategies handle this gap differently.

        Strategies:
        - "scurve_full": S-curve extrapolation to all w > t (current default)
        - "hybrid": Exponential interpolation below first_error_w, S-curve above
        - "conservative": Zero below first_error_w (underestimates)
        - "power_law": Power-law extrapolation below first_error_w
        - "auto": Automatically select best strategy based on gap analysis

        Returns:
            (best_ler, results_dict) where results_dict maps strategy name to LER
        """
        results = {}

        # Calculate using all strategies
        results["scurve_full"] = self._calc_LER_scurve_full()
        results["hybrid"] = self._calc_LER_hybrid()
        results["conservative"] = self._calc_LER_conservative()
        results["power_law"] = self._calc_LER_power_law()
        results["bounded_linear"] = self._calc_LER_bounded_scurve()

        # Strategy selection based on gap analysis
        if strategy == "auto":
            strategy = self._select_best_strategy()

        best_ler = results.get(strategy, results["scurve_full"])
        return best_ler, results

    def _select_best_strategy(self) -> str:
        """
        Automatically select the best LER calculation strategy based on:
        1. Gap size between binomial peak (ep) and first_error_w
        2. S-curve fit quality (R^2)
        3. Number of events at lowest sampled weights

        Returns strategy name.
        """
        N = self._num_noise
        p = self._error_rate
        ep = int(p * N)
        first_error_w = self._has_logical_errorw

        # Gap analysis
        gap = first_error_w - ep
        sigma = max(1, int(np.sqrt(p * (1.0 - p) * N)))

        # If gap is small (< 2 sigma), S-curve extrapolation is reliable
        if gap < 2 * sigma:
            return "scurve_full"

        # If gap is moderate (2-4 sigma), use hybrid approach
        if gap < 4 * sigma:
            return "hybrid"

        # If gap is large and we have good fit, use S-curve
        if self._R_square_score >= 0.98:
            return "scurve_full"

        # For large gaps with poor fit, use power law
        return "power_law"

    def _calc_LER_scurve_full(self) -> float:
        """
        Calculate LER using S-curve extrapolation to ALL weights > t.

        This is the most aggressive extrapolation strategy. It uses the
        fitted S-curve model for all weights, including those below
        first_error_w where no empirical data exists.

        At low error rates, this can OVERESTIMATE if the S-curve
        extrapolates too high at low weights.
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = max(1, int(np.sqrt(p * (1.0 - p) * N)))
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        # Extend to saturation if needed
        if maxw < self._saturatew:
            maxw = self._saturatew

        for w in range(minw, maxw + 1):
            binom_w = binomial_weight(N, w, p)

            if w <= self._t:
                sub_PL = 0.0
            else:
                # S-curve extrapolation for ALL w > t
                sub_PL = min(
                    modified_sigmoid_function(w, self._a, self._b, self._c, self._t),
                    0.5,
                )
            LER += sub_PL * binom_w

        return LER

    def _calc_LER_hybrid(self) -> float:
        """
        Calculate LER using hybrid exponential interpolation.

        - w <= t: PL = 0 (fault-tolerant)
        - t < w < first_error_w: Exponential interpolation
        - w >= first_error_w: S-curve model

        The exponential interpolation smoothly connects PL~0 at w=t+1
        to the S-curve value at w=first_error_w.

        This strategy tends to UNDERESTIMATE at low error rates because
        the exponential decay is too aggressive.
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = max(1, int(np.sqrt(p * (1.0 - p) * N)))
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        if maxw < self._saturatew:
            maxw = self._saturatew

        first_error_w = self._has_logical_errorw
        pl_at_first = modified_sigmoid_function(
            first_error_w, self._a, self._b, self._c, self._t
        )
        pl_at_first = min(pl_at_first, 0.5)

        for w in range(minw, maxw + 1):
            binom_w = binomial_weight(N, w, p)

            sub_PL = hybrid_pl_exponential(
                w, self._a, self._b, self._c, self._t, first_error_w, pl_at_first
            )
            LER += sub_PL * binom_w

        return LER

    def _calc_LER_conservative(self) -> float:
        """
        Calculate LER using conservative strategy (zero below first_error_w).

        - w <= t: PL = 0 (fault-tolerant)
        - t < w < first_error_w: PL = 0 (conservative assumption)
        - w >= first_error_w: Empirical if available, else S-curve

        This strategy SEVERELY UNDERESTIMATES at low error rates when
        the gap between ep and first_error_w is large.
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = max(1, int(np.sqrt(p * (1.0 - p) * N)))
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        if maxw < self._saturatew:
            maxw = self._saturatew

        first_error_w = self._has_logical_errorw

        for w in range(minw, maxw + 1):
            binom_w = binomial_weight(N, w, p)

            if w < first_error_w:
                # Conservative: assume zero below first observed error
                sub_PL = 0.0
            elif w in self._estimated_subspaceLER:
                # Use empirical data where available
                sub_PL = self._estimated_subspaceLER[w]
            else:
                # S-curve for weights above first_error_w without data
                sub_PL = min(
                    modified_sigmoid_function(w, self._a, self._b, self._c, self._t),
                    0.5,
                )

            LER += sub_PL * binom_w

        return LER

    def _calc_LER_bounded_scurve(self) -> float:
        """
        Calculate LER using BOUNDED S-curve extrapolation.

        KEY INSIGHT: The S-curve model `0.5 / (1 + exp(a*w + b + c/sqrt(w-t)))`
        blows up as w → t because of the pole term `c/sqrt(w-t)`.

        At low error rates (p < 0.001), the binomial distribution centers at
        weights very close to t, where the S-curve extrapolation gives
        nonsensical values.

        This strategy bounds the extrapolation by:
        1. Using a linear interpolation between PL=0 at w=t and PL at first_error_w
        2. This is much more physically reasonable than the pole explosion
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = max(1, int(np.sqrt(p * (1.0 - p) * N)))
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        if maxw < self._saturatew:
            maxw = self._saturatew

        first_error_w = self._has_logical_errorw
        t = self._t

        # Get PL at first_error_w from the S-curve
        pl_at_first = modified_sigmoid_function(
            first_error_w, self._a, self._b, self._c, t
        )
        pl_at_first = min(pl_at_first, 0.5)

        for w in range(minw, maxw + 1):
            binom_w = binomial_weight(N, w, p)

            if w <= t:
                sub_PL = 0.0
            elif w < first_error_w:
                # LINEAR interpolation from 0 at w=t+1 to pl_at_first at first_error_w
                # This is much more reasonable than S-curve pole explosion
                span = first_error_w - (t + 1)
                if span > 0:
                    # Linear: PL(w) = pl_at_first * (w - (t+1)) / span
                    sub_PL = pl_at_first * (w - (t + 1)) / span
                else:
                    sub_PL = 0.0
            else:
                # Use S-curve for w >= first_error_w
                sub_PL = min(
                    modified_sigmoid_function(w, self._a, self._b, self._c, t), 0.5
                )

            LER += sub_PL * binom_w

        return LER

    def _calc_LER_power_law(self) -> float:
        """
        Calculate LER using power-law extrapolation below first_error_w.

        - w <= t: PL = 0 (fault-tolerant)
        - t < w < first_error_w: PL = C * (w - t)^alpha (power law fit)
        - w >= first_error_w: Empirical if available, else S-curve

        The power law parameters are fitted to the lowest observed weights.
        This strategy can OVERESTIMATE if the power law is too steep.
        """
        LER = 0.0
        N = self._num_noise
        p = self._error_rate

        sigma = max(1, int(np.sqrt(p * (1.0 - p) * N)))
        ep = int(p * N)
        minw = max(self._t + 1, ep - self._k_range * sigma)
        maxw = min(N, ep + self._k_range * sigma)

        if maxw < self._saturatew:
            maxw = self._saturatew

        first_error_w = self._has_logical_errorw

        # Fit power law to lowest observed weights
        C, alpha_power = fit_power_law_to_lowest_weights(
            self._estimated_subspaceLER, self._t, num_points=4
        )

        # Fallback if fitting failed
        if C <= 0 or not np.isfinite(C):
            # Use S-curve extrapolation as fallback
            return self._calc_LER_scurve_full()

        for w in range(minw, maxw + 1):
            binom_w = binomial_weight(N, w, p)

            if w <= self._t:
                sub_PL = 0.0
            elif w < first_error_w:
                # Power law extrapolation
                sub_PL = min(C * np.power(w - self._t, alpha_power), 0.5)
            elif w in self._estimated_subspaceLER:
                sub_PL = self._estimated_subspaceLER[w]
            else:
                sub_PL = min(
                    modified_sigmoid_function(w, self._a, self._b, self._c, self._t),
                    0.5,
                )

            LER += sub_PL * binom_w

        return LER

    def get_gap_analysis(self) -> Dict:
        """
        Analyze the gap between binomial peak and first observable error.

        Returns diagnostic information useful for understanding bias sources.
        """
        N = self._num_noise
        p = self._error_rate
        ep = int(p * N)
        sigma = max(1, int(np.sqrt(p * (1.0 - p) * N)))

        first_error_w = self._has_logical_errorw
        gap = first_error_w - ep

        return {
            "ep": ep,
            "sigma": sigma,
            "first_error_w": first_error_w,
            "gap": gap,
            "gap_in_sigma": gap / sigma if sigma > 0 else float("inf"),
            "t": self._t,
            "saturate_w": self._saturatew,
            "r_squared": self._R_square_score,
            "n_sampled_weights": len(self._estimated_subspaceLER),
            "critical_minw": max(self._t + 1, ep - self._k_range * sigma),
            "critical_maxw": min(N, ep + self._k_range * sigma),
        }

    # ------------------------------------------------------------------
    #  Main entry point: iterative, time-budgeted estimation
    # ------------------------------------------------------------------

    def calculate_LER_from_file(
        self,
        filepath: str,
        pvalue: float,
        codedistance: int,
        figname: Optional[str] = None,
        titlename: Optional[str] = None,
        savefigures: bool = True,
    ):
        """
        Iteratively calculate the LER from the given circuit file.

        The only user-facing hyperparameters are:
          - time_budget (set in constructor)
          - error_rate (pvalue)
          - codedistance

        All other parameters (sampling strategy, convergence criteria, etc.)
        are automatically tuned internally.

        Args:
            filepath: Path to the STIM circuit file
            pvalue: Physical error rate
            codedistance: Code distance (used to compute t = (d-1)/2)
            figname: Base filename for debug figures (optional)
            titlename: Title for figures (optional)
            savefigures: Whether to save intermediate figures

        Returns:
            Estimated logical error rate (LER)
        """
        self._error_rate = pvalue
        self._circuit_level_code_distance = codedistance
        self._t = max(0, (codedistance - 1) // 2)

        # Generate default figure name if not provided
        if figname is None:
            figname = f"scaler_p{pvalue:.2e}_d{codedistance}_"

        self.parse_from_file(filepath)

        # Reset stats
        self._subspace_LE_count.clear()
        self._subspace_sample_used.clear()
        self._estimated_subspaceLER.clear()
        self._LER = 0.0
        self._a = 0.0  # Reset fit parameters
        self._b = 0.0
        self._c = 0.0
        self._remaining_time_budget = float(self._time_budget)

        start_time = time.perf_counter()

        # Determine S-curve bracket
        print("=" * 60)
        print(f"ScaLER: Stratified LER Estimation")
        print(f"  Time budget: {self._time_budget:.1f}s")
        print(f"  Error rate: {pvalue:.2e}")
        print(f"  Code distance: {codedistance} (t={self._t})")
        print("=" * 60)

        print("\nPhase 1: Determining S-curve bounds...")
        self.determine_lower_w()
        print(f"  Lower bound (first logical error): w = {self._has_logical_errorw}")
        self.determine_saturated_w()
        print(f"  Saturation bound: w = {self._saturatew}")

        # Measure sampling rate
        print("\nPhase 2: Calibrating sampling rate...")
        self.measure_sample_rates()
        if self._remaining_time_budget <= 0.0:
            print("Time budget exhausted during calibration.")
            return None

        # ----------------------------------------------------------
        # Initial warmup sampling: build initial S-curve estimate
        # ----------------------------------------------------------
        print("\nPhase 3: Initial S-curve estimation...")
        WARMUP_MIN_SHOTS = 10_000
        warmup_seconds = 0.1 * self._remaining_time_budget
        warmup_shots_est = int(warmup_seconds * self._sampling_rate)
        warmup_shots = max(WARMUP_MIN_SHOTS, warmup_shots_est)
        warmup_shots = min(warmup_shots, self._MAX_SHOTS_PER_STEP)

        wlist0 = evenly_spaced_ints(self._has_logical_errorw, self._saturatew, 6)
        shots_per_w = max(100, warmup_shots // max(1, len(wlist0)))
        slist0 = [shots_per_w] * len(wlist0)

        total_warmup = sum(slist0)
        if total_warmup > self._MAX_SHOTS_PER_STEP and total_warmup > 0:
            factor = self._MAX_SHOTS_PER_STEP / total_warmup
            slist0 = [max(1, int(round(s * factor))) for s in slist0]

        elapsed = self._sampling_step(wlist0, slist0)
        self._remaining_time_budget -= elapsed

        # First fit
        self.fit_log_S_model(
            filename=figname + "warmup.pdf" if savefigures else None,
            savefigure=savefigures,
            time_val=time.perf_counter() - start_time,
        )
        theta_prev = (self._a, self._b, self._c)
        pl_prev: Optional[float] = self._calc_LER_from_fit()

        print(f"  Initial sweet spot estimate: w = {self._sweet_spot}")
        print(f"  Initial LER estimate: {pl_prev:.3e}")
        print(f"  Initial R²: {self._R_square_score:.4f}")

        # ----------------------------------------------------------
        # Iterative refinement with progressive sampling
        # ----------------------------------------------------------
        print("\nPhase 4: Iterative refinement...")

        # Convergence criteria (internal, not user-exposed)
        stable_count = 0
        param_tol = 0.03
        r2_target = 0.98
        pl_tol = 0.15
        max_iters = 15

        iter_idx = 0
        while (
            self._remaining_time_budget > 0.0
            and stable_count < 2
            and iter_idx < max_iters
        ):
            iter_idx += 1
            print(f"\n--- Iteration {iter_idx} ---")

            wlist, slist = self.next_step()
            if not wlist:
                print("  No more weights to sample.")
                break

            elapsed = self._sampling_step(wlist, slist)
            self._remaining_time_budget -= elapsed
            if self._remaining_time_budget <= 0.0:
                print("  Time budget exhausted.")
                break

            self.fit_log_S_model(
                filename=figname + f"iter{iter_idx}.pdf" if savefigures else None,
                savefigure=savefigures,
                time_val=time.perf_counter() - start_time,
            )
            theta_new = (self._a, self._b, self._c)
            pl_new = self._calc_LER_from_fit()

            params_ok = self.params_stable(
                theta_new, theta_prev, param_tol, self._R_square_score, r2_target
            )
            band_ok = self._band_well_sampled()
            pl_ok = self._pl_stable(pl_new, pl_prev, pl_tol)

            print(f"  Sweet spot: w={self._sweet_spot}, R²={self._R_square_score:.4f}")
            prev_str = f"{pl_prev:.3e}" if pl_prev else "N/A"
            print(f"  LER: {pl_new:.3e} (prev: {prev_str})")
            print(f"  Convergence: params={params_ok}, band={band_ok}, pl={pl_ok}")

            if params_ok and band_ok and pl_ok:
                stable_count += 1
                print(f"  [OK] Stable (count={stable_count}/2)")
            else:
                stable_count = 0

            theta_prev = theta_new
            pl_prev = pl_new

        # ----------------------------------------------------------
        # Final results with bias diagnostics
        # ----------------------------------------------------------
        total_time = time.perf_counter() - start_time
        ler_bias_reduced = self._calc_LER_bias_reduced()
        ler_empirical = self._calc_LER_empirical_only()
        ler_fitted = self._calc_LER_fitted_only()
        ler_lower, ler_upper = self._estimate_LER_uncertainty()

        # Save final figure
        self.fit_log_S_model(
            filename=figname + "final.pdf" if savefigures else None,
            savefigure=savefigures,
            time_val=total_time,
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Estimated LER (bias-reduced): {ler_bias_reduced:.4e}")
        print(f"  Uncertainty interval: [{ler_lower:.4e}, {ler_upper:.4e}]")
        print(f"  LER (empirical only): {ler_empirical:.4e}")
        print(f"  LER (fitted only): {ler_fitted:.4e}")
        print(f"  Final R²: {self._R_square_score:.4f}")
        print(f"  Sweet spot: w = {self._sweet_spot}")
        print(f"  Total samples: {sum(self._subspace_sample_used.values()):,}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Time remaining: {max(0, self._remaining_time_budget):.1f}s")
        print("=" * 60)

        self._LER = ler_bias_reduced
        return ler_bias_reduced


if __name__ == "__main__":
    # Example usage with only time_budget as the main hyperparameter
    filepath = "C:/Users/yezhu/GitRepos/ScaLERQEC/stimprograms/surface/surface9"

    # Create Scaler with time budget (the only user-exposed hyperparameter)
    scaler = Scaler(error_rate=0.001, time_budget=60)  # 60 seconds

    # Run LER estimation
    ler = scaler.calculate_LER_from_file(
        filepath=filepath,
        pvalue=0.001,
        codedistance=9,
        figname="surface9_",  # Base name for output figures
        savefigures=True,
    )

    print(f"\nFinal LER estimate: {ler:.4e}")
