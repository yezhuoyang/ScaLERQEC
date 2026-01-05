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
    evenly_spaced_ints,
    modified_linear_function,
    modified_linear_function_with_d,
    modified_sigmoid_function,
    refined_sweet_spot,
    sigma_estimator,
)
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
        self, low: int, high: int, shots: int = 2000, epsilon: float = 0.002
    ) -> int:
        """
        Find the smallest w in [low, high] such that PL(w) > epsilon.
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
                # and self._subspace_LE_count.get(x, 0) >= (self._MIN_NUM_LE_EVENT // 10)
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
        # 5. Update sweet spot
        # -----------------------------
        alpha = -1.0 / self._a
        w_sweet = int(refined_sweet_spot(alpha, self._c, self._t, ratio=self._ratio))
        # if w_sweet < ep:
        #     w_sweet = ep
        if w_sweet <= self._t:
            w_sweet = self._t + 1
        w_sweet = min(self._saturatew, max(self._t + 1, w_sweet))
        self._sweet_spot = w_sweet

        sweet_spot_y = modified_linear_function_with_d(
            self._sweet_spot, self._a, self._b, self._c, self._t
        )

        sample_cost_list = [self._subspace_sample_used[x] for x in x_list]

        # -----------------------------
        # 6. Plot for debugging
        # -----------------------------
        if not savefigure:
            return

        # x-range for fitted curve
        x_fit = np.linspace(self._t + 1, max(x_list), 1000)
        y_fit = modified_linear_function_with_d(
            x_fit, self._a, self._b, self._c, self._t
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        # Bars for y_list (log-space)
        ax.bar(
            x_list,
            y_list,
            width=0.6,
            align="center",
            color="orange",
            edgecolor="orange",
            label="Data histogram (log-S)",
        )

        # Error bars in the same units as y_list (just using sigma_list directly,
        # same as in your old code for debugging purposes)
        ax.errorbar(
            x_list,
            y_list,
            yerr=sigma_list,
            fmt="o",
            color="black",
            capsize=3,
            markersize=1,
            elinewidth=1,
            label="Error bars",
        )

        # Fitted curve
        ax.plot(
            x_fit,
            y_fit,
            label=f"Fitted line, R2={self._R_square_score:.4f}",
            color="blue",
            linestyle="--",
        )

        # Sweet spot marker
        ax.scatter(
            self._sweet_spot,
            sweet_spot_y,
            color="purple",
            marker="o",
            s=50,
            label="Sweet Spot",
        )
        ax.text(
            self._sweet_spot * 1.1,
            sweet_spot_y * 1.1,
            "Sweet Spot",
            ha="center",
            color="purple",
            fontsize=10,
        )

        # Fault-tolerant region
        ax.axvspan(0, self._t, color="green", alpha=0.15)
        ax.text(
            self._t / 2,
            max(y_list) * 1.8,
            "Fault\ntolerant",
            ha="center",
            color="green",
            fontsize=8,
        )

        # Curve fitting region
        ax.axvspan(self._t, self._saturatew, color="yellow", alpha=0.10)
        ax.text(
            (self._t + self._saturatew) / 2,
            max(y_list) * 1.2,
            "Curve fitting",
            ha="center",
            fontsize=15,
        )

        # Critical 5σ region
        ax.axvspan(self._minw, self._maxw, color="gray", alpha=0.2)
        ax.axvline(
            self._minw, color="red", linestyle="--", linewidth=1.2, label=r"$w_{\min}$"
        )
        ax.axvline(
            self._maxw,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label=r"$w_{\max}$",
        )
        ax.text(
            (self._minw + self._maxw) / 2,
            max(y_list) * 1.8,
            r"$5\sigma$ Critical Region",
            ha="center",
            fontsize=10,
        )

        # Saturation region
        ax.axvspan(self._saturatew, self._saturatew + 12, color="red", alpha=0.15)
        ax.text(
            self._saturatew + 6,
            max(y_list) * 2.8,
            "Saturation",
            ha="center",
            color="red",
            fontsize=10,
        )

        # Sample cost annotations (scientific notation)
        num_points_to_annotate = min(5, len(x_list))
        indices = np.linspace(0, len(x_list) - 1, num=num_points_to_annotate, dtype=int)
        for i in indices:
            x, y, s = x_list[i], y_list[i], sample_cost_list[i]
            if s > 0:
                s_str = "{0:.1e}".format(s)
                base, exp = s_str.split("e")
                label = r"${0}\times 10^{{{1}}}$".format(base, int(exp))
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=7,
                )

        # Side annotation box (simplified for Scaler)
        text_lines = [
            r"$N_{LE}^{Clip}=%d$" % self._MIN_NUM_LE_EVENT,
            r"$r_{sweet}=%.2f$" % self._ratio,
            r"$\alpha=%.4f$" % alpha,
            r"$\mu =%.4f$" % (alpha * self._b),
            r"$\beta=%.4f$" % self._c,
            r"$w_{\min}=%d$" % self._minw,
            r"$w_{\max}=%d$" % self._maxw,
            r"$w_{sweet}=%d$" % self._sweet_spot,
            r"$\#\mathrm{detector}=%d$" % self._num_detector,
            r"$\#\mathrm{noise}=%d$" % self._num_noise,
        ]
        if self._LER > 0:
            text_lines.append(
                r"$P_L={0}\times 10^{{{1}}}$".format(
                    *"{0:.2e}".format(self._LER).split("e")
                )
            )
        if time_val is not None:
            text_lines.append(r"$\mathrm{Time}=%.2f\,\mathrm{s}$" % time_val)

        fig.subplots_adjust(right=0.75)
        fig.text(
            0.78,
            0.5,
            "\n".join(text_lines),
            fontsize=7,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
        )

        ax.set_xlabel("Weight")
        ax.set_ylabel(r"$\log\left(\frac{0.5}{\mathrm{LER}} - 1\right)$")
        ax.set_title("Fitted log-S-curve")
        ax.legend(fontsize=8)
        fig.tight_layout()

        # Choose filename if none provided
        if filename is None:
            filename = f"logS_fit_debug_p{self._error_rate:.3g}_d{self._circuit_level_code_distance}.pdf"

        print(f"Saving log-S fit debug figure to: {filename}")
        fig.savefig(filename, format="pdf", bbox_inches="tight")
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
    #  Decide the next sampling step
    # ------------------------------------------------------------------

    def _choose_candidate_weights(self) -> List[int]:
        """
        Choose a small set of candidate weights to sample next.

        Policy:
        - Work in the union of the [minw, maxw] band and the current sweet spot.
        - Clip to [t+1, saturatew].
        - Always include w_sweet and a few neighbours, plus a coarse grid
            so that coverage between sweet spot and the ends is roughly uniform.
        """
        # 1) Determine the center (sweet spot), with a reasonable fallback.
        if self._sweet_spot is None:
            ep = int(self._error_rate * self._num_noise)
            center = max(self._t + 1, ep)
        else:
            center = int(self._sweet_spot)

        # 2) Start from the union of [minw, maxw] and {center}
        #    (so the bracket is guaranteed to contain the sweet spot).
        left_bracket = min(self._minw, center)
        right_bracket = max(self._maxw, center)

        # 3) Clip by physical limits [t+1, saturatew]
        left_bracket = max(self._t + 1, left_bracket)
        right_bracket = min(self._saturatew, right_bracket)

        # If everything collapsed, fall back to [t+1, saturatew].
        if right_bracket < left_bracket:
            left_bracket = self._t + 1
            right_bracket = self._saturatew

        # Final safety: if still weird, just return the center.
        if right_bracket < left_bracket:
            return [center]

        # Clamp center into the bracket
        if center < left_bracket:
            center = left_bracket
        if center > right_bracket:
            center = right_bracket

        W = set()

        # 4) Always include the sweet spot and a couple of neighbours
        for delta in range(-2, 3):  # center-2, -1, 0, +1, +2
            w = center + delta
            if left_bracket <= w <= right_bracket:
                W.add(w)

        # 5) Add a small grid across the bracket for uniform coverage
        def evenly_spaced_ints(lo: int, hi: int, k: int) -> List[int]:
            if k <= 1 or hi <= lo:
                return [lo, hi]
            return sorted(
                set(int(round(lo + i * (hi - lo) / (k - 1))) for i in range(k))
            )

        grid_points = evenly_spaced_ints(left_bracket, right_bracket, 5)
        for w in grid_points:
            if left_bracket <= w <= right_bracket:
                W.add(w)

        # 6) Also ensure the bracket endpoints themselves are present
        W.add(left_bracket)
        W.add(right_bracket)

        wlist = sorted(W)
        print("  Candidate weights (choose_candidate_weights):", wlist)
        return wlist

    def next_step(self) -> Tuple[List[int], List[int]]:
        """
        Decide the next step of the stratified sampling process.

        New policy:
          - Always sample at all candidate weights in the bracket.
          - Allocate more shots near the sweet spot and to never-sampled weights,
            but keep a roughly uniform coverage across the whole bracket.
        """
        # If we cannot take any more shots, stop.
        if self._sampling_rate <= 0.0 or self._remaining_time_budget <= 0.0:
            return [], []

        # Total shots we are allowed to take this step
        remaining_shots = int(self._remaining_time_budget * self._sampling_rate)
        if remaining_shots <= 0:
            return [], []

        # Enforce per-step hard cap for memory control
        step_shots = min(remaining_shots, self._MAX_SHOTS_PER_STEP)

        # Choose candidate weights
        wlist = self._choose_candidate_weights()
        if not wlist:
            return [], []

        # ------ Shot allocation policy ------
        # Priority factors for each weight
        factors = []
        center = self._sweet_spot

        for w in wlist:
            f = 1.0

            # 1) Strongly favour the sweet spot and its immediate neighbours
            if center is not None:
                d = abs(w - center)
                if d <= 1:
                    f *= 6.0  # very close to sweet spot
                elif d <= 2:
                    f *= 3.0  # near sweet spot

            # 2) Boost completely unsampled subspaces
            if self._subspace_sample_used.get(w, 0) == 0:
                f *= 4.0

            factors.append(f)

        total_factor = sum(factors)
        if total_factor <= 0.0:
            # Fallback: uniform allocation
            shots_per_w = max(1, step_shots // len(wlist))
            return wlist, [shots_per_w] * len(wlist)

        # Minimum shots so that even far-from-sweet weights keep getting data
        min_shots_per_w = max(
            500, int(step_shots * 0.02)
        )  # at least 2% per weight, ≥500

        slist: List[int] = []
        remaining = step_shots

        for i, (w, f) in enumerate(zip(wlist, factors)):
            if i == len(wlist) - 1:
                # Give whatever shots are left to the last weight,
                # so rounding doesn't lose shots.
                s = max(1, remaining)
            else:
                raw = step_shots * (f / total_factor)
                s = max(min_shots_per_w, int(round(raw)))
                remaining -= s
                if remaining < 0:
                    remaining = 0

            slist.append(s)

        # Debug print
        print("  Candidate weights (choose_candidate_weights):", wlist)
        print("  Allocated shots:", slist, " (total =", sum(slist), ")")

        return wlist, slist

    # ------------------------------------------------------------------
    #  Final LER integration from fitted S-curve
    # ------------------------------------------------------------------

    def _calc_LER_from_fit(self) -> float:
        """
        Integrate the fitted S-curve (or empirical subspace LER where available)
        against the binomial distribution over error weights.
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
                sub_PL = modified_sigmoid_function(
                    w, self._a, self._b, self._c, self._t
                )
            LER += sub_PL * binomial_weight(N, w, p)

        self._LER = LER
        return LER

    # ------------------------------------------------------------------
    #  Main entry point: iterative, time-budgeted estimation
    # ------------------------------------------------------------------

    def calculate_LER_from_file(
        self,
        filepath: str,
        pvalue: float,
        codedistance: int,
        figname,
        titlename,
        repeat: int = 1,
    ):
        """
        Iteratively calculate the LER from the given circuit file.

        Steps:
          1. Parse the circuit from the file, compile stim and QEPG graph.
          2. Measure sampling rate (shots/second).
          3. Iteratively:
             - choose subspaces,
             - sample within remaining time budget (and per-step shot cap),
             - refit S-curve parameters.
          4. Stop when:
             - parameters are stable,
             - the band around sweet spot is well-sampled,
             - and the overall PL estimate is stable.
        """
        self._error_rate = pvalue
        self._circuit_level_code_distance = codedistance
        self._t = max(0, (codedistance - 1) // 2)

        self.parse_from_file(filepath)

        # Reset stats
        self._subspace_LE_count.clear()
        self._subspace_sample_used.clear()
        self._estimated_subspaceLER.clear()
        self._LER = 0.0
        self._remaining_time_budget = float(self._time_budget)

        # Determine S-curve bracket
        print("Determining S-curve lower bound...")
        self.determine_lower_w()
        print(f"  Found has_logical_errorw = {self._has_logical_errorw}")
        print("Determining S-curve saturated bound...")
        self.determine_saturated_w()
        print(f"  Found saturatew = {self._saturatew}")

        # Measure sampling rate
        print("Measuring sampling rate...")
        self.measure_sample_rates()
        if self._remaining_time_budget <= 0.0:
            print("Time budget exhausted during calibration.")
            return None

        # ----------------------------------------------------------
        # Initial warmup sampling: bounded by a fixed max shots
        # ----------------------------------------------------------
        WARMUP_MIN_SHOTS = 10_000
        # Nominal warmup based on time, but capped hard:
        warmup_seconds = 0.1 * self._remaining_time_budget
        warmup_shots_est = int(warmup_seconds * self._sampling_rate)
        warmup_shots = max(WARMUP_MIN_SHOTS, warmup_shots_est)
        warmup_shots = min(warmup_shots, self._MAX_SHOTS_PER_STEP)

        wlist0 = evenly_spaced_ints(self._has_logical_errorw, self._saturatew, 6)
        shots_per_w = max(100, warmup_shots // max(1, len(wlist0)))
        slist0 = [shots_per_w] * len(wlist0)

        # Ensure total shots respects the cap (after integer division)
        total_warmup = sum(slist0)
        if total_warmup > self._MAX_SHOTS_PER_STEP and total_warmup > 0:
            factor = self._MAX_SHOTS_PER_STEP / total_warmup
            slist0 = [max(1, int(round(s * factor))) for s in slist0]

        elapsed = self._sampling_step(wlist0, slist0)
        self._remaining_time_budget -= elapsed

        # First fit
        self.fit_log_S_model(
            filename=figname + "first.pdf", savefigure=True, time_val=None
        )
        theta_prev = (self._a, self._b, self._c)

        # First PL estimate
        pl_prev: Optional[float] = self._calc_LER_from_fit()

        # ----------------------------------------------------------
        # Iterative refinement
        # ----------------------------------------------------------
        stable_count = 0
        param_tol = 0.03  # stricter parameter tolerance
        r2_target = 0.98  # stricter R^2 requirement
        pl_tol = 0.20  # require PL to be stable within 20% (tune)
        max_iters = 10

        iter_idx = 0
        while (
            self._remaining_time_budget > 0.0
            and stable_count < 2
            and iter_idx < max_iters
        ):
            iter_idx += 1
            print(f"\n=== Iteration {iter_idx} ===")

            wlist, slist = self.next_step()
            if not wlist:
                # Nothing more to do under this policy
                print("No more weights to sample under current policy.")
                break

            elapsed = self._sampling_step(wlist, slist)
            self._remaining_time_budget -= elapsed
            if self._remaining_time_budget <= 0.0:
                print("Time budget exhausted during iterative refinement.")
                break

            self.fit_log_S_model(
                filename=figname + f"iter{iter_idx}.pdf", savefigure=True, time_val=None
            )
            theta_new = (self._a, self._b, self._c)
            pl_new = self._calc_LER_from_fit()

            print("Current sweet spot weight:", self._sweet_spot)

            params_ok = self.params_stable(
                theta_new, theta_prev, param_tol, self._R_square_score, r2_target
            )
            band_ok = self._band_well_sampled()
            pl_ok = self._pl_stable(pl_new, pl_prev, pl_tol)

            print(
                f"  R^2={self._R_square_score:.4f}, band_ok={band_ok}, "
                f"params_ok={params_ok}, PL_old={pl_prev:.3e} PL_new={pl_new:.3e}"
            )

            if params_ok and band_ok and pl_ok:
                stable_count += 1
                print(f"  All criteria satisfied (count={stable_count}).")
            else:
                stable_count = 0
                print("  Not yet globally stable; continue sampling.")

            theta_prev = theta_new
            pl_prev = pl_new

        # Final LER estimate
        ler_est = self._calc_LER_from_fit()
        print(f"\nEstimated PL ≈ {ler_est:.3e}")
        print(f"R^2 of final fit: {self._R_square_score:.4f}")
        print(f"Remaining time budget: {self._remaining_time_budget:.2f} s")
        return ler_est


if __name__ == "__main__":
    filepath = "C:/Users/yezhu/GitRepos/ScaLERQEC/stimprograms/surface/surface9"
    scaler = Scaler(error_rate=0.001, time_budget=3600)
    scaler.calculate_LER_from_file(
        filepath,
        pvalue=0.001,
        codedistance=9,
        figname="test.png",
        titlename="Test Circuit",
        repeat=1,
    )

    # montecalc = MonteLERcalc(MIN_NUM_LE_EVENT=1000)
    # montecalc.calculate_LER_from_file(
    #     samplebudget=1_000_000,
    #     filepath=filepath,
    #     pvalue=0.001,
    #     repeat=5,
    # )
