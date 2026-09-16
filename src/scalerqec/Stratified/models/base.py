from __future__ import annotations

"""Abstract base class for S-curve models used in ScaLER.

Every S-curve model must subclass :class:`ScurveModelBase` and implement
methods for predicting P_L(w), transforming between probability and
linearised (log-logit) space, fitting parameters to measured data, and
computing the optimal sampling weight (sweet spot).

The base class provides a concrete :meth:`fit` method that performs
weighted nonlinear least squares in the transformed ``y(w)`` space with
bias correction.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


class ScurveModelBase(ABC):
    """Abstract base class for S-curve models used in ScaLER.

    Subclasses must implement methods for:

    * **Prediction**: :meth:`predict` -- compute P_L(w).
    * **Transform / inverse**: :meth:`transform`, :meth:`inverse_transform`
      -- convert between probability and log-logit space.
    * **Linear prediction**: :meth:`linear_prediction` -- compute y(w) in
      transformed space.
    * **Fitting helpers**: :meth:`_get_fit_function`,
      :meth:`_get_initial_guess`, :meth:`_get_bounds`,
      :meth:`_set_params_from_fit`.
    * **Sweet-spot calculation**: :meth:`calculate_sweet_spot`.

    The concrete :meth:`fit` method (provided here) performs weighted
    nonlinear least squares via ``scipy.optimize.curve_fit`` in
    log-logit space, applying bias correction to the observed data.

    Attributes:
        _t (int): Fault-tolerant threshold, ``(d - 1) / 2`` where *d*
            is the code distance.
        _gamma (float): Sweet-spot tuning parameter for the condition
            ``d^2 y / dw^2 = gamma * |dy/dw|``.
        _params (Dict[str, float]): Fitted model parameters.
        _r_squared (float): R^2 goodness-of-fit from the last call to
            :meth:`fit`.
    """

    def __init__(self, t: int = 0, gamma: float = 0.05):
        """Initialize the S-curve model.

        Args:
            t: Fault-tolerant threshold ``(d - 1) / 2``.
            gamma: Sweet-spot tuning parameter controlling where the
                S-curve transitions from curved to linear.
        """
        self._t = t
        self._gamma = gamma
        self._params: Dict[str, float] = {}
        self._r_squared: float = 0.0
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether a fit has succeeded (default parameters are not a fit)."""
        return self._is_fitted

    @property
    def t(self) -> int:
        """Fault-tolerant threshold."""
        return self._t

    @t.setter
    def t(self, value: int) -> None:
        """Set fault-tolerant threshold."""
        self._t = value

    @property
    def gamma(self) -> float:
        """Sweet spot tuning parameter."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        """Set sweet spot tuning parameter."""
        self._gamma = value

    @property
    def r_squared(self) -> float:
        """R² score from the last fit."""
        return self._r_squared

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for display/logging."""
        pass

    @property
    @abstractmethod
    def param_names(self) -> Tuple[str, ...]:
        """Names of the model parameters."""
        pass

    @abstractmethod
    def predict(
        self, w: float | NDArray[np.floating[Any]]
    ) -> float | NDArray[np.floating[Any]]:
        """
        Compute P_L(w) for given weight(s).

        Args:
            w: Weight value(s), must be > t for valid predictions

        Returns:
            Logical error rate P_L in range (0, 0.5)
        """
        pass

    @abstractmethod
    def transform(
        self, p_w: float | NDArray[np.floating[Any]]
    ) -> float | NDArray[np.floating[Any]]:
        """
        Transform P_L to linearized space y(w).

        This is typically y = log(0.5/P_L - 1).

        Args:
            p_w: Logical error rate(s) in range (0, 0.5)

        Returns:
            Transformed value(s) in linear space
        """
        pass

    @abstractmethod
    def inverse_transform(
        self, y: float | NDArray[np.floating[Any]]
    ) -> float | NDArray[np.floating[Any]]:
        """
        Transform y(w) back to P_L.

        Args:
            y: Value(s) in transformed linear space

        Returns:
            Logical error rate P_L in range (0, 0.5)
        """
        pass

    @abstractmethod
    def linear_prediction(
        self, w: float | NDArray[np.floating[Any]]
    ) -> float | NDArray[np.floating[Any]]:
        """
        Compute y(w) in transformed linear space.

        Args:
            w: Weight value(s)

        Returns:
            Predicted y(w) values
        """
        pass

    @abstractmethod
    def _get_fit_function(self):
        """
        Get the function to use for curve fitting.

        Returns:
            A callable suitable for scipy.optimize.curve_fit
        """
        pass

    @abstractmethod
    def _get_initial_guess(
        self, x_list: List[float], y_list: List[float]
    ) -> Tuple[float, ...]:
        """
        Get initial parameter guess for fitting.

        Args:
            x_list: Weight values
            y_list: Transformed y values

        Returns:
            Tuple of initial parameter values
        """
        pass

    @abstractmethod
    def _get_bounds(
        self, initial_guess: Tuple[float, ...]
    ) -> Tuple[List[float], List[float]]:
        """
        Get parameter bounds for fitting.

        Args:
            initial_guess: Initial parameter values

        Returns:
            (lower_bounds, upper_bounds) for each parameter
        """
        pass

    @abstractmethod
    def _set_params_from_fit(self, popt: Tuple[float, ...]) -> None:
        """
        Set model parameters from fitted values.

        Args:
            popt: Optimized parameter values from curve_fit
        """
        pass

    @abstractmethod
    def calculate_sweet_spot(self) -> int:
        """
        Calculate the optimal sampling weight (sweet spot).

        Uses the general formula: d²y/dw² = γ * dy/dw

        Returns:
            Sweet spot weight value (integer > t)
        """
        pass

    def fit(
        self,
        weights: List[int],
        p_values: List[float],
        sample_counts: List[int],
        le_counts: List[int],
    ) -> None:
        """
        Fit model parameters to data.

        Args:
            weights: List of weight values
            p_values: List of estimated P_L values for each weight
            sample_counts: Number of samples at each weight
            le_counts: Number of logical errors at each weight
        """
        # Filter valid data points
        valid_indices = [
            i
            for i, (p, le) in enumerate(zip(p_values, le_counts))
            if 0.0 < p < 0.5 and le > 0
        ]

        if len(valid_indices) < len(self.param_names):
            # Not enough data to fit
            return

        x_list = [float(weights[i]) for i in valid_indices]
        y_list = []
        sigma_list = []

        for i in valid_indices:
            p = p_values[i]
            n = sample_counts[i]
            m = le_counts[i]

            # Transform to linear space with bias correction
            y = self.transform(p)
            bias = self._bias_estimator(n, m)
            y_corrected = y - bias

            y_list.append(y_corrected)
            sigma_list.append(self._sigma_estimator(n, m))

        # Get initial guess and bounds
        initial_guess = self._get_initial_guess(x_list, y_list)
        lower, upper = self._get_bounds(initial_guess)

        # Perform curve fit
        try:
            popt, _ = curve_fit(
                self._get_fit_function(),
                x_list,
                y_list,
                p0=initial_guess,
                bounds=(lower, upper),
                sigma=sigma_list,
                maxfev=50_000,
            )

            self._set_params_from_fit(popt)

            # Compute R²
            y_pred = [self.linear_prediction(x) for x in x_list]
            self._r_squared = self._r_squared_score(y_list, y_pred)
            self._is_fitted = True

        except (RuntimeError, ValueError) as e:
            # Fit failed, keep existing parameters
            print(f"Warning: Curve fitting failed: {e}")

    def set_params(self, **params: float) -> None:
        """Set model parameters."""
        self._params.update(params)

    def get_params(self) -> Dict[str, float]:
        """Get model parameters."""
        return self._params.copy()

    @staticmethod
    def _sigma_estimator(n: int, m: int) -> float:
        """
        Estimate standard deviation of y(w) = log(0.5/P_w - 1).

        Args:
            n: Number of samples
            m: Number of logical errors

        Returns:
            Estimated standard deviation
        """
        if m <= 0 or n <= 1 or n <= 2 * m:
            return 1.0  # Default when formula is invalid

        try:
            result = np.sqrt(n**2 * (n - m) / (m * (n - 1) * (n - 2 * m) ** 2))
            return float(result) if np.isfinite(result) else 1.0
        except (ZeroDivisionError, ValueError):
            return 1.0

    @staticmethod
    def _bias_estimator(n: int, m: int) -> float:
        """
        Estimate bias in y(w) = log(0.5/P_w - 1).

        Args:
            n: Number of samples
            m: Number of logical errors

        Returns:
            Estimated bias
        """
        if m <= 0 or n <= 1 or n <= 2 * m:
            return 0.0

        try:
            result = 0.5 * (n / m) * (n - 4 * m) / (n - 2 * m) ** 2 * (n - m) / (n - 1)
            return float(result) if np.isfinite(result) else 0.0
        except (ZeroDivisionError, ValueError):
            return 0.0

    @staticmethod
    def _r_squared_score(y_true: List[float], y_pred: List[float]) -> float:
        """
        Compute R² (coefficient of determination).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            R² score (clipped to [0, 1])
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
        ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        r2 = 1.0 - ss_res / ss_tot
        return float(max(0.0, r2))
