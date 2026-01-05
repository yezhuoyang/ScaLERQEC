import numpy as np
from typing import Tuple, Callable


def scurve_function(x, center, sigma):
    return 0.5 / (1 + np.exp(-(x - center) / sigma))
    # return 0*x


# Define the inverse transform: y → 1/2 * 1 / (1 + e^y)
def inv_logit_half(y):
    return 0.5 / (1 + np.exp(y))


def linear_function(x, a, b):
    """
    Linear function for curve fitting.
    """
    return a * x + b


def modified_linear_function_with_d(x, a, b, c, d):
    eps = 1e-12
    delta = (x - d) ** 0.5
    delta = np.where(np.abs(delta) < eps, np.sign(delta) * eps, delta)
    return a * x + b + c / delta


# Strategy A: keep the model safe near the pole
def modified_linear_function(d):
    def tempfunc(x, a, b, c, d=d):
        return modified_linear_function_with_d(x, a, b, c, d)

    return tempfunc


def modified_sigmoid_function(x, a, b, c, d):
    """
    Modified sigmoid function for curve fitting.
    This function is used to fit the S-curve.
    """
    z = a * x + b + c / ((x - d) ** 0.5)
    # ignore overflows in exp → exp(z) becomes np.inf, so 0.5/(1+inf) = 0.0
    with np.errstate(over="ignore"):
        y = 0.5 / (1 + np.exp(z))
    return y


def quadratic_function(x, a, b, c):
    """
    Linear function for curve fitting.
    """
    return a * x**2 + b * x + c


def poly_function(x, a, b, c, d):
    """
    Linear function for curve fitting.
    """
    return a * x**3 + b * x**2 + c * x + d


# Redefine turning point where the 2nd term is still significant in dy/dw
def refined_sweet_spot(alpha, beta, t, ratio=0.05):
    # We define turning point by solving: 1/alpha = ratio * (1/2) * beta / (w - t)^{3/2}
    # => (w - t)^{3/2} = (ratio * beta * alpha) / 2
    # => w = t + [(ratio * beta * alpha / 2)]^{2/3}
    return t + ((ratio * beta * alpha / 2) ** (2 / 3))


"""
Return the estimated sigma of y(w)
"""


def sigma_estimator(N, M):
    return np.sqrt(N**2 * (N - M) / (M * (N - 1) * (N - 2 * M) ** 2))


"""
Return the estimated sigma of Pw
"""


def subspace_sigma_estimator(N, M):
    return np.sqrt(M * (N - M) / (N - 1)) / N


def bias_estimator(N, M):
    """
    Bias = E[y(w)] - y(w)
    Estimated by: (1/2) * f''(P_w) * Var(P_w)
    where f(x) = ln(1/(2x) - 1)
    """
    # Pw = M / N
    # var_Pw = sigma_estimator(N, M)**2
    # f2 = 4 / (1 - 2 * Pw)**2 + 1 / Pw**2
    # bias = 0.5 * f2 * var_Pw
    # return 0
    return 1 / 2 * (N / M) * (N - 4 * M) / (N - 2 * M) ** 2 * (N - M) / (N - 1)


def show_bias_estimator(N, M):
    """
    Bias = E[y(w)] - y(w)
    Estimated by: (1/2) * f''(P_w) * Var(P_w)
    where f(x) = ln(1/(2x) - 1)
    """
    Pw = M / N
    return (1 - Pw) / (2 * Pw * N)


def evenly_spaced_ints(minw, maxw, N):
    if N == 1:
        return [minw]
    if N > (maxw - minw + 1):
        return list(range(minw, maxw + 1))

    # Use high-resolution linspace, round, then deduplicate
    raw = np.linspace(minw, maxw, num=10 * N)
    rounded = sorted(set(map(int, raw)))

    # Pick N evenly spaced indices from the unique set
    indices = np.linspace(0, len(rounded) - 1, num=N, dtype=int)
    return [rounded[i] for i in indices]


# =============================================================================
# NEW S-CURVE MODELS FOR COMPREHENSIVE COMPARISON
# =============================================================================


def paper_scurve_function(w, mu: float, alpha: float, beta: float, t: int):
    """
    Paper's S-curve: f_t[mu, alpha, beta](w) = 0.5 / (1 + exp(-((w-mu)/alpha) + beta/sqrt(w-t)))

    This is the exact formulation from the paper (Definition 3.1).
    Returns 0 for w <= t (fault-tolerant region).

    Parameters:
    - w: weight (scalar or array)
    - mu: center parameter
    - alpha: scale parameter (related to slope)
    - beta: pole strength parameter
    - t: threshold (d-1)/2 where d is code distance
    """
    w = np.asarray(w, dtype=float)
    is_scalar = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=float)
    mask = w > t

    if np.any(mask):
        w_valid = w[mask]
        z = -((w_valid - mu) / alpha) + beta / np.sqrt(w_valid - t)
        with np.errstate(over="ignore"):
            result[mask] = 0.5 / (1.0 + np.exp(z))

    return float(result[0]) if is_scalar else result


def paper_log_scurve(w, mu: float, alpha: float, beta: float, t: int):
    """
    Log-transformed S-curve for fitting: y = -((w-mu)/alpha) + beta/sqrt(w-t)
    This is what we fit to log(0.5/PL - 1).

    Used for curve fitting in log-space.
    """
    w = np.asarray(w, dtype=float)
    eps = 1e-12
    delta = np.sqrt(np.maximum(w - t, eps))
    return -((w - mu) / alpha) + beta / delta


def paper_log_scurve_curried(t: int) -> Callable:
    """Return a curried version of paper_log_scurve for curve_fit."""

    def func(w, mu, alpha, beta):
        return paper_log_scurve(w, mu, alpha, beta, t)

    return func


def convert_params_to_paper(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """
    Convert current params (a, b, c) to paper params (mu, alpha, beta).

    Current model: 0.5 / (1 + exp(a*x + b + c/sqrt(x-d)))
    Paper model: 0.5 / (1 + exp(-((w-mu)/alpha) + beta/sqrt(w-t)))

    Relationship:
    - a = -1/alpha => alpha = -1/a
    - b = mu/alpha => mu = b * alpha = -b/a
    - c = beta
    """
    if abs(a) < 1e-12:
        raise ValueError("Parameter a is too close to zero")
    alpha = -1.0 / a
    mu = -b / a  # = b * alpha
    beta = c
    return mu, alpha, beta


def convert_params_from_paper(
    mu: float, alpha: float, beta: float
) -> Tuple[float, float, float]:
    """
    Convert paper params (mu, alpha, beta) to current params (a, b, c).
    """
    if abs(alpha) < 1e-12:
        raise ValueError("Parameter alpha is too close to zero")
    a = -1.0 / alpha
    b = mu / alpha
    c = beta
    return a, b, c


def quadratic_scurve_function(w, a: float, b: float, c: float, d: float, t: int):
    """
    Quadratic S-curve model: 0.5 / (1 + exp(a*x^2 + b*x + c + d/sqrt(x-t)))

    Adds a quadratic term to capture potential curvature in the S-curve.
    May better fit codes with non-linear scaling behavior.
    """
    w = np.asarray(w, dtype=float)
    is_scalar = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=float)
    mask = w > t

    if np.any(mask):
        w_valid = w[mask]
        eps = 1e-12
        delta = np.sqrt(np.maximum(w_valid - t, eps))
        z = a * w_valid**2 + b * w_valid + c + d / delta
        with np.errstate(over="ignore"):
            result[mask] = 0.5 / (1.0 + np.exp(z))

    return float(result[0]) if is_scalar else result


def quadratic_log_scurve(w, a: float, b: float, c: float, d: float, t: int):
    """Log-transformed quadratic S-curve for fitting."""
    w = np.asarray(w, dtype=float)
    eps = 1e-12
    delta = np.sqrt(np.maximum(w - t, eps))
    return a * w**2 + b * w + c + d / delta


def quadratic_log_scurve_curried(t: int) -> Callable:
    """Return a curried version of quadratic_log_scurve for curve_fit."""

    def func(w, a, b, c, d):
        return quadratic_log_scurve(w, a, b, c, d, t)

    return func


def double_pole_scurve_function(w, a: float, b: float, c: float, e: float, t: int):
    """
    Double-pole S-curve model: 0.5 / (1 + exp(a*x + b + c/sqrt(x-t) + e/(x-t)))

    Adds a second pole term 1/(x-t) to capture steeper transitions near the threshold.
    The 1/(x-t) term dominates closer to t than the 1/sqrt(x-t) term.
    """
    w = np.asarray(w, dtype=float)
    is_scalar = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=float)
    mask = w > t

    if np.any(mask):
        w_valid = w[mask]
        eps = 1e-12
        delta = np.maximum(w_valid - t, eps)
        sqrt_delta = np.sqrt(delta)
        z = a * w_valid + b + c / sqrt_delta + e / delta
        with np.errstate(over="ignore"):
            result[mask] = 0.5 / (1.0 + np.exp(z))

    return float(result[0]) if is_scalar else result


def double_pole_log_scurve(w, a: float, b: float, c: float, e: float, t: int):
    """Log-transformed double-pole S-curve for fitting."""
    w = np.asarray(w, dtype=float)
    eps = 1e-12
    delta = np.maximum(w - t, eps)
    sqrt_delta = np.sqrt(delta)
    return a * w + b + c / sqrt_delta + e / delta


def double_pole_log_scurve_curried(t: int) -> Callable:
    """Return a curried version of double_pole_log_scurve for curve_fit."""

    def func(w, a, b, c, e):
        return double_pole_log_scurve(w, a, b, c, e, t)

    return func


def variable_power_scurve_function(
    w, a: float, b: float, c: float, gamma: float, t: int
):
    """
    Variable-power S-curve model: 0.5 / (1 + exp(a*x + b + c/(x-t)^gamma))

    The power gamma in the pole term is a free parameter.
    - gamma = 0.5: equivalent to current model
    - gamma > 0.5: steeper near threshold
    - gamma < 0.5: gentler near threshold
    """
    w = np.asarray(w, dtype=float)
    is_scalar = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=float)
    mask = w > t

    if np.any(mask):
        w_valid = w[mask]
        eps = 1e-12
        delta = np.maximum(w_valid - t, eps)
        pole_term = c / np.power(delta, gamma)
        z = a * w_valid + b + pole_term
        with np.errstate(over="ignore"):
            result[mask] = 0.5 / (1.0 + np.exp(z))

    return float(result[0]) if is_scalar else result


def variable_power_log_scurve(w, a: float, b: float, c: float, gamma: float, t: int):
    """Log-transformed variable-power S-curve for fitting."""
    w = np.asarray(w, dtype=float)
    eps = 1e-12
    delta = np.maximum(w - t, eps)
    return a * w + b + c / np.power(delta, gamma)


def variable_power_log_scurve_curried(t: int) -> Callable:
    """Return a curried version of variable_power_log_scurve for curve_fit."""

    def func(w, a, b, c, gamma):
        return variable_power_log_scurve(w, a, b, c, gamma, t)

    return func


def linear_only_scurve_function(w, a: float, b: float, t: int):
    """
    Linear-only S-curve model: 0.5 / (1 + exp(a*x + b))

    No pole term - simplest possible model.
    May work well when the S-curve is far from the threshold.
    """
    w = np.asarray(w, dtype=float)
    is_scalar = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=float)
    mask = w > t

    if np.any(mask):
        w_valid = w[mask]
        z = a * w_valid + b
        with np.errstate(over="ignore"):
            result[mask] = 0.5 / (1.0 + np.exp(z))

    return float(result[0]) if is_scalar else result


def linear_only_log_scurve(w, a: float, b: float, t: int):
    """Log-transformed linear-only S-curve for fitting."""
    w = np.asarray(w, dtype=float)
    return a * w + b


def linear_only_log_scurve_curried(t: int) -> Callable:
    """Return a curried version of linear_only_log_scurve for curve_fit."""

    def func(w, a, b):
        return linear_only_log_scurve(w, a, b, t)

    return func


# =============================================================================
# HYBRID PL MODELS FOR LOW-WEIGHT EXTRAPOLATION
# =============================================================================


def hybrid_pl_exponential(
    w,
    a: float,
    b: float,
    c: float,
    t: int,
    first_error_w: int,
    pl_at_first: float,
):
    """
    Hybrid model with exponential interpolation below first_error_w.

    - w <= t: PL = 0 (fault-tolerant region)
    - t < w < first_error_w: exponential interpolation from ~0 to pl_at_first
    - w >= first_error_w: S-curve model

    The exponential interpolation is: PL(w) = pl_at_first * exp(-lambda * (first_error_w - w))
    where lambda is chosen so PL(t+1) ~ epsilon (very small).
    """
    w = np.asarray(w, dtype=float)
    is_scalar = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=float)

    # Region 1: w <= t (fault-tolerant)
    # result already 0

    # Region 2: t < w < first_error_w (exponential interpolation)
    mask_interp = (w > t) & (w < first_error_w)
    if np.any(mask_interp):
        epsilon = 1e-10  # Target PL at w=t+1
        span = first_error_w - (t + 1)
        if span > 0 and pl_at_first > epsilon:
            lam = np.log(pl_at_first / epsilon) / span
            result[mask_interp] = pl_at_first * np.exp(
                -lam * (first_error_w - w[mask_interp])
            )

    # Region 3: w >= first_error_w (S-curve)
    mask_scurve = w >= first_error_w
    if np.any(mask_scurve):
        result[mask_scurve] = np.minimum(
            modified_sigmoid_function(w[mask_scurve], a, b, c, t), 0.5
        )

    return float(result[0]) if is_scalar else result


def hybrid_pl_power_law(
    w,
    t: int,
    first_error_w: int,
    C: float,
    alpha_power: float,
    empirical_data: dict,
    a: float,
    b: float,
    c: float,
):
    """
    Hybrid model with power-law extrapolation below first_error_w.

    - w <= t: PL = 0 (fault-tolerant)
    - t < w < first_error_w: PL(w) = C * (w - t)^alpha_power
    - w >= first_error_w: empirical if available, else S-curve

    The power law parameters C and alpha_power should be fit to the lowest
    observed weights.
    """
    w = np.asarray(w, dtype=float)
    is_scalar = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=float)

    # Region 1: w <= t
    # result already 0

    # Region 2: t < w < first_error_w (power-law extrapolation)
    mask_extrap = (w > t) & (w < first_error_w)
    if np.any(mask_extrap):
        w_valid = w[mask_extrap]
        result[mask_extrap] = np.minimum(C * np.power(w_valid - t, alpha_power), 0.5)

    # Region 3: w >= first_error_w
    mask_observed = w >= first_error_w
    if np.any(mask_observed):
        for i, w_val in enumerate(w):
            if w_val >= first_error_w:
                w_int = int(w_val)
                if w_int in empirical_data:
                    result[i] = empirical_data[w_int]
                else:
                    result[i] = min(modified_sigmoid_function(w_val, a, b, c, t), 0.5)

    return float(result[0]) if is_scalar else result


def fit_power_law_to_lowest_weights(
    empirical_data: dict, t: int, num_points: int = 3
) -> Tuple[float, float]:
    """
    Fit power law PL(w) = C * (w - t)^alpha to the lowest observed weights.

    Returns (C, alpha) or (0, 0) if fitting fails.
    """
    # Get lowest weights with positive PL
    observed = sorted(
        [(w, pl) for w, pl in empirical_data.items() if w > t and pl > 0]
    )[:num_points]

    if len(observed) < 2:
        return 0.0, 0.0

    x_fit = np.array([np.log(w - t) for w, _ in observed])
    y_fit = np.array([np.log(pl) for _, pl in observed])

    if np.any(~np.isfinite(x_fit)) or np.any(~np.isfinite(y_fit)):
        return 0.0, 0.0

    # Linear regression: log(PL) = log(C) + alpha * log(w - t)
    n = len(x_fit)
    x_mean = np.mean(x_fit)
    y_mean = np.mean(y_fit)

    denom = np.sum((x_fit - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0, 0.0

    alpha = np.sum((x_fit - x_mean) * (y_fit - y_mean)) / denom
    log_C = y_mean - alpha * x_mean
    C = np.exp(log_C)

    return C, alpha


# =============================================================================
# MODEL REGISTRY FOR EASY COMPARISON
# =============================================================================

SCURVE_MODELS = {
    "current": {
        "sigmoid_func": modified_sigmoid_function,
        "log_func": modified_linear_function_with_d,
        "log_curried": modified_linear_function,
        "n_params": 3,  # a, b, c (plus fixed t)
        "param_names": ["a", "b", "c"],
    },
    "paper": {
        "sigmoid_func": paper_scurve_function,
        "log_func": paper_log_scurve,
        "log_curried": paper_log_scurve_curried,
        "n_params": 3,  # mu, alpha, beta
        "param_names": ["mu", "alpha", "beta"],
    },
    "quadratic": {
        "sigmoid_func": quadratic_scurve_function,
        "log_func": quadratic_log_scurve,
        "log_curried": quadratic_log_scurve_curried,
        "n_params": 4,  # a, b, c, d
        "param_names": ["a", "b", "c", "d"],
    },
    "double_pole": {
        "sigmoid_func": double_pole_scurve_function,
        "log_func": double_pole_log_scurve,
        "log_curried": double_pole_log_scurve_curried,
        "n_params": 4,  # a, b, c, e
        "param_names": ["a", "b", "c", "e"],
    },
    "variable_power": {
        "sigmoid_func": variable_power_scurve_function,
        "log_func": variable_power_log_scurve,
        "log_curried": variable_power_log_scurve_curried,
        "n_params": 4,  # a, b, c, gamma
        "param_names": ["a", "b", "c", "gamma"],
    },
    "linear_only": {
        "sigmoid_func": linear_only_scurve_function,
        "log_func": linear_only_log_scurve,
        "log_curried": linear_only_log_scurve_curried,
        "n_params": 2,  # a, b
        "param_names": ["a", "b"],
    },
}
