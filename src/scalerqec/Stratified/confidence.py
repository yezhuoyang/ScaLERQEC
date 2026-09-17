"""Conservative finite-sample bounds for fixed-budget importance profiles.

Theorem 4 of Maurer and Pontil (2009), applied to both tails and union-bounded
over sampled strata. These are pointwise bounds, not a confidence band for an
entire p continuum. As elsewhere, probability arithmetic uses floating point.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneralNoiseConfidenceBounds:
    p: float
    confidence: float
    lower: float
    upper: float
    zero_failure_weights: tuple[int, ...]


def _factor_logs(factor, p):
    """Log probabilities without multiplying tiny activation probabilities."""
    if factor.chain:
        survival = 0.0
        probabilities = []
        for rate in factor.rates:
            activation = (
                math.log(rate) + math.log(p) if rate > 0 and p > 0 else -math.inf
            )
            probabilities.append(survival + activation)
            survival += math.log1p(-rate * p) if rate * p < 1 else -math.inf
        return np.array([survival] + probabilities)
    rate = factor.rates[0]
    activation = math.log(rate) + math.log(p) if rate > 0 and p > 0 else -math.inf
    return np.array(
        [math.log1p(-rate * p) if rate * p < 1 else -math.inf]
        + [activation + math.log(f) if f > 0 else -math.inf for f in factor.fractions]
    )


def _maximum_log_likelihood(model, p, limit, reference_p=None):
    """Exact max-product DP over all supported histories of each Pauli weight."""
    reference_p = model.reference_p if reference_p is None else reference_p
    best = np.full(limit + 1, -np.inf)
    best[0] = 0.0
    for factor in model._factors:
        reference = _factor_logs(factor, reference_p)
        target = _factor_logs(factor, p)
        ratio = np.full(len(reference), -np.inf)
        np.subtract(target, reference, out=ratio, where=np.isfinite(reference))
        updated = np.full_like(best, -np.inf)
        for weight in np.unique(factor.weights):
            if weight > limit:
                continue
            score = ratio[factor.weights == weight].max()
            np.maximum(
                updated[weight:],
                best[: limit + 1 - weight] + score,
                out=updated[weight:],
            )
        best = updated
    return best


def confidence_bounds(profile, p, *, confidence=0.95):
    """Bound full LER, including omitted weights and unobserved failures.

    Requires independent samples and fixed, preselected sample counts within
    strata. Not valid after optional stopping or decoder selection using these
    same samples. For M preselected p values, use 1-(1-confidence)/M at each
    point to obtain simultaneous coverage by a further union bound.
    """
    p = profile.model._validate_p(p)
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one.")
    limit = int(profile.weights.max())
    maxima = _maximum_log_likelihood(profile.model, p, limit)
    logs = profile._log_likelihood(p)
    mass, tail = profile.model._weight_distribution_with_tail(p, limit)
    unsampled = np.ones(limit + 1, dtype=bool)
    unsampled[profile.sampled_weights] = False
    lower, upper = 0.0, float(tail + mass[unsampled].sum())
    log_delta = math.log(4 * len(profile.sampled_weights) / (1 - confidence))
    no_failures = []
    for w, n in zip(profile.sampled_weights, profile.counts):
        chosen = profile.weights == w
        if not profile.failures[chosen].any():
            no_failures.append(int(w))
        if mass[w] == 0:
            continue
        # Normalize before exponentiation: 0 <= F R / max(R) <= 1.
        values = (
            np.exp(np.minimum(0.0, logs[chosen] - maxima[w])) * profile.failures[chosen]
        )
        mean = float(values.mean())
        variance = float(values.var(ddof=1))
        radius = math.sqrt(2 * variance * log_delta / n) + 7 * log_delta / (3 * (n - 1))
        log_bound = math.log(profile._reference_mass[w]) + maxima[w]

        def contribution(value, cap=mass[w], log_scale=log_bound):
            if value <= 0:
                return 0.0
            return math.exp(min(math.log(cap), log_scale + math.log(value)))

        lower += contribution(mean - radius)
        upper += contribution(mean + radius)
    return GeneralNoiseConfidenceBounds(
        p, confidence, min(1.0, lower), min(1.0, upper), tuple(no_failures)
    )
