"""Experimental Bernstein profiling of linear Pauli noise.

Uniformize primitive c*p trials to C*p followed by p-independent thinning.
The latent trial count T is NOT Pauli weight W. Joint (T,W) histograms retain
the user's weight convention, including zero-weight classical noise.
See docs/uniformized_profiling.md for the proof and confidence assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from .adaptive import AccuracyEstimate
from .confidence import bounded_empirical_interval
from .general_noise import _integer
from .noise_polynomial import LERPolynomial


def bounded_kl_interval(mean, n, delta):
    """Two-sided Chernoff interval for IID observations in [0,1].

    Bernoulli observations are not required: convexity bounds the MGF of
    any bounded observation by that of a Bernoulli with the same mean.
    This is a fixed-n interval. The caller must allocate delta across looks.
    """
    mean, delta = float(mean), float(delta)
    n = _integer(n, "n", 1)
    if not 0 <= mean <= 1 or not 0 < delta < 1:
        raise ValueError("mean must be in [0,1] and delta in (0,1).")
    threshold = math.log(2 / delta) / n
    if mean == 0:
        return 0.0, -math.expm1(-threshold)
    if mean == 1:
        return math.exp(-threshold), 1.0

    def divergence(q):
        if q <= 0 or q >= 1:
            return math.inf
        return mean * math.log(mean / q) + (1 - mean) * (
            math.log1p(-mean) - math.log1p(-q)
        )

    left, right = 0.0, mean
    for _ in range(60):
        mid = (left + right) / 2
        if divergence(mid) > threshold:
            left = mid
        else:
            right = mid
    lower = left  # Outward rounding of the numerical inversion.
    left, right = mean, 1.0
    for _ in range(60):
        mid = (left + right) / 2
        if divergence(mid) > threshold:
            right = mid
        else:
            left = mid
    return lower, right


class UniformizedSampler:
    """Sample conditional on latent T without a weight suffix table.

    Public for audits and fixed-stratum experiments. ``sample`` returns
    detector/observable bits and actual Pauli weights, in that order.
    Each positive categorical activation is one primitive trial. Every
    positive conditional E/ELSE branch is another primitive trial, including
    unused later branches; retaining these independent latent trials is exact.
    """

    def __init__(self, model):
        self.model = model
        factors, branches, rates = [], [], []
        for j, factor in enumerate(model._factors):
            for r, rate in enumerate(factor.rates):
                if rate > 0:
                    factors.append(j)
                    branches.append(r + 1 if factor.chain else 0)
                    rates.append(rate)
        self.num_trials = len(rates)
        self.rate = max(rates, default=0.0)
        self.factors = np.asarray(factors, dtype=np.int64)
        self.branches = np.asarray(branches, dtype=np.int64)
        self.accept = np.asarray(rates) / self.rate if rates else np.empty(0)
        if rates and np.any(self.accept == 0):
            raise FloatingPointError("A positive thinning probability underflowed.")
        self.offsets = np.r_[
            0, np.cumsum([len(f.weights) for f in model._factors], dtype=np.int64)
        ]
        self.weights = (
            np.concatenate([f.weights for f in model._factors])
            if model._factors
            else np.empty(0, dtype=np.int64)
        )
        max_marks = max((len(f.fractions) for f in model._factors), default=0)
        self.mark_cdf = np.ones((len(model._factors), max_marks))
        for j, factor in enumerate(model._factors):
            if not factor.chain and len(factor.fractions):
                cdf = np.cumsum(factor.fractions)
                cdf[-1] = 1.0
                if np.any((np.diff(np.r_[0.0, cdf]) <= 0) & (factor.fractions > 0)):
                    raise FloatingPointError(
                        "A positive categorical mark lost floating-point sampling support."
                    )
                self.mark_cdf[j, : len(cdf)] = cdf
        if model._responses is None:
            model.compile_responses()
        # Sparse response representation avoids dense gathers per fault. The
        # existing compiler still owns its dense tables; no suffix DP is built.
        columns, indptr = [], [0]
        for responses in model._responses:
            for response in responses:
                selected = np.flatnonzero(response)
                columns.append(selected)
                indptr.append(indptr[-1] + len(selected))
        self.columns = (
            np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)
        )
        self.indptr = np.asarray(indptr, dtype=np.int64)

    def sample(self, trial_counts, rng, *, outcome_buffer=None):
        counts = np.asarray(trial_counts)
        if (
            counts.ndim != 1
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
            or np.any(counts > self.num_trials)
        ):
            raise ValueError("trial_counts must be an integer vector in [0,M].")
        n, factor_count = len(counts), len(self.model._factors)
        if outcome_buffer is not None:
            max_outcome = max(
                (len(f.weights) - 1 for f in self.model._factors), default=0
            )
            if (
                not isinstance(outcome_buffer, np.ndarray)
                or outcome_buffer.shape != (n, factor_count)
                or outcome_buffer.dtype.kind not in "iu"
                or np.iinfo(outcome_buffer.dtype).max < max_outcome
                or not outcome_buffer.flags.writeable
            ):
                raise ValueError(
                    "outcome_buffer must be a writable integer (shots,factors) array with sufficient range."
                )
            outcome_buffer.fill(0)
        width = self.model.num_detectors + self.model.num_observables
        bits = np.zeros((n, width), dtype=bool)
        pauli_weights = np.zeros(n, dtype=np.int64)
        # Bound temporary event storage even at p=1/C. A single shot may have
        # M trials; metadata and that shot necessarily require O(M) space.
        start = 0
        while start < n:
            stop, total = start, 0
            while stop < n and (stop == start or total + int(counts[stop]) <= 200_000):
                total += int(counts[stop])
                stop += 1
            rows = np.repeat(
                np.arange(start, stop), counts[start:stop].astype(np.int64)
            )
            trials = np.concatenate(
                [
                    rng.choice(self.num_trials, int(t), replace=False)
                    for t in counts[start:stop]
                ]
            )
            accepted = rng.random(len(trials)) < self.accept[trials]
            rows, trials = rows[accepted], trials[accepted]
            fs, outcomes = self.factors[trials], self.branches[trials].copy()
            marked = outcomes == 0
            outcomes[marked] = 1 + (
                rng.random(np.count_nonzero(marked))[:, None]
                >= self.mark_cdf[fs[marked]]
            ).sum(axis=1)
            # First accepted ELSE branch wins; later latent trials are ignored.
            order = np.lexsort((outcomes, fs, rows))
            rows, fs, outcomes = rows[order], fs[order], outcomes[order]
            first = (
                np.r_[True, (rows[1:] != rows[:-1]) | (fs[1:] != fs[:-1])]
                if len(rows)
                else np.empty(0, dtype=bool)
            )
            rows, fs, outcomes = rows[first], fs[first], outcomes[first]
            if outcome_buffer is not None:
                outcome_buffer[rows, fs] = outcomes
            ids = self.offsets[fs] + outcomes
            np.add.at(pauli_weights, rows, self.weights[ids])
            # Cap each sparse gather too (an individual response may be dense).
            cursor = 0
            lengths = self.indptr[ids + 1] - self.indptr[ids]
            cumulative = np.r_[0, np.cumsum(lengths)]
            while cursor < len(ids):
                end = max(
                    cursor + 1,
                    int(
                        np.searchsorted(
                            cumulative, cumulative[cursor] + 1_000_000, side="right"
                        )
                        - 1
                    ),
                )
                end = min(end, len(ids))
                lens = lengths[cursor:end]
                repeated = np.repeat(np.arange(cursor, end), lens)
                local_starts = np.repeat(np.r_[0, np.cumsum(lens)[:-1]], lens)
                indices = (
                    self.indptr[ids[repeated]] + np.arange(len(repeated)) - local_starts
                )
                np.logical_xor.at(
                    bits.ravel(), rows[repeated] * width + self.columns[indices], True
                )
                cursor = end
            start = stop
        return bits, pauli_weights


@dataclass(frozen=True)
class BernsteinProfile:
    """Reusable polynomial and joint latent-count/Pauli-weight profile.

    ``joint_counts`` entries are (T,W,shots,failures). Polynomial coefficients
    are importance estimates, not exact conditional failure probabilities.
    Intervals cover only the declared p grid, simultaneously over all looks.
    """

    status: str
    reason: str
    estimates: tuple[AccuracyEstimate, ...]
    confidence: float
    shots: int
    seconds: float
    num_trials: int
    uniform_rate: float
    proposal_probabilities: tuple[float, ...]
    joint_counts: tuple[tuple[int, int, int, int], ...]
    _polynomial: LERPolynomial = field(repr=False)
    _log_proposal: np.ndarray = field(repr=False)

    @property
    def converged(self):
        return self.status == "accuracy_met"

    def to_polynomial(self, *, allow_unconverged=False):
        if not self.converged and not allow_unconverged:
            raise RuntimeError(f"Accuracy target was not met: {self.reason}")
        return self._polynomial

    def weight_polynomial(self, weight, *, allow_unconverged=False):
        """Estimated P(failure AND W=weight); no per-weight accuracy claim."""
        weight = _integer(weight, "weight")
        self.to_polynomial(allow_unconverged=allow_unconverged)
        failures = np.zeros(self.num_trials + 1, dtype=np.int64)
        for t, w, _, failed in self.joint_counts:
            if w == weight:
                failures[t] += failed
        metadata = self._polynomial.metadata
        metadata.update(
            {
                "pauli_weight": weight,
                "accuracy_certified": False,
                "interval_scope": "none: joint-weight contribution",
            }
        )
        return _make_polynomial(
            self._polynomial.reference_p,
            self._polynomial.max_p,
            self.uniform_rate,
            self.num_trials,
            failures,
            self.shots,
            self._log_proposal,
            metadata,
        )


def _make_polynomial(p0, max_p, rate, m, failures, shots, log_proposal, metadata):
    ts = np.flatnonzero(failures)
    if not rate:
        rates, misses = [], np.empty((len(ts), 0), dtype=np.int64)
        reference_logs = np.zeros(len(ts))
    else:
        rates, misses = [rate], (m - ts)[:, None]
        reference_logs = binom.logpmf(ts, m, rate * p0)
    logs = (
        np.log(failures[ts])
        - math.log(max(1, shots))
        - log_proposal[ts]
        + reference_logs
    )
    return LERPolynomial(p0, rates, ts, misses, logs, max_p=max_p, metadata=metadata)


def sample_bernstein_profile(
    model,
    decoder,
    probabilities,
    *,
    relative_error=0.1,
    absolute_error=0.0,
    confidence=0.99,
    max_shots=1_000_000,
    max_seconds=60.0,
    batch_size=512,
    seed=None,
):
    """Automatic, table-free, full-support profiling on a fixed finite p grid.

    A fixed equal mixture of target p values and the interior reference p is
    used to draw T. This defensive proposal bounds likelihood ratios, including
    unobserved failures. Statistical coverage assumes a fixed deterministic
    decoder and independent draws. Time limits are cooperative, between batches.
    Endpoint targets are supported; the reference component covers all T.
    """
    started = perf_counter()
    ps = np.asarray(probabilities, dtype=float)
    if ps.ndim != 1 or not len(ps) or not np.isfinite(ps).all():
        raise ValueError("probabilities must be a nonempty finite vector.")
    for p in ps:
        model._validate_p(p)
    for value, name in [
        (relative_error, "relative_error"),
        (absolute_error, "absolute_error"),
    ]:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if relative_error == absolute_error == 0:
        raise ValueError("At least one requested error tolerance must be positive.")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1).")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive.")
    max_shots = _integer(max_shots, "max_shots", 1)
    batch_size = _integer(batch_size, "batch_size", 1)
    sampler = UniformizedSampler(model)
    m, rate = sampler.num_trials, sampler.rate
    anchors = np.unique(np.r_[ps, model.reference_p])
    if (m + 1) * len(anchors) > 20_000_000:
        raise ValueError(
            "The trial-count by p-grid likelihood table is too large; use fewer p targets."
        )
    qs = anchors * rate if rate else np.zeros(len(anchors))
    grid_q = ps * rate if rate else np.zeros(len(ps))
    if rate and np.any((anchors > 0) & (qs == 0)):
        raise FloatingPointError("A positive uniformized probability underflowed.")
    ts = np.arange(m + 1)
    log_proposal = logsumexp(binom.logpmf(ts[:, None], m, qs), axis=1) - math.log(
        len(anchors)
    )
    # Each target is a component with mass 1/J, hence P_target/Q <= J.
    bound = len(anchors)
    normalized_ratio = np.exp(
        binom.logpmf(ts[:, None], m, grid_q) - log_proposal[:, None] - math.log(bound)
    )
    if not np.isfinite(normalized_ratio).all() or np.any(normalized_ratio > 1 + 1e-10):
        raise FloatingPointError("Invalid defensive-mixture likelihood bounds.")
    normalized_ratio = np.minimum(normalized_ratio, 1)
    rng = np.random.default_rng(seed)
    failures_by_t = np.zeros(m + 1, dtype=np.int64)
    joint = {}
    lower, upper = np.zeros(len(ps)), np.ones(len(ps))
    total, look = 0, 0
    estimates = tuple(AccuracyEstimate(float(p), 0, 0, 1, 1, False) for p in ps)
    status, reason = "budget_exhausted", "Maximum sample count reached."
    while total < max_shots:
        if perf_counter() - started >= max_seconds:
            reason = "Cooperative time budget reached."
            break
        # A deterministic warm-up schedule avoids a large first decode batch.
        # Keeping checkpoint sizes predeclared preserves the error-spending proof.
        n = min(batch_size, 1 << min(look, batch_size.bit_length()), max_shots - total)
        components = rng.integers(len(anchors), size=n)
        counts = rng.binomial(m, qs[components])
        bits, weights = sampler.sample(counts, rng)
        failed = model._failures(
            decoder, bits, model.num_detectors, model.num_observables
        )
        failures_by_t += np.bincount(counts[failed], minlength=m + 1)
        keys, inv, sizes = np.unique(
            np.column_stack([counts, weights]),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        failed_sizes = np.bincount(inv, weights=failed, minlength=len(keys)).astype(
            np.int64
        )
        for key, size, failure_count in zip(keys, sizes, failed_sizes):
            k = tuple(map(int, key))
            previous = joint.get(k, (0, 0))
            joint[k] = (previous[0] + int(size), previous[1] + int(failure_count))
        total += n
        means = failures_by_t @ normalized_ratio / total
        # The histogram contains every nonzero observation; all successes
        # contribute zero to both moments. Use centered sums to avoid
        # cancellation when the observations are almost constant.
        variances = np.zeros(len(ps))
        if total > 1:
            zero_count = total - int(failures_by_t.sum())
            variances = (
                failures_by_t @ (normalized_ratio - means) ** 2 + zero_count * means**2
            ) / (total - 1)
        delta = (1 - confidence) / (len(ps) * (look + 1) * (look + 2))
        for j, mean in enumerate(means):
            lo, hi = bounded_kl_interval(float(mean), total, delta / 2)
            if total > 1:
                elo, ehi = bounded_empirical_interval(
                    float(mean), float(variances[j]), total, delta / 2
                )
                lo, hi = max(lo, elo), min(hi, ehi)
            lower[j] = max(lower[j], bound * lo)
            upper[j] = min(upper[j], bound * hi)
        if np.any(lower > upper):
            raise RuntimeError(
                "Sequential confidence intervals became inconsistent; no certificate issued."
            )
        values = means * bound
        errors = np.maximum(values - lower, upper - values)
        met = errors <= absolute_error + relative_error * lower
        estimates = tuple(
            AccuracyEstimate(
                float(p), float(value), float(lo), float(hi), float(error), bool(ok)
            )
            for p, value, lo, hi, error, ok in zip(
                ps, values, lower, upper, errors, met
            )
        )
        look += 1
        if met.all():
            status, reason = (
                "accuracy_met",
                "Simultaneous finite-grid accuracy target met.",
            )
            break
    metadata = {
        "method": "uniformized_bernstein_defensive_mixture",
        "accuracy_certified": status == "accuracy_met",
        "status": status,
        "confidence": confidence,
        "coverage_grid": ps.tolist(),
        "relative_error": float(relative_error),
        "absolute_error": float(absolute_error),
        "interval_scope": "declared finite grid only; sequential KL and empirical Bernstein bounds",
        "shots": total,
        "latent_trials": m,
        "uniform_rate": rate,
        "proposal_probabilities": anchors.tolist(),
        "pauli_weight": "support at original noise locations",
        "stopped_estimator_unbiased": False,
    }
    polynomial = _make_polynomial(
        model.reference_p,
        model.max_p,
        rate,
        m,
        failures_by_t,
        total,
        log_proposal,
        metadata,
    )
    log_proposal.flags.writeable = False
    return BernsteinProfile(
        status,
        reason,
        estimates,
        confidence,
        total,
        perf_counter() - started,
        m,
        rate,
        tuple(map(float, anchors)),
        tuple((t, w, n, f) for (t, w), (n, f) in sorted(joint.items())),
        polynomial,
        log_proposal,
    )
