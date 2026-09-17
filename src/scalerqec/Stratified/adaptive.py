"""Accuracy-controlled Pauli-weight profiling with safe sequential stopping.

Fixed conditional mixture proposals, deterministic doubling checkpoints, and
summable error spending give simultaneous confidence over strata, checkpoints,
and a requested finite p grid. Small strata can instead be enumerated exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from .confidence import _factor_logs, _maximum_log_likelihood
from .noise_polynomial import LERPolynomial

_MAX_TABLE_CELLS = 50_000_000


@dataclass(frozen=True)
class AccuracyEstimate:
    p: float
    ler: float
    lower: float
    upper: float
    error_bound: float
    accuracy_met: bool


@dataclass(frozen=True)
class AccuracyControlledProfile:
    """A polynomial and simultaneous intervals at a preselected finite grid.

    Certification is statistical, at ``confidence``, under the documented
    model/sampling assumptions; it is not a continuum band or exact arithmetic.
    The stopped point estimator need not be unbiased at its random stopping time.
    """

    status: str
    reason: str
    estimates: tuple[AccuracyEstimate, ...]
    confidence: float
    relative_error: float
    absolute_error: float
    shots: int
    exact_histories: int
    sample_counts: dict[int, int]
    exact_weights: tuple[int, ...]
    proposal_probabilities: tuple[float, ...]
    seconds: float
    _polynomial: LERPolynomial = field(repr=False)

    @property
    def converged(self):
        return self.status == "accuracy_met"

    def to_polynomial(self, *, allow_unconverged=False):
        """Export with coverage-grid provenance; reject unresolved runs by default."""
        if not self.converged and not allow_unconverged:
            raise RuntimeError(f"Accuracy target was not met: {self.reason}")
        return self._polynomial


def _integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _log_ratios(active, misses, probabilities, model):
    ps = np.asarray(probabilities)
    logs = np.zeros((len(active), len(ps)))
    nonzero = ps > 0
    logs[:, nonzero] = active[:, None] * (
        np.log(ps[nonzero]) - math.log(model.reference_p)
    )
    logs[np.ix_(active > 0, ~nonzero)] = -np.inf
    for g, rate in enumerate(model.rates):
        inside = rate * ps < 1
        logs[:, inside] += misses[:, g, None] * (
            np.log1p(-rate * ps[inside]) - math.log1p(-rate * model.reference_p)
        )
        logs[np.ix_(misses[:, g] > 0, ~inside)] = -np.inf
    return logs


def _support_table(model, limit, cap):
    """Capped counts of supported histories, for bounded exact enumeration."""
    table = np.zeros((len(model._factors) + 1, limit + 1))
    table[-1, 0] = 1
    for j in range(len(model._factors) - 1, -1, -1):
        factor = model._factors[j]
        supported = factor.probabilities(model.reference_p) > 0
        counts = np.bincount(factor.weights[supported])
        table[j] = np.minimum(cap, np.convolve(table[j + 1], counts)[: limit + 1])
    return table


def _enumerate_stratum(model, decoder, weight, count, table, batch_size=4096):
    """Unrank every supported history in a small stratum, in bounded batches."""
    if model._responses is None:
        model.compile_responses()
    for first in range(0, count, batch_size):
        ranks = np.arange(first, min(count, first + batch_size), dtype=np.int64)
        n = len(ranks)
        remaining = np.full(n, weight, dtype=np.int64)
        bits = np.zeros((n, model.num_detectors + model.num_observables), dtype=bool)
        active = np.zeros(n, dtype=np.int64)
        misses = np.zeros((n, len(model.rates)), dtype=np.int64)
        log_probability = np.zeros(n)
        for j, factor in enumerate(model._factors):
            options = np.flatnonzero(factor.probabilities(model.reference_p) > 0)
            residual = remaining[:, None] - factor.weights[options]
            counts = table[j + 1, np.maximum(0, residual)].copy()
            counts[residual < 0] = 0
            cumulative = counts.cumsum(axis=1)
            chosen = (ranks[:, None] >= cumulative).sum(axis=1)
            before = np.column_stack([np.zeros(n), cumulative[:, :-1]])
            ranks -= before[np.arange(n), chosen].astype(np.int64)
            outcomes = options[chosen]
            remaining -= factor.weights[outcomes]
            bits ^= model._responses[j][outcomes]
            active += outcomes != 0
            log_probability += _factor_logs(factor, model.reference_p)[outcomes]
            for r, g in enumerate(np.searchsorted(model.rates, factor.rates)):
                misses[:, g] += (
                    (outcomes == 0) | (outcomes > r + 1)
                    if factor.chain
                    else outcomes == 0
                )
        if remaining.any() or ranks.any():
            raise RuntimeError("Exact stratum enumeration did not exhaust its ranks.")
        failures = model._failures(
            decoder, bits, model.num_detectors, model.num_observables
        )
        yield failures, active, misses, log_probability


def _polynomial(model, coefficients, metadata=None):
    keys = np.asarray(list(coefficients), dtype=np.int64).reshape(
        -1, 1 + len(model.rates)
    )
    return LERPolynomial(
        model.reference_p,
        model.rates,
        keys[:, 0],
        keys[:, 1:],
        np.asarray(list(coefficients.values())),
        max_p=model.max_p,
        metadata=metadata,
    )


def _group_failures(failures, active, misses):
    keys, inverse, counts = np.unique(
        np.column_stack([active[failures], misses[failures]]),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return keys, inverse, counts


@dataclass
class _Stratum:
    weight: int
    mass: np.ndarray
    rng: np.random.Generator
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    m2: np.ndarray
    n: int = 0
    look: int = 0
    exact: bool = False
    histogram: dict = field(default_factory=dict)
    coefficients: dict = field(default_factory=dict)
    proposal_indices: np.ndarray | None = None
    log_normalizers: np.ndarray | None = None
    log_bound: np.ndarray | None = None


def sample_until_accuracy(
    model,
    decoder,
    probabilities,
    *,
    relative_error=0.1,
    absolute_error=0.0,
    confidence=0.99,
    max_shots=1_000_000,
    max_seconds=60.0,
    exact_budget=100_000,
    seed=None,
):
    """Sample automatically until accuracy is established or resources run out.

    On the simultaneous confidence event, an ``accuracy_met`` result satisfies
    |estimate-LER| <= absolute_error + relative_error*LER at EVERY requested p.
    The same polynomial is returned for reuse; certification covers only this
    finite grid. Samples stop at predetermined doubling checkpoints, so the
    confidence calculation remains valid under adaptive allocation/stopping.

    ``max_shots`` and ``max_seconds`` are resource caps, not accuracy settings.
    The time cap is cooperative, checked between complete work units; response
    compilation/one work unit may exceed it. ``exact_budget`` bounds the number
    of deterministically enumerated histories separately from random samples.
    """
    started = perf_counter()
    ps = np.asarray(list(probabilities), dtype=float)
    if ps.ndim != 1 or not 0 < len(ps) <= 1024:
        raise ValueError("probabilities must contain between 1 and 1024 p values.")
    ps = np.unique([model._validate_p(p) for p in ps])
    relative_error, absolute_error, confidence = map(
        float, (relative_error, absolute_error, confidence)
    )
    if (
        not math.isfinite(relative_error)
        or not math.isfinite(absolute_error)
        or relative_error < 0
        or absolute_error < 0
        or relative_error + absolute_error <= 0
    ):
        raise ValueError(
            "Accuracy tolerances must be finite, nonnegative, and not both zero."
        )
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one.")
    max_shots = _integer(max_shots, "max_shots")
    exact_budget = _integer(exact_budget, "exact_budget")
    if exact_budget > 1_000_000_000:
        raise ValueError(
            "exact_budget exceeds the supported enumeration counter range."
        )
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite, or None.")
    # Every active original outcome must have a representable proposal probability.
    for factor in model._factors:
        if np.any(
            np.isfinite(_factor_logs(factor, model.reference_p))
            & (factor.probabilities(model.reference_p) <= 0)
        ):
            raise FloatingPointError(
                "Reference channel support underflowed; rescale the circuit reference."
            )
    chosen = (
        ps
        if len(ps) <= 8
        else ps[np.unique(np.linspace(0, len(ps) - 1, 5).astype(int))]
    )
    anchors = np.unique(np.r_[chosen, model.reference_p])
    for p in anchors:
        for factor in model._factors:
            if np.any(
                np.isfinite(_factor_logs(factor, p)) & (factor.probabilities(p) <= 0)
            ):
                raise FloatingPointError(
                    "A requested proposal channel probability underflowed."
                )
    entropy = np.random.SeedSequence(seed).entropy
    states = {}
    plans = [model._sampling_plan(p) for p in anchors]
    limit = min(16, model.max_weight)
    shots, exact_histories = 0, 0
    tables, counts, mass, tail = None, None, None, None
    status, reason = (
        "budget_exhausted",
        "Resource budget exhausted before meeting the requested accuracy.",
    )

    def prepare(new_limit):
        if (len(model._factors) + 1) * (new_limit + 1) * (
            len(anchors) + 1
        ) > _MAX_TABLE_CELLS:
            return False
        nonlocal tables, counts, mass, tail
        tables = [model._suffix_table(new_limit, p) for p in anchors]
        counts = _support_table(model, new_limit, exact_budget + 1)
        mass, tail = model._weight_distributions_with_tail(ps, new_limit)
        for w in range(new_limit + 1):
            if w not in states and np.any(mass[:, w] > 0):
                states[w] = _Stratum(
                    w,
                    mass[:, w],
                    np.random.default_rng(
                        np.random.SeedSequence(entropy, spawn_key=(w,))
                    ),
                    np.zeros(len(ps)),
                    mass[:, w].copy(),
                    np.zeros(len(ps)),
                    np.zeros(len(ps)),
                )
        return True

    prepared = prepare(limit)
    if not prepared:
        mass, tail = np.zeros((len(ps), 0)), np.ones(len(ps))
        reason = "Conditional tables exceed the memory limit."

    def coefficients():
        result = {}
        for state in states.values():
            for key, value in state.coefficients.items():
                result[key] = float(np.logaddexp(result.get(key, -np.inf), value))
        return result

    def snapshot():
        poly = _polynomial(model, coefficients())
        values = poly(ps)
        lower = sum((s.lower for s in states.values()), np.zeros(len(ps)))
        upper = sum((s.upper for s in states.values()), tail.copy())
        lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        # Products can underflow even when every individual channel is supported.
        # A zero interval at positive p requires an actual exhaustive zero proof.
        numerical_zero = (ps > 0) & (values == 0) & (upper == 0)
        if np.any(numerical_zero):
            fully_checked = (
                counts is not None
                and counts.shape[1] > model.max_weight
                and all(
                    count == 0 or (w in states and states[w].exact)
                    for w, count in enumerate(counts[0])
                )
            )
            possible_failure = np.isfinite(
                _log_ratios(poly.active, poly.misses, ps, model)
            ).any(axis=0)
            numerical_zero &= ~np.asarray(fully_checked & ~possible_failure)
            upper[numerical_zero] = 1.0
        error = np.maximum(values - lower, upper - values)
        met = (error <= absolute_error + relative_error * lower) & ~numerical_zero
        return poly, values, lower, upper, error, met, numerical_zero

    while prepared:
        poly, values, lower, upper, error, met, numerical_zero = snapshot()
        if np.all(met):
            status, reason = (
                "accuracy_met",
                "Simultaneous requested-grid accuracy target met.",
            )
            break
        if np.any(numerical_zero):
            status, reason = (
                "numerical_limit",
                "Probability underflow prevents distinguishing an unverified zero from a positive LER.",
            )
            break
        if max_seconds is not None and perf_counter() - started >= max_seconds:
            reason = "Time budget exhausted before establishing accuracy."
            break
        # This choice affects efficiency only; coverage uses preassigned budgets.
        scale = np.maximum(
            absolute_error + relative_error * np.maximum(values, lower),
            np.maximum(upper, np.finfo(float).tiny) * 0.01,
        )
        priorities = {
            w: float(np.max((s.upper - s.lower) / scale))
            for w, s in states.items()
            if not s.exact
        }
        tail_priority = float(np.max(tail / scale))
        if limit < model.max_weight and tail_priority > max(
            priorities.values(), default=0.0
        ):
            limit = min(model.max_weight, max(limit + 1, 2 * limit))
            if not prepare(limit):
                reason = "Additional weight strata exceed the memory limit."
                break
            continue
        if not priorities:
            reason = "Floating-point resolution prevents establishing the requested tolerance."
            break
        state = states[max(priorities, key=priorities.get)]
        w = state.weight
        next_count = max(256, 2 * state.n)
        additional = next_count - state.n
        exact_count = int(counts[0, w])
        if 0 < exact_count <= exact_budget - exact_histories and (
            exact_count <= 4 * additional or shots + additional > max_shots
        ):
            state.coefficients = {}
            exact_mean = np.zeros(len(ps))
            for fail, active, misses, log_p0 in _enumerate_stratum(
                model, decoder, w, exact_count, counts
            ):
                keys, inverse, _ = _group_failures(fail, active, misses)
                logs = np.full(len(keys), -np.inf)
                np.logaddexp.at(logs, inverse, log_p0[fail])
                for key, log_value in zip(keys, logs):
                    key = tuple(key)
                    state.coefficients[key] = float(
                        np.logaddexp(state.coefficients.get(key, -np.inf), log_value)
                    )
                ratios = _log_ratios(active[fail], misses[fail], ps, model)
                exact_mean += np.exp(log_p0[fail, None] + ratios).sum(axis=0)
            state.exact = True
            state.lower = state.upper = exact_mean
            state.histogram.clear()
            exact_histories += exact_count
            continue
        if shots + additional > max_shots:
            reason = "Sample budget cannot reach the next valid confidence checkpoint."
            break
        if state.proposal_indices is None:
            indices = np.array(
                [
                    j
                    for j, table in enumerate(tables)
                    if table[0, w] >= model._conditional_mass_floor()
                ],
                dtype=int,
            )
            if not len(indices):
                status = "numerical_limit"
                reason = (
                    "All conditional proposal masses underflowed or have insufficient numerical precision for a required weight."
                )
                break
            state.proposal_indices = indices
            state.log_normalizers = np.array(
                [-math.log(len(indices)) - math.log(tables[j][0, w]) for j in indices]
            )
            bound = []
            for j, p in enumerate(ps):
                if state.mass[j] == 0:
                    bound.append(-np.inf)
                    continue
                at_target = [k for k in indices if anchors[k] == p]
                if at_target:
                    bound.append(
                        math.log(tables[at_target[0]][0, w]) + math.log(len(indices))
                    )
                else:
                    interior = [k for k in indices if 0 < anchors[k] < model.max_p]
                    closest = min(
                        interior, key=lambda k: abs(math.log(anchors[k]) - math.log(p))
                    )
                    maximum = _maximum_log_likelihood(
                        model, p, limit, anchors[closest]
                    )[w]
                    bound.append(
                        math.log(tables[closest][0, w])
                        + math.log(len(indices))
                        + maximum
                    )
            state.log_bound = np.asarray(bound)
        indices = state.proposal_indices
        # Random mixture membership makes each complete batch an IID mixture sample.
        for first in range(0, additional, 4096):
            batch = min(4096, additional - first)
            memberships = state.rng.integers(len(indices), size=batch)
            for slot, k in enumerate(indices):
                number = int(np.sum(memberships == slot))
                if not number:
                    continue
                bits, active, misses = model._sample_stratum(
                    w, number, state.rng, tables[k], plans[k]
                )
                fail = model._failures(
                    decoder, bits, model.num_detectors, model.num_observables
                )
                log_q = logsumexp(
                    _log_ratios(active, misses, anchors[indices], model)
                    + state.log_normalizers,
                    axis=1,
                )
                target_logs = _log_ratios(active, misses, ps, model) - log_q[:, None]
                normalized = np.zeros_like(target_logs)
                valid = np.isfinite(state.log_bound)
                normalized[:, valid] = np.exp(
                    np.minimum(0.0, target_logs[:, valid] - state.log_bound[valid])
                )
                normalized *= fail[:, None]
                mean = normalized.mean(axis=0)
                m2 = ((normalized - mean) ** 2).sum(axis=0)
                delta = mean - state.mean
                new_n = state.n + number
                state.m2 += m2 + delta**2 * state.n * number / new_n
                state.mean += delta * number / new_n
                state.n = new_n
                keys, _, freq = _group_failures(fail, active, misses)
                for key, freq1 in zip(keys, freq):
                    key = tuple(key)
                    state.histogram[key] = state.histogram.get(key, 0) + int(freq1)
        shots += additional
        # Sum_{w>=0}1/[(w+1)(w+2)] = Sum_{r>=0}1/[(r+1)(r+2)] = 1.
        log_delta = (
            math.log(4 * len(ps))
            - math.log1p(-confidence)
            + math.log(w + 1)
            + math.log(w + 2)
            + math.log(state.look + 1)
            + math.log(state.look + 2)
        )
        radius = np.sqrt(2 * state.m2 / (state.n - 1) * log_delta / state.n)
        radius += 7 * log_delta / (3 * (state.n - 1))

        def scale_bound(numbers, selected=state):
            result = np.zeros(len(ps))
            valid = (numbers > 0) & (selected.mass > 0)
            result[valid] = np.exp(
                np.minimum(
                    np.log(selected.mass[valid]),
                    selected.log_bound[valid] + np.log(numbers[valid]),
                )
            )
            return result

        state.lower = np.maximum(state.lower, scale_bound(state.mean - radius))
        state.upper = np.minimum(state.upper, scale_bound(state.mean + radius))
        state.look += 1
        keys = np.asarray(list(state.histogram), dtype=np.int64).reshape(
            -1, len(model.rates) + 1
        )
        denominator = logsumexp(
            _log_ratios(keys[:, 0], keys[:, 1:], anchors[indices], model)
            + state.log_normalizers,
            axis=1,
        )
        state.coefficients = {
            key: math.log(count) - math.log(state.n) - den
            for (key, count), den in zip(state.histogram.items(), denominator)
        }
        if np.any(state.lower > state.upper):
            status, reason = (
                "inconsistent_bounds",
                "Sequential confidence intervals became inconsistent.",
            )
            state.lower, state.upper = np.zeros(len(ps)), state.mass.copy()
            break
    poly, values, lower, upper, error, met, _ = snapshot()
    estimates = tuple(
        AccuracyEstimate(float(p), float(v), float(lo), float(hi), float(err), bool(ok))
        for p, v, lo, hi, err, ok in zip(ps, values, lower, upper, error, met)
    )
    provenance = {
        "status": status,
        "confidence": confidence,
        "relative_error": relative_error,
        "absolute_error": absolute_error,
        "certified_p_values": ps.tolist() if status == "accuracy_met" else [],
        "requested_p_values": ps.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "meaning": "Stopped polynomial estimate. Accuracy covers only the listed finite grid, not other p values.",
    }
    poly = _polynomial(model, coefficients(), {"accuracy_control": provenance})
    return AccuracyControlledProfile(
        status,
        reason,
        estimates,
        confidence,
        relative_error,
        absolute_error,
        shots,
        exact_histories,
        {w: s.n for w, s in states.items() if s.n},
        tuple(w for w, s in states.items() if s.exact),
        tuple(map(float, anchors)),
        perf_counter() - started,
        poly,
    )
