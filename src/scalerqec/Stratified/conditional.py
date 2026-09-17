"""Log-space conditional Pauli-weight sampling by skipping identity runs.

For suffix mass Z(j,w), the probability of no activation before k, conditional
on remaining weight w at j, is prod_{i=j}^{k-1} P_i(0) Z(k,w)/Z(j,w).
This survival function is monotone, so binary search locates the next fault.
Zero-weight nonidentity outcomes are activations too; they are never skipped.
"""

import numpy as np

from .confidence import _factor_logs


def _log_weight_step(factor, probabilities, suffix, output):
    limit = len(output) - 1
    for weight in np.unique(factor.weights):
        if weight <= limit:
            log_mass = np.logaddexp.reduce(probabilities[factor.weights == weight])
            np.logaddexp(
                output[weight:],
                suffix[: limit + 1 - weight] + log_mass,
                out=output[weight:],
            )


def log_weight_distribution(model, p, limit):
    """Only the root masses, using O(limit) workspace for detached profiles."""
    mass = np.full(limit + 1, -np.inf)
    mass[0] = 0
    for factor in reversed(model._factors):
        updated = np.full_like(mass, -np.inf)
        _log_weight_step(factor, _factor_logs(factor, p), mass, updated)
        mass = updated
    return mass


class ConditionalTable:
    def __init__(self, model, limit, p):
        factors = model._factors
        n = len(factors)
        self.logs = np.full((n + 1, limit + 1), -np.inf)
        self.logs[-1, 0] = 0
        width = max((len(f.weights) for f in factors), default=1)
        self.outcome_logs = np.full((n, width), -np.inf)
        self.weights = np.zeros((n, width), dtype=np.int64)
        self.offsets = np.zeros(n, dtype=np.int64)
        self.baseline_misses = np.zeros(len(model.rates), dtype=np.int64)
        self.rate_indices = []
        self.first_rate = np.zeros(n, dtype=np.int64)
        self.chain = np.array([f.chain for f in factors])
        offset = 0
        for j, f in enumerate(factors):
            self.offsets[j] = offset
            offset += len(f.weights)
            self.outcome_logs[j, : len(f.weights)] = _factor_logs(f, p)
            self.weights[j, : len(f.weights)] = f.weights
            indices = np.searchsorted(model.rates, f.rates)
            self.rate_indices.append(indices)
            self.first_rate[j] = indices[0]
            np.add.at(self.baseline_misses, indices, 1)
        for j in range(n - 1, -1, -1):
            f = factors[j]
            _log_weight_step(
                f,
                self.outcome_logs[j, : len(f.weights)],
                self.logs[j + 1],
                self.logs[j],
            )
        identity = self.outcome_logs[:, 0]
        self.identity_prefix = np.r_[
            0.0, np.cumsum(np.where(np.isfinite(identity), identity, 0.0))
        ]
        self.zero_prefix = np.r_[0, np.cumsum(~np.isfinite(identity))]
        zero_activations = np.any(
            (self.weights[:, 1:] == 0) & np.isfinite(self.outcome_logs[:, 1:]), axis=1
        )
        self.zero_activation_suffix = np.r_[np.cumsum(zero_activations[::-1])[::-1], 0]

    def __getitem__(self, index):
        # Compatibility for probability diagnostics; sampling uses logs directly.
        return np.exp(self.logs[index])

    def sample(self, model, weight, shots, rng, outcome_buffer=None):
        if not np.isfinite(self.logs[0, weight]):
            raise ValueError("Cannot sample an impossible Pauli-weight stratum.")
        if model._responses is None:
            model.compile_responses()
        bit_count = model.num_detectors + model.num_observables
        if not hasattr(model, "_packed_responses"):
            model._packed_responses = (
                np.concatenate(
                    [np.packbits(response, axis=1) for response in model._responses]
                )
                if model._responses
                else np.empty((0, (bit_count + 7) // 8), dtype=np.uint8)
            )
        bits = np.zeros((shots, (bit_count + 7) // 8), dtype=np.uint8)
        active = np.zeros(shots, dtype=np.int64)
        misses = np.broadcast_to(self.baseline_misses, (shots, len(model.rates))).copy()
        if outcome_buffer is not None:
            outcome_buffer.fill(0)
        rows = np.arange(shots)
        start = np.zeros(shots, dtype=np.int64)
        remaining = np.full(shots, weight, dtype=np.int64)
        n = len(model._factors)
        while len(rows):
            lo, hi = start.copy(), np.full(len(rows), n + 1, dtype=np.int64)
            # log1p(-U) is finite and includes the U=0 endpoint safely.
            threshold = np.log1p(-rng.random(len(rows)))
            denominator = self.logs[start, remaining]
            # With no remaining Pauli weight and no zero-weight activation,
            # the all-identity suffix is certain. Do not let cancellation in
            # logarithms turn this exact case into a spurious tiny fault chance.
            finished = (remaining == 0) & (self.zero_activation_suffix[start] == 0)
            lo[finished] = n
            while np.any(hi - lo > 1):
                mid = (hi + lo) // 2
                survival = (
                    self.identity_prefix[mid]
                    - self.identity_prefix[start]
                    + self.logs[mid, remaining]
                    - denominator
                )
                survival[self.zero_prefix[mid] != self.zero_prefix[start]] = -np.inf
                # >= skips zero-probability intervals when the RNG returns 0.
                before = survival >= threshold
                lo = np.where(before, mid, lo)
                hi = np.where(before, hi, mid)
            live = lo < n
            if np.any(remaining[~live]):
                raise FloatingPointError(
                    "Conditional survival search lost a required fault."
                )
            rows, remaining, location = rows[live], remaining[live], lo[live]
            if not len(rows):
                break
            residual = remaining[:, None] - self.weights[location, 1:]
            logs = (
                self.outcome_logs[location, 1:]
                + self.logs[location[:, None] + 1, np.maximum(residual, 0)]
            )
            logs[residual < 0] = -np.inf
            peak = logs.max(axis=1)
            if not np.isfinite(peak).all():
                raise FloatingPointError(
                    "Conditional activation has no supported outcome."
                )
            masses = np.exp(logs - peak[:, None])
            cdf = masses.cumsum(axis=1)
            draw = rng.random(len(rows)) * cdf[:, -1]
            outcome = (draw[:, None] >= cdf).sum(axis=1) + 1
            bits[rows] ^= model._packed_responses[self.offsets[location] + outcome]
            active[rows] += 1
            remaining -= self.weights[location, outcome]
            if outcome_buffer is not None:
                outcome_buffer[rows, location] = outcome
            # Vectorize categorical factors; only ELSE chains need branch counts.
            categorical = ~self.chain[location]
            misses[rows[categorical], self.first_rate[location[categorical]]] -= 1
            for j in np.unique(location[~categorical]):
                selected = location == j
                indices = self.rate_indices[j]
                for r, g in enumerate(indices):
                    affected = selected & (outcome <= r + 1)
                    misses[rows[affected], g] -= 1
            start = location + 1
        return np.unpackbits(bits, axis=1, count=bit_count).astype(bool), active, misses
