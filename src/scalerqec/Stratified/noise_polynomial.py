"""A reusable polynomial estimated from a general-noise weighted profile.

The factored basis avoids the cancellation of high-degree power coefficients.
All coefficients are fixed when the profile is created; no curve fitting occurs.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def _frozen(values, dtype):
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _nonnegative_integers(values, name):
    array = np.asarray(values)
    if (
        array.dtype.kind not in "iu"
        or np.any(array < 0)
        or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must contain nonnegative integers.")
    return array


def _precision(value, name, minimum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


class LERPolynomial:
    r"""Estimated polynomial in the positive factored likelihood basis.

    .. math::

        \widehat L(p) = \sum_s A_s (p/p_0)^{K_s}
          \prod_g [(1-c_g p)/(1-c_g p_0)]^{M_{sg}}.

    ``log_coefficients`` stores log(A_s), allowing tiny reference contributions
    to become measurable at other p without underflow. Repeated terms are
    combined exactly algebraically (up to floating-point summation).

    This object contains the estimated function, not its sampling uncertainty.
    Use the originating profile for SE and omitted-weight bounds. Increasing
    export precision does not remove Monte Carlo coefficient uncertainty.
    """

    def __init__(
        self,
        reference_p,
        rates,
        active,
        misses,
        log_coefficients,
        *,
        max_p,
        metadata=None,
    ):
        reference_p, max_p = float(reference_p), float(max_p)
        rates = np.asarray(rates, dtype=float)
        active = _nonnegative_integers(active, "active")
        misses = _nonnegative_integers(misses, "misses")
        logs = np.asarray(log_coefficients, dtype=float)
        if (
            not math.isfinite(reference_p)
            or reference_p <= 0
            or math.isnan(max_p)
            or reference_p >= max_p
            or rates.ndim != 1
            or not np.isfinite(rates).all()
            or np.any(rates < 0)
            or np.any(rates * reference_p >= 1)
        ):
            raise ValueError(
                "Invalid reference probability, domain, or rate coefficients."
            )
        n = active.size
        if (
            active.shape != (n,)
            or misses.shape != (n, len(rates))
            or logs.shape != (n,)
            or not np.isfinite(logs).all()
        ):
            raise ValueError(
                "Polynomial term arrays have incompatible shapes or nonfinite coefficients."
            )
        if len(rates) and rates.max() > 0 and max_p > 1 / rates.max():
            raise ValueError(
                "Polynomial domain exceeds the channel probability domain."
            )
        keys = np.column_stack([active, misses])
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        combined = np.full(len(unique), -np.inf)
        np.logaddexp.at(combined, inverse, logs)
        self.reference_p = reference_p
        self.max_p = max_p
        self.rates = _frozen(rates, np.float64)
        self.active = _frozen(unique[:, 0], np.int64)
        self.misses = _frozen(unique[:, 1:], np.int64)
        self.log_coefficients = _frozen(combined, np.float64)
        self._metadata = json.dumps(metadata or {}, allow_nan=False)

    @property
    def metadata(self):
        return json.loads(self._metadata)

    @property
    def num_terms(self):
        return len(self.active)

    @property
    def degree(self):
        """Degree upper bound; cancellations may reduce the actual degree."""
        return int(
            np.max(self.active + self.misses[:, self.rates > 0].sum(axis=1), initial=0)
        )

    def _probabilities(self, values):
        p = np.asarray(values, dtype=float)
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > self.max_p):
            raise ValueError(f"p must be finite and between 0 and {self.max_p}.")
        return p

    def __call__(self, probabilities):
        """Evaluate scalar or array p, with bounded temporary memory."""
        p = self._probabilities(probabilities)
        flat = p.ravel()
        result = np.zeros(len(flat))
        if self.num_terms:
            batch = max(1, 1_000_000 // self.num_terms)
            for start in range(0, len(flat), batch):
                values = flat[start : start + batch]
                logs = np.broadcast_to(
                    self.log_coefficients[:, None], (self.num_terms, len(values))
                ).copy()
                nonzero = values > 0
                logs[:, nonzero] += self.active[:, None] * (
                    np.log(values[nonzero]) - math.log(self.reference_p)
                )
                logs[np.ix_(self.active > 0, ~nonzero)] = -np.inf
                for g, rate in enumerate(self.rates):
                    inside = rate * values < 1
                    logs[:, inside] += self.misses[:, g, None] * (
                        np.log1p(-rate * values[inside])
                        - math.log1p(-rate * self.reference_p)
                    )
                    logs[np.ix_(self.misses[:, g] > 0, ~inside)] = -np.inf
                with np.errstate(over="raise"):
                    result[start : start + batch] = np.exp(logsumexp(logs, axis=0))
        result = result.reshape(p.shape)
        return float(result) if p.ndim == 0 else result

    def power_coefficients(self, *, precision=50, max_degree=100):
        """Return Decimal coefficients (constant first), with guarded expansion.

        Prefer the factored callable for numerical evaluation. Even accurately
        exported coefficients can catastrophically cancel in ordinary floating
        point for large circuits. The guard requires explicit opt-in to costly
        expansion; the factored polynomial has no such degree restriction.
        """
        precision = _precision(precision, "precision", 16)
        max_degree = _precision(max_degree, "max_degree", 0)
        if self.degree > max_degree:
            raise ValueError(
                f"Polynomial degree {self.degree} exceeds max_degree={max_degree}; use its factored form or explicitly increase the guard."
            )
        with localcontext() as context:
            context.prec = precision
            p0 = Decimal(str(self.reference_p))
            rates = [Decimal(str(rate)) for rate in self.rates]
            coefficients = [Decimal(0)] * (self.degree + 1)
            for k, ms, log_a in zip(self.active, self.misses, self.log_coefficients):
                k = int(k)
                amplitude = Decimal(str(log_a)).exp() / p0**k
                term = [Decimal(1)]
                for rate, m in zip(rates, ms):
                    m = int(m)
                    if not m or not rate:
                        continue
                    amplitude /= (1 - rate * p0) ** m
                    factor = [
                        Decimal(math.comb(m, j)) * (-rate) ** j for j in range(m + 1)
                    ]
                    product = [Decimal(0)] * (len(term) + m)
                    for i, a in enumerate(term):
                        for j, b in enumerate(factor):
                            product[i + j] += a * b
                    term = product
                for j, coefficient in enumerate(term):
                    coefficients[k + j] += amplitude * coefficient
            return tuple(+coefficient for coefficient in coefficients)

    def to_sympy(self, symbol="p", *, expanded=False, precision=30, max_degree=100):
        """Export a symbolic function of p, without fitting p-grid values."""
        import sympy as sp

        precision = _precision(precision, "precision", 16)
        p = sp.Symbol(symbol) if isinstance(symbol, str) else symbol
        if not isinstance(p, sp.Symbol):
            raise TypeError("symbol must be a name or a SymPy Symbol.")
        if expanded:
            coefficients = self.power_coefficients(
                precision=precision, max_degree=max_degree
            )
            return sp.Add(
                *(
                    sp.Float(str(c), precision) * p**j
                    for j, c in enumerate(coefficients)
                    if c
                )
            )
        p0 = sp.Float(str(self.reference_p), precision)
        terms = []
        for k, ms, log_a in zip(self.active, self.misses, self.log_coefficients):
            a = sp.exp(sp.Float(str(log_a), precision)).evalf(precision)
            factors = [a, (p / p0) ** int(k)]
            for rate, m in zip(self.rates, ms):
                if rate and m:
                    c = sp.Float(str(rate), precision)
                    factors.append(((1 - c * p) / (1 - c * p0)) ** int(m))
            terms.append(sp.Mul(*factors))
        return sp.Add(*terms)

    def save(self, path):
        manifest = json.dumps(
            {
                "format": "scalerqec.ler-polynomial",
                "version": 1,
                "reference_p": self.reference_p,
                "max_p": self.max_p if math.isfinite(self.max_p) else None,
                "metadata": self.metadata,
            },
            allow_nan=False,
        )
        with Path(path).open("wb") as file:
            np.savez_compressed(
                file,
                manifest=np.array(manifest),
                rates=self.rates,
                active=self.active,
                misses=self.misses,
                log_coefficients=self.log_coefficients,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"]))
            if (
                manifest.get("format") != "scalerqec.ler-polynomial"
                or manifest.get("version") != 1
            ):
                raise ValueError("Unsupported LER polynomial format.")
            return cls(
                manifest["reference_p"],
                data["rates"],
                data["active"],
                data["misses"],
                data["log_coefficients"],
                max_p=manifest["max_p"] if manifest["max_p"] is not None else math.inf,
                metadata=manifest["metadata"],
            )
