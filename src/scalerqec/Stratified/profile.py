"""Reusable failure spectra for uniform independent DEPOLARIZE1 noise.

A profile belongs to one circuit, set of fault locations, and fixed decoder.
Reweighting is exact for its stored spectrum; an extrapolated spectrum still
has unquantified systematic error. No sampling or fitting occurs here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binom


def _probabilities(values):
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return result


def _readonly(values, dtype):
    array = np.asarray(values, dtype=dtype)
    # Bytes backing prevents callers from re-enabling writes to the snapshot.
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class LERCurve:
    """LER and model-dependence diagnostics at each requested p.

    ``modeled_ler`` is the contribution from weights filled by the fitted
    model, and ``modeled_probability_mass`` is their binomial mass. Neither
    is an error bar or a bound on systematic error. Measured weights also
    have sampling uncertainty, including weights with zero failures.
    """

    p: np.ndarray
    ler: np.ndarray
    modeled_ler: np.ndarray
    modeled_probability_mass: np.ndarray


class LERProfile:
    """A detached, serializable conditional failure spectrum indexed by weight.

    Args:
        conditional_ler: N+1 failure probabilities, including weights 0 and N.
        modeled_weights: Boolean mask identifying fitted/extrapolated entries.
        sample_counts: Number of sampled trials at each weight, if available.
        failure_counts: Number of failures among those trials, if available.
        metadata: JSON-compatible provenance, including decoder reference p.

    Arrays are immutable copies. Metadata is copied on access. Arbitrary p in
    [0,1] can be evaluated, but this does not validate the model at that p or
    retune the decoder. Dense monomial coefficients are deliberately avoided;
    this binomial (Bernstein) representation is numerically better behaved.
    """

    FORMAT = "scalerqec.ler-profile"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        conditional_ler,
        *,
        modeled_weights=None,
        sample_counts=None,
        failure_counts=None,
        metadata=None,
    ):
        rates = _probabilities(conditional_ler)
        if rates.ndim != 1 or rates.size == 0:
            raise ValueError(
                "conditional_ler must be a nonempty one-dimensional spectrum."
            )
        self._conditional_ler = _readonly(rates, np.float64)
        size = rates.size
        mask = (
            np.zeros(size, dtype=bool)
            if modeled_weights is None
            else np.asarray(modeled_weights)
        )
        if mask.shape != rates.shape or mask.dtype.kind != "b":
            raise ValueError(
                "modeled_weights must be a boolean mask matching the spectrum."
            )
        self._modeled_weights = _readonly(mask, bool)

        def counts(values, name):
            raw = (
                np.zeros(size, dtype=np.int64) if values is None else np.asarray(values)
            )
            if (
                raw.shape != rates.shape
                or raw.dtype.kind not in "iu"
                or np.any(raw < 0)
                or np.any(raw > np.iinfo(np.int64).max)
            ):
                raise ValueError(
                    f"{name} must contain nonnegative int64 counts matching the spectrum."
                )
            return _readonly(raw, np.int64)

        self._sample_counts = counts(sample_counts, "sample_counts")
        self._failure_counts = counts(failure_counts, "failure_counts")
        if np.any(self._failure_counts > self._sample_counts):
            raise ValueError("failure_counts cannot exceed sample_counts.")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object.")
        self._metadata_json = json.dumps(metadata or {}, allow_nan=False)

    @property
    def num_noise(self):
        return self._conditional_ler.size - 1

    @property
    def conditional_ler(self):
        return self._conditional_ler

    @property
    def modeled_weights(self):
        return self._modeled_weights

    @property
    def sample_counts(self):
        return self._sample_counts

    @property
    def failure_counts(self):
        return self._failure_counts

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    def curve(self, p_values, *, max_working_elements=1_000_000):
        """Evaluate a grid with diagnostics and bounded temporary array sizes.

        All weights 0..N are included. No normal/Poisson approximation or tail
        cutoff is used, including at p=0, p=1, and very small LER. Complexity
        is O((N+1) * number of p values); memory is O(N + grid size + chunk).
        """
        p = _probabilities(p_values)
        if (
            isinstance(max_working_elements, bool)
            or not isinstance(max_working_elements, (int, np.integer))
            or max_working_elements < 1
        ):
            raise ValueError("max_working_elements must be a positive integer.")
        flat = p.ravel()
        result = np.zeros((flat.size, 3))
        # Tile both axes so even a very large circuit respects the chunk limit.
        weights_per_chunk = min(self.num_noise + 1, max_working_elements)
        p_per_chunk = max(1, max_working_elements // weights_per_chunk)
        for start in range(0, flat.size, p_per_chunk):
            stop = min(start + p_per_chunk, flat.size)
            for low in range(0, self.num_noise + 1, weights_per_chunk):
                high = min(low + weights_per_chunk, self.num_noise + 1)
                weights = np.arange(low, high)
                mass = binom.pmf(
                    weights[None, :], self.num_noise, flat[start:stop, None]
                )
                rates = self._conditional_ler[low:high]
                mask = self._modeled_weights[low:high]
                result[start:stop, 0] += mass @ rates
                result[start:stop, 1] += mass @ (rates * mask)
                result[start:stop, 2] += mass @ mask.astype(float)
        return LERCurve(
            _readonly(p, np.float64),
            *(_readonly(result[:, i].reshape(p.shape), np.float64) for i in range(3)),
        )

    def evaluate(self, p_values, *, max_working_elements=1_000_000):
        """Return a float for scalar p, or an array with the shape of p_values."""
        result = self.curve(p_values, max_working_elements=max_working_elements).ler
        return float(result) if result.ndim == 0 else result

    def to_polynomial(self):
        """Export the existing SID weighted spectrum as a polynomial in p.

        This preserves all fitted/extrapolated entries and their model bias.
        It does not reinterpret a SID spectrum as a general-noise profile.
        The Bernstein sum is represented in the common positive factored basis.
        """
        from .noise_polynomial import LERPolynomial

        weights = np.flatnonzero(self._conditional_ler > 0)
        logs = binom.logpmf(weights, self.num_noise, 0.5) + np.log(
            self._conditional_ler[weights]
        )
        return LERPolynomial(
            0.5,
            [1.0],
            weights,
            (self.num_noise - weights)[:, None],
            logs,
            max_p=1.0,
            metadata={
                "noise_model": "uniform independent DEPOLARIZE1",
                "profile_metadata": self.metadata,
                "modeled_weights": np.flatnonzero(self._modeled_weights).tolist(),
                "meaning": "Polynomial of the stored SID spectrum; fitting/extrapolation uncertainty is unchanged.",
            },
        )

    def save(self, path):
        """Write a portable JSON snapshot; no pickle or executable objects."""
        data = {
            "format": self.FORMAT,
            "schema_version": self.SCHEMA_VERSION,
            "num_noise": self.num_noise,
            "conditional_ler": self._conditional_ler.tolist(),
            "modeled_weights": self._modeled_weights.tolist(),
            "sample_counts": self._sample_counts.tolist(),
            "failure_counts": self._failure_counts.tolist(),
            "metadata": self.metadata,
        }
        Path(path).write_text(
            json.dumps(data, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        """Load and validate a snapshot, rejecting incompatible schemas."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "format",
            "schema_version",
            "num_noise",
            "conditional_ler",
            "modeled_weights",
            "sample_counts",
            "failure_counts",
            "metadata",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Invalid LER profile fields.")
        if (
            data["format"] != cls.FORMAT
            or type(data["schema_version"]) is not int
            or data["schema_version"] != cls.SCHEMA_VERSION
        ):
            raise ValueError("Unsupported LER profile format or schema version.")
        profile = cls(
            data["conditional_ler"],
            modeled_weights=data["modeled_weights"],
            sample_counts=data["sample_counts"],
            failure_counts=data["failure_counts"],
            metadata=data["metadata"],
        )
        if type(data["num_noise"]) is not int or data["num_noise"] != profile.num_noise:
            raise ValueError("num_noise does not match the spectrum length.")
        return profile

    def plot(self, p_values, *, ax=None, **plot_kwargs):
        """Plot LER versus p and return a Matplotlib axes (does not call show)."""
        import matplotlib.pyplot as plt

        p = _probabilities(p_values)
        if p.ndim != 1 or p.size == 0:
            raise ValueError("Plotting requires a nonempty one-dimensional p grid.")
        rates = self.evaluate(p)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(p, rates, **plot_kwargs)
        if np.all(p > 0):
            ax.set_xscale("log")
        if np.all(rates > 0):
            ax.set_yscale("log")
        ax.set_xlabel("Physical error probability p (uniform SID)")
        ax.set_ylabel("Logical error rate (fixed decoder)")
        return ax
