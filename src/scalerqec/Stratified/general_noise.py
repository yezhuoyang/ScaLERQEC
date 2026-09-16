"""Experimental exact Pauli-weight stratification for noise proportional to p.

See ``docs/general_noise_math.md`` for the estimator and its assumptions. Stim
interprets the circuit; no detector-error-model approximation is used to sample
noise. This module intentionally does not change Scaler's uniform-SID contract.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

_MEASUREMENTS = {
    "M",
    "MX",
    "MY",
    "MR",
    "MRX",
    "MRY",
    "MXX",
    "MYY",
    "MZZ",
    "MPP",
    "MPAD",
}
_SINGLE = {
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "PAULI_CHANNEL_1",
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
}
_DOUBLE = {"DEPOLARIZE2", "PAULI_CHANNEL_2"}
_PAULIS_2 = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]


def _integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _instruction(name, targets, args=(), tag=""):
    return stim.CircuitInstruction(name, targets, args, tag=tag)


def _circuit(instruction=None):
    result = stim.Circuit()
    if instruction is not None:
        result.append(instruction)
    return result


def _pauli_targets(pauli, qubits):
    factories = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
    return [factories[a](q) for a, q in zip(pauli, qubits) if a != "I"]


def _pauli_weight(targets):
    # Repeated factors at one location multiply, disregarding global phase.
    support = {}
    for t in targets:
        axis = 1 if t.is_x_target else 3 if t.is_y_target else 2
        support[t.value] = support.get(t.value, 0) ^ axis
    return sum(v != 0 for v in support.values())


@dataclass
class _Factor:
    rates: tuple[float, ...]
    fractions: np.ndarray
    weights: np.ndarray
    replacements: list[list[tuple[int, stim.Circuit]]]
    chain: bool = False

    def probabilities(self, p):
        if not self.chain:
            return np.r_[1 - self.rates[0] * p, self.fractions * (self.rates[0] * p)]
        survival = 1.0
        probs = [0.0]
        for rate in self.rates:
            probs.append(survival * rate * p)
            survival *= 1 - rate * p
        probs[0] = survival
        return np.asarray(probs)


@dataclass(frozen=True)
class GeneralNoiseEstimate:
    """Partial LER, sampling SE, and a separate bound on omitted strata.

    ``minimum_ess`` is a likelihood-overlap diagnostic, not a guarantee that
    rare failures were observed. Standard errors are not confidence bounds.
    """

    p: float
    ler: float
    standard_error: float
    missing_probability_mass: float
    minimum_ess: float
    observed_failures: int


class LinearNoiseModel:
    """A Stim circuit whose noise arguments at reference_p define c * p.

    All noise probabilities, including measurement flips and conditional ELSE
    probabilities, scale together. Zero arguments stay zero. Clifford syntax
    is interpreted by Stim, including repeats and record-controlled Paulis.
    Intrinsically nondeterministic ideal detectors/observables are rejected.

    Weight is inserted Pauli support. Classical measurement flips and heralded
    identity outcomes have weight zero. Noise locations are never commuted or
    merged. Sweep bits have their default zero values.
    """

    def __init__(self, circuit, reference_p, *, compile_responses=True):
        self.reference_p = float(reference_p)
        if not math.isfinite(self.reference_p) or self.reference_p <= 0:
            raise ValueError("reference_p must be finite and positive.")
        self.circuit = stim.Circuit(str(circuit)).flattened()
        self._pieces = []
        self._factors = []
        self._argument_rates = []
        chain = None
        for op in self.circuit:
            name, targets, args = op.name, op.targets_copy(), op.gate_args_copy()
            tag = op.tag
            is_noise = stim.gate_data(name).is_noisy_gate or name == "MPAD"
            if is_noise:
                self._argument_rates.extend(a / self.reference_p for a in args)
            if name in {"E", "ELSE_CORRELATED_ERROR"}:
                slot = self._add_piece(stim.Circuit())
                replacement = [(slot, _circuit(_instruction("E", targets, [1], tag)))]
                rate = args[0] / self.reference_p
                if name == "E":
                    chain = _Factor(
                        (rate,),
                        np.empty(0),
                        np.array([0, _pauli_weight(targets)]),
                        [[], replacement],
                        True,
                    )
                    self._factors.append(chain)
                else:
                    if chain is None:
                        raise ValueError("ELSE_CORRELATED_ERROR has no preceding E.")
                    chain.rates += (rate,)
                    chain.weights = np.append(chain.weights, _pauli_weight(targets))
                    chain.replacements.append(replacement)
            elif name in _SINGLE | _DOUBLE:
                size = 2 if name in _DOUBLE else 1
                for start in range(0, len(targets), size):
                    ts = targets[start : start + size]
                    qubits = [t.value for t in ts]
                    heralded = name.startswith("HERALDED")
                    base = (
                        _circuit(_instruction("MPAD", [0]))
                        if heralded
                        else stim.Circuit()
                    )
                    slot = self._add_piece(base)
                    if name.endswith("_ERROR"):
                        paulis, ps = [name[0]], args
                    elif name == "DEPOLARIZE1":
                        paulis, ps = list("XYZ"), [args[0] / 3] * 3
                    elif name == "DEPOLARIZE2":
                        paulis, ps = _PAULIS_2, [args[0] / 15] * 15
                    elif name == "PAULI_CHANNEL_1":
                        paulis, ps = list("XYZ"), args
                    elif name == "PAULI_CHANNEL_2":
                        paulis, ps = _PAULIS_2, args
                    elif name == "HERALDED_ERASE":
                        paulis, ps = list("IXYZ"), [args[0] / 4] * 4
                    else:
                        paulis, ps = list("IXYZ"), args
                    replacements = [[]]
                    for pauli in paulis:
                        if heralded:
                            forced = [int(pauli == a) for a in "IXYZ"]
                            op1 = _instruction(
                                "HERALDED_PAULI_CHANNEL_1", ts, forced, tag
                            )
                        else:
                            op1 = _instruction(
                                "E", _pauli_targets(pauli, qubits), [1], tag
                            )
                        replacements.append([(slot, _circuit(op1))])
                    self._add_categorical(
                        ps, [sum(a != "I" for a in s) for s in paulis], replacements
                    )
            elif name in _MEASUREMENTS:
                for ts in self._measurement_groups(name, targets):
                    slot = self._add_piece(_circuit(_instruction(name, ts, (), tag)))
                    if args:
                        forced = _circuit(_instruction(name, ts, [1], tag))
                        self._add_categorical(args, [0], [[], [(slot, forced)]])
            elif name in {"I_ERROR", "II_ERROR"}:
                # Stochastic identity annotations have no observable effect;
                # marginalize them exactly, but retain their parameter domain.
                self._add_piece(stim.Circuit())
            elif is_noise:
                raise NotImplementedError(f"Unsupported Stim noise instruction: {name}")
            else:
                self._add_piece(_circuit(op))

        all_rates = [r for factor in self._factors for r in factor.rates]
        self.rates = np.unique(all_rates)
        max_rate = max(self._argument_rates + all_rates + [0.0])
        self.max_p = 1 / max_rate if max_rate else math.inf
        if self.reference_p >= self.max_p:
            raise ValueError(
                "reference_p must be strictly inside the noise family's probability domain."
            )
        self.max_weight = sum(int(f.weights.max()) for f in self._factors)
        self.ideal_circuit = self._forced_circuit({})
        # Only used to validate noiseless determinism; never sampled as noise.
        self.ideal_circuit.detector_error_model()
        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables
        self._responses = None
        if compile_responses:
            self.compile_responses()

    @classmethod
    def from_stabcode(cls, code, reference_p, **kwargs):
        """Use the existing StabIR compiler and its attached noise model."""
        if code.stimcirc is None:
            code.construct_circuit()
        if code.stimcirc is None:
            raise ValueError("StabCode did not produce a Stim circuit.")
        return cls(code.stimcirc, reference_p, **kwargs)

    def _add_piece(self, piece):
        self._pieces.append(piece)
        return len(self._pieces) - 1

    def _add_categorical(self, probs, weights, replacements):
        total = math.fsum(probs)
        fractions = np.asarray(probs) / total if total else np.zeros(len(probs))
        self._factors.append(
            _Factor(
                (total / self.reference_p,),
                fractions,
                np.array([0] + weights),
                replacements,
            )
        )

    @staticmethod
    def _measurement_groups(name, targets):
        if name == "MPP":
            groups = []
            for target in targets:
                if target.is_combiner or (groups and groups[-1][-1].is_combiner):
                    groups[-1].append(target)
                else:
                    groups.append([target])
            return groups
        size = 2 if name in {"MXX", "MYY", "MZZ"} else 1
        return [targets[i : i + size] for i in range(0, len(targets), size)]

    def _validate_p(self, p):
        p = float(p)
        if not math.isfinite(p) or p < 0 or p > self.max_p:
            raise ValueError(f"p must be finite and between 0 and {self.max_p}.")
        return p

    def circuit_at(self, p):
        """Actual Stim circuit at p, for independent Monte Carlo comparisons."""
        p = self._validate_p(p)
        result = stim.Circuit()
        for op in self.circuit:
            args = op.gate_args_copy()
            if stim.gate_data(op.name).is_noisy_gate or op.name == "MPAD":
                args = [min(1.0, a / self.reference_p * p) for a in args]
            result.append(_instruction(op.name, op.targets_copy(), args, op.tag))
        return result

    def _forced_circuit(self, outcomes):
        replacements = {}
        for index, outcome in outcomes.items():
            replacements.update(self._factors[index].replacements[outcome])
        result = stim.Circuit()
        for i, piece in enumerate(self._pieces):
            result += replacements.get(i, piece)
        return result

    def compile_responses(self):
        """Compile exact forced-fault responses using Stim's Clifford simulator."""
        converter = self.ideal_circuit.compile_m2d_converter()
        responses = []
        width = self.num_detectors + self.num_observables
        for j, factor in enumerate(self._factors):
            columns = np.zeros((len(factor.weights), width), dtype=np.bool_)
            for a in range(1, len(factor.weights)):
                if factor.probabilities(self.reference_p)[a] == 0:
                    continue
                forced = self._forced_circuit({j: a})
                measurements = forced.compile_sampler(seed=0).sample(2)
                bits = converter.convert(
                    measurements=measurements, append_observables=True
                )
                if not np.array_equal(bits[0], bits[1]):
                    raise ValueError(
                        "Fault response is not deterministic for this circuit."
                    )
                columns[a] = bits[0]
            responses.append(columns)
        self._responses = responses

    def weight_distribution(self, p, *, max_weight=None):
        """Exact generating-function coefficients; an optional prefix saves work."""
        p = self._validate_p(p)
        limit = (
            self.max_weight
            if max_weight is None
            else min(_integer(max_weight, "max_weight"), self.max_weight)
        )
        return self._weight_distribution_with_tail(p, limit)[0]

    def _weight_distribution_with_tail(self, p, limit):
        # Accumulate overflow positively. 1-sum(prefix) loses tiny tails to
        # cancellation and can misleadingly report a zero truncation bound.
        mass = np.zeros(limit + 1)
        mass[0] = 1
        tail = 0.0
        for factor in self._factors:
            polynomial = np.bincount(factor.weights, weights=factor.probabilities(p))
            expanded = np.convolve(mass, polynomial)
            tail += float(expanded[limit + 1 :].sum())
            mass = expanded[: limit + 1]
        return mass, tail

    def _suffix_table(self, limit):
        table = np.zeros((len(self._factors) + 1, limit + 1))
        table[-1, 0] = 1
        for j in range(len(self._factors) - 1, -1, -1):
            f = self._factors[j]
            polynomial = np.bincount(
                f.weights, weights=f.probabilities(self.reference_p)
            )
            table[j] = np.convolve(table[j + 1], polynomial)[: limit + 1]
        return table

    def _sample_stratum(self, weight, shots, rng, table):
        if self._responses is None:
            self.compile_responses()
        remaining = np.full(shots, weight, dtype=np.int64)
        bits = np.zeros(
            (shots, self.num_detectors + self.num_observables), dtype=np.bool_
        )
        active = np.zeros(shots, dtype=np.int64)
        misses = np.zeros((shots, len(self.rates)), dtype=np.int64)
        for j, factor in enumerate(self._factors):
            probs = factor.probabilities(self.reference_p)
            # Group equal-weight outcomes before the DP draw. Choice inside a
            # weight group is categorical and independent of the suffix.
            unique_weights = np.unique(factor.weights)
            group_probs = np.array(
                [probs[factor.weights == v].sum() for v in unique_weights]
            )
            residual = remaining[:, None] - unique_weights
            masses = group_probs * table[j + 1, np.maximum(residual, 0)]
            masses[residual < 0] = 0
            total = masses.sum(axis=1)
            if np.any(total <= 0):
                raise FloatingPointError("Conditional weight probability underflowed.")
            cumulative = np.cumsum(masses, axis=1)
            group = np.sum(
                rng.random(shots)[:, None] * total[:, None] >= cumulative, axis=1
            )
            group = np.minimum(group, len(unique_weights) - 1)
            outcomes = np.zeros(shots, dtype=np.int64)
            for g, value in enumerate(unique_weights):
                selected = np.flatnonzero(group == g)
                if not len(selected):
                    continue
                options = np.flatnonzero((factor.weights == value) & (probs > 0))
                option_probs = probs[options] / probs[options].sum()
                outcomes[selected] = rng.choice(options, len(selected), p=option_probs)
            remaining -= factor.weights[outcomes]
            bits ^= self._responses[j][outcomes]
            active += outcomes != 0
            for r, rate in enumerate(factor.rates):
                g = np.searchsorted(self.rates, rate)
                missed = (
                    ((outcomes == 0) | (outcomes > r + 1))
                    if factor.chain
                    else outcomes == 0
                )
                misses[:, g] += missed
        if remaining.any():
            raise RuntimeError(
                "Conditional sampler did not achieve its requested Pauli weight."
            )
        return bits, active, misses

    @staticmethod
    def _failures(decoder, bits, num_detectors, num_observables):
        det = bits[:, :num_detectors]
        predicted = (
            decoder.decode_batch(det)
            if hasattr(decoder, "decode_batch")
            else decoder(det)
        )
        predicted = np.asarray(predicted)
        if predicted.ndim == 1 and num_observables == 1:
            predicted = predicted[:, None]
        if (
            predicted.shape != (len(bits), num_observables)
            or not np.isin(predicted, [0, 1]).all()
        ):
            raise ValueError(
                "Decoder must return one binary prediction per shot and observable."
            )
        return np.any(predicted != bits[:, num_detectors:], axis=1)

    def sample_profile(
        self,
        decoder,
        *,
        shots_per_weight,
        max_weight=None,
        seed=None,
        batch_size=4096,
        metadata=None,
    ):
        """Sample once with fixed allocation, then evaluate many p values.

        The same decoder must be used in subsequent comparison experiments.
        Unsampled high weights are bounded, not extrapolated. Each reachable
        stratum requires >=2 trials to estimate its sampling variance.
        """
        shots = _integer(shots_per_weight, "shots_per_weight", 2)
        batch_size = _integer(batch_size, "batch_size", 1)
        limit = (
            self.max_weight
            if max_weight is None
            else min(_integer(max_weight, "max_weight"), self.max_weight)
        )
        # Avoid an accidental unbounded allocation for expanded large circuits.
        if (len(self._factors) + 1) * (limit + 1) > 50_000_000:
            raise ValueError(
                "Conditional sampler table is too large; specify a smaller max_weight."
            )
        table = self._suffix_table(limit)
        rng = np.random.default_rng(seed)
        records = []
        for w, mass in enumerate(table[0]):
            if mass == 0:
                continue
            for start in range(0, shots, batch_size):
                count = min(batch_size, shots - start)
                bits, active, misses = self._sample_stratum(w, count, rng, table)
                failures = self._failures(
                    decoder, bits, self.num_detectors, self.num_observables
                )
                records.append((np.full(count, w), failures, active, misses))
        if not records:
            raise FloatingPointError("All selected stratum probabilities underflowed.")
        weights, failures, active, misses = (
            np.concatenate([row[k] for row in records]) for k in range(4)
        )
        return GeneralNoiseProfile(
            self, weights, failures, active, misses, metadata=metadata
        )

    def enumerate_histories(self, decoder, p, *, max_histories=1_000_000):
        """Exact small-circuit oracle, including q_w(p) and the full LER."""
        p = self._validate_p(p)
        count = math.prod(len(f.weights) for f in self._factors)
        if count > _integer(max_histories, "max_histories", 1):
            raise ValueError(f"Enumeration needs {count} histories.")
        if self._responses is None:
            self.compile_responses()
        mass = np.zeros(self.max_weight + 1)
        failure_mass = mass.copy()
        probabilities = [f.probabilities(p) for f in self._factors]
        for history in itertools.product(
            *(range(len(f.weights)) for f in self._factors)
        ):
            probability = math.prod(
                probs[a] for probs, a in zip(probabilities, history)
            )
            if not probability:
                continue
            bits = np.zeros(
                (1, self.num_detectors + self.num_observables), dtype=np.bool_
            )
            weight = 0
            for j, a in enumerate(history):
                weight += int(self._factors[j].weights[a])
                bits[0] ^= self._responses[j][a]
            mass[weight] += probability
            if self._failures(decoder, bits, self.num_detectors, self.num_observables)[
                0
            ]:
                failure_mass[weight] += probability
        return {
            "ler": float(failure_mass.sum()),
            "weight_mass": mass,
            "failure_mass": failure_mass,
            "conditional_ler": np.divide(
                failure_mass, mass, out=np.zeros_like(mass), where=mass > 0
            ),
        }


class GeneralNoiseProfile:
    """Detached sufficient history statistics for one circuit/noise/decoder.

    Storage and reweighting are experimental (format version 1). No decoder is
    pickled. Loaded profiles evaluate without compiling responses or resampling.
    """

    def __init__(self, model, weights, failures, active, misses, *, metadata=None):
        # Detach from the caller's compiled model. No response compilation or
        # decoding occurs when taking this compact snapshot.
        self.model = LinearNoiseModel(
            model.circuit, model.reference_p, compile_responses=False
        )
        for values in (weights, active, misses):
            array = np.asarray(values)
            if (
                array.dtype.kind not in "iu"
                or np.any(array < 0)
                or np.any(array > np.iinfo(np.int64).max)
            ):
                raise ValueError("History counts must be nonnegative integers.")
        if not np.isin(failures, [0, 1]).all():
            raise ValueError("Failure indicators must be binary.")
        self.weights = np.asarray(weights, dtype=np.int64).copy()
        self.failures = np.asarray(failures, dtype=np.bool_).copy()
        self.active = np.asarray(active, dtype=np.int64).copy()
        self.misses = np.asarray(misses, dtype=np.int64).copy()
        n = len(self.weights)
        if (
            self.weights.shape != (n,)
            or self.failures.shape != (n,)
            or self.active.shape != (n,)
            or self.misses.shape != (n, len(model.rates))
        ):
            raise ValueError("Profile arrays have incompatible shapes.")
        if (
            not n
            or np.any(self.weights < 0)
            or np.any(self.weights > model.max_weight)
            or np.any(self.active < 0)
            or np.any(self.active > len(model._factors))
            or np.any(self.misses < 0)
        ):
            raise ValueError("Invalid profile statistics.")
        maximum_misses = np.zeros(len(model.rates), dtype=np.int64)
        for factor in model._factors:
            for rate in factor.rates:
                maximum_misses[np.searchsorted(model.rates, rate)] += 1
        if np.any(self.misses > maximum_misses):
            raise ValueError("History miss counts exceed the number of noise branches.")
        self.sampled_weights, self.counts = np.unique(self.weights, return_counts=True)
        if np.any(self.counts < 2):
            raise ValueError("Every sampled stratum needs at least two trials.")
        self._reference_mass = model.weight_distribution(
            model.reference_p, max_weight=int(self.weights.max())
        )
        if np.any(self._reference_mass[self.sampled_weights] <= 0):
            raise ValueError("Sampled an impossible or underflowed stratum.")
        self.metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        for array in (
            self.weights,
            self.failures,
            self.active,
            self.misses,
            self.sampled_weights,
            self.counts,
        ):
            array.flags.writeable = False

    def _log_likelihood(self, p):
        p0 = self.model.reference_p
        log_ratio = np.zeros(len(self.weights))
        if p == 0:
            log_ratio[self.active > 0] = -np.inf
        else:
            log_ratio += self.active * (math.log(p) - math.log(p0))
        for g, rate in enumerate(self.model.rates):
            if rate * p >= 1:
                log_ratio[self.misses[:, g] > 0] = -np.inf
            else:
                log_ratio += self.misses[:, g] * (
                    math.log1p(-rate * p) - math.log1p(-rate * p0)
                )
        return log_ratio

    def _likelihood(self, p):
        with np.errstate(over="raise"):
            return np.exp(self._log_likelihood(p))

    def evaluate(self, p):
        p = self.model._validate_p(p)
        log_ratio = self._log_likelihood(p)
        total = standard_error = 0.0
        minimum_ess = math.inf
        target_mass, tail = self.model._weight_distribution_with_tail(
            p, int(self.weights.max())
        )
        for w, n in zip(self.sampled_weights, self.counts):
            selected = self.weights == w
            logs = log_ratio[selected]
            peak = logs.max()
            # Combine the stratum probability with the likelihood in log
            # space BEFORE exponentiation. R itself can overflow even when
            # Z_w(p0)*R is small. Scaled moments also avoid squaring huge R.
            scaled = np.exp(logs - peak) if np.isfinite(peak) else np.zeros(n)
            scale = math.exp(math.log(self._reference_mass[w]) + peak)
            values = scaled * self.failures[selected]
            total += scale * values.mean()
            standard_error = math.hypot(
                standard_error, scale * values.std(ddof=1) / math.sqrt(n)
            )
            if target_mass[w] > 0:
                ess = (
                    float(scaled.sum() ** 2 / (scaled @ scaled))
                    if np.any(scaled)
                    else 0.0
                )
                minimum_ess = min(minimum_ess, ess)
        missing_weights = np.ones(len(target_mass), dtype=bool)
        missing_weights[self.sampled_weights] = False
        missing = float(np.clip(tail + target_mass[missing_weights].sum(), 0, 1))
        return GeneralNoiseEstimate(
            p,
            float(total),
            standard_error,
            missing,
            0.0 if math.isinf(minimum_ess) else minimum_ess,
            int(self.failures.sum()),
        )

    def curve(self, probabilities):
        return [self.evaluate(p) for p in probabilities]

    def save(self, path):
        manifest = json.dumps(
            {
                "format": "scalerqec.general-noise-profile",
                "version": 1,
                "circuit": str(self.model.circuit),
                "reference_p": self.model.reference_p,
                "metadata": self.metadata,
            },
            allow_nan=False,
        )
        with Path(path).open("wb") as file:
            np.savez_compressed(
                file,
                manifest=np.array(manifest),
                weights=self.weights,
                failures=self.failures,
                active=self.active,
                misses=self.misses,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"]))
            if (
                manifest.get("format") != "scalerqec.general-noise-profile"
                or manifest.get("version") != 1
            ):
                raise ValueError("Unsupported general-noise profile format.")
            model = LinearNoiseModel(
                manifest["circuit"], manifest["reference_p"], compile_responses=False
            )
            return cls(
                model,
                data["weights"],
                data["failures"],
                data["active"],
                data["misses"],
                metadata=manifest["metadata"],
            )
