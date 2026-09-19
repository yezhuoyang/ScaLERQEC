"""Independent original-instruction replay of explicit Pauli fault histories.

This parser deliberately does not use LinearNoiseModel's factors, replacement
circuits, or fault responses. It covers the noise grammar used by the large
validation matrix and rejects other channels explicitly.
"""

import math
from dataclasses import dataclass

import numpy as np
import stim


@dataclass
class Event:
    probabilities_at_reference: list
    options: list
    weights: list
    paulis: list
    chain: bool = False

    def probabilities(self, p, p0):
        rates = np.asarray(self.probabilities_at_reference) * (p / p0)
        if self.chain:
            survival = 1.0
            values = []
            for q in rates:
                values.append(survival * q)
                survival *= 1 - q
            return np.array([survival] + values)
        return np.r_[1 - math.fsum(rates), rates]


def pauli_circuit(paulis):
    c = stim.Circuit()
    for q, a in paulis:
        c.append(a, [q])
    return c


def support(paulis):
    axes = {"X": 1, "Y": 3, "Z": 2}
    product = {}
    for q, a in paulis:
        product[q] = product.get(q, 0) ^ axes[a]
    return sum(bool(v) for v in product.values())


class ReplayOracle:
    def __init__(self, circuit, p0):
        self.reference_p = p0
        self.circuit = circuit
        self.pieces = []
        self.events = []
        chain = None
        for op in circuit.flattened():
            name = op.name
            ts = op.targets_copy()
            args = op.gate_args_copy()
            if name in {"E", "ELSE_CORRELATED_ERROR"}:
                slot = len(self.pieces)
                self.pieces.append(stim.Circuit())
                paulis = [
                    (t.value, "X" if t.is_x_target else "Y" if t.is_y_target else "Z")
                    for t in ts
                ]
                if name == "E":
                    chain = Event([], [{}], [0], [[]], True)
                    self.events.append(chain)
                if chain is None:
                    raise ValueError("Missing E before ELSE")
                chain.probabilities_at_reference.append(args[0])
                chain.options.append({slot: pauli_circuit(paulis)})
                chain.paulis.append(paulis)
                chain.weights.append(support(paulis))
                continue
            if name in {
                "DEPOLARIZE1",
                "DEPOLARIZE2",
                "PAULI_CHANNEL_1",
                "PAULI_CHANNEL_2",
                "X_ERROR",
                "Y_ERROR",
                "Z_ERROR",
            }:
                size = 2 if name in {"DEPOLARIZE2", "PAULI_CHANNEL_2"} else 1
                for first in range(0, len(ts), size):
                    qubits = [t.value for t in ts[first : first + size]]
                    slot = len(self.pieces)
                    self.pieces.append(stim.Circuit())
                    if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
                        words = [name[0]]
                        probs = args
                    else:
                        words = (
                            list("XYZ")
                            if size == 1
                            else [
                                a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"
                            ]
                        )
                        probs = (
                            [args[0] / len(words)] * len(words)
                            if name.startswith("DEPOLARIZE")
                            else args
                        )
                    paulis = [
                        [(q, a) for q, a in zip(qubits, s) if a != "I"] for s in words
                    ]
                    self.events.append(
                        Event(
                            list(probs),
                            [{}] + [{slot: pauli_circuit(x)} for x in paulis],
                            [0] + [support(x) for x in paulis],
                            [[]] + paulis,
                        )
                    )
                continue
            if name in {"M", "MX", "MY", "MR", "MRX", "MRY"} and args:
                for t in ts:
                    slot = len(self.pieces)
                    base = stim.Circuit()
                    base.append(name, [t])
                    self.pieces.append(base)
                    forced = stim.Circuit()
                    forced.append(name, [t], 1.0)
                    self.events.append(
                        Event(list(args), [{}, {slot: forced}], [0, 0], [[], []])
                    )
                continue
            if stim.gate_data(name).is_noisy_gate and args and any(args):
                raise NotImplementedError(
                    f"Independent replay grammar does not cover {name}"
                )
            piece = stim.Circuit()
            piece.append(op)
            self.pieces.append(piece)
        self.converter = circuit.compile_m2d_converter()

    def forced(self, outcomes):
        chosen = {}
        for event, a in zip(self.events, outcomes):
            if a:
                chosen.update(event.options[int(a)])
        c = stim.Circuit()
        for j, piece in enumerate(self.pieces):
            c += chosen.get(j, piece)
        return c

    def bits(self, outcomes):
        measurements = self.forced(outcomes).compile_sampler(seed=20260916).sample(2)
        values = self.converter.convert(
            measurements=measurements, append_observables=True
        )
        if not np.array_equal(values[0], values[1]):
            raise AssertionError("Nondeterministic forced oracle")
        return values[0]

    def log_probability(self, outcomes, p):
        return sum(
            math.log(event.probabilities(p, self.reference_p)[int(a)])
            for event, a in zip(self.events, outcomes)
        )


def audit_sampled_histories(model, oracle, *, seed=20260916):
    """Replay actual low- and typical-weight draws and their likelihood ratios."""
    if len(oracle.events) != len(model._factors):
        raise AssertionError("Noise location count differs")
    for event, factor in zip(oracle.events, model._factors):
        np.testing.assert_array_equal(event.weights, factor.weights)
        for p in [0.001, 0.003, 0.01]:
            np.testing.assert_allclose(
                event.probabilities(p, oracle.reference_p),
                factor.probabilities(p),
                atol=1e-15,
                rtol=1e-12,
            )
    mean = sum(
        np.dot(e.weights, e.probabilities(oracle.reference_p, oracle.reference_p))
        for e in oracle.events
    )
    weights = sorted(
        {min(2, model.max_weight), min(model.max_weight, max(1, round(mean)))}
    )
    table = model._suffix_table(max(weights))
    plans = None
    checked = 0
    checked_per_weight = {}
    largest_ratio_error = 0.0
    for w in weights:
        if not np.isfinite(table.logs[0, w]):
            continue
        histories = np.empty((4, len(oracle.events)), dtype=np.int64)
        bits, active, misses = model._sample_stratum(
            w,
            4,
            np.random.default_rng(seed + w),
            table,
            plans,
            outcome_buffer=histories,
        )
        for i, history in enumerate(histories):
            np.testing.assert_array_equal(bits[i], oracle.bits(history))
            actual_w = sum(e.weights[int(a)] for e, a in zip(oracle.events, history))
            if actual_w != w:
                raise AssertionError("Draw has wrong Pauli weight")
            baseline = oracle.log_probability(history, oracle.reference_p)
            for p in [0.001, 0.003]:
                exact = oracle.log_probability(history, p) - baseline
                recorded = active[i] * math.log(p / model.reference_p) + sum(
                    int(m) * math.log((1 - c * p) / (1 - c * model.reference_p))
                    for c, m in zip(model.rates, misses[i])
                )
                largest_ratio_error = max(largest_ratio_error, abs(exact - recorded))
                if not math.isclose(exact, recorded, abs_tol=2e-7, rel_tol=1e-9):
                    raise AssertionError("Incorrect history reweighting ratio")
            checked += 1
            checked_per_weight[w] = checked_per_weight.get(w, 0) + 1
    return {
        "status": "passed",
        "factors_checked": len(oracle.events),
        "histories_replayed": checked,
        "histories_per_weight": checked_per_weight,
        "weights": weights,
        "max_log_ratio_difference": largest_ratio_error,
    }


def hgp_all_columns(model, oracle, hx, hz, lz):
    """Independent CSS commutation checks every outcome, without Stim replay."""
    checks_x = np.vstack([hx, np.zeros_like(hz), np.zeros_like(lz)]).astype(bool)
    checks_z = np.vstack([np.zeros_like(hx), hz, lz]).astype(bool)
    count = 0
    for event, responses in zip(oracle.events, model._responses):
        for a, paulis in enumerate(event.paulis):
            if event.probabilities(model.reference_p, model.reference_p)[a] == 0:
                continue
            expected = np.zeros(checks_x.shape[0], dtype=bool)
            for q, axis in paulis:
                if axis in "XY":
                    expected ^= checks_z[:, q]
                if axis in "ZY":
                    expected ^= checks_x[:, q]
            np.testing.assert_array_equal(expected, responses[a])
            count += 1
    return count
