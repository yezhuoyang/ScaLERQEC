"""Quantum Error Propagation Graph (QEPG) construction in pure Python.

This module implements backward propagation of Pauli errors through a
Clifford circuit to determine how each single-qubit error at every noise
source affects the circuit's detectors and logical observable. The result
is an error propagation matrix that maps noise configurations to
detector/observable outcome vectors.
"""

import numpy as np

from .clifford import CliffordCircuit, Measurement, Reset, pauliNoise


class QEPGpython:
    """Quantum Error Propagation Graph built via backward Pauli propagation.

    Given a :class:`CliffordCircuit`, this class constructs a binary
    propagation matrix that encodes, for every noise source and every Pauli
    error type (X, Y, Z), which detectors and/or the logical observable are
    flipped.

    The propagation matrix ``_propMatrix`` has shape
    ``(3 * num_noise, num_detectors + 1)`` and is organized in three
    row blocks:

    - Rows ``[0, num_noise)``: X-error propagation for each noise source.
    - Rows ``[num_noise, 2*num_noise)``: Y-error propagation.
    - Rows ``[2*num_noise, 3*num_noise)``: Z-error propagation.

    The last column of each row indicates whether the corresponding error
    flips the logical observable.

    Args:
        circuit: A fully constructed :class:`CliffordCircuit` with gates,
            noise sources, measurements, detectors, and observable defined.
    """

    def __init__(self, circuit: CliffordCircuit):
        self._circuit = circuit
        self._total_meas = self._circuit._totalMeas
        self._total_noise = self._circuit._totalnoise
        self._propMatrix = np.zeros(
            (3 * self._total_noise, len(self._circuit.parityMatchGroup) + 1),
            dtype="uint8",
        )

    def backword_graph_construction(self) -> None:
        """Build the error propagation matrix by reverse-traversing the circuit.

        Processes the gate list from last to first. Three per-qubit tracking
        arrays (``current_x_prop``, ``current_y_prop``, ``current_z_prop``)
        record how a Pauli X, Y, or Z error on each qubit would propagate
        to detectors and the observable at the current point in the reverse
        traversal.

        Gate-type rules applied during backward propagation:

        - **Measurement**: An X or Y error on the measured qubit flips the
          detectors that include this measurement (and the observable if
          applicable). Z errors commute with Z-basis measurement and have
          no effect.
        - **Reset**: Clears all propagation on the reset qubit, since the
          qubit state is discarded.
        - **CNOT**: X propagates forward from control to target; Z propagates
          backward from target to control. Y propagation combines both
          effects.
        - **Hadamard**: Swaps X and Z propagation on the gate's qubit.
        - **Noise source**: Snapshots the current propagation state for this
          qubit into the corresponding rows of ``_propMatrix``.

        After this method completes, ``_propMatrix`` is fully populated and
        ready for querying via :meth:`sample_x_error`, :meth:`sample_y_error`,
        :meth:`sample_z_error`, or :meth:`sample_noise_vector`.
        """
        nqubit = self._circuit._qubit_num

        column_size = len(self._circuit.parityMatchGroup) + 1
        current_x_prop = np.zeros((nqubit, column_size), dtype="uint8")
        current_y_prop = np.zeros((nqubit, column_size), dtype="uint8")
        current_z_prop = np.zeros((nqubit, column_size), dtype="uint8")

        current_noise_index = self._circuit._totalnoise - 1
        current_meas_index = self._total_meas - 1

        total_noise = self._total_noise
        T = len(self._circuit._gatelists)

        for t in range(T - 1, -1, -1):
            # Update current_x_prop, current_y_prop, current_z_prop based on the current gate and measurement
            gate = self._circuit._gatelists[t]
            """
            If the gate is a oiginal noise, add edges to the graph based on current propogation
            """
            if isinstance(gate, pauliNoise):
                noiseindex = current_noise_index
                # print("Noise!")
                self._propMatrix[noiseindex, :] = current_x_prop[
                    gate._qubitindex, :
                ].copy()
                self._propMatrix[total_noise + noiseindex, :] = current_y_prop[
                    gate._qubitindex, :
                ].copy()
                self._propMatrix[total_noise * 2 + noiseindex, :] = current_z_prop[
                    gate._qubitindex, :
                ].copy()
                current_noise_index -= 1
                continue
            """
            When there is a measurement, update the current propogation based on the measurement
            We just need to consider the propagation of X and Y because only 
            the X and Y error can be detected by the measurement
            """
            if isinstance(gate, Measurement):
                measureindex = current_meas_index

                qindex = gate._qubitindex
                if measureindex in self._circuit.observable:
                    current_x_prop[qindex][column_size - 1] ^= 1

                for parityIdx in self._circuit.get_measIdx_to_parityIdx(measureindex):
                    current_x_prop[qindex][parityIdx] ^= 1

                current_y_prop[qindex] = current_x_prop[qindex].copy()
                current_z_prop[qindex] = 0

                current_meas_index -= 1
                continue

            if isinstance(gate, Reset):
                current_x_prop[gate._qubitindex, :] = 0
                current_y_prop[gate._qubitindex, :] = 0
                current_z_prop[gate._qubitindex, :] = 0
                continue

            # CNOT backward propagation: X spreads from target→control, Z from control→target.
            # Y_control uses target.X (updated above) and Y_target uses control.Z (updated above).
            # This order is correct because Y = iXZ, so Y_ctrl depends on the NEW X of target
            # and Y_tgt depends on the NEW Z of control (post-CNOT Heisenberg picture).
            if gate._name == "CNOT":
                control = gate._control
                target = gate._target
                current_x_prop[control, :] = (
                    current_x_prop[control, :] + current_x_prop[target, :]
                ) % 2
                current_z_prop[target, :] = (
                    current_z_prop[control, :] + current_z_prop[target, :]
                ) % 2
                current_y_prop[control, :] = (
                    current_y_prop[control, :] + current_x_prop[target, :]
                ) % 2
                current_y_prop[target, :] = (
                    current_y_prop[target, :] + current_z_prop[control, :]
                ) % 2
                continue

            """
            Deal with propagation by H gate
            If there is a H gate, we need to swap the X and Z propagations
            """
            if gate._name == "H":
                qubitindex = gate._qubitindex
                tmp_row = current_x_prop[qubitindex, :].copy()
                current_x_prop[qubitindex, :] = current_z_prop[qubitindex, :]
                current_z_prop[qubitindex, :] = tmp_row
                continue

            """
            Phase (S) gate: S†XS = Y, S†YS = -X, S†ZS = Z
            In GF(2): swap X and Y propagation, Z unchanged.
            """
            if gate._name == "P":
                qubitindex = gate._qubitindex
                tmp_row = current_x_prop[qubitindex, :].copy()
                current_x_prop[qubitindex, :] = current_y_prop[qubitindex, :]
                current_y_prop[qubitindex, :] = tmp_row
                continue

            """
            Pauli X/Y/Z: identity in GF(2) (signs vanish).
            """
            if gate._name in ("X", "Y", "Z"):
                continue

    def sample_x_error(self, noise_index: int) -> list[int]:
        """Get the detector/observable flip vector for an X error at a noise source.

        Args:
            noise_index: The noise source index (0-based).

        Returns:
            A list of length ``num_detectors + 1`` with binary entries.
            A ``1`` at position *i* means detector *i* is flipped; a ``1``
            in the last position means the observable is flipped.
        """
        return list(self._propMatrix[noise_index, :])

    def sample_y_error(self, noise_index: int) -> list[int]:
        """Get the detector/observable flip vector for a Y error at a noise source.

        Args:
            noise_index: The noise source index (0-based).

        Returns:
            A list of length ``num_detectors + 1`` with binary entries.
        """
        return list(self._propMatrix[self._total_noise + noise_index, :])

    def sample_z_error(self, noise_index: int) -> list[int]:
        """Get the detector/observable flip vector for a Z error at a noise source.

        Args:
            noise_index: The noise source index (0-based).

        Returns:
            A list of length ``num_detectors + 1`` with binary entries.
        """
        return list(self._propMatrix[2 * self._total_noise + noise_index, :])

    def sample_nonuniform_batch(
        self,
        noise_probs: np.ndarray,
        num_shots: int,
        correlated_pairs: list | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample detector/observable outcomes with per-source error probabilities.

        Delegates to the C++ QEPG backend for SIMD-accelerated, OpenMP-parallel
        non-uniform sampling. Falls back to a Python implementation if the C++
        graph is not available.

        Args:
            noise_probs: Array of shape ``(num_noise, 3)`` with columns
                ``[px, py, pz]`` giving per-source error probabilities.
            num_shots: Number of Monte Carlo shots to generate.
            correlated_pairs: Optional list of ``CorrelatedNoisePair``
                objects for DEPOLARIZE2 events.
            rng: Optional numpy random Generator (only used in Python fallback).

        Returns:
            A tuple ``(detector_outcomes, observable_outcomes)`` where
            ``detector_outcomes`` has shape ``(num_shots, num_detectors)``
            and ``observable_outcomes`` has shape ``(num_shots,)``, both
            with dtype ``uint8``.
        """
        N = self._total_noise
        assert noise_probs.shape == (N, 3), (
            f"noise_probs shape {noise_probs.shape} != expected ({N}, 3)"
        )

        # Use C++ backend if available
        if hasattr(self, "_cpp_graph") and self._cpp_graph is not None:
            from scalerqec import qepg as qepg_cpp

            probs = np.ascontiguousarray(noise_probs, dtype=np.float64)

            if correlated_pairs:
                corr_a = np.array(
                    [p.source_a for p in correlated_pairs], dtype=np.uint64
                )
                corr_b = np.array(
                    [p.source_b for p in correlated_pairs], dtype=np.uint64
                )
                corr_p = np.array([p.prob for p in correlated_pairs], dtype=np.float64)
            else:
                corr_a = np.empty(0, dtype=np.uint64)
                corr_b = np.empty(0, dtype=np.uint64)
                corr_p = np.empty(0, dtype=np.float64)

            det, obs = qepg_cpp.return_samples_nonuniform_to_numpy(
                self._cpp_graph, probs, corr_a, corr_b, corr_p, num_shots
            )
            return det, obs

        # Python fallback
        return self._sample_nonuniform_batch_python(
            noise_probs, num_shots, correlated_pairs, rng
        )

    def _sample_nonuniform_batch_python(
        self,
        noise_probs: np.ndarray,
        num_shots: int,
        correlated_pairs: list | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Python fallback for non-uniform sampling (used when C++ is unavailable)."""
        if rng is None:
            rng = np.random.default_rng()

        N = self._total_noise
        column_size = self._propMatrix.shape[1]

        cum = np.cumsum(noise_probs, axis=1)
        rand = rng.random((num_shots, N))

        x_fired = rand < cum[np.newaxis, :, 0]
        y_fired = (rand >= cum[np.newaxis, :, 0]) & (rand < cum[np.newaxis, :, 1])
        z_fired = (rand >= cum[np.newaxis, :, 1]) & (rand < cum[np.newaxis, :, 2])

        result = np.zeros((num_shots, column_size), dtype=np.uint8)

        for fired, offset in [(x_fired, 0), (y_fired, N), (z_fired, 2 * N)]:
            shot_indices, source_indices = np.nonzero(fired)
            if len(shot_indices) == 0:
                continue
            for src in np.unique(source_indices):
                mask = source_indices == src
                shots_for_src = shot_indices[mask]
                result[shots_for_src] ^= self._propMatrix[offset + src]

        if correlated_pairs:
            from ..Monte.noise_model_parser import TWO_QUBIT_PAULIS

            for pair in correlated_pairs:
                fire = rng.random(num_shots) < pair.prob
                num_fired = np.count_nonzero(fire)
                if num_fired == 0:
                    continue

                pauli_idx = rng.integers(0, 15, size=num_fired)
                full_pauli = np.full(num_shots, -1, dtype=np.int32)
                full_pauli[fire] = pauli_idx

                for pidx, (pa, pb) in enumerate(TWO_QUBIT_PAULIS):
                    mask = full_pauli == pidx
                    if not np.any(mask):
                        continue
                    shots_mask = np.nonzero(mask)[0]
                    if pa > 0:
                        result[shots_mask] ^= self._propMatrix[
                            (pa - 1) * N + pair.source_a
                        ]
                    if pb > 0:
                        result[shots_mask] ^= self._propMatrix[
                            (pb - 1) * N + pair.source_b
                        ]

        detector_outcomes = result[:, :-1]
        observable_outcomes = result[:, -1]

        return detector_outcomes, observable_outcomes

    def sample_noise_vector(self, noise_vector: np.ndarray) -> np.ndarray:
        """Compute the detector/observable outcome for an arbitrary error pattern.

        Multiplies a binary noise vector (indicating which X/Y/Z errors are
        active at which noise sources) by the propagation matrix to obtain
        the combined detector and observable flip pattern.

        Args:
            noise_vector: A binary array of length ``3 * num_noise``. The
                first ``num_noise`` entries correspond to X errors, the next
                to Y errors, and the last to Z errors, matching the row
                layout of ``_propMatrix``.

        Returns:
            An array of length ``num_detectors + 1`` giving the XOR-combined
            detector flips and observable flip.

        Raises:
            AssertionError: If ``noise_vector`` length does not equal
                ``3 * num_noise``.
        """
        assert len(noise_vector) == 3 * self._total_noise
        return noise_vector @ self._propMatrix


if __name__ == "__main__":
    stim_str = ""
    filepath = "C:/Users/username/Documents/Sampling/stimprograms/1cnot"
    with open(filepath, "r", encoding="utf-8") as f:
        stim_str = f.read()
    circuit = CliffordCircuit(3)
    circuit.set_error_rate(0.01)
    circuit.compile_from_stim_circuit_str(stim_str)

    graph = QEPGpython(circuit)
    graph.backword_graph_construction()

    print(graph.sample_noise_vector(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
