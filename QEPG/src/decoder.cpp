#include "decoder.hpp"

#include <stdexcept>
#include <limits>
#include <queue>
#include <algorithm>

namespace QEPG {

namespace {

/**
 * Internal representation of a single Pauli error generator as an edge
 * between (at most) two detectors or a detector and a virtual boundary.
 *
 * noise_index:
 *   row index in parityPropMatrixTranspose_ (0 .. 3*total_noise-1)
 *
 * u, v:
 *   detector indices in [0, num_det) or the virtual boundary index = num_det.
 */
struct NoiseEdge {
    std::size_t noise_index;
    int         u;            // detector index [0, num_det) or boundary = num_det
    int         v;            // same as u
    bool        flips_logical;
};

} // namespace

PMDecodedResult decode_perfect_matching(
    const QEPG& graph,
    const Row&  syndrome_det)
{
    const std::size_t num_det      = graph.get_total_detector();
    const std::size_t total_noise  = graph.get_total_noise();
    const auto&       parityT      = graph.get_parityPropMatrixTrans(); // rows: 3*total_noise, cols: num_det+1

    if (syndrome_det.size() != num_det) {
        throw std::invalid_argument(
            "decode_perfect_matching: syndrome_det has wrong length");
    }
    if (parityT.empty()) {
        throw std::runtime_error(
            "decode_perfect_matching: parity propagation matrix is empty");
    }

    const std::size_t num_rows      = parityT.size();         // expected 3 * total_noise
    const std::size_t expected_cols = num_det + 1;            // detectors + logical observable

    if (num_rows != 3 * total_noise) {
        // Not fatal, but indicates mismatch between QEPG and decoder assumptions.
        // For now we do not throw here; we just rely on column indexing below.
    }

    // ------------------------- build NoiseEdge list -------------------------
    //
    // Each row of parityPropMatrixTranspose_ encodes which detectors (0..num_det-1)
    // and whether the logical observable (column num_det) flip under the
    // corresponding Pauli error generator.
    //
    // We restrict to rows that flip 1 or 2 detectors:
    //   - 1 detector  => edge detector <-> virtual boundary
    //   - 2 detectors => edge detector <-> detector
    //
    // More-than-2 detector flips are ignored in this first implementation.
    const int boundary_node = static_cast<int>(num_det); // virtual node index

    std::vector<NoiseEdge> edges;
    edges.reserve(num_rows);

    for (std::size_t i = 0; i < num_rows; ++i) {
        const Row& row = parityT[i];
        if (row.size() < expected_cols) {
            throw std::runtime_error(
                "decode_perfect_matching: parity matrix row has fewer bits than expected");
        }

        // Collect all detectors flipped by this error.
        std::vector<int> dets;
        dets.reserve(4);
        for (std::size_t j = 0; j < num_det; ++j) {
            if (row.test(j)) {
                dets.push_back(static_cast<int>(j));
            }
        }
        const bool flips_logical = row.test(num_det);

        if (dets.empty()) {
            // Pure logical error; syndrome alone cannot fix it. Ignore here.
            continue;
        } else if (dets.size() == 1) {
            // Single detector: treat as edge to boundary node.
            edges.push_back(NoiseEdge{ i, dets[0], boundary_node, flips_logical });
        } else if (dets.size() == 2) {
            edges.push_back(NoiseEdge{ i, dets[0], dets[1], flips_logical });
        } else {
            // >2 detectors: outside simple matching graph model.
            // TODO: decompose into a path and create multiple edges.
            continue;
        }
    }

    // Build adjacency from detectors (and boundary) to incident NoiseEdges.
    std::vector<std::vector<std::size_t>> adjacency(num_det + 1); // +1 for boundary
    for (std::size_t e = 0; e < edges.size(); ++e) {
        const auto& ne = edges[e];
        if (ne.u >= 0 && ne.u <= boundary_node) {
            adjacency[static_cast<std::size_t>(ne.u)].push_back(e);
        }
        if (ne.v >= 0 && ne.v <= boundary_node && ne.v != ne.u) {
            adjacency[static_cast<std::size_t>(ne.v)].push_back(e);
        }
    }

    // ------------------------- greedy matching decoder ----------------------
    //
    // remaining_syndrome: detectors whose parity is still unexplained
    // chosen_errors: which parity rows (noise terms) are turned "on"
    Row remaining_syndrome = syndrome_det;
    Row chosen_errors(num_rows);
    chosen_errors.reset();
    bool logical_flip = false;

    // For each detector with an unsatisfied detection event, we try to pick a
    // single NoiseEdge incident on it that most "helpfully" reduces syndrome.
    //
    // Strategy:
    //   1. Prefer an edge (u,v) where v is another currently active detector;
    //      this cancels two detection events at once.
    //   2. If none exists, fall back to an edge (u, boundary).
    //   3. If neither exists, leave that detector unmatched (syndrome remains).
    for (std::size_t d = 0; d < num_det; ++d) {
        if (!remaining_syndrome.test(d)) {
            continue;
        }

        const int det = static_cast<int>(d);

        std::size_t best_edge_idx = static_cast<std::size_t>(-1);
        bool        best_edge_connects_two_defects = false;

        // Explore all edges incident on detector d.
        for (std::size_t e_idx : adjacency[det]) {
            const auto& ne    = edges[e_idx];
            const int   other = (ne.u == det) ? ne.v : ne.u;

            if (other == boundary_node) {
                // Candidate that cancels this one defect if no better option.
                if (!best_edge_connects_two_defects &&
                    best_edge_idx == static_cast<std::size_t>(-1)) {
                    best_edge_idx = e_idx;
                }
            } else {
                // Edge to another detector; best option if that detector is
                // also currently in the syndrome.
                if (other >= 0 &&
                    other < static_cast<int>(num_det) &&
                    remaining_syndrome.test(static_cast<std::size_t>(other))) {
                    best_edge_idx                   = e_idx;
                    best_edge_connects_two_defects = true;
                    break; // good enough for this simple heuristic
                }
            }
        }

        if (best_edge_idx == static_cast<std::size_t>(-1)) {
            // No usable edge; in a full decoder you might search multi-step paths
            // or backtrack. For this first implementation we leave this defect.
            continue;
        }

        const auto& ne = edges[best_edge_idx];

        // Toggle this error generator (row in parity matrix).
        const bool was_set = chosen_errors.test(ne.noise_index);
        chosen_errors.set(ne.noise_index, !was_set);

        // Toggle the detectors it affects.
        if (ne.u >= 0 && ne.u < static_cast<int>(num_det)) {
            remaining_syndrome.flip(static_cast<std::size_t>(ne.u));
        }
        if (ne.v >= 0 && ne.v < static_cast<int>(num_det)) {
            remaining_syndrome.flip(static_cast<std::size_t>(ne.v));
        }

        if (ne.flips_logical) {
            logical_flip = !logical_flip;
        }
    }

    // At this point remaining_syndrome may still be non-zero if the greedy
    // strategy could not resolve all defects using single edges. In regimes
    // where the QEPG graph matches the matching-graph assumptions, you can
    // add an assertion here if desired.

    PMDecodedResult result;
    result.error_bits   = std::move(chosen_errors);
    result.logical_flip = logical_flip;
    return result;
}

} // namespace QEPG
