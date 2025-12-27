#pragma once

#include "QEPG.hpp"
#include <vector>

namespace QEPG {

/**
 * Result of decoding a single-shot detector syndrome using a (greedy) perfect
 * matching–style decoder.
 *
 * error_bits has length 3 * total_noise and is indexed as:
 *   [0 .. total_noise-1]               : X-channel error at each noise location
 *   [total_noise .. 2*total_noise-1]   : Y-channel
 *   [2*total_noise .. 3*total_noise-1] : Z-channel
 *
 * logical_flip is the inferred value of the logical observable (1 = flip).
 */
struct PMDecodedResult {
    Row  error_bits;
    bool logical_flip;
};

/**
 * Decode a single detector syndrome using the QEPG parity propagation graph.
 *
 * @param graph         Compiled QEPG graph (parity propagation matrix built).
 * @param syndrome_det  Detector syndrome for one shot as a Row of length
 *                      graph.get_total_detector(). Bit j is 1 iff detector j
 *                      fired in this shot.
 *
 * @return PMDecodedResult containing the inferred error pattern over all
 *         Pauli error channels and the inferred logical observable value.
 *
 * Notes:
 *  - This first version uses a simple greedy matching strategy over the
 *    single-error edges extracted from the QEPG parity propagation matrix.
 *  - It assumes each Pauli error term flips at most two detectors, so that
 *    the parity-propagation graph is a matching graph. Rows that flip more
 *    than two detectors are currently ignored.
 *  - The code is structured so that you can later plug in a true
 *    minimum-weight perfect matching implementation (e.g., Blossom) without
 *    changing this public interface.
 */
PMDecodedResult decode_perfect_matching(
    const QEPG& graph,
    const Row&  syndrome_det);

} // namespace QEPG
