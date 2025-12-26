#pragma once

#include <cstddef>
#include <vector>
#include "QEPG.hpp"

namespace cuda_accel {

struct DeviceGraph {
    std::size_t num_noise   = 0;   // N
    std::size_t num_det     = 0;   // D
    // Device buffers:
    //   noise_to_det[i * num_det + j] = 0/1: does noise i flip detector j?
    //   noise_to_obs[i]               = 0/1: does noise i flip the observable?
    unsigned char* d_noise_to_det = nullptr;
    unsigned char* d_noise_to_obs = nullptr;
};

// Build device representation from QEPG graph
DeviceGraph make_device_graph(const QEPG::QEPG& graph);

// Free device buffers (safe to call on default-constructed / already-freed)
void free_device_graph(DeviceGraph& dg);

// Main CUDA sampler:
// per_shot_weight.size() == total_shots
// Writes:
//   h_det_out: size = total_shots * num_det (row-major: [shot, det])
//   h_obs_out: size = total_shots
void sample_many_weights_separate_obs(
    const DeviceGraph&                 dg,
    const std::vector<std::size_t>&    per_shot_weight,
    unsigned char*                     h_det_out,
    unsigned char*                     h_obs_out);

} // namespace cuda_accel
