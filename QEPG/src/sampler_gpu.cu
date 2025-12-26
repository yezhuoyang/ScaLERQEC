#include "cuda_accel.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cstdint>

using std::size_t;

namespace cuda_accel {

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err__ = (expr); \
        if (err__ != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(err__) \
            ); \
        } \
    } while (0)

// --------------------- RNG (very simple xorshift64) ------------------------

__device__ inline uint64_t xorshift64(uint64_t& state) {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}


DeviceGraph make_device_graph(const QEPG::QEPG& graph) {
    DeviceGraph dg;

    // This is the 3 * total_noise_ rows: [0..N-1]=X, [N..2N-1]=Y, [2N..3N-1]=Z
    const auto& rows = graph.get_parityPropMatrixTrans();
    if (rows.empty()) {
        throw std::runtime_error("QEPG parity matrix is empty");
    }

    // IMPORTANT: num_noise is the number of rows in the transpose,
    // not graph.get_total_noise().
    dg.num_noise = rows.size();                  // <= FIX
    dg.num_det   = graph.get_total_detector();

    const std::size_t expected_bits = dg.num_det + 1;  // detectors + 1 observable bit

    if (rows.front().size() != expected_bits) {
        throw std::runtime_error(
            "Row bit length != num_det + 1 (detectors + obs)");
    }

    // Host buffers: one row per “effective noise term” (X/Y/Z channel)
    std::vector<unsigned char> h_noise_to_det(dg.num_noise * dg.num_det);
    std::vector<unsigned char> h_noise_to_obs(dg.num_noise);

    for (std::size_t i = 0; i < dg.num_noise; ++i) {
        const auto& bitrow = rows[i];
        if (bitrow.size() != expected_bits) {
            throw std::runtime_error("Inconsistent row bit length in parity matrix");
        }

        // detector bits
        for (std::size_t j = 0; j < dg.num_det; ++j) {
            h_noise_to_det[i * dg.num_det + j] =
                static_cast<unsigned char>(bitrow.test(j));
        }

        // last bit = observable
        h_noise_to_obs[i] =
            static_cast<unsigned char>(bitrow.test(dg.num_det));
    }

    // Device allocation + copy
    CUDA_CHECK(cudaMalloc(&dg.d_noise_to_det,
                          h_noise_to_det.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&dg.d_noise_to_obs,
                          h_noise_to_obs.size() * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(dg.d_noise_to_det,
                          h_noise_to_det.data(),
                          h_noise_to_det.size() * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.d_noise_to_obs,
                          h_noise_to_obs.data(),
                          h_noise_to_obs.size() * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    return dg;
}



void free_device_graph(DeviceGraph& dg) {
    if (dg.d_noise_to_det) {
        cudaFree(dg.d_noise_to_det);
        dg.d_noise_to_det = nullptr;
    }
    if (dg.d_noise_to_obs) {
        cudaFree(dg.d_noise_to_obs);
        dg.d_noise_to_obs = nullptr;
    }
    dg.num_noise = 0;
    dg.num_det   = 0;
}

// --------------------- Kernel: one thread per shot -------------------------

// NOTE: for a "minimal working" version we cap the weight per shot.
// You can lift this later by using dynamic allocations or different layout.
constexpr int MAX_WEIGHT_PER_SHOT = 64;

__global__ void sample_kernel(
    const unsigned char* __restrict__ d_noise_to_det,
    const unsigned char* __restrict__ d_noise_to_obs,
    std::size_t                       num_noise,
    std::size_t                       num_det,
    const std::size_t* __restrict__   d_per_shot_weight,
    std::size_t                       total_shots,
    uint64_t                          global_seed,
    unsigned char* __restrict__       d_det_out,
    unsigned char* __restrict__       d_obs_out)
{
    const std::size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_shots) return;

    std::size_t w = d_per_shot_weight[s];
    if (w > num_noise) {
        // invalid, but we just clamp to num_noise for safety
        w = num_noise;
    }
    if (w > MAX_WEIGHT_PER_SHOT) {
        // You can choose to assert or clamp. For now: clamp.
        w = MAX_WEIGHT_PER_SHOT;
    }

    // Thread-local RNG state
    uint64_t state = global_seed ^ (0x9E3779B97F4A7C15ull * (s + 1));

    // Floyd-style sampling of w distinct indices in [0, num_noise)
    int chosen[MAX_WEIGHT_PER_SHOT];
    int count = 0;
    while (count < static_cast<int>(w)) {
        uint64_t r = xorshift64(state);
        std::size_t candidate = static_cast<std::size_t>(r % num_noise);

        bool duplicate = false;
        for (int k = 0; k < count; ++k) {
            if (static_cast<std::size_t>(chosen[k]) == candidate) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            chosen[count++] = static_cast<int>(candidate);
        }
    }

    // Output row pointers
    unsigned char* det_row = d_det_out + s * num_det;
    unsigned char  obs_val = 0;

    // Initialize detector row to 0
    for (std::size_t j = 0; j < num_det; ++j) {
        det_row[j] = 0;
    }

    // Accumulate parities
    for (int k = 0; k < count; ++k) {
        const std::size_t idx = static_cast<std::size_t>(chosen[k]);
        const unsigned char* row_det =
            d_noise_to_det + idx * num_det;
        const unsigned char row_obs =
            d_noise_to_obs[idx];

        // XOR across detectors
        for (std::size_t j = 0; j < num_det; ++j) {
            det_row[j] ^= row_det[j];
        }
        // XOR observable
        obs_val ^= row_obs;
    }

    d_obs_out[s] = obs_val;
}

// --------------------- Host wrapper ----------------------------------------

void sample_many_weights_separate_obs(
    const DeviceGraph&                 dg,
    const std::vector<std::size_t>&    per_shot_weight,
    unsigned char*                     h_det_out,
    unsigned char*                     h_obs_out)
{
    const std::size_t total_shots = per_shot_weight.size();
    if (total_shots == 0) return;

    // Basic sanity
    if (!dg.d_noise_to_det || !dg.d_noise_to_obs) {
        throw std::runtime_error("DeviceGraph not initialized");
    }

    // Max weight check (for the simple fixed-size buffer in kernel)
    std::size_t max_w = 0;
    for (auto w : per_shot_weight) {
        if (w > max_w) max_w = w;
    }
    if (max_w > MAX_WEIGHT_PER_SHOT) {
        throw std::runtime_error(
            "per_shot_weight exceeds MAX_WEIGHT_PER_SHOT in CUDA sampler");
    }

    // Device buffer for weights
    std::size_t* d_per_shot_weight = nullptr;
    CUDA_CHECK(cudaMalloc(&d_per_shot_weight,
                          total_shots * sizeof(std::size_t)));
    CUDA_CHECK(cudaMemcpy(d_per_shot_weight,
                          per_shot_weight.data(),
                          total_shots * sizeof(std::size_t),
                          cudaMemcpyHostToDevice));

    // Device output buffers
    unsigned char* d_det_out = nullptr;
    unsigned char* d_obs_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_det_out,
                          total_shots * dg.num_det * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_obs_out,
                          total_shots * sizeof(unsigned char)));

    // Launch kernel
    const int block_size = 128;
    const int grid_size  =
        static_cast<int>((total_shots + block_size - 1) / block_size);

    const uint64_t global_seed = 0x1234567890ABCDEFull; // you can make this configurable

    sample_kernel<<<grid_size, block_size>>>(
        dg.d_noise_to_det,
        dg.d_noise_to_obs,
        dg.num_noise,
        dg.num_det,
        d_per_shot_weight,
        total_shots,
        global_seed,
        d_det_out,
        d_obs_out);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(
        h_det_out,
        d_det_out,
        total_shots * dg.num_det * sizeof(unsigned char),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_obs_out,
        d_obs_out,
        total_shots * sizeof(unsigned char),
        cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_per_shot_weight);
    cudaFree(d_det_out);
    cudaFree(d_obs_out);
}

} // namespace cuda_accel
