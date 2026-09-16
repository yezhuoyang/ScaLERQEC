/**
 * @file sampler.cpp
 * @brief Implementation of random Pauli error sampling methods.
 *
 * Contains the sampling algorithms (Floyd, removal, Monte Carlo) for
 * generating random error configurations, and the batch methods that
 * use OpenMP to parallelize sample generation across threads.
 */

#include "sampler.hpp"
#include "chrono"
#include <atomic>
#include <bit>
#include <thread>
#include <cmath>
#include <numbers>
#include <algorithm>


namespace SAMPLE{

/// Convert uint64 to uniform double in [0, 1) with 53-bit precision.
static inline double to_double_01(std::uint64_t v) noexcept {
    return (v >> 11) * (1.0 / (std::uint64_t{1} << 53));
}

/// Sample from Poisson(lambda) using Knuth's algorithm for small lambda,
/// normal approximation for large lambda.
static inline std::size_t sample_poisson(Xoshiro256pp& rng, double lambda) noexcept {
    if (lambda < 30.0) {
        // Knuth's algorithm: exact for small lambda, O(lambda) per sample
        double L = std::exp(-lambda);
        std::size_t k = 0;
        double p = 1.0;
        do {
            ++k;
            p *= to_double_01(rng());
        } while (p > L);
        return k - 1;
    } else {
        // Normal approximation: N(lambda, sqrt(lambda)) via Box-Muller
        double u1 = to_double_01(rng());
        double u2 = to_double_01(rng());
        if (u1 < 1e-300) u1 = 1e-300;  // avoid log(0)
        constexpr double two_pi = 2.0 * std::numbers::pi;
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
        double x = lambda + std::sqrt(lambda) * z;
        return x > 0.0 ? static_cast<std::size_t>(x + 0.5) : 0;
    }
}

/*---------------------------------------ctor----------*/
sampler::sampler()=default;

sampler::sampler(size_t num_total_paulierror)
    : num_total_pauliError_(num_total_paulierror),
      collision_bitmap_(num_total_paulierror, 0),
      scratch_sample_() {
    scratch_sample_.reserve(num_total_paulierror);
};

sampler::~sampler()=default;

/*---------------------------------------Sample one vector with fixed weight----------*/


/**
 * @brief Generate a single error sample using removal-based sampling.
 *
 * Creates a set of all noise location indices, then randomly removes
 * positions until exactly `weight` remain. Each remaining position
 * receives a uniformly random Pauli type (X, Y, or Z).
 *
 * @note More efficient than Floyd's insertion method when weight is
 *       close to num_total_pauliError_ (avoids collision overhead).
 */
inline std::size_t sampler::generate_sample_removal(size_t weight, std::mt19937& gen){
    // Mark all positions as active
    std::fill(collision_bitmap_.begin(), collision_bitmap_.begin() + num_total_pauliError_, 1);
    size_t remaining = num_total_pauliError_;

    // Remove random positions until only `weight` remain
    std::uniform_int_distribution<size_t> posdistrib(0, num_total_pauliError_-1);
    while(remaining > weight){
        size_t pos = posdistrib(gen);
        if(collision_bitmap_[pos]){
            collision_bitmap_[pos] = 0;
            --remaining;
        }
    }

    // Collect remaining positions into scratch buffer
    scratch_sample_.clear();
    std::uniform_int_distribution<size_t> typedistrib(1, 3);
    for(size_t i = 0; i < num_total_pauliError_; ++i){
        if(collision_bitmap_[i]){
            scratch_sample_.push_back(singlePauli{i, typedistrib(gen)});
        }
    }
    // Clear bitmap for positions we used
    std::memset(collision_bitmap_.data(), 0, num_total_pauliError_);
    return scratch_sample_.size();
}

/**
 * @brief Generate a single error sample using Floyd's insertion algorithm.
 *
 * Selects `weight` distinct noise locations uniformly at random. Uses
 * a hash set to reject collisions. Automatically delegates to
 * generate_sample_removal() when weight > num_total_pauliError_ / 2
 * to avoid excessive collisions.
 *
 * Each selected position receives a uniformly random Pauli type (1=X, 2=Y, 3=Z).
 */
inline std::size_t sampler::generate_sample_Floyd(size_t weight, std::mt19937& gen){
    // Hybrid strategy: use removal when weight > half of total
    if(weight > num_total_pauliError_ / 2){
        return generate_sample_removal(weight, gen);
    }

    // Addition strategy with bitmap collision detection (no heap allocation)
    scratch_sample_.clear();

    std::uniform_int_distribution<size_t> posdistrib(0, num_total_pauliError_-1);
    std::uniform_int_distribution<size_t> typedistrib(1, 3);

    while(scratch_sample_.size() < weight){
        size_t newpos = posdistrib(gen);
        if(!collision_bitmap_[newpos]){
            collision_bitmap_[newpos] = 1;
            scratch_sample_.push_back(singlePauli{newpos, typedistrib(gen)});
        }
    }
    // Clear only the used positions (O(weight) not O(N))
    for(std::size_t i = 0; i < scratch_sample_.size(); ++i){
        collision_bitmap_[scratch_sample_[i].qindex] = 0;
    }
    return scratch_sample_.size();
}


/**
 * @brief Generate a single error sample using Bernoulli (Monte Carlo) sampling.
 *
 * Iterates over the first ErrorSize noise locations. Each location independently
 * has an error with probability error_prob. Activated locations receive a
 * uniformly random Pauli type.
 */
inline std::size_t sampler::generate_sample_Monte(double error_prob,size_t ErrorSize,std::mt19937& gen){
    scratch_sample_.clear();
    std::uniform_int_distribution<size_t> typedistrib(1, 3);
    std::bernoulli_distribution dist(error_prob);
    for(size_t pos=0;pos<ErrorSize;++pos){
        if(dist(gen)){
            scratch_sample_.push_back(singlePauli{pos, typedistrib(gen)});
        }
    }
    return scratch_sample_.size();
}


/*----------- Xoshiro256++ overloads -----------*/

inline std::size_t sampler::generate_sample_removal(size_t weight, Xoshiro256pp& gen){
    std::memset(collision_bitmap_.data(), 1, num_total_pauliError_);
    size_t remaining = num_total_pauliError_;

    while(remaining > weight){
        size_t pos = gen.bounded(num_total_pauliError_);
        if(collision_bitmap_[pos]){
            collision_bitmap_[pos] = 0;
            --remaining;
        }
    }

    scratch_sample_.clear();
    for(size_t i = 0; i < num_total_pauliError_; ++i){
        if(collision_bitmap_[i]){
            scratch_sample_.push_back(singlePauli{i, 1 + gen.bounded(3)});
        }
    }
    std::memset(collision_bitmap_.data(), 0, num_total_pauliError_);
    return scratch_sample_.size();
}

inline std::size_t sampler::generate_sample_Floyd(size_t weight, Xoshiro256pp& gen){
    if(weight > num_total_pauliError_ / 2){
        return generate_sample_removal(weight, gen);
    }

    scratch_sample_.clear();

    while(scratch_sample_.size() < weight){
        size_t newpos = gen.bounded(num_total_pauliError_);
        if(!collision_bitmap_[newpos]){
            collision_bitmap_[newpos] = 1;
            scratch_sample_.push_back(singlePauli{newpos, 1 + gen.bounded(3)});
        }
    }
    for(std::size_t i = 0; i < scratch_sample_.size(); ++i){
        collision_bitmap_[scratch_sample_[i].qindex] = 0;
    }
    return scratch_sample_.size();
}

inline std::size_t sampler::generate_sample_Monte(double error_prob, size_t ErrorSize, Xoshiro256pp& gen){
    scratch_sample_.clear();
    // Use threshold on upper 32 bits for speed (sufficient precision for error rates)
    const double threshold = error_prob;
    for(size_t pos = 0; pos < ErrorSize; ++pos){
        // Use upper 32 bits of random value for threshold test
        if(to_double_01(gen()) < threshold){
            scratch_sample_.push_back(singlePauli{pos, 1 + gen.bounded(3)});
        }
    }
    return scratch_sample_.size();
}


inline std::size_t sampler::generate_sample_Monte_sparse(double error_prob, size_t ErrorSize, Xoshiro256pp& gen){
    // Sparse O(k) algorithm: sample k ~ Binomial(N,p), then pick k locations via Floyd's.
    // For small p this is dramatically faster than the O(N) dense Bernoulli scan.
    scratch_sample_.clear();

    const std::size_t k = std::binomial_distribution<std::size_t>(ErrorSize, error_prob)(gen);

    if(k == 0) return 0;
    if(k >= ErrorSize) {
        for(size_t pos = 0; pos < ErrorSize; ++pos)
            scratch_sample_.push_back(singlePauli{pos, 1 + gen.bounded(3)});
        return scratch_sample_.size();
    }

    if(k <= ErrorSize / 2){
        while(scratch_sample_.size() < k){
            size_t newpos = gen.bounded(ErrorSize);
            if(!collision_bitmap_[newpos]){
                collision_bitmap_[newpos] = 1;
                scratch_sample_.push_back(singlePauli{newpos, 1 + gen.bounded(3)});
            }
        }
        for(std::size_t i = 0; i < scratch_sample_.size(); ++i)
            collision_bitmap_[scratch_sample_[i].qindex] = 0;
    } else {
        std::memset(collision_bitmap_.data(), 1, ErrorSize);
        size_t remaining = ErrorSize;
        while(remaining > k){
            size_t pos = gen.bounded(ErrorSize);
            if(collision_bitmap_[pos]){
                collision_bitmap_[pos] = 0;
                --remaining;
            }
        }
        for(size_t i = 0; i < ErrorSize; ++i){
            if(collision_bitmap_[i])
                scratch_sample_.push_back(singlePauli{i, 1 + gen.bounded(3)});
        }
        std::memset(collision_bitmap_.data(), 0, ErrorSize);
    }

    return scratch_sample_.size();
}




/**
 * @brief Generate many samples in parallel with fixed Pauli weight.
 *
 * Uses OpenMP to distribute sample generation across threads. Each thread
 * maintains a thread-local Mersenne Twister PRNG seeded from a global seed
 * XORed with the thread ID hash for reproducibility within a run.
 *
 * Each sample is generated via generate_sample_Floyd() and then its
 * detector/observable outcome is computed via calculate_parity_output_from_one_sample().
 */
void sampler::generate_many_output_samples(const QEPG::QEPG& graph,std::vector<QEPG::Row>& samplecontainer, size_t pauliweight, size_t samplenumber){
    if (pauliweight > num_total_pauliError_) throw std::invalid_argument("weight exceeds number of fault locations");

    samplecontainer.resize(samplenumber);

    static const std::uint64_t global_seed = std::random_device{}();
    static std::atomic<uint64_t> call_counter{0};
    const uint64_t call_id = call_counter.fetch_add(1, std::memory_order_relaxed);
    const size_t n_err = num_total_pauliError_;
    const auto& flat = graph.get_parityPropMatrixTransFlat();
    const std::size_t n_noise = flat.n_rows() / 3;
    const std::size_t n_words = flat.words_per_row();
    const std::size_t n_cols = flat.n_cols();

    #pragma omp parallel
    {
        sampler local_sampler(n_err);
        Xoshiro256pp rng(global_seed ^ std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ (call_id * 0x9E3779B97F4A7C15ULL));

        // Per-thread aligned result buffer
        std::uint64_t* result_buf = simd::aligned_alloc_u64(flat.stride_words());

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            std::size_t n = local_sampler.generate_sample_Floyd(pauliweight, rng);

            // Zero result, accumulate XOR via SIMD, copy to Row
            simd::zero_words(result_buf, n_words);
            local_sampler.calculate_parity_output_flat(
                flat, n_noise, local_sampler.scratch_data(), n, result_buf);

            samplecontainer[i] = QEPG::Row(n_cols, result_buf, n_words);
        }

        simd::aligned_free(result_buf);
    }
}


/**
 * @brief Generate many Monte Carlo samples in parallel.
 *
 * Same parallelization strategy as generate_many_output_samples() but uses
 * Bernoulli sampling (generate_sample_Monte) instead of fixed-weight sampling.
 */
void sampler::generate_many_output_samples_Monte(const QEPG::QEPG& graph,std::vector<QEPG::Row>& samplecontainer,double error_prob, size_t samplenumber){
    samplecontainer.resize(samplenumber);
    size_t total_error=graph.get_total_noise();
    static const std::uint64_t global_seed = std::random_device{}();
    static std::atomic<uint64_t> call_counter{0};
    const uint64_t call_id = call_counter.fetch_add(1, std::memory_order_relaxed);
    const size_t n_err = num_total_pauliError_;
    const auto& flat = graph.get_parityPropMatrixTransFlat();
    const std::size_t n_noise = flat.n_rows() / 3;
    const std::size_t n_words = flat.words_per_row();
    const std::size_t n_cols = flat.n_cols();

    #pragma omp parallel
    {
        sampler local_sampler(n_err);
        Xoshiro256pp rng(global_seed ^ std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ (call_id * 0x9E3779B97F4A7C15ULL));
        std::uint64_t* result_buf = simd::aligned_alloc_u64(flat.stride_words());

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            std::size_t n = local_sampler.generate_sample_Monte(error_prob, total_error, rng);

            simd::zero_words(result_buf, n_words);
            local_sampler.calculate_parity_output_flat(
                flat, n_noise, local_sampler.scratch_data(), n, result_buf);

            samplecontainer[i] = QEPG::Row(n_cols, result_buf, n_words);
        }

        simd::aligned_free(result_buf);
    }
}


/**
 * @brief Enumerate all error configurations of a given weight (stub).
 *
 * Intended to enumerate all C(N, weight) * 3^weight configurations
 * using recursion. Currently not implemented.
 */
void sampler::generate_all_samples_with_fixed_weight(const QEPG::QEPG& graph,std::vector<QEPG::Row>& samplecontainer,size_t pauliweight){

}


/**
 * @brief Generate many samples and also return the noise vectors (single-threaded).
 *
 * Unlike the parallel batch methods, this version stores both the computed
 * detector/observable outcomes and the raw noise configurations that produced them.
 * Useful for debugging or for algorithms that need to inspect which specific
 * errors caused a given syndrome.
 */
void sampler::generate_many_output_samples_with_noise_vector(const QEPG::QEPG& graph,std::vector<std::vector<singlePauli>>& noisecontainer,std::vector<QEPG::Row>& samplecontainer, size_t pauliweight, size_t samplenumber){
    if (pauliweight > num_total_pauliError_) throw std::invalid_argument("weight exceeds number of fault locations");

    samplecontainer.reserve(samplenumber);
    noisecontainer.reserve(samplenumber);
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    for(size_t i=0;i<samplenumber;i++){
        std::size_t n = generate_sample_Floyd(pauliweight, gen);
        samplecontainer.push_back(calculate_parity_output_from_one_sample(graph, scratch_data(), n));
        noisecontainer.emplace_back(scratch_sample_.begin(), scratch_sample_.begin() + n);
    }
}


/// Helper: unpack result_buf (uint64_t words) into numpy byte buffers.
/// Extracts individual bits into uint8 bytes (0 or 1).
/// Uses 8-bit-at-a-time extraction for ~8x fewer loop iterations.
static inline void unpack_result_to_numpy(
    const std::uint64_t* result_buf,
    std::uint8_t* det_dst,
    std::uint8_t& obs_dst,
    std::size_t n_det,
    std::size_t n_words) noexcept
{
    // Unpack 8 bits at a time from each byte of each word
    std::size_t bit_idx = 0;
    const std::size_t full_bytes = n_det / 8;
    const auto* byte_ptr = reinterpret_cast<const std::uint8_t*>(result_buf);

    for (std::size_t b = 0; b < full_bytes; ++b) {
        std::uint8_t byte = byte_ptr[b];
        det_dst[bit_idx + 0] = (byte     ) & 1;
        det_dst[bit_idx + 1] = (byte >> 1) & 1;
        det_dst[bit_idx + 2] = (byte >> 2) & 1;
        det_dst[bit_idx + 3] = (byte >> 3) & 1;
        det_dst[bit_idx + 4] = (byte >> 4) & 1;
        det_dst[bit_idx + 5] = (byte >> 5) & 1;
        det_dst[bit_idx + 6] = (byte >> 6) & 1;
        det_dst[bit_idx + 7] = (byte >> 7) & 1;
        bit_idx += 8;
    }

    // Remaining bits (< 8)
    if (bit_idx < n_det) {
        std::uint8_t byte = byte_ptr[full_bytes];
        for (std::size_t k = 0; bit_idx + k < n_det; ++k)
            det_dst[bit_idx + k] = (byte >> k) & 1;
    }

    // Observable is the last bit (at position n_det)
    obs_dst = static_cast<std::uint8_t>(
        (result_buf[n_det / 64] >> (n_det % 64)) & 1);
}


void sampler::generate_many_output_samples_to_numpy(
    const QEPG::QEPG& graph,
    std::uint8_t* det_buf, std::uint8_t* obs_buf,
    std::size_t n_det, size_t pauliweight, size_t samplenumber)
{
    if (pauliweight > num_total_pauliError_) throw std::invalid_argument("weight exceeds number of fault locations");

    static const std::uint64_t global_seed = std::random_device{}();
    static std::atomic<uint64_t> call_counter{0};
    const uint64_t call_id = call_counter.fetch_add(1, std::memory_order_relaxed);
    const size_t n_err = num_total_pauliError_;
    const auto& flat = graph.get_parityPropMatrixTransFlat();
    const std::size_t n_noise = flat.n_rows() / 3;
    const std::size_t n_words = flat.words_per_row();

    #pragma omp parallel
    {
        sampler local_sampler(n_err);
        Xoshiro256pp rng(global_seed ^ std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ (call_id * 0x9E3779B97F4A7C15ULL));
        std::uint64_t* result_buf = simd::aligned_alloc_u64(flat.stride_words());

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            std::size_t n = local_sampler.generate_sample_Floyd(pauliweight, rng);

            simd::zero_words(result_buf, n_words);
            local_sampler.calculate_parity_output_flat(
                flat, n_noise, local_sampler.scratch_data(), n, result_buf);

            unpack_result_to_numpy(result_buf,
                det_buf + i * n_det, obs_buf[i], n_det, n_words);
        }

        simd::aligned_free(result_buf);
    }
}


void sampler::generate_many_output_samples_Monte_to_numpy(
    const QEPG::QEPG& graph,
    std::uint8_t* det_buf, std::uint8_t* obs_buf,
    std::size_t n_det, double error_prob, size_t samplenumber)
{
    static const std::uint64_t global_seed = std::random_device{}();
    static std::atomic<uint64_t> call_counter{0};
    const uint64_t call_id = call_counter.fetch_add(1, std::memory_order_relaxed);
    const size_t n_err = num_total_pauliError_;
    const size_t total_error = graph.get_total_noise();
    const auto& flat = graph.get_parityPropMatrixTransFlat();
    const std::size_t n_noise = flat.n_rows() / 3;
    const std::size_t n_words = flat.words_per_row();


    #pragma omp parallel
    {
        sampler local_sampler(n_err);
        Xoshiro256pp rng(global_seed ^ std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ (call_id * 0x9E3779B97F4A7C15ULL));
        std::uint64_t* result_buf = simd::aligned_alloc_u64(flat.stride_words());

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            simd::zero_words(result_buf, n_words);

            // Fused sample + XOR: sample k ~ Binomial(N,p), generate k
            // error locations, and XOR each row directly into result_buf.
            const std::size_t k = std::binomial_distribution<std::size_t>(total_error, error_prob)(rng);

            if (k > 0 && k < total_error) {
                local_sampler.generate_and_xor_sparse(
                    flat, n_noise, total_error, k, rng, result_buf);
            } else if (k >= total_error) {
                for (std::size_t pos = 0; pos < total_error; ++pos) {
                    std::size_t type = 1 + rng.bounded(3);
                    flat.xor_row_into(pos + (type - 1) * n_noise, result_buf);
                }
            }

            unpack_result_to_numpy(result_buf,
                det_buf + i * n_det, obs_buf[i], n_det, n_words);
        }

        simd::aligned_free(result_buf);
    }
}


void sampler::generate_many_output_samples_nonuniform_to_numpy(
    const QEPG::QEPG& graph,
    std::uint8_t* det_buf, std::uint8_t* obs_buf,
    std::size_t n_det,
    const double* noise_probs,
    std::size_t num_noise,
    const CorrelatedPair* corr_pairs,
    std::size_t num_corr_pairs,
    std::size_t samplenumber)
{
    const auto& flat = graph.get_parityPropMatrixTransFlat();
    const std::size_t n_noise_flat = flat.n_rows() / 3;
    const std::size_t n_words = flat.words_per_row();

    // Pre-compute per-source total probability
    std::vector<double> ptotal(num_noise);
    double P_all = 0.0;
    for (std::size_t i = 0; i < num_noise; ++i) {
        ptotal[i] = noise_probs[3 * i] + noise_probs[3 * i + 1] + noise_probs[3 * i + 2];
        P_all += ptotal[i];
    }

    // Pre-compute conditional Pauli type thresholds
    std::vector<double> cond_x(num_noise, 0.0);
    std::vector<double> cond_xy(num_noise, 0.0);
    for (std::size_t i = 0; i < num_noise; ++i) {
        if (ptotal[i] > 0.0) {
            cond_x[i]  = noise_probs[3 * i] / ptotal[i];
            cond_xy[i] = (noise_probs[3 * i] + noise_probs[3 * i + 1]) / ptotal[i];
        }
    }

    // Decide sparse vs dense
    double p_max = 0.0;
    for (std::size_t i = 0; i < num_noise; ++i)
        if (ptotal[i] > p_max) p_max = ptotal[i];
    const bool use_sparse = (p_max < 0.01) && (P_all < 0.06 * num_noise);

    // Exact sparse sampling by geometric skipping and Bernoulli thinning.
    // Unlike Poisson draws with replacement, each source fires at most once.
    const double log_survival = p_max > 0.0 && p_max < 1.0 ? std::log1p(-p_max) : 0.0;

    static const std::uint64_t global_seed = std::random_device{}();
    static std::atomic<uint64_t> call_counter{0};
    const uint64_t call_id = call_counter.fetch_add(1, std::memory_order_relaxed);

    #pragma omp parallel
    {
        Xoshiro256pp rng(global_seed ^ std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ (call_id * 0x9E3779B97F4A7C15ULL));
        std::uint64_t* result_buf = simd::aligned_alloc_u64(flat.stride_words());

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            simd::zero_words(result_buf, n_words);

            if (use_sparse) {
                if (p_max > 0.0) {
                    std::size_t src = 0;
                    while (src < num_noise) {
                        const double skip = std::floor(std::log1p(-to_double_01(rng())) / log_survival);
                        if (skip >= static_cast<double>(num_noise - src)) break;
                        src += static_cast<std::size_t>(skip);
                        if (to_double_01(rng()) < ptotal[src] / p_max) {
                            const double r = to_double_01(rng());
                            const std::size_t type = r < cond_x[src] ? 1 : (r < cond_xy[src] ? 2 : 3);
                            flat.xor_row_into(src + (type - 1) * n_noise_flat, result_buf);
                        }
                        ++src;
                    }
                }
            } else {
                for (std::size_t s = 0; s < num_noise; ++s) {
                    const double r = to_double_01(rng());
                    if (r < ptotal[s]) {
                        const std::size_t type = r < noise_probs[3*s] ? 1 :
                            (r < noise_probs[3*s] + noise_probs[3*s+1] ? 2 : 3);
                        flat.xor_row_into(s + (type - 1) * n_noise_flat, result_buf);
                    }
                }
            }

            // --- Correlated pairs (DEPOLARIZE2) ---
            for (std::size_t c = 0; c < num_corr_pairs; ++c) {
                const double r = to_double_01(rng());
                if (r < corr_pairs[c].prob) {
                    std::size_t pidx = rng.bounded(15);
                    std::size_t pa = TWO_QUBIT_PAULIS[pidx][0];
                    std::size_t pb = TWO_QUBIT_PAULIS[pidx][1];
                    if (pa > 0) {
                        flat.xor_row_into(
                            corr_pairs[c].source_a + (pa - 1) * n_noise_flat,
                            result_buf);
                    }
                    if (pb > 0) {
                        flat.xor_row_into(
                            corr_pairs[c].source_b + (pb - 1) * n_noise_flat,
                            result_buf);
                    }
                }
            }

            // --- Unpack to numpy ---
            unpack_result_to_numpy(result_buf,
                det_buf + i * n_det, obs_buf[i], n_det, n_words);
        }

        simd::aligned_free(result_buf);
    }
}


}
