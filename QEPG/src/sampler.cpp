#include "sampler.hpp"
#include <chrono>
#include <thread>


namespace SAMPLE{

/*---------------------------------------ctor----------*/
sampler::sampler()=default;

sampler::sampler(size_t num_total_paulierror):num_total_pauliError_(num_total_paulierror){};

sampler::~sampler()=default;

/*---------------------------------------Sample one vector with fixed weight----------*/        


inline std::vector<singlePauli> sampler::generate_sample_Floyd(size_t weight, std::mt19937& gen){

    std::vector<singlePauli> result;
    result.reserve(weight);

    // Uniform distribution in the range [1, 6]
    std::uniform_int_distribution<> posdistrib(0, num_total_pauliError_-1);
    std::uniform_int_distribution<> typedistrib(1, 3);
    std::unordered_set<size_t> usedpos;


    while(result.size()<weight){
        size_t newpos=(size_t)posdistrib(gen);
        if(usedpos.insert(newpos).second){
            result.emplace_back(std::move(singlePauli{ newpos, (size_t)typedistrib(gen)}));  // OK
        }
    }
    return result;
}


inline std::vector<singlePauli> sampler::generate_sample_Monte(double error_prob,size_t ErrorSize,std::mt19937& gen){
    std::vector<singlePauli> result;
    // Special case: zero error rate means no errors
    if (error_prob <= 0.0) {
        return result;
    }
    result.reserve(size_t(error_prob*ErrorSize));
    std::uniform_int_distribution<> typedistrib(1, 3);
    std::bernoulli_distribution dist(error_prob);
    for(size_t pos=0;pos<ErrorSize;++pos){
        if(dist(gen)){
            result.emplace_back(std::move(singlePauli{ pos, (size_t)typedistrib(gen)}));
        }
    }
    return result;
}






void sampler::generate_many_output_samples(const QEPG::QEPG& graph,std::vector<QEPG::Row>& samplecontainer, size_t pauliweight, size_t samplenumber){
    //samplecontainer.reserve(samplenumber);
    samplecontainer.resize(samplenumber);

    static const std::uint64_t global_seed = std::random_device{}();   // log this if you need to replay

    // 2. Parallel region
    #pragma omp parallel
    {
        thread_local std::mt19937 rng{
            static_cast<std::mt19937::result_type>(
                global_seed ^                                    // same run → same base
                std::hash<std::thread::id>{}(std::this_thread::get_id()))  // thread-specific part
        };


        // 3. Work-share the loop
        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            auto sample = generate_sample_Floyd(pauliweight, rng);
            samplecontainer[i] = calculate_parity_output_from_one_sample(graph, sample);
        }
    }
}


void sampler::generate_many_output_samples_Monte(const QEPG::QEPG& graph,std::vector<QEPG::Row>& samplecontainer,double error_prob, size_t samplenumber){
    //samplecontainer.reserve(samplenumber);
    samplecontainer.resize(samplenumber);
    size_t total_error=graph.get_total_noise();
    static const std::uint64_t global_seed = std::random_device{}();   // log this if you need to replay
    // 2. Parallel region
    #pragma omp parallel
    {
        thread_local std::mt19937 rng{
            static_cast<std::mt19937::result_type>(
                global_seed ^                                    // same run → same base
                std::hash<std::thread::id>{}(std::this_thread::get_id()))  // thread-specific part
        };
        // 3. Work-share the loop
        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            auto sample = generate_sample_Monte(error_prob,total_error, rng);
            samplecontainer[i] = calculate_parity_output_from_one_sample(graph, sample);
        }
    }
}




/*
In this implementation, we also return the generated random noise vector 
*/
void sampler::generate_many_output_samples_with_noise_vector(const QEPG::QEPG& graph,std::vector<std::vector<singlePauli>>& noisecontainer,std::vector<QEPG::Row>& samplecontainer, size_t pauliweight, size_t samplenumber){
    samplecontainer.reserve(samplenumber);
    noisecontainer.reserve(samplenumber);
    std::random_device rd;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    // Mersenne Twister engine seeded with rd()
    std::mt19937 gen(seed);
    for(size_t i=0;i<samplenumber;i++){
        std::vector<singlePauli> sample = generate_sample_Floyd(pauliweight,gen);
        // Calculate parity BEFORE moving sample
        samplecontainer.push_back(calculate_parity_output_from_one_sample(graph,sample));
        noisecontainer.push_back(std::move(sample));
    }
}


/*---------------------------------------SIMD-accelerated sampling methods----------*/

inline void sampler::calculate_parity_simd(
    uint64_t* result,
    const SIMD::SIMDMatrix& matrix,
    const std::vector<singlePauli>& sample,
    size_t n_noise)
{
    // Zero the result buffer
    matrix.zero_buffer(result);

    // XOR each error's row into the result
    for (const singlePauli& noise : sample) {
        size_t row_idx;
        if (noise.type == PAULIX) {
            row_idx = noise.qindex;
        } else if (noise.type == PAULIY) {
            row_idx = noise.qindex + n_noise;
        } else {  // PAULIZ
            row_idx = noise.qindex + n_noise * 2;
        }
        matrix.xor_row_into(result, row_idx);
    }
}

inline QEPG::Row sampler::buffer_to_bitset(const uint64_t* src, size_t /*words*/, size_t bits) {
    QEPG::Row result(bits);
    // Calculate the actual number of blocks the bitset needs
    size_t num_blocks = result.num_blocks();
    // Use boost's from_block_range to efficiently copy blocks (only copy needed blocks)
    boost::from_block_range(src, src + num_blocks, result);
    return result;
}

void sampler::generate_many_output_samples_simd(
    const QEPG::QEPG& graph,
    std::vector<QEPG::Row>& samplecontainer,
    size_t pauliweight,
    size_t samplenumber)
{
    // Get the parity propagation matrix
    const auto& dm = graph.get_parityPropMatrixTrans();
    if (dm.empty()) {
        samplecontainer.clear();
        return;
    }

    const size_t n_rows = dm.size();
    const size_t n_noise = n_rows / 3;
    const size_t n_cols = dm[0].size();

    // Create SIMD-aligned copy of the matrix (done once, shared across threads)
    SIMD::SIMDMatrix simd_matrix(n_rows, n_cols);
    simd_matrix.copy_from_bitset_matrix(dm);

    // Prepare output container
    samplecontainer.resize(samplenumber);

    static const std::uint64_t global_seed = std::random_device{}();

    // Parallel sampling with SIMD parity computation
    #pragma omp parallel
    {
        // Thread-local RNG
        thread_local std::mt19937 rng{
            static_cast<std::mt19937::result_type>(
                global_seed ^ std::hash<std::thread::id>{}(std::this_thread::get_id()))
        };

        // Thread-local aligned result buffer
        uint64_t* local_result = simd_matrix.allocate_result_buffer();

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            // Generate random error pattern
            auto sample = generate_sample_Floyd(pauliweight, rng);

            // Compute parity using SIMD
            calculate_parity_simd(local_result, simd_matrix, sample, n_noise);

            // Convert to boost::dynamic_bitset
            samplecontainer[i] = buffer_to_bitset(local_result, simd_matrix.words_per_row(), n_cols);
        }

        // Free thread-local buffer
        SIMD::aligned_free_u64(local_result);
    }
}

void sampler::generate_many_output_samples_Monte_simd(
    const QEPG::QEPG& graph,
    std::vector<QEPG::Row>& samplecontainer,
    double error_prob,
    size_t samplenumber)
{
    // Get the parity propagation matrix
    const auto& dm = graph.get_parityPropMatrixTrans();
    if (dm.empty()) {
        samplecontainer.clear();
        return;
    }

    const size_t n_rows = dm.size();
    const size_t n_noise = n_rows / 3;
    const size_t n_cols = dm[0].size();
    const size_t total_error = graph.get_total_noise();

    // Create SIMD-aligned copy of the matrix
    SIMD::SIMDMatrix simd_matrix(n_rows, n_cols);
    simd_matrix.copy_from_bitset_matrix(dm);

    // Prepare output container
    samplecontainer.resize(samplenumber);

    static const std::uint64_t global_seed = std::random_device{}();

    // Parallel Monte Carlo sampling with SIMD parity computation
    #pragma omp parallel
    {
        // Thread-local RNG
        thread_local std::mt19937 rng{
            static_cast<std::mt19937::result_type>(
                global_seed ^ std::hash<std::thread::id>{}(std::this_thread::get_id()))
        };

        // Thread-local aligned result buffer
        uint64_t* local_result = simd_matrix.allocate_result_buffer();

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            // Generate random error pattern with Monte Carlo
            auto sample = generate_sample_Monte(error_prob, total_error, rng);

            // Compute parity using SIMD
            calculate_parity_simd(local_result, simd_matrix, sample, n_noise);

            // Convert to boost::dynamic_bitset
            samplecontainer[i] = buffer_to_bitset(local_result, simd_matrix.words_per_row(), n_cols);
        }

        // Free thread-local buffer
        SIMD::aligned_free_u64(local_result);
    }
}


/*---------------------------------------Optimized buffer-only SIMD methods----------*/

// xorshift128+ - very fast PRNG, passes BigCrush
inline uint64_t xorshift128plus(uint64_t* state) {
    uint64_t s1 = state[0];
    const uint64_t s0 = state[1];
    state[0] = s0;
    s1 ^= s1 << 23;
    state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return state[1] + s0;
}

inline void sampler::generate_sample_fast(
    size_t* positions,
    uint8_t* types,
    size_t weight,
    size_t n_noise,
    uint64_t* rng_state)
{
    // Fisher-Yates partial shuffle approach for sampling without replacement
    // For small weight relative to n_noise, rejection sampling is faster
    if (weight == 0) return;

    if (weight * 10 < n_noise) {
        // Rejection sampling for sparse case
        size_t count = 0;
        // Use a simple bitset for tracking used positions (stack allocated for small n_noise)
        thread_local std::vector<bool> used;
        used.assign(n_noise, false);

        while (count < weight) {
            uint64_t r = xorshift128plus(rng_state);
            size_t pos = r % n_noise;
            if (!used[pos]) {
                used[pos] = true;
                positions[count] = pos;
                // Generate type (1, 2, or 3) from next random bits
                types[count] = static_cast<uint8_t>(1 + (xorshift128plus(rng_state) % 3));
                ++count;
            }
        }
    } else {
        // For denser sampling, use partial Fisher-Yates
        thread_local std::vector<size_t> indices;
        if (indices.size() < n_noise) {
            indices.resize(n_noise);
        }
        // Always reinitialize indices to 0,1,2,... before each sample
        // This is critical for correctness - the previous implementation
        // left indices in a shuffled state which biased subsequent samples
        for (size_t i = 0; i < n_noise; ++i) indices[i] = i;

        for (size_t i = 0; i < weight; ++i) {
            uint64_t r = xorshift128plus(rng_state);
            size_t j = i + (r % (n_noise - i));
            std::swap(indices[i], indices[j]);
            positions[i] = indices[i];
            types[i] = static_cast<uint8_t>(1 + (xorshift128plus(rng_state) % 3));
        }
    }
}

inline size_t sampler::generate_sample_Monte_fast(
    size_t* positions,
    uint8_t* types,
    size_t n_noise,
    double error_prob,
    uint64_t* rng_state)
{
    // Special case: zero error rate means no errors
    if (error_prob <= 0.0) {
        return 0;
    }

    size_t count = 0;
    // Use geometric distribution to skip ahead
    // For low error rates, this is much faster than checking each position
    if (error_prob < 0.1) {
        // Geometric skipping
        double log_1_minus_p = std::log(1.0 - error_prob);
        size_t pos = 0;
        while (pos < n_noise) {
            // Generate geometric random variable
            double u = (xorshift128plus(rng_state) >> 11) * (1.0 / 9007199254740992.0); // [0,1)
            if (u < 1e-15) u = 1e-15; // Avoid log(0)
            size_t skip = static_cast<size_t>(std::log(u) / log_1_minus_p);
            pos += skip;
            if (pos < n_noise) {
                positions[count] = pos;
                types[count] = static_cast<uint8_t>(1 + (xorshift128plus(rng_state) % 3));
                ++count;
                ++pos;
            }
        }
    } else {
        // Direct Bernoulli for high error rates
        const uint64_t threshold = static_cast<uint64_t>(error_prob * 18446744073709551616.0);
        for (size_t pos = 0; pos < n_noise; ++pos) {
            if (xorshift128plus(rng_state) < threshold) {
                positions[count] = pos;
                types[count] = static_cast<uint8_t>(1 + (xorshift128plus(rng_state) % 3));
                ++count;
            }
        }
    }
    return count;
}

inline void sampler::unpack_buffer_to_bytes(
    const uint64_t* src,
    uint8_t* dst,
    size_t n_bits)
{
    size_t full_words = n_bits / 64;
    size_t remaining = n_bits % 64;

    // Unpack full 64-bit words
    for (size_t w = 0; w < full_words; ++w) {
        uint64_t word = src[w];
        uint8_t* out = dst + w * 64;
        // Unroll for better performance
        for (int b = 0; b < 64; b += 8) {
            out[b + 0] = static_cast<uint8_t>((word >> (b + 0)) & 1);
            out[b + 1] = static_cast<uint8_t>((word >> (b + 1)) & 1);
            out[b + 2] = static_cast<uint8_t>((word >> (b + 2)) & 1);
            out[b + 3] = static_cast<uint8_t>((word >> (b + 3)) & 1);
            out[b + 4] = static_cast<uint8_t>((word >> (b + 4)) & 1);
            out[b + 5] = static_cast<uint8_t>((word >> (b + 5)) & 1);
            out[b + 6] = static_cast<uint8_t>((word >> (b + 6)) & 1);
            out[b + 7] = static_cast<uint8_t>((word >> (b + 7)) & 1);
        }
    }

    // Handle remaining bits
    if (remaining > 0) {
        uint64_t word = src[full_words];
        uint8_t* out = dst + full_words * 64;
        for (size_t b = 0; b < remaining; ++b) {
            out[b] = static_cast<uint8_t>((word >> b) & 1);
        }
    }
}

void sampler::generate_samples_to_buffer_simd(
    const SIMD::SIMDMatrix& simd_matrix,
    uint8_t* det_output,
    uint8_t* obs_output,
    size_t det_row_stride,
    size_t n_det,
    size_t n_noise,
    size_t pauliweight,
    size_t samplenumber,
    size_t begin_index)
{
    if (samplenumber == 0) return;

    const size_t n_cols = n_det + 1;  // detectors + observable

    static const uint64_t global_seed = std::random_device{}();

    #pragma omp parallel
    {
        // Thread-local RNG state (xorshift128+)
        uint64_t rng_state[2];
        uint64_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        rng_state[0] = global_seed ^ thread_id ^ 0x853c49e6748fea9bULL;
        rng_state[1] = global_seed ^ (thread_id << 32) ^ 0xda3e39cb94b95bdbULL;
        // Warm up the RNG
        for (int i = 0; i < 10; ++i) xorshift128plus(rng_state);

        // Thread-local aligned result buffer
        uint64_t* local_result = simd_matrix.allocate_result_buffer();

        // Thread-local scratch for sample generation
        thread_local std::vector<size_t> positions;
        thread_local std::vector<uint8_t> types;
        if (positions.size() < pauliweight) {
            positions.resize(pauliweight);
            types.resize(pauliweight);
        }

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            const size_t row_idx = begin_index + static_cast<size_t>(i);

            // Generate random error pattern
            generate_sample_fast(positions.data(), types.data(), pauliweight, n_noise, rng_state);

            // Zero the result buffer
            simd_matrix.zero_buffer(local_result);

            // XOR each error's row into the result
            for (size_t j = 0; j < pauliweight; ++j) {
                size_t mat_row;
                if (types[j] == PAULIX) {
                    mat_row = positions[j];
                } else if (types[j] == PAULIY) {
                    mat_row = positions[j] + n_noise;
                } else {  // PAULIZ
                    mat_row = positions[j] + n_noise * 2;
                }
                simd_matrix.xor_row_into(local_result, mat_row);
            }

            // Unpack detector bits directly to output
            uint8_t* det_row = det_output + row_idx * det_row_stride;
            unpack_buffer_to_bytes(local_result, det_row, n_det);

            // Extract observable (last bit)
            size_t obs_word = n_det / 64;
            size_t obs_bit = n_det % 64;
            obs_output[row_idx] = static_cast<uint8_t>((local_result[obs_word] >> obs_bit) & 1);
        }

        SIMD::aligned_free_u64(local_result);
    }
}

void sampler::generate_samples_Monte_to_buffer_simd(
    const SIMD::SIMDMatrix& simd_matrix,
    uint8_t* det_output,
    uint8_t* obs_output,
    size_t det_row_stride,
    size_t n_det,
    size_t n_noise,
    double error_prob,
    size_t samplenumber,
    size_t begin_index)
{
    if (samplenumber == 0) return;

    const size_t n_cols = n_det + 1;

    static const uint64_t global_seed = std::random_device{}();

    #pragma omp parallel
    {
        // Thread-local RNG state
        uint64_t rng_state[2];
        uint64_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        rng_state[0] = global_seed ^ thread_id ^ 0x853c49e6748fea9bULL;
        rng_state[1] = global_seed ^ (thread_id << 32) ^ 0xda3e39cb94b95bdbULL;
        for (int i = 0; i < 10; ++i) xorshift128plus(rng_state);

        // Thread-local aligned result buffer
        uint64_t* local_result = simd_matrix.allocate_result_buffer();

        // Thread-local scratch for sample generation (worst case: all positions)
        thread_local std::vector<size_t> positions;
        thread_local std::vector<uint8_t> types;
        if (positions.size() < n_noise) {
            positions.resize(n_noise);
            types.resize(n_noise);
        }

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(samplenumber); ++i) {
            const size_t row_idx = begin_index + static_cast<size_t>(i);

            // Generate random error pattern with Monte Carlo
            size_t num_errors = generate_sample_Monte_fast(
                positions.data(), types.data(), n_noise, error_prob, rng_state);

            // Zero the result buffer
            simd_matrix.zero_buffer(local_result);

            // XOR each error's row into the result
            for (size_t j = 0; j < num_errors; ++j) {
                size_t mat_row;
                if (types[j] == PAULIX) {
                    mat_row = positions[j];
                } else if (types[j] == PAULIY) {
                    mat_row = positions[j] + n_noise;
                } else {  // PAULIZ
                    mat_row = positions[j] + n_noise * 2;
                }
                simd_matrix.xor_row_into(local_result, mat_row);
            }

            // Unpack detector bits directly to output
            uint8_t* det_row = det_output + row_idx * det_row_stride;
            unpack_buffer_to_bytes(local_result, det_row, n_det);

            // Extract observable (last bit)
            size_t obs_word = n_det / 64;
            size_t obs_bit = n_det % 64;
            obs_output[row_idx] = static_cast<uint8_t>((local_result[obs_word] >> obs_bit) & 1);
        }

        SIMD::aligned_free_u64(local_result);
    }
}


}

