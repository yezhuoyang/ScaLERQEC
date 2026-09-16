/**
 * @file LERcalculator.cpp
 * @brief Implementation of the high-level sampling API for Python interop.
 *
 * Contains internal conversion utilities (bitset-to-boolean, bitset-to-NumPy)
 * and the implementations of all LERcalculator namespace functions declared
 * in LERcalculator.hpp. Conversion functions use OpenMP for parallel unpacking
 * of bitset rows into contiguous NumPy arrays.
 */

#include "LERcalculator.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>




namespace LERcalculator{

namespace {
void validate_output_size(std::size_t detectors, std::size_t shots) {
    const auto limit = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    if (detectors >= limit || shots > limit / (detectors + 1))
        throw std::invalid_argument("Requested sample output is too large");
}
void validate_probability(double p) {
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("Probability must be finite and in [0, 1]");
}
std::size_t validate_batches(const QEPG::QEPG& graph,
        const std::vector<std::size_t>& weights, const std::vector<std::size_t>& shots) {
    if (weights.size() != shots.size())
        throw std::invalid_argument("weights and shots must have equal lengths");
    std::size_t total = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] > graph.get_total_noise())
            throw std::invalid_argument("weight exceeds number of fault locations");
        if (shots[i] > std::numeric_limits<std::size_t>::max() - total)
            throw std::invalid_argument("shot count overflows");
        total += shots[i];
    }
    validate_output_size(graph.get_total_detector(), total);
    return total;
}
}




/**
 * @brief Convert a vector of GF(2) bitset rows to a 2D boolean vector.
 *
 * @param[out] result          Output vector, populated with one vector<bool> per bitset row.
 * @param[in]  samplecontainer Input bitset rows to convert.
 */
void convert_bitset_row_to_boolean(std::vector<std::vector<bool>>& result,const std::vector<QEPG::Row>& samplecontainer){
        result.reserve(samplecontainer.size()); // Reserve space

        // Convert each DynamicBitset row to std::vector<bool>
        for (const auto& bitset_row : samplecontainer) {
            std::vector<bool> bool_row(bitset_row.size());
            for (size_t i = 0; i < bitset_row.size(); ++i) {
                bool_row[i] = bitset_row[i]; // Access individual bits
            }
            result.push_back(bool_row);
        }
}



/**
 * @brief Convert bitset rows to a 2D NumPy bool array using parallel block unpacking.
 *
 * Releases the Python GIL during conversion. Each row of the bitset is unpacked
 * word-by-word (64 bits at a time) into the flat NumPy buffer. Uses OpenMP
 * to parallelize across rows.
 *
 * @param rows The bitset rows to convert.
 * @return A NumPy bool array of shape (n_rows, n_cols).
 */
inline py::array_t<bool> bitset_rows_to_numpy(const std::vector<QEPG::Row>& rows)
{
    using bitset_t  = QEPG::Row;
    using block_t   = bitset_t::block_type;           // usually uint64_t

    const std::size_t n_rows = rows.size();
    const std::size_t n_cols = n_rows ? rows.front().size() : 0;
    if(n_cols==0)
        return py::array_t<bool>({n_rows, n_cols});

    py::array_t<bool> out({n_rows, n_cols});
    auto req=out.request();
    auto* base=static_cast<std::uint8_t*>(req.ptr);
    const auto row_stride=n_cols;

    //release the GIL so other python threads can run
    py::gil_scoped_release release;


    //How many bits stored in this block, either 32 or 64
    constexpr std::size_t WORD_BITS = std::numeric_limits<block_t>::digits;


    //-----parallel over rows-------------------------------
    #pragma omp parallel for schedule(static)
    for(long long r=0; r < static_cast<long long>(n_rows);++r)
    {
        const QEPG::Row& bits=rows[static_cast<std::size_t>(r)];
        const std::size_t n_blk=bits.num_blocks();

        /* --- thread-local scratch buffer (namespace scope ==> OK in MSVC) */
        static thread_local std::vector<block_t> tl_buf;
        tl_buf.resize(n_blk);                               // realloc only if needed
        bits.to_block_range(tl_buf.begin());        // fill the buffer

        /*  2. Unpack into the Numpy row (64 bits -> 64 bytes)*/
        std::uint8_t* dst=base+r*row_stride;

        for(std::size_t b=0; b+1 <n_blk;++b){
            std::uint64_t word=static_cast<std::uint64_t>(tl_buf[b]);
            for(int k=0;k<WORD_BITS;++k,word>>=1)
                dst[b*WORD_BITS+k]=static_cast<std::uint8_t>(word&1);
        }

        const std::size_t rem_bits=n_cols&(WORD_BITS-1);
        if(rem_bits){
            std::uint64_t word = static_cast<std::uint64_t>(tl_buf[n_blk-1]);
            for(std::size_t k=0;k<rem_bits;++k, word>>=1)
                dst[(n_blk-1)*WORD_BITS + k]=static_cast<std::uint8_t>(word & 1);
        }
    }
    return out;
}




/**
 * @brief Convert bitset rows to separate detector and observable boolean vectors.
 *
 * Splits each bitset row: all bits except the last go into the detector result,
 * and the last bit goes into the observable result.
 *
 * @param[out] result          Detector outcomes (each row has size n_cols - 1).
 * @param[out] obsresult       Observable outcomes (one bool per sample).
 * @param[in]  samplecontainer Input bitset rows.
 */
inline void convert_bitset_row_to_boolean_separate_obs(std::vector<std::vector<bool>>& result,std::vector<bool>& obsresult,const std::vector<QEPG::Row>& samplecontainer){
        result.reserve(samplecontainer.size()); // Reserve space
        obsresult.reserve(samplecontainer.size());
        // Convert each DynamicBitset row to std::vector<bool>
        for (const auto& bitset_row : samplecontainer) {
            std::vector<bool> bool_row(bitset_row.size()-1);
            for (size_t i = 0; i < bitset_row.size()-1; ++i) {
                bool_row[i] = bitset_row[i]; // Access individual bits
            }
            result.push_back(bool_row);
            obsresult.push_back(bitset_row[bitset_row.size()-1]);
        }
}


/**
 * @brief Convert bitset rows into pre-allocated NumPy arrays with separate detector and observable outputs.
 *
 * Unpacks bitset rows directly into contiguous NumPy memory with OpenMP parallelism.
 * Writes into the arrays starting at begin_index, allowing multiple weight batches
 * to be concatenated into the same output arrays.
 *
 * @param[in,out] detectionresult Pre-allocated NumPy array of shape (N, num_detectors).
 * @param[in,out] obsresult       Pre-allocated NumPy array of shape (N,).
 * @param[in]     begin_index     Row offset at which to start writing in the output arrays.
 * @param[in]     samplecontainer Input bitset rows to unpack.
 *
 * @throws std::runtime_error If the output arrays have wrong shape or are too small.
 */
inline void convert_bitset_row_to_boolean_separate_obs_numpy(
        pybind11::array_t<bool>&        detectionresult,   // shape (N, k)
        pybind11::array_t<bool>&        obsresult,         // shape (N,)
        const std::size_t               begin_index,
        const std::vector<QEPG::Row>&   samplecontainer)
{
    namespace py = pybind11;
    using bitset_t = QEPG::Row;
    using block_t  = bitset_t::block_type;                 // 32- or 64-bit

    const std::size_t n_rows = samplecontainer.size();
    if (n_rows == 0) return;                               // nothing to do

    /* ------- detector-column count (k) must be constant ---------------- */
    const std::size_t n_det = samplecontainer.front().size() - 1;   // last = obs

    /* ------- basic shape / bounds checks ------------------------------ */
    auto det_info = detectionresult.request();
    auto obs_info = obsresult.request();

    if (det_info.ndim != 2 || det_info.shape[1] != n_det)
        throw std::runtime_error("detectionresult has wrong shape");
    if (det_info.shape[0] < begin_index + n_rows
        || obs_info.shape[0] < begin_index + n_rows)
        throw std::runtime_error("output arrays are too small");

    /* ------- raw pointers & strides (bytes) --------------------------- */
    auto* det_base = static_cast<std::uint8_t*>(det_info.ptr);
    const std::size_t det_row_stride =
        static_cast<std::size_t>(det_info.strides[0]);     // bytes per row

    auto* obs_base = static_cast<std::uint8_t*>(obs_info.ptr);

    /* ------- constants ------------------------------------------------ */
    constexpr std::size_t WORD_BITS =
        std::numeric_limits<block_t>::digits;              // 32 or 64

    /* ------- work outside the GIL ------------------------------------ */
    py::gil_scoped_release release;

    #pragma omp parallel default(none) shared(samplecontainer, det_base, obs_base) firstprivate(n_rows, begin_index, det_row_stride, n_det, WORD_BITS)
    {
        std::vector<block_t> tl_buf;   // scratch per thread

        #pragma omp for schedule(static)
        for (long long r = 0; r < static_cast<long long>(n_rows); ++r)
        {
            const bitset_t& bits   = samplecontainer[static_cast<std::size_t>(r)];
            const std::size_t n_blk = bits.num_blocks();

            /* -- obtain packed words ---------------------------------- */
            tl_buf.resize(n_blk);
            bits.to_block_range(tl_buf.begin());

            /* -- detector destination row ----------------------------- */
            std::uint8_t* det_dst =
                det_base + (begin_index + r) * det_row_stride;

            /* -- full words ------------------------------------------- */
            const std::size_t n_blk_det = n_det / WORD_BITS;
            for (std::size_t b = 0; b < n_blk_det; ++b) {
                block_t w = tl_buf[b];
                for (std::size_t k = 0; k < WORD_BITS; ++k, w >>= 1)
                    det_dst[b * WORD_BITS + k] =
                        static_cast<std::uint8_t>(w & 1);
            }

            /* -- tail bits -------------------------------------------- */
            const std::size_t rem = n_det & (WORD_BITS - 1);
            if (rem) {
                block_t w = tl_buf[n_blk_det];
                for (std::size_t k = 0; k < rem; ++k, w >>= 1)
                    det_dst[n_blk_det * WORD_BITS + k] =
                        static_cast<std::uint8_t>(w & 1);
            }

            /* -- observable bit --------------------------------------- */
            obs_base[begin_index + r] =
                static_cast<std::uint8_t>(bits[n_det]);
        }
    }
}



/// @copydoc LERcalculator::return_samples_with_fixed_QEPG
std::vector<std::vector<bool>> return_samples_with_fixed_QEPG(const QEPG::QEPG& graph,size_t weight, size_t shots){
    SAMPLE::sampler sampler(graph.get_total_noise());
    std::vector<QEPG::Row> samplecontainer;
    sampler.generate_many_output_samples(graph,samplecontainer,weight,shots);
    std::vector<std::vector<bool>> result;
    convert_bitset_row_to_boolean(result,samplecontainer);
    return std::move(result);
}

/// @copydoc LERcalculator::return_samples_with_fixed_QEPG_numpy
std::pair<py::array_t<std::uint8_t>,py::array_t<std::uint8_t>> return_samples_with_fixed_QEPG_numpy(const QEPG::QEPG& graph,size_t weight, size_t shots){
    const std::size_t n_det = graph.get_total_detector();
    validate_output_size(n_det, shots);
    if (weight > graph.get_total_noise()) throw std::invalid_argument("weight exceeds number of fault locations");
    SAMPLE::sampler sampler(graph.get_total_noise());

    // Allocate NumPy buffers directly
    py::array_t<std::uint8_t> detectorresult({shots, n_det});
    py::array_t<std::uint8_t> obsresult(shots);

    auto det_info = detectorresult.request();
    auto obs_info = obsresult.request();
    auto* det_ptr = static_cast<std::uint8_t*>(det_info.ptr);
    auto* obs_ptr = static_cast<std::uint8_t*>(obs_info.ptr);

    // Fused sampling: write directly into NumPy buffers, no intermediate Row allocation
    {
        py::gil_scoped_release release;
        sampler.generate_many_output_samples_to_numpy(graph, det_ptr, obs_ptr, n_det, weight, shots);
    }

    return {std::move(detectorresult), std::move(obsresult)};
}





/// @copydoc LERcalculator::return_samples
 std::vector<std::vector<bool>> return_samples(const std::string& prog_str,size_t weight, size_t shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);
    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();
    SAMPLE::sampler sampler(c.get_num_noise());
    std::vector<QEPG::Row> samplecontainer;
    sampler.generate_many_output_samples(graph,samplecontainer,weight,shots);
    std::vector<std::vector<bool>> result;
    convert_bitset_row_to_boolean(result,samplecontainer);
    return std::move(result);
}



/// @copydoc LERcalculator::return_samples_numpy
py::array_t<bool> return_samples_numpy(const std::string& prog_str,size_t weight, size_t shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();


    SAMPLE::sampler sampler(c.get_num_noise());

    std::vector<QEPG::Row> samplecontainer;

    sampler.generate_many_output_samples(graph,samplecontainer,weight,shots);

    py::array_t<bool>  result;
    result=bitset_rows_to_numpy(samplecontainer);
    return std::move(result);
}





/// @copydoc LERcalculator::return_all_samples_with_fixed_weights
 std::vector<std::vector<bool>> return_all_samples_with_fixed_weights(const std::string& prog_str,const size_t& weight){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();


    SAMPLE::sampler sampler(c.get_num_noise());

    std::vector<QEPG::Row> samplecontainer;

    sampler.generate_all_samples_with_fixed_weight(graph,samplecontainer,weight);


    std::vector<std::vector<bool>> result;
    convert_bitset_row_to_boolean(result,samplecontainer);

    return result;
}


/// @copydoc LERcalculator::return_samples_with_noise_vector
std::pair<std::vector<std::vector<std::pair<int,int>>> ,std::vector<std::vector<bool>>>
return_samples_with_noise_vector(const std::string & prog_str,size_t weight, size_t shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();


    SAMPLE::sampler sampler(c.get_num_noise());

    std::vector<QEPG::Row> samplecontainer;
    std::vector<std::vector<SAMPLE::singlePauli>> noisecontainer;

    sampler.generate_many_output_samples_with_noise_vector(graph,noisecontainer,samplecontainer,weight,shots);


    std::vector<std::vector<bool>> sampleresult;
    convert_bitset_row_to_boolean(sampleresult,samplecontainer);

    std::vector<std::vector<std::pair<int,int>>> noisegenerated;
    noisegenerated.reserve(shots);
    for(std::vector<SAMPLE::singlePauli> tmpnoisevector: noisecontainer){
        std::vector<std::pair<int,int>> outputnoisevector;
        for(SAMPLE::singlePauli tmpnoise: tmpnoisevector){
              outputnoisevector.push_back(std::pair<int,int>{tmpnoise.qindex,tmpnoise.type});
        }
        noisegenerated.push_back(outputnoisevector);
    }

    return std::pair<std::vector<std::vector<std::pair<int,int>>> ,std::vector<std::vector<bool>>>{std::move(noisegenerated),std::move(sampleresult)};
}



/// @copydoc LERcalculator::return_samples_many_weights
std::vector<std::vector<std::vector<bool>>> return_samples_many_weights(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();

    validate_batches(graph, weight, shots);
    SAMPLE::sampler sampler(c.get_num_noise());

    std::vector<QEPG::Row> samplecontainer;
    std::vector<std::vector<bool>> tmpresult;
    tmpresult.reserve(weight.size());
    std::vector<std::vector<std::vector<bool>>> result;
    result.reserve(weight.size());

    for(size_t i=0;i<weight.size();++i){
        samplecontainer.clear();
        tmpresult.clear();
        sampler.generate_many_output_samples(graph,samplecontainer,weight[i],shots[i]);
        convert_bitset_row_to_boolean(tmpresult,samplecontainer);
        result.emplace_back(tmpresult);
    }
    return std::move(result);
}


/// @copydoc LERcalculator::compile_QEPG
QEPG::QEPG compile_QEPG(const std::string& prog_str){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);
    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();
    return std::move(graph);
}


/// @copydoc LERcalculator::return_samples_many_weights_numpy
std::vector<py::array_t<bool>> return_samples_many_weights_numpy(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();

    validate_batches(graph, weight, shots);
    SAMPLE::sampler sampler(c.get_num_noise());

    std::vector<QEPG::Row> samplecontainer;
    std::vector<py::array_t<bool>> result;
    result.reserve(weight.size());

    for(size_t i=0;i<weight.size();++i){
        samplecontainer.clear();
        py::array_t<bool> tmpresult;
        sampler.generate_many_output_samples(graph,samplecontainer,weight[i],shots[i]);
        tmpresult=bitset_rows_to_numpy(samplecontainer);
        result.emplace_back(std::move(tmpresult));
    }
    return std::move(result);
}


/// @copydoc LERcalculator::return_samples_Monte_separate_obs_with_QEPG
std::pair<py::array_t<std::uint8_t>,py::array_t<std::uint8_t>> return_samples_Monte_separate_obs_with_QEPG(const QEPG::QEPG& graph,const double& error_rate, const size_t& shot){
    validate_probability(error_rate);
    validate_output_size(graph.get_total_detector(), shot);
    const std::size_t n_det = graph.get_total_detector();
    SAMPLE::sampler sampler(graph.get_total_noise());

    // Allocate NumPy buffers directly — no intermediate vector<Row>
    py::array_t<std::uint8_t> detectorresult({shot, n_det});
    py::array_t<std::uint8_t> obsresult(shot);

    auto det_info = detectorresult.request();
    auto obs_info = obsresult.request();
    auto* det_ptr = static_cast<std::uint8_t*>(det_info.ptr);
    auto* obs_ptr = static_cast<std::uint8_t*>(obs_info.ptr);

    {
        py::gil_scoped_release release;
        sampler.generate_many_output_samples_Monte_to_numpy(graph, det_ptr, obs_ptr, n_det, error_rate, shot);
    }

    return {std::move(detectorresult), std::move(obsresult)};
}




/// @copydoc LERcalculator::return_samples_many_weights_separate_obs_with_QEPG
std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs_with_QEPG(const QEPG::QEPG& graph,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    const auto shot_sum = validate_batches(graph, weight, shots);
    const auto n_det = graph.get_total_detector();
    SAMPLE::sampler sampler(graph.get_total_noise());
    py::array_t<bool> detectorresult({shot_sum, n_det});
    py::array_t<bool> obsresult(shot_sum);
    auto* det = reinterpret_cast<std::uint8_t*>(detectorresult.mutable_data());
    auto* obs = reinterpret_cast<std::uint8_t*>(obsresult.mutable_data());
    {
        py::gil_scoped_release release;
        std::size_t offset = 0;
        for (std::size_t i = 0; i < weight.size(); ++i) {
            if (shots[i] == 0) continue;
            sampler.generate_many_output_samples_to_numpy(graph, det + offset*n_det,
                obs + offset, n_det, weight[i], shots[i]);
            offset += shots[i];
        }
    }
    return {std::move(detectorresult), std::move(obsresult)};
}



/// @copydoc LERcalculator::return_samples_many_weights_separate_obs
 std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();

    return return_samples_many_weights_separate_obs_with_QEPG(graph, weight, shots);
}




/// @copydoc LERcalculator::return_detector_matrix
std::vector<std::vector<bool>> return_detector_matrix(const std::string& prog_str){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    c.print_circuit();
    graph.backward_graph_construction();
    graph.print_detectorMatrix();
    const std::vector<QEPG::Row>& parityMtrans=graph.get_parityPropMatrixTrans();
    const size_t row_size=parityMtrans.size();
    const size_t col_size=parityMtrans.empty() ? 0 : parityMtrans[0].size();


    // 2. Allocate the whole target matrix in one go
    std::vector<std::vector<bool>> result(row_size,std::vector<bool>(col_size));

    for(size_t row=0;row<row_size;row++){
        for(size_t column=0;column<col_size;column++){
            result[row][column]=parityMtrans[row][column];
        }
    }
    return result;
}


/// @copydoc LERcalculator::return_samples_nonuniform_to_numpy
std::pair<py::array_t<std::uint8_t>, py::array_t<std::uint8_t>>
return_samples_nonuniform_to_numpy(
    const QEPG::QEPG& graph,
    py::array_t<double> noise_probs_arr,
    py::array_t<std::size_t> corr_sources_a_arr,
    py::array_t<std::size_t> corr_sources_b_arr,
    py::array_t<double> corr_probs_arr,
    std::size_t shot)
{
    validate_output_size(graph.get_total_detector(), shot);
    auto probs = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(noise_probs_arr);
    auto sources_a = py::array_t<std::size_t, py::array::c_style | py::array::forcecast>::ensure(corr_sources_a_arr);
    auto sources_b = py::array_t<std::size_t, py::array::c_style | py::array::forcecast>::ensure(corr_sources_b_arr);
    auto probabilities = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(corr_probs_arr);
    if (!probs || !sources_a || !sources_b || !probabilities)
        throw std::invalid_argument("Invalid noise arrays");
    const auto num_noise = graph.get_total_noise();
    if (probs.ndim() != 2 || probs.shape(0) != num_noise || probs.shape(1) != 3)
        throw std::invalid_argument("noise_probs must have shape (graph.num_noise, 3)");
    const double* noise_probs = probs.data();
    for (std::size_t i = 0; i < num_noise; ++i) {
        double sum = 0;
        for (std::size_t j = 0; j < 3; ++j) {
            validate_probability(noise_probs[3*i+j]);
            sum += noise_probs[3*i+j];
        }
        if (sum > 1.0 + 1e-14) throw std::invalid_argument("Pauli probabilities sum to more than one");
    }
    if (sources_a.ndim() != 1 || sources_b.ndim() != 1 || probabilities.ndim() != 1 ||
        sources_a.size() != sources_b.size() || sources_a.size() != probabilities.size())
        throw std::invalid_argument("Correlated arrays must be one-dimensional with equal lengths");
    const std::size_t num_corr = sources_a.size();
    std::vector<SAMPLE::CorrelatedPair> corr_pairs(num_corr);
    for (std::size_t i = 0; i < num_corr; ++i) {
        const auto a = sources_a.data()[i], b = sources_b.data()[i];
        if (a >= num_noise || b >= num_noise || a == b)
            throw std::invalid_argument("Correlated source indices must be distinct and in range");
        validate_probability(probabilities.data()[i]);
        corr_pairs[i] = {a, b, probabilities.data()[i]};
    }

    // Allocate output numpy arrays
    const std::size_t n_det = graph.get_total_detector();
    py::array_t<std::uint8_t> det_result({shot, n_det});
    py::array_t<std::uint8_t> obs_result(shot);

    auto det_info = det_result.request();
    auto obs_info = obs_result.request();
    auto* det_buf = static_cast<std::uint8_t*>(det_info.ptr);
    auto* obs_buf = static_cast<std::uint8_t*>(obs_info.ptr);

    // Run the sampler
    SAMPLE::sampler sampler(graph.get_total_noise());
    {
        py::gil_scoped_release release;
        sampler.generate_many_output_samples_nonuniform_to_numpy(
            graph, det_buf, obs_buf, n_det,
            noise_probs, num_noise,
            num_corr > 0 ? corr_pairs.data() : nullptr, num_corr,
            shot);
    }

    return {std::move(det_result), std::move(obs_result)};
}


}
