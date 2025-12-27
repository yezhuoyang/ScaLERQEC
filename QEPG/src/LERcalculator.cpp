#include "LERcalculator.hpp"




namespace LERcalculator{




void convert_bitset_row_to_boolean(std::vector<std::vector<bool>>& result,const std::vector<QEPG::Row>& samplecontainer){
        result.reserve(samplecontainer.size()); // Reserve space

        // Convert each boost::dynamic_bitset<> to std::vector<bool>
        for (const auto& bitset_row : samplecontainer) {
            std::vector<bool> bool_row(bitset_row.size());
            for (size_t i = 0; i < bitset_row.size(); ++i) {
                bool_row[i] = bitset_row[i]; // Access individual bits
            }
            result.push_back(bool_row);
        }
}




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
        boost::to_block_range(bits, tl_buf.begin());        // fill the buffer

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





inline void convert_bitset_row_to_boolean_separate_obs(std::vector<std::vector<bool>>& result,std::vector<bool>& obsresult,const std::vector<QEPG::Row>& samplecontainer){
        result.reserve(samplecontainer.size()); // Reserve space
        obsresult.reserve(samplecontainer.size());
        // Convert each boost::dynamic_bitset<> to std::vector<bool>
        for (const auto& bitset_row : samplecontainer) {
            std::vector<bool> bool_row(bitset_row.size()-1);
            for (size_t i = 0; i < bitset_row.size()-1; ++i) {
                bool_row[i] = bitset_row[i]; // Access individual bits
            }
            result.push_back(bool_row);
            obsresult.push_back(bitset_row[bitset_row.size()-1]);
        }
}



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

    #pragma omp parallel                                                \
        default(none) shared(samplecontainer, det_base, obs_base)
    {
        static thread_local std::vector<block_t> tl_buf;   // scratch per thread

        #pragma omp for schedule(static)
        for (long long r = 0; r < static_cast<long long>(n_rows); ++r)
        {
            const bitset_t& bits   = samplecontainer[static_cast<std::size_t>(r)];
            const std::size_t n_blk = bits.num_blocks();

            /* -- obtain packed words ---------------------------------- */
            tl_buf.resize(n_blk);
            boost::to_block_range(bits, tl_buf.begin());

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



std::vector<std::vector<bool>> return_samples_with_fixed_QEPG(const QEPG::QEPG& graph,size_t weight, size_t shots){
    SAMPLE::sampler sampler(graph.get_total_noise());
    std::vector<QEPG::Row> samplecontainer;
    sampler.generate_many_output_samples(graph,samplecontainer,weight,shots);
    std::vector<std::vector<bool>> result;
    convert_bitset_row_to_boolean(result,samplecontainer);
    return std::move(result);
}






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



std::vector<std::vector<std::vector<bool>>> return_samples_many_weights(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();

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


QEPG::QEPG compile_QEPG(const std::string& prog_str){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);
    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();
    return std::move(graph);
}


std::vector<py::array_t<bool>> return_samples_many_weights_numpy(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();

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


std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_Monte_separate_obs_with_QEPG(const QEPG::QEPG& graph,const double& error_rate, const size_t& shot){
    SAMPLE::sampler sampler(graph.get_total_noise());
    std::vector<QEPG::Row> samplecontainer;
    py::array_t<bool> detectorresult({shot,graph.get_total_detector()});
    py::array_t<bool> obsresult(shot);
    sampler.generate_many_output_samples_Monte(graph,samplecontainer,error_rate,shot);
    convert_bitset_row_to_boolean_separate_obs_numpy(detectorresult,obsresult,0,samplecontainer);
    return std::pair<py::array_t<bool>,py::array_t<bool>>{std::move(detectorresult),std::move(obsresult)};
}




std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs_with_QEPG(const QEPG::QEPG& graph,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    SAMPLE::sampler sampler(graph.get_total_noise());
    std::vector<QEPG::Row> samplecontainer;
    size_t shot_sum=0;
    for(int i=0;i<weight.size();i++){
        shot_sum+=shots[i];
    }
    py::array_t<bool> detectorresult({shot_sum,graph.get_total_detector()});
    py::array_t<bool> obsresult(shot_sum);
    auto begin_index=0;
    for(size_t i=0;i<weight.size();++i){
        samplecontainer.clear();
        sampler.generate_many_output_samples(graph,samplecontainer,weight[i],shots[i]);
        convert_bitset_row_to_boolean_separate_obs_numpy(detectorresult,obsresult,begin_index,samplecontainer);
        begin_index+=shots[i];
    }
    return std::pair<py::array_t<bool>,py::array_t<bool>>{std::move(detectorresult),std::move(obsresult)};
}


std::pair<py::array_t<bool>, py::array_t<bool>>
return_samples_many_weights_separate_obs_with_QEPG_cuda(
    const QEPG::QEPG&          graph,
    const std::vector<size_t>& weight,
    const std::vector<size_t>& shots)
{
    if (weight.size() != shots.size()) {
        throw std::runtime_error("weight.size() != shots.size()");
    }

    // Total number of shots
    size_t shot_sum = 0;
    for (size_t i = 0; i < shots.size(); ++i) {
        shot_sum += shots[i];
    }
    if (shot_sum == 0) {
        // Return empty arrays with correct shapes
        py::array_t<bool> empty_det({
            py::ssize_t(0),
            py::ssize_t(graph.get_total_detector())
        });

        py::array_t<bool> empty_obs(0);
        return {std::move(empty_det), std::move(empty_obs)};
    }

    const size_t num_det = graph.get_total_detector();

    // Allocate Python arrays (C-contiguous, row-major)
    py::array_t<bool> detectorresult({shot_sum, num_det});
    py::array_t<bool> obsresult(shot_sum);

    auto det_info = detectorresult.request();
    auto obs_info = obsresult.request();

    if (det_info.ndim != 2 ||
        static_cast<std::size_t>(det_info.shape[0]) != shot_sum ||
        static_cast<std::size_t>(det_info.shape[1]) != num_det)
    {
        throw std::runtime_error("detectorresult has unexpected shape/strides");
    }
    if (obs_info.ndim != 1 ||
        static_cast<std::size_t>(obs_info.shape[0]) != shot_sum)
    {
        throw std::runtime_error("obsresult has unexpected shape/strides");
    }

    // Build per-shot weight array (explode by weights and group shots)
    std::vector<std::size_t> per_shot_weight;
    per_shot_weight.reserve(shot_sum);
    for (size_t i = 0; i < weight.size(); ++i) {
        for (size_t s = 0; s < shots[i]; ++s) {
            per_shot_weight.push_back(weight[i]);
        }
    }

    // Underlying py::array_t<bool> storage is byte-addressable
    auto* det_ptr = static_cast<unsigned char*>(det_info.ptr);
    auto* obs_ptr = static_cast<unsigned char*>(obs_info.ptr);

    {
        // Release GIL while doing heavy GPU work
        py::gil_scoped_release release;

        // Build device graph and run CUDA sampler
        auto dgraph = cuda_accel::make_device_graph(graph);
        try {
            cuda_accel::sample_many_weights_separate_obs(
                dgraph,
                per_shot_weight,
                det_ptr,
                obs_ptr);
        } catch (...) {
            cuda_accel::free_device_graph(dgraph);
            throw;
        }
        cuda_accel::free_device_graph(dgraph);
    }

    return {std::move(detectorresult), std::move(obsresult)};
}




 std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();

    SAMPLE::sampler sampler(c.get_num_noise());


    std::vector<QEPG::Row> samplecontainer;

    size_t shot_sum=0;
    for(int i=0;i<weight.size();i++){
        shot_sum+=shots[i];
    }

    py::array_t<bool> detectorresult({shot_sum,c.get_num_detector()});
    py::array_t<bool> obsresult(shot_sum);

    auto begin_index=0;
    for(size_t i=0;i<weight.size();++i){
        samplecontainer.clear();
        sampler.generate_many_output_samples(graph,samplecontainer,weight[i],shots[i]);
        convert_bitset_row_to_boolean_separate_obs_numpy(detectorresult,obsresult,begin_index,samplecontainer);
        begin_index+=shots[i];
    }
    return std::pair<py::array_t<bool>,py::array_t<bool>>{std::move(detectorresult),std::move(obsresult)};
}




std::vector<std::vector<bool>> return_detector_matrix(const std::string& prog_str){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    c.print_circuit();
    graph.backward_graph_construction();
    graph.print_detectorMatrix();
    const std::vector<QEPG::Row>& parityMtrans=graph.get_parityPropMatrixTrans();
    const size_t row_size=parityMtrans.size();
    const size_t col_size=parityMtrans[0].size();


    // 2. Allocate the whole target matrix in one go
    std::vector<std::vector<bool>> result(row_size,std::vector<bool>(col_size));

    for(size_t row=0;row<row_size;row++){
        for(size_t column=0;column<col_size;column++){
            result[row][column]=parityMtrans[row][column];
        }
    }
    return result;
}


double calculate_LER_Monte_with_decoder(
    const std::string& prog_str,
    const double&      error_rate,
    const std::size_t  shots)
{
    // Compile Stim-style string into Clifford circuit and QEPG graph
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c, c.get_num_detector(), c.get_num_noise());
    graph.backward_graph_construction();

    return calculate_LER_Monte_with_decoder_QEPG(graph, error_rate, shots);
}



double calculate_LER_Monte_with_decoder_QEPG(
    const QEPG::QEPG& graph,
    const double&      error_rate,
    const std::size_t  shots)
{
    if (shots == 0) {
        return 0.0;
    }

    const std::size_t num_det   = graph.get_total_detector();
    const std::size_t total_err = graph.get_total_noise();

    if (num_det == 0 || total_err == 0) {
        throw std::runtime_error(
            "calculate_LER_Monte_with_decoder_QEPG: graph has zero detectors or noise terms");
    }

    // 1) Sample detector+observable outcomes using your existing Monte Carlo sampler.
    SAMPLE::sampler sampler(total_err);
    std::vector<QEPG::Row> samplecontainer;
    samplecontainer.reserve(shots);

    sampler.generate_many_output_samples_Monte(
        graph,
        samplecontainer,
        error_rate,
        shots);

    if (samplecontainer.empty()) {
        return 0.0;
    }

    // Sanity check: each row should be [num_det detector bits | 1 logical bit]
    const std::size_t expected_row_size = num_det + 1;
    if (samplecontainer.front().size() != expected_row_size) {
        throw std::runtime_error(
            "calculate_LER_Monte_with_decoder_QEPG: sample row size != num_det + 1");
    }

    // 2) For each shot:
    //    - extract detector syndrome (first num_det bits)
    //    - extract true logical measurement (last bit)
    //    - run decoder on the syndrome
    //    - compare decoded logical value with true logical value
    std::size_t logical_failures = 0;

    QEPG::Row det_syndrome(num_det);   // reusable buffer

    for (std::size_t s = 0; s < samplecontainer.size(); ++s) {
        const QEPG::Row& full_row = samplecontainer[s];

        // Copy detector bits into a length-num_det bitset
        for (std::size_t j = 0; j < num_det; ++j) {
            det_syndrome[j] = full_row.test(j);
        }

        const bool true_logical = full_row.test(num_det);

        // Decode using the new perfect-matching-style decoder
        QEPG::PMDecodedResult decoded =
            QEPG::decode_perfect_matching(graph, det_syndrome);

        const bool decoded_logical = decoded.logical_flip;

        if (decoded_logical != true_logical) {
            ++logical_failures;
        }
    }

    return static_cast<double>(logical_failures)
         / static_cast<double>(shots);
}





}