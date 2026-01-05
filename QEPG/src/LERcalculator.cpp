#include "LERcalculator.hpp"
#include "simd_matrix.hpp"




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

    #pragma omp parallel
    {
        // Thread-local scratch buffer for unpacking bitset blocks
        std::vector<block_t> tl_buf;

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
    for(size_t i=0;i<weight.size();i++){
        shot_sum+=shots[i];
    }
    py::array_t<bool> detectorresult({shot_sum,graph.get_total_detector()});
    py::array_t<bool> obsresult(shot_sum);
    size_t begin_index=0;
    for(size_t i=0;i<weight.size();++i){
        samplecontainer.clear();
        sampler.generate_many_output_samples(graph,samplecontainer,weight[i],shots[i]);
        convert_bitset_row_to_boolean_separate_obs_numpy(detectorresult,obsresult,begin_index,samplecontainer);
        begin_index+=shots[i];
    }
    return std::pair<py::array_t<bool>,py::array_t<bool>>{std::move(detectorresult),std::move(obsresult)};
}



 std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    clifford::cliffordcircuit c;
    c.compile_from_rewrited_stim_string(prog_str);

    QEPG::QEPG graph(c,c.get_num_detector(),c.get_num_noise());
    graph.backward_graph_construction();

    SAMPLE::sampler sampler(c.get_num_noise());


    std::vector<QEPG::Row> samplecontainer;

    size_t shot_sum=0;
    for(size_t i=0;i<weight.size();i++){
        shot_sum+=shots[i];
    }

    py::array_t<bool> detectorresult({shot_sum,c.get_num_detector()});
    py::array_t<bool> obsresult(shot_sum);

    size_t begin_index=0;
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


/*---------------------------------------SIMD-accelerated sampling functions----------*/

std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs_with_QEPG_simd(const QEPG::QEPG& graph,const std::vector<size_t>& weight, const std::vector<size_t>& shots){
    // Get matrix dimensions
    const auto& dm = graph.get_parityPropMatrixTrans();
    if (dm.empty()) {
        return std::pair<py::array_t<bool>,py::array_t<bool>>{
            py::array_t<bool>({size_t(0), graph.get_total_detector()}),
            py::array_t<bool>(size_t(0))
        };
    }

    const size_t n_rows = dm.size();
    const size_t n_noise = n_rows / 3;
    const size_t n_det = graph.get_total_detector();
    const size_t n_cols = n_det + 1;  // detectors + observable

    // Calculate total shots
    size_t shot_sum = 0;
    for (size_t i = 0; i < weight.size(); i++) {
        shot_sum += shots[i];
    }

    // Allocate output arrays
    py::array_t<bool> detectorresult({shot_sum, n_det});
    py::array_t<bool> obsresult(shot_sum);

    // Get raw pointers
    auto det_info = detectorresult.request();
    auto obs_info = obsresult.request();
    uint8_t* det_ptr = static_cast<uint8_t*>(det_info.ptr);
    uint8_t* obs_ptr = static_cast<uint8_t*>(obs_info.ptr);
    const size_t det_row_stride = static_cast<size_t>(det_info.strides[0]);

    // Build SIMD matrix once (shared across all weight batches)
    SIMD::SIMDMatrix simd_matrix(n_rows, n_cols);
    simd_matrix.copy_from_bitset_matrix(dm);

    // Create sampler
    SAMPLE::sampler sampler(n_noise);

    // Release GIL for parallel computation
    py::gil_scoped_release release;

    // Process each weight batch
    size_t begin_index = 0;
    for (size_t i = 0; i < weight.size(); ++i) {
        sampler.generate_samples_to_buffer_simd(
            simd_matrix,
            det_ptr,
            obs_ptr,
            det_row_stride,
            n_det,
            n_noise,
            weight[i],
            shots[i],
            begin_index
        );
        begin_index += shots[i];
    }

    return std::pair<py::array_t<bool>,py::array_t<bool>>{std::move(detectorresult),std::move(obsresult)};
}


std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_Monte_separate_obs_with_QEPG_simd(const QEPG::QEPG& graph,const double& error_rate, const size_t& shot){
    // Get matrix dimensions
    const auto& dm = graph.get_parityPropMatrixTrans();
    if (dm.empty()) {
        return std::pair<py::array_t<bool>,py::array_t<bool>>{
            py::array_t<bool>({size_t(0), graph.get_total_detector()}),
            py::array_t<bool>(size_t(0))
        };
    }

    const size_t n_rows = dm.size();
    const size_t n_noise = n_rows / 3;
    const size_t n_det = graph.get_total_detector();
    const size_t n_cols = n_det + 1;

    // Allocate output arrays
    py::array_t<bool> detectorresult({shot, n_det});
    py::array_t<bool> obsresult(shot);

    // Get raw pointers
    auto det_info = detectorresult.request();
    auto obs_info = obsresult.request();
    uint8_t* det_ptr = static_cast<uint8_t*>(det_info.ptr);
    uint8_t* obs_ptr = static_cast<uint8_t*>(obs_info.ptr);
    const size_t det_row_stride = static_cast<size_t>(det_info.strides[0]);

    // Build SIMD matrix
    SIMD::SIMDMatrix simd_matrix(n_rows, n_cols);
    simd_matrix.copy_from_bitset_matrix(dm);

    // Create sampler
    SAMPLE::sampler sampler(n_noise);

    // Release GIL for parallel computation
    py::gil_scoped_release release;

    // Generate samples directly to output buffer
    sampler.generate_samples_Monte_to_buffer_simd(
        simd_matrix,
        det_ptr,
        obs_ptr,
        det_row_stride,
        n_det,
        n_noise,
        error_rate,
        shot,
        0  // begin_index
    );

    return std::pair<py::array_t<bool>,py::array_t<bool>>{std::move(detectorresult),std::move(obsresult)};
}


}