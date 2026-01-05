#ifndef SAMPLE_HPP
#define SAMPLE_HPP
#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>
#include <random>
#include <unordered_set>
#include "QEPG.hpp"
#include "simd_matrix.hpp"
#include <omp.h>

namespace SAMPLE{



const size_t PAULIX = 1;
const size_t PAULIY = 2;
const size_t PAULIZ = 3;

struct singlePauli{
    size_t qindex;
    size_t type;
};






class sampler{


    public:
    
    /*---------------------------------------ctor----------*/
        sampler();
        explicit sampler(size_t num_total_paulierror);
        ~sampler();


    /*---------------------------------------Sample one vector with fixed weight----------*/        

        /*
        Generate a single sample with weight error by Floyd method. 
        */
        inline std::vector<singlePauli> generate_sample_Floyd(size_t weight,std::mt19937& gen);

        inline std::vector<singlePauli> generate_sample_Monte(double error_prob ,size_t ErrorSize,std::mt19937& gen);

        inline QEPG::Row calculate_parity_output_from_one_sample(const QEPG::QEPG& graph,const std::vector<singlePauli>& sample){
            const auto&dm=graph.get_parityPropMatrixTrans();
            const std::size_t n_rows=dm.size();
            const std::size_t n_noise=int(n_rows/3);
            const std::size_t n_cols=n_rows ? dm[0].size():0;
            QEPG::Row result(n_cols);
            for(singlePauli noise: sample){
                size_t pos=noise.qindex;
                size_t type=noise.type;
                if(type==SAMPLE::PAULIX){
                    result^=dm[pos];
                }
                else if(type==SAMPLE::PAULIY){
                    result^=dm[pos+n_noise];
                }
                else if(type==SAMPLE::PAULIZ){
                    result^=dm[pos+n_noise*2];
                }
            }
            return result;
        }


        void generate_many_output_samples(const QEPG::QEPG& graph,std::vector<QEPG::Row>& samplecontainer,size_t pauliweight , size_t samplenumber);

        void generate_many_output_samples_Monte(const QEPG::QEPG& graph,std::vector<QEPG::Row>& samplecontainer,double error_prob, size_t samplenumber);

        void generate_many_output_samples_with_noise_vector(const QEPG::QEPG& graph,std::vector<std::vector<singlePauli>>& noisecontainer,std::vector<QEPG::Row>& samplecontainer, size_t pauliweight, size_t samplenumber);

    /*---------------------------------------SIMD-accelerated sampling methods----------*/

        /**
         * @brief SIMD-accelerated fixed-weight sampling (OpenMP + AVX2)
         * @param graph The QEPG graph
         * @param samplecontainer Output container for samples
         * @param pauliweight Number of Pauli errors per sample
         * @param samplenumber Number of samples to generate
         */
        void generate_many_output_samples_simd(
            const QEPG::QEPG& graph,
            std::vector<QEPG::Row>& samplecontainer,
            size_t pauliweight,
            size_t samplenumber);

        /**
         * @brief SIMD-accelerated Monte Carlo sampling (OpenMP + AVX2)
         * @param graph The QEPG graph
         * @param samplecontainer Output container for samples
         * @param error_prob Error probability per noise location
         * @param samplenumber Number of samples to generate
         */
        void generate_many_output_samples_Monte_simd(
            const QEPG::QEPG& graph,
            std::vector<QEPG::Row>& samplecontainer,
            double error_prob,
            size_t samplenumber);

    /*---------------------------------------Optimized SIMD methods (buffer-only, no bitset)----------*/

        /**
         * @brief Highly optimized SIMD sampling that writes directly to output buffers
         * @param simd_matrix Pre-built SIMD matrix (caller should reuse for multiple calls)
         * @param det_output Output buffer for detector results (row-major, 1 byte per bit)
         * @param obs_output Output buffer for observable results (1 byte per sample)
         * @param det_row_stride Stride between detector rows in bytes
         * @param n_det Number of detectors
         * @param n_noise Number of noise locations
         * @param pauliweight Number of Pauli errors per sample
         * @param samplenumber Number of samples to generate
         * @param begin_index Starting index in output buffers
         */
        void generate_samples_to_buffer_simd(
            const SIMD::SIMDMatrix& simd_matrix,
            uint8_t* det_output,
            uint8_t* obs_output,
            size_t det_row_stride,
            size_t n_det,
            size_t n_noise,
            size_t pauliweight,
            size_t samplenumber,
            size_t begin_index);

        /**
         * @brief Highly optimized Monte Carlo SIMD sampling that writes directly to output buffers
         */
        void generate_samples_Monte_to_buffer_simd(
            const SIMD::SIMDMatrix& simd_matrix,
            uint8_t* det_output,
            uint8_t* obs_output,
            size_t det_row_stride,
            size_t n_det,
            size_t n_noise,
            double error_prob,
            size_t samplenumber,
            size_t begin_index);

    private:

        size_t     num_total_pauliError_;

        /**
         * @brief SIMD kernel: calculate parity using aligned matrix
         * @param result Aligned result buffer
         * @param matrix SIMD-aligned parity matrix
         * @param sample Vector of Pauli errors
         * @param n_noise Number of noise locations
         */
        inline void calculate_parity_simd(
            uint64_t* result,
            const SIMD::SIMDMatrix& matrix,
            const std::vector<singlePauli>& sample,
            size_t n_noise);

        /**
         * @brief Convert aligned uint64_t buffer to boost::dynamic_bitset
         * @param src Aligned source buffer
         * @param words Number of uint64_t words
         * @param bits Number of actual bits
         * @return boost::dynamic_bitset with the result
         */
        inline QEPG::Row buffer_to_bitset(const uint64_t* src, size_t words, size_t bits);

        /**
         * @brief Optimized sample generation using xorshift128+ for faster RNG
         * @param positions Output array for error positions
         * @param types Output array for error types (1=X, 2=Y, 3=Z)
         * @param weight Number of errors to generate
         * @param n_noise Total number of noise locations
         * @param rng_state Two uint64_t values for xorshift state
         */
        inline void generate_sample_fast(
            size_t* positions,
            uint8_t* types,
            size_t weight,
            size_t n_noise,
            uint64_t* rng_state);

        /**
         * @brief Optimized Monte Carlo sample generation
         * @param positions Output array for error positions (pre-allocated to n_noise)
         * @param types Output array for error types
         * @param n_noise Total number of noise locations
         * @param error_prob Error probability
         * @param rng_state RNG state
         * @return Number of errors generated
         */
        inline size_t generate_sample_Monte_fast(
            size_t* positions,
            uint8_t* types,
            size_t n_noise,
            double error_prob,
            uint64_t* rng_state);

        /**
         * @brief Unpack uint64_t buffer to byte array for detector output
         */
        inline void unpack_buffer_to_bytes(
            const uint64_t* src,
            uint8_t* dst,
            size_t n_bits);



};




}
#endif




