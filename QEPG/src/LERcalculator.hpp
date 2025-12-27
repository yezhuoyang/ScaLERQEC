#ifndef LER_HPP
#define LER_HPP
#pragma once

#include "QEPG.hpp"
#include "sampler.hpp"
#include "cuda_accel.hpp"
#include "decoder.hpp"
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>          // <-- defines py::array_t
namespace LERcalculator{

namespace py = pybind11;

 struct noisesample{
      int type;
      size_t position;
 };

 std::vector<std::vector<bool>> return_samples(const std::string & prog_str,size_t weight, size_t shots);

 std::vector<std::vector<bool>> return_samples_with_fixed_QEPG(const QEPG::QEPG& graph,size_t weight, size_t shots);



 std::pair<std::vector<std::vector<std::pair<int,int>>> ,std::vector<std::vector<bool>>>  return_samples_with_noise_vector(const std::string & prog_str,size_t weight, size_t shots);


 std::vector<std::vector<std::vector<bool>>> return_samples_many_weights(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots);


 
 std::vector<py::array_t<bool>> return_samples_many_weights_numpy(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots);

 std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots);


 std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs_with_QEPG(const QEPG::QEPG& graph,const std::vector<size_t>& weight, const std::vector<size_t>& shots);


 std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs_with_QEPG_cuda(const QEPG::QEPG& graph,const std::vector<size_t>& weight, const std::vector<size_t>& shots);

 std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_Monte_separate_obs_with_QEPG(const QEPG::QEPG& graph,const double& error_rate, const size_t& shot);


 std::vector<std::vector<bool>> return_all_samples_with_fixed_weights(const std::string& prog_str,const size_t& weight);


 std::vector<std::vector<bool>> return_detector_matrix(const std::string& prog_str);


 py::array_t<bool> return_samples_numpy(const std::string& prog_str,size_t weight, size_t shots);


 QEPG::QEPG compile_QEPG(const std::string& prog_str);


double calculate_LER_Monte_with_decoder_QEPG(
    const QEPG::QEPG& graph,
    const double&      error_rate,
    const std::size_t  shots);

double calculate_LER_Monte_with_decoder(
    const std::string& prog_str,
    const double&      error_rate,
    const std::size_t  shots);


}


#endif