#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     
#include <pybind11/operators.h>

#include "src/clifford.hpp"
#include "src/QEPG.hpp"
#include "src/sampler.hpp"
#include "src/LERcalculator.hpp"
#include <boost/dynamic_bitset.hpp>

namespace py = pybind11;


namespace SAMPLE {
    class sampler; // Forward declare if binding methods/ctors
}
// Declare other classes if needed for binding their members
namespace clifford {
    class cliffordcircuit;
}
namespace QEPG {
    class QEPG;
}


namespace LERcalculator{
    std::vector<std::vector<bool>> return_samples(const std::string& prog_str, size_t weight, size_t shots);
    std::vector<std::vector<std::vector<bool>>> return_samples_many_weights(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots);
    std::vector<std::vector<bool>> return_detector_matrix(const std::string& prog_str);
    std::pair<std::vector<std::vector<std::pair<int,int>>> ,std::vector<std::vector<bool>>>  return_samples_with_noise_vector(const std::string & prog_str,size_t weight, size_t shots);
    std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots);
    py::array_t<bool> return_samples_numpy(const std::string& prog_str,size_t weight, size_t shots);
    std::vector<py::array_t<bool>> return_samples_many_weights_numpy(const std::string& prog_str,const std::vector<size_t>& weight, const std::vector<size_t>& shots);
    QEPG::QEPG compile_QEPG(const std::string& prog_str);
    std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs_with_QEPG(const QEPG::QEPG& graph,const std::vector<size_t>& weight, const std::vector<size_t>& shots);
    std::vector<std::vector<bool>> return_samples_with_fixed_QEPG(const QEPG::QEPG& graph,size_t weight, size_t shots);
    std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_Monte_separate_obs_with_QEPG(const QEPG::QEPG& graph,const double& error_rate, const size_t& shot);
    // SIMD-accelerated functions
    std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_many_weights_separate_obs_with_QEPG_simd(const QEPG::QEPG& graph,const std::vector<size_t>& weight, const std::vector<size_t>& shots);
    std::pair<py::array_t<bool>,py::array_t<bool>> return_samples_Monte_separate_obs_with_QEPG_simd(const QEPG::QEPG& graph,const double& error_rate, const size_t& shot);
}
   

// --- Bindings ---
PYBIND11_MODULE(qepg, m) {
    m.doc() = "Pybind11 bindings for QEPG library";

    // 1. Bind boost::dynamic_bitset
    py::class_<boost::dynamic_bitset<>>(m, "DynamicBitset") // Python name for the bitset type

        .def("size", &boost::dynamic_bitset<>::size, "Get the size of the bitset")
        .def("test", &boost::dynamic_bitset<>::test, py::arg("pos"), "Test if the bit at position pos is set")
        .def(py::self == py::self)

        // Add conversion to Python list for easier access
        .def("to_list", [](const boost::dynamic_bitset<>& self) {
             std::vector<bool> list(self.size());
             for(size_t i=0; i<self.size(); ++i) list[i] = self[i];
             return list;
         }, "Convert the bitset to a list of booleans")

        // Add a useful string representation
        .def("__repr__", [](const boost::dynamic_bitset<>& self) {
            std::string s = "<DynamicBitset ";
            if (self.size() > 40) { // Truncate long outputs
                 for(size_t i=0; i<20; ++i) s += (i < self.size() ? (self[i] ? '1' : '0') : '-');
                 s += "...";
                 for(size_t i=self.size()-20; i<self.size(); ++i) s += (self[i] ? '1' : '0');
             } else { // Show full bitstring for small bitsets
                 for(size_t i=0; i<self.size(); ++i) s += (self[i] ? '1' : '0');
             }
            s += ">";
            return s;
        })
        ;

    // 2. Bind clifford::cliffordcircuit class
    py::class_<clifford::cliffordcircuit>(m, "CliffordCircuit")
        .def(py::init<>()) // Default constructor
        .def("compile_from_rewrited_stim_string", &clifford::cliffordcircuit::compile_from_rewrited_stim_string,
             py::arg("prog_str"), "Compile circuit from Stim string")
        // Expose getters used in return_samples or potentially useful in Python
        .def("get_num_detector", &clifford::cliffordcircuit::get_num_detector)
        .def("get_num_noise", &clifford::cliffordcircuit::get_num_noise)
        .def("get_num_qubit", &clifford::cliffordcircuit::get_num_qubit)
        // Bind other methods of CliffordCircuit if needed
        ;

    // 3. Bind QEPG::QEPG class
    py::class_<QEPG::QEPG>(m, "QEPGGraph")
        .def(py::init<clifford::cliffordcircuit, size_t, size_t>(), // Takes a CliffordCircuit object
             py::arg("circuit"), py::arg("num_detector"), py::arg("num_noise"))
        .def("backward_graph_construction", &QEPG::QEPG::backward_graph_construction)
        ;

    // 4. Bind SAMPLE::sampler class
    py::class_<SAMPLE::sampler>(m, "Sampler")
        .def(py::init<size_t>(), py::arg("num_total_paulierror"))
        ;

    m.def("return_samples", &LERcalculator::return_samples,
          py::arg("prog_str"), py::arg("weight"), py::arg("shots"),
          py::return_value_policy::move,
          "Function that returns samples based on a circuit and parameters");

    m.def("return_samples_with_fixed_QEPG", &LERcalculator::return_samples_with_fixed_QEPG, // Use &SAMPLE::return_samples
          py::arg("graph"), py::arg("weight"), py::arg("shots"),
          py::return_value_policy::move,
          "Function that returns samples based on a QEPG");



    m.def("return_samples_Monte_separate_obs_with_QEPG",&LERcalculator::return_samples_Monte_separate_obs_with_QEPG,
        py::arg("graph"), py::arg("error_rate"),py::arg("shot"),
        py::return_value_policy::move,       
        "Function that returns samples based on a QEPG with monte carlo method");



    m.def("return_samples_numpy", &LERcalculator::return_samples_numpy, 
          py::arg("prog_str"), py::arg("weight"), py::arg("shots"),
          py::return_value_policy::move,
          "Function that directly return numpy array");

    m.def("return_samples_many_weights", &LERcalculator::return_samples_many_weights, // Use &SAMPLE::return_samples
        py::arg("prog_str"), py::arg("weight"), py::arg("shots"),
        py::return_value_policy::move,
        "Function that returns samples of a list of weights based on a circuit and parameters");

        
    m.def("return_samples_many_weights_numpy", &LERcalculator::return_samples_many_weights_numpy, // Use &SAMPLE::return_samples
        py::arg("prog_str"), py::arg("weight"), py::arg("shots"),
        py::return_value_policy::move,
        "Function that returns samples of a list of weights based on a circuit and parameters, it return numpy vector directly");


    m.def("return_detector_matrix", &LERcalculator::return_detector_matrix, // Use &SAMPLE::return_detector_matrix
          py::arg("prog_str"),
          "Function that returns the detector matrix");

    m.def("return_samples_with_noise_vector",
        &LERcalculator::return_samples_with_noise_vector,
        py::arg("prog_str"), py::arg("weight"), py::arg("shots"),
        py::return_value_policy::move);   // avoid an extra copy on the Python side

    m.def("return_samples_many_weights_separate_obs",
        &LERcalculator::return_samples_many_weights_separate_obs,
        py::arg("prog_str"), py::arg("weight"), py::arg("shots"),
        py::return_value_policy::move);   // avoid an extra copy on the Python side

    m.def(
        "compile_QEPG",
        &LERcalculator::compile_QEPG,
        py::arg("prog_str"),
        R"pbdoc(
            compile_QEPG(prog_str: str) → QEPGGraph
            Parse a Stim‐style program string into a QEPGGraph object,
            run its backward_graph_construction() pass, and return it.
        )pbdoc",
        py::return_value_policy::move  // make sure we move the returned graph into Python
    );
    

    m.def("return_samples_many_weights_separate_obs_with_QEPG",
        &LERcalculator::return_samples_many_weights_separate_obs_with_QEPG,
        py::arg("graph"), py::arg("weight"), py::arg("shots"),
        py::return_value_policy::move);   // avoid an extra copy on the Python side

    // SIMD-accelerated sampling functions
    m.def("return_samples_many_weights_separate_obs_with_QEPG_simd",
        &LERcalculator::return_samples_many_weights_separate_obs_with_QEPG_simd,
        py::arg("graph"), py::arg("weight"), py::arg("shots"),
        py::return_value_policy::move,
        "SIMD-accelerated version of return_samples_many_weights_separate_obs_with_QEPG");

    m.def("return_samples_Monte_separate_obs_with_QEPG_simd",
        &LERcalculator::return_samples_Monte_separate_obs_with_QEPG_simd,
        py::arg("graph"), py::arg("error_rate"), py::arg("shot"),
        py::return_value_policy::move,
        "SIMD-accelerated version of return_samples_Monte_separate_obs_with_QEPG");

}