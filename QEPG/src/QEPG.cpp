/**
 * @file QEPG.cpp
 * @brief Implementation of the QEPG backward error propagation graph construction.
 *
 * Contains the core backward_graph_construction() algorithm that traverses
 * the Clifford circuit in reverse to build the parity propagation matrix,
 * as well as matrix utility functions (transpose, multiplication over GF(2))
 * and the older compute_parityPropMatrix() path.
 */

#include "QEPG.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <unordered_set>

namespace QEPG{

/*------------------------------------------ctor----*/

QEPG::QEPG(){

}

QEPG::QEPG(const clifford::cliffordcircuit& othercircuit, size_t total_detectors, size_t total_noise):
                    circuit_(othercircuit),
                    total_detectors_(total_detectors),
                    total_noise_(total_noise),
                    parityPropMatrixTranspose_(3* total_noise_,Row(total_detectors+1))
                    {

}


QEPG::~QEPG(){

}



const size_t& QEPG::get_total_noise() const noexcept{
    return total_noise_;
}

const size_t& QEPG::get_total_detector() const noexcept{
    return total_detectors_;
}



/*--------------------print QEPG graph---------------------------------------*/



/// @brief Print detector matrix, its transpose, and parity propagation matrix transpose to stdout.
void QEPG::print_detectorMatrix(char zero, char one) const{

    std::cout<<"-----------detectorMatrix:----------------------\n";
    for(const auto& row:detectorMatrix_){
        for(std::size_t c=0;c<row.size();++c){
            std::cout<<(row.test(c)? one:zero);
        }
        std::cout<<"\n";
    }
    std::cout<<"-----------detectorMatrix(Transpose):----------------------\n";
    for(const auto& row:detectorMatrixTranspose_){
        for(std::size_t c=0;c<row.size();++c){
            std::cout<<(row.test(c)? one:zero);
        }
        std::cout<<"\n";
    }

    std::cout<<"-----------ParitygroupMatrixTranspose:----------------------\n";
    for(const auto& row:parityPropMatrixTranspose_){
        for(std::size_t c=0;c<row.size();++c){
            std::cout<<(row.test(c)? one:zero);
        }
        std::cout<<"\n";
    }
}




/*---------------Construction of the QEPG graph-------------------------------*/




/**
 * @brief Transpose a GF(2) bit matrix in place.
 *
 * Given a matrix with n_rows rows of width n_cols, produces a transposed
 * matrix with n_cols rows of width n_rows. Uses find_first/find_next
 * iteration for sparse-friendly traversal.
 *
 * @param[in]  mat      Input matrix (n_rows x n_cols).
 * @param[out] matTrans Output transposed matrix (n_cols x n_rows).
 */
inline void transpose_matrix(const std::vector<Row>& mat,std::vector<Row>& matTrans){
    const std::size_t n_rows=mat.size();
    const std::size_t n_cols=n_rows ? mat[0].size():0;
    matTrans.assign(n_cols, Row(n_rows));
    for(std::size_t r=0; r<n_rows; ++r){
        const Row& src= mat[r];
        for(std::size_t c=src.find_first();c!=Row::npos; c=src.find_next(c)){
            matTrans[c].set(r);
        }
    }
}

/**
 * @brief Build the parity propagation matrix by backward traversal of the circuit.
 *
 * Algorithm overview:
 * 1. Allocate per-qubit X, Y, Z propagation vectors (each a GF(2) row of
 *    length num_detectors + 1).
 * 2. Walk the gate list from last to first. For each gate:
 *    - DEPOLARIZE1: snapshot the current X/Y/Z propagation of that qubit
 *      into the corresponding rows of parityPropMatrixTranspose_.
 *    - M (measurement): look up which detectors/observable reference this
 *      measurement and set the corresponding bits in the X and Y propagation
 *      vectors for the measured qubit (Z-basis measurement detects X and Y errors).
 *    - R (reset): clear all propagation vectors for the reset qubit (errors
 *      before a reset cannot propagate past it).
 *    - cnot: apply the symplectic CNOT update rules:
 *        X_control ^= X_target  (X propagates forward on control)
 *        Z_target  ^= Z_control (Z propagates backward on target)
 *        Y_control ^= X_target  (Y = iXZ, so X part propagates)
 *        Y_target  ^= Z_control (Y = iXZ, so Z part propagates)
 *    - h (Hadamard): swap X and Z propagation vectors for that qubit.
 *
 * After completion, parityPropMatrixTranspose_ has 3*N rows (X block [0,N),
 * Y block [N,2N), Z block [2N,3N)) and (num_detectors+1) columns.
 */
void QEPG::backward_graph_construction(){

    size_t gate_size=circuit_.get_gate_num();

    /*
    Store the propagation from pauli noise to qubits
    */
    const size_t total_meas=circuit_.get_num_meas();


    /*
    Directly store the propagation from pauli noise to qubits
    */
    const size_t num_detectors=circuit_.get_detector_parity_group().size();

    std::vector<Row> current_x_parity_prop(circuit_.get_num_qubit(),Row(num_detectors+1));
    std::vector<Row> current_y_parity_prop(circuit_.get_num_qubit(),Row(num_detectors+1));
    std::vector<Row> current_z_parity_prop(circuit_.get_num_qubit(),Row(num_detectors+1));


    size_t current_meas_index=circuit_.get_num_meas()-1;
    size_t current_noise_index=total_noise_-1;

    const clifford::paritygroup& observable=circuit_.get_observable_parity_group();
    const std::unordered_set<size_t> observable_set(observable.indexlist.begin(), observable.indexlist.end());

    // NOTE: String comparison per gate is a known performance bottleneck for large
    // circuits. A future optimization would replace Gate::name with enum GateKind + switch.
    for(size_t t = gate_size; t-- > 0;){

        const auto& gate=circuit_.get_gate(t);
        const std::string& name=gate.name;

        /*
        *   First case, when the gate is a depolarization noise
        */
        if(name=="DEPOLARIZE1"){
                size_t qindex=gate.qubits[0];
                /*
                Uptill now, the fate of this noise is determined
                So in priciple, we can update the parity propagation
                */
                parityPropMatrixTranspose_[current_noise_index]=current_x_parity_prop[qindex];   //current_x_prop(circuit_.get_num_qubit(),Row(3* total_noise_))
                parityPropMatrixTranspose_[total_noise_+current_noise_index]=current_y_parity_prop[qindex];
                parityPropMatrixTranspose_[total_noise_*2+current_noise_index]=current_z_parity_prop[qindex];

                current_noise_index--;
                continue;
        }
        /*
        *   When the gate is a measurement
        */
        if(name=="M"){
            size_t qindex=gate.qubits[0];
            /*
            Update all affected parity/detector measurement
            */
            const clifford::parityIndexgroup& tmpmeasuregroup=circuit_.get_measure_to_parity_index(current_meas_index);
            for(size_t parityindex: tmpmeasuregroup.indexlist){
                    current_x_parity_prop[qindex][parityindex] = !current_x_parity_prop[qindex][parityindex];
            }
            /*
            This measurement will flip the observable
            */
            if(observable_set.count(current_meas_index)){
                    current_x_parity_prop[qindex][num_detectors] = !current_x_parity_prop[qindex][num_detectors];
            }
            // MZ destroys phase errors. X persists and toggles each referenced
            // measurement; repeated references therefore accumulate by XOR.
            current_y_parity_prop[qindex] = current_x_parity_prop[qindex];
            current_z_parity_prop[qindex].reset();
            current_meas_index--;
            continue;
        }
        /*
        *   When the gate is a reset
        */
        if(name=="R"){
            size_t qindex=gate.qubits[0];

            current_x_parity_prop[qindex].reset();
            current_y_parity_prop[qindex].reset();
            current_z_parity_prop[qindex].reset();
            continue;
        }
        /*
        *   When the gate is a CNOT
        */
        if(name=="cnot"){
            size_t qcontrol=gate.qubits[0];
            size_t qtarget=gate.qubits[1];
            current_x_parity_prop[qcontrol]^=current_x_parity_prop[qtarget];
            current_z_parity_prop[qtarget]^=current_z_parity_prop[qcontrol];
            current_y_parity_prop[qcontrol]^=current_x_parity_prop[qtarget];
            current_y_parity_prop[qtarget]^=current_z_parity_prop[qcontrol];
            continue;
        }

        if(name=="h"){
            size_t qindex=gate.qubits[0];
            current_x_parity_prop[qindex].swap(current_z_parity_prop[qindex]);
            continue;
        }
        /*
        *   Phase (S) gate: S†XS = Y, S†YS = -X, S†ZS = Z
        *   In GF(2) (signs ignored): swap X and Y propagation, Z unchanged.
        */
        if(name=="p"){
            size_t qindex=gate.qubits[0];
            current_x_parity_prop[qindex].swap(current_y_parity_prop[qindex]);
            continue;
        }
        /*
        *   Pauli X/Y/Z gates are self-inverse Cliffords.
        *   Their conjugation only introduces signs (e.g., X†ZX = -Z),
        *   which vanish in GF(2). Propagation is identity (no-op).
        */
        if(name=="x" || name=="y" || name=="z"){
            continue;
        }
    }

    // Build contiguous SIMD-aligned copy for fast sampling
    build_flat_parity_table();
}


const std::vector<Row>& QEPG::get_detectorMatrix() const noexcept{
    return detectorMatrix_;
}


const std::vector<Row>& QEPG::get_dectorMatrixTrans() const noexcept{
    return detectorMatrixTranspose_;
}

const std::vector<Row>& QEPG::get_parityPropMatrix() const noexcept{
    return parityPropMatrix_;
}



const std::vector<Row>& QEPG::get_parityPropMatrixTrans() const noexcept{
    return parityPropMatrixTranspose_;
}

const qepg_bits::FlatBitTable& QEPG::get_parityPropMatrixTransFlat() const noexcept{
    return parityPropMatrixTransFlat_;
}

void QEPG::build_flat_parity_table(){
    if(parityPropMatrixTranspose_.empty()) return;
    const std::size_t n_rows = parityPropMatrixTranspose_.size();
    const std::size_t n_cols = parityPropMatrixTranspose_[0].size();

    parityPropMatrixTransFlat_ = qepg_bits::FlatBitTable(n_rows, n_cols);

    for(std::size_t r = 0; r < n_rows; ++r){
        const Row& src = parityPropMatrixTranspose_[r];
        std::uint64_t* dst = parityPropMatrixTransFlat_.row_ptr(r);
        // Copy block data directly
        src.to_block_range(dst);
    }
}

/**
 * @brief Multiply two GF(2) matrices using transpose-and-AND-popcount.
 *
 * Computes result[i][j] = popcount(mat1[i] AND mat2^T[j]) mod 2.
 * Transposes mat2 first, then performs the row-by-row inner products.
 *
 * @param mat1 Left matrix (R1 x C).
 * @param mat2 Right matrix (C x C2).
 * @return Result matrix (R1 x C2) over GF(2).
 */
std::vector<Row> bitset_matrix_multiplication(const std::vector<Row>& mat1,const std::vector<Row>& mat2){
    const size_t row1=mat1.size();
    const size_t col1=row1? mat1[0].size():0;
    const size_t row2=mat2.size();
    const size_t col2=row1? mat2[0].size():0;
    std::vector<Row> result(row1,Row(col2));
    std::vector<Row> mat2transpose;
    transpose_matrix(mat2,mat2transpose);
    for(size_t i=0;i<row1;i++){
        for(size_t j=0;j<col2;j++){
            result[i][j]=and_popcount(mat1[i],mat2transpose[j])%2? true:false;
        }
    }
    return result;
}


/**
 * @brief Compute parity propagation matrix via explicit matrix multiplication (legacy path).
 *
 * Builds the parity group matrix from detector and observable definitions
 * (a binary matrix mapping detectors to measurements), then multiplies it
 * by the detector matrix (mapping noise locations to measurements) to get
 * the parity propagation matrix (mapping noise locations to detectors).
 * Finally transposes the result.
 *
 * @note This path is superseded by backward_graph_construction() which
 *       achieves the same result without the intermediate detector matrix.
 */
void QEPG::compute_parityPropMatrix(){

    using clock     = std::chrono::steady_clock;          // monotonic, good for benchmarking
    using microsec  = std::chrono::microseconds;
    auto t0 = clock::now();                               // start timer

    const std::vector<clifford::paritygroup>& detector_parity_group=circuit_.get_detector_parity_group();
    const clifford::paritygroup& observable_group=circuit_.get_observable_parity_group();
    const size_t row_size=detector_parity_group.size()+1;
    const size_t col_size=circuit_.get_num_meas();

    std::vector<Row> paritygroupMatrix(row_size,Row(col_size));


    for(size_t i=0; i<detector_parity_group.size();i++){
        for(size_t index: detector_parity_group[i].indexlist){
            paritygroupMatrix[i][index]=true;
        }
    }
    for(size_t index: observable_group.indexlist){
        paritygroupMatrix[detector_parity_group.size()][index]=true;
    }

    parityPropMatrix_=bitset_matrix_multiplication(paritygroupMatrix,detectorMatrix_);
    transpose_matrix(parityPropMatrix_,parityPropMatrixTranspose_);

}




}
