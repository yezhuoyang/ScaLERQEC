/**
 * @file clifford.cpp
 * @brief Implementation of the cliffordcircuit class and STIM string parser.
 *
 * Implements gate addition methods (each Clifford gate auto-inserts a
 * DEPOLARIZE1 noise channel), the STIM string compiler that parses
 * instructions line by line, and helper functions for tokenizing and
 * parsing rec[...] references in DETECTOR/OBSERVABLE lines.
 */

#include "clifford.hpp"
#include <iomanip>
#include <iostream>
#include <string_view>
#include <charconv>   // std::from_chars

namespace clifford{



/*-------------------------------------------ctor */

cliffordcircuit::cliffordcircuit() = default;

cliffordcircuit::cliffordcircuit(size_t n_qubit)
        : num_qubit_(n_qubit){}

/*-------------------------------------------configuration*/


void cliffordcircuit::set_error_rate(double p){error_rate_=p;}


/*--------------------------------------------Add quantum noise*/


/// @brief Append an X_ERROR noise channel (does not increment noise counter).
void cliffordcircuit::add_XError(size_t qindex) {
    circuit_.push_back({"X_ERROR", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

/// @brief Append a Z_ERROR noise channel (does not increment noise counter).
void cliffordcircuit::add_ZError(size_t qindex) {
    circuit_.push_back({"Z_ERROR", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

/// @brief Append a DEPOLARIZE1 noise channel and increment the noise counter.
void cliffordcircuit::add_depolarize1(size_t qindex) {
    circuit_.push_back({"DEPOLARIZE1", {qindex}});
    num_noise_++;
    num_qubit_=std::max(num_qubit_,qindex+1);
}



/*--------------------------------------------1 qubit gate*/


/// @brief Append Hadamard gate preceded by depolarizing noise.
void cliffordcircuit::add_hadamard(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"h", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

/// @brief Append Phase (S) gate preceded by depolarizing noise.
void cliffordcircuit::add_phase(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"p", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

/// @brief Append Pauli-X gate preceded by depolarizing noise.
void cliffordcircuit::add_pauliX(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"x", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

/// @brief Append Pauli-Y gate preceded by depolarizing noise.
void cliffordcircuit::add_pauliy(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"y", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

/// @brief Append Pauli-Z gate preceded by depolarizing noise.
void cliffordcircuit::add_pauliz(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"z", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}


/*--------------------------------------------2 qubit gate---------------------------*/


/**
 * @brief Append a CNOT gate with depolarizing noise on both control and target qubits.
 *
 * Two DEPOLARIZE1 channels are inserted (one per qubit) before the CNOT instruction.
 */
void cliffordcircuit::add_cnot(size_t qcontrol, size_t qtarget){
    add_depolarize1(qcontrol);
    add_depolarize1(qtarget);
    circuit_.push_back({"cnot", {qcontrol,qtarget}});
    num_qubit_=std::max(num_qubit_,qcontrol+1);
    num_qubit_=std::max(num_qubit_,qtarget+1);
}


/*--------------------------------------------Reset/Measurement gadget---------------*/

/// @brief Append a reset operation (no noise channel).
void cliffordcircuit::add_reset(size_t qindex){
    //add_depolarize1(qindex);
    circuit_.push_back({"R", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

/**
 * @brief Append a measurement with an associated depolarizing noise channel.
 *
 * Records the gate index of this measurement in measureindexList_ and creates
 * an empty parityIndexgroup entry in measure_to_parity_index_ so that
 * subsequent DETECTOR lines can populate the inverse mapping.
 */
void cliffordcircuit::add_measurement(size_t qindex){
    add_depolarize1(qindex);
    measureindexList_.push_back(circuit_.size());
    measure_to_parity_index_.emplace_back(parityIndexgroup{{}});
    circuit_.push_back({"M", {qindex}});
    num_meas_++;
    num_qubit_=std::max(num_qubit_,qindex+1);
}


/*---------------------------------------------visualizatation-----------------------*/


/// @brief Print the full circuit structure to stdout for debugging.
void cliffordcircuit::print_circuit() const{
    std::cout << '\n'
              << "----------------------- Clifford circuit-----------------------\n"
              << "    qubits      : " << num_qubit_      << '\n'
              << "    error rate  : " << std::fixed << std::setprecision(4)
                                          << error_rate_    << '\n'
              << "    noise terms : " << num_noise_      << '\n'
              << "    num measures  : " << num_meas_ << '\n'
              << "    detectors   : " << num_detectors_  << '\n'
              << "-------------------------------------------------------------\n";

    const std::size_t n = circuit_.size();
    const int width = static_cast<int>(std::to_string(n - 1).size());  // digits

    for (std::size_t i = 0; i < n; ++i) {
        const auto& g = circuit_[i];

        std::cout << std::setw(width) << i << ": "
                << std::left << std::setw(11) << g.name  // longest = "DEPOLARIZE1"
                << ' ';

        for (auto q : g.qubits)
            std::cout << 'q' << q << ' ';
        std::cout << '\n';
    }
    std::cout<<"----------------------Detector measurements---------------------------------------\n";
    int index=0;
    for(const auto& paritygroup : detectors_){
        std::cout<<"Detector["<<index<<"]: ";
        for(const auto& index: paritygroup.indexlist)
             std::cout<<index<<" ";
        std::cout<<"\n";
        index++;
    }
    std::cout<<"----------------------Measure index to Detector measurements---------------------------------------\n";
    index=0;
    for(const auto& mgroup : measure_to_parity_index_){
        std::cout<<"M["<<index<<"]: ";
        for(const auto& index: mgroup.indexlist)
             std::cout<<index<<" ";
        std::cout<<"\n";
        index++;
    }
    std::cout<<"----------------------Observable measurements---------------------------------------\n";
    std::cout<<"Observable[0]: ";
    for(const auto& index: observable_.indexlist)
        std::cout<<index<<" ";
    std::cout<<"\n";
    std::cout<<"----------------------Measurement to gate index---------------------------------------\n";
    index=0;
    for(const auto& Mindex:measureindexList_){
        std::cout<<"M["<<index<<"]: "<<Mindex<<"\n";
        index++;
    }

}



/*--Get gate by index-------------------------------------*/
const Gate& cliffordcircuit::get_gate(size_t gateindex) const{return circuit_.at(gateindex);}

/*Get member-------------------------------------------*/
size_t cliffordcircuit::get_num_qubit() const{return num_qubit_;}
size_t cliffordcircuit::get_gate_num() const{return circuit_.size();}

void cliffordcircuit::set_num_qubit(size_t num_qubit) {num_qubit_=num_qubit;}

size_t cliffordcircuit::get_num_meas() const{
    return num_meas_;
}

size_t cliffordcircuit::get_num_noise() const{
    return num_noise_;
}

size_t cliffordcircuit::get_num_detector() const{
    return num_detectors_;
}

const std::vector<paritygroup>& cliffordcircuit::get_detector_parity_group() const{
    return detectors_;
}

const paritygroup& cliffordcircuit::get_observable_parity_group() const{
    return observable_;
}


const parityIndexgroup& cliffordcircuit::get_measure_to_parity_index(const size_t& mindex) const{
    return measure_to_parity_index_[mindex];
}

/*Helper functions for parsing------------------------------------------------------------*/


/**
 * @brief Iterate over each line in a string_view, invoking a callback per line.
 *
 * Handles both Unix (LF) and Windows (CRLF) line endings.
 *
 * @tparam Callback Callable accepting a std::string_view.
 * @param sv       The input text to split into lines.
 * @param callback Function called once per line (excluding line endings).
 */
template <typename Callback>
void for_each_line(std::string_view sv, Callback&& callback){
    while(!sv.empty()){
        auto pos = sv.find_first_of("\n\r");
        std::string_view line=sv.substr(0,pos);
        callback(line);

        if(pos == sv.npos) break;
        const char separator = sv[pos];
        sv.remove_prefix(pos+1);
        if(separator == '\r' && !sv.empty() && sv.front() == '\n')
            sv.remove_prefix(1);
    }
}


/**
 * @brief Extract the next whitespace-delimited token from a string_view.
 *
 * Advances sv past the consumed token and any leading whitespace.
 *
 * @param[in,out] sv The remaining input; modified to skip past the returned token.
 * @return The next token, or an empty string_view if no tokens remain.
 */
inline std::string_view next_token(std::string_view& sv){
    //First, we discard all leading spaces/tabs
    const auto first=sv.find_first_not_of(" \t");
    if(first==std::string_view::npos){
        sv={};
        return {};
    }
    sv.remove_prefix(first);

    //Token is substring [0, pos)
    const auto pos=sv.find_first_of(" \t");
    std::string_view token=sv.substr(0,pos);
    sv.remove_prefix(token.size());
    return token;
}


/**
 * @brief Parse an integer from a string_view token.
 * @param tok The token to parse.
 * @return The parsed integer value.
 */
inline int to_int(std::string_view tok){
    int value{};
    std::from_chars(tok.data(),tok.data()+tok.size(),value);
    return value;
}




/**
 * @brief Consume one rec[<signed_int>] from the front of a string_view.
 *
 * On success, returns the integer inside the brackets and advances sv
 * to just after the closing ']'.
 *
 * @param[in,out] sv Input string_view; advanced past the parsed rec[...].
 * @return The signed integer value inside the rec[] brackets.
 * @throws std::runtime_error On malformed input (missing 'rec[' or ']').
 */
inline int parse_one_rec(std::string_view& sv)
{
    constexpr std::string_view tag = "rec[";
    const auto pos = sv.find(tag);
    if (pos == std::string_view::npos)
        throw std::runtime_error{"missing 'rec['"};

    sv.remove_prefix(pos + tag.size());          // skip "rec["

    const char* begin = sv.data();
    auto end_offset   = sv.find(']');
    if (end_offset == std::string_view::npos)
        throw std::runtime_error{"missing ']'"};

    int value{};
    auto res = std::from_chars(begin, begin + end_offset, value);
    if (res.ec != std::errc() || res.ptr != begin + end_offset)
        throw std::runtime_error{"bad integer inside rec[]"};

    sv.remove_prefix(end_offset + 1);            // drop "<int>]"
    return value;
}


/**
 * @brief Parse all rec[...] occurrences in a DETECTOR or OBSERVABLE line.
 *
 * Skips the coordinate block "(x, y, z)" at the start of the line, then
 * collects every rec[<int>] token that follows.
 *
 * @param rest The line content after the DETECTOR/OBSERVABLE keyword.
 * @return Vector of signed integer indices in the order they appear.
 * @throws std::runtime_error If no rec[...] tokens are found.
 */
inline std::vector<int> parse_detector_recs(std::string_view rest)
{
    // 1.  Skip the coordinate block "(... , ...)"
    auto close_paren = rest.find(')');
    if (close_paren != std::string_view::npos)
        rest.remove_prefix(close_paren + 1);

    // 2.  Collect every rec[...] that follows
    std::vector<int> indices;
    while (true) {
        auto next = rest.find("rec[");
        if (next == std::string_view::npos) break;   // no more
        // parse_one_rec will advance `rest`
        indices.push_back(parse_one_rec(rest));
    }
    if (indices.empty())
        throw std::runtime_error{"DETECTOR line has no rec[...] tokens"};
    return indices;
}



/**
 * @brief Parse a non-negative integer from a string_view token.
 *
 * @param tok The token to parse.
 * @return The parsed std::size_t value.
 * @throws std::invalid_argument If the token is not a valid non-negative integer.
 * @throws std::out_of_range If the value exceeds std::size_t range.
 */
inline std::size_t to_size_t(std::string_view tok){
    std::size_t value{};
    const char* begin = tok.data();
    const char* end= begin + tok.size();

    auto [ptr, ec] = std::from_chars(begin,end,value,10);

    if (ec == std::errc::invalid_argument || ptr != end)
        throw std::invalid_argument{"token is not a non-negative integer"};
    if (ec == std::errc::result_out_of_range)
        throw std::out_of_range{"integer value exceeds std::size_t range"};

    return value;
}






/**
 * @brief Parse a normalized STIM program string and populate the circuit.
 *
 * Processes each line of the input, recognizing instructions:
 * - "M <qubit>" : measurement (calls add_measurement)
 * - "R <qubit>" : reset (calls add_reset)
 * - "H <qubit>" : Hadamard (calls add_hadamard)
 * - "CX <control> <target>" : CNOT (calls add_cnot)
 * - "DETECTOR(...) rec[i] rec[j] ..." : defines a detector from measurement
 *   indices relative to the current measurement count (negative offsets)
 * - "OBSERVABLE_INCLUDE(...) rec[i] ..." : defines the logical observable
 *
 * For DETECTOR lines, two mappings are maintained:
 * 1. detector_index -> list of measurement indices (stored in detectors_)
 * 2. measurement_index -> list of detector indices (stored in measure_to_parity_index_)
 */
void cliffordcircuit::compile_from_rewrited_stim_string(std::string stim_str){

    // Compilation replaces the circuit, including all record-index mappings.
    *this = cliffordcircuit();

    for_each_line(stim_str, [this](std::string_view line){
        std::string_view rest=line;
        std::string_view op=next_token(rest);

        if(op=="M"){
            size_t qindex=to_size_t(next_token(rest));
            add_measurement(qindex);
        }
        else if(op=="R"){
            size_t qindex=to_size_t(next_token(rest));
            add_reset(qindex);
        }
        else if(op=="H"){
            size_t qindex=to_size_t(next_token(rest));
            add_hadamard(qindex);
        }
        else if(op=="S"){
            size_t qindex=to_size_t(next_token(rest));
            add_phase(qindex);
        }
        else if(op=="X"){
            size_t qindex=to_size_t(next_token(rest));
            add_pauliX(qindex);
        }
        else if(op=="Y"){
            size_t qindex=to_size_t(next_token(rest));
            add_pauliy(qindex);
        }
        else if(op=="Z"){
            size_t qindex=to_size_t(next_token(rest));
            add_pauliz(qindex);
        }
        else if(op=="CX"){
            size_t qcontrol=to_size_t(next_token(rest));
            size_t qtarget=to_size_t(next_token(rest));
            add_cnot(qcontrol,qtarget);
        }
        else if(op.substr(0,8)=="DETECTOR"){
           std::vector<int> intlist= parse_detector_recs(rest);
           /*
           We keep track of two mapping: The first mapping is from detector index to all
           measurement indices it contains. The second mapping is the inverse of this mapping:
           from measurement indices to all detector index it affects.
           */
           paritygroup measuregroup;
           for(int index: intlist){
               if (index >= 0 || static_cast<size_t>(-static_cast<long long>(index)) > num_meas_)
                   throw std::invalid_argument("DETECTOR record reference is out of range");
               // index is a negative offset from num_meas_
               size_t meas_idx = static_cast<size_t>(static_cast<ptrdiff_t>(num_meas_) + index);
               measuregroup.indexlist.push_back(meas_idx);
               measure_to_parity_index_[meas_idx].indexlist.emplace_back(num_detectors_);
           }
           detectors_.push_back(measuregroup);
           num_detectors_++;
        }
        else if(op.substr(0,10)=="OBSERVABLE"){
            std::vector<int> intlist= parse_detector_recs(rest);
            paritygroup measuregroup;
            for(int index: intlist){
                 if (index >= 0 || static_cast<size_t>(-static_cast<long long>(index)) > num_meas_)
                     throw std::invalid_argument("OBSERVABLE record reference is out of range");
                 size_t meas_idx = static_cast<size_t>(static_cast<ptrdiff_t>(num_meas_) + index);
                 measuregroup.indexlist.push_back(meas_idx);
            }
            observable_=measuregroup;
        }
    });


}



}
