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


void cliffordcircuit::add_XError(size_t qindex) {
    circuit_.push_back({"X_ERROR", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

void cliffordcircuit::add_ZError(size_t qindex) {
    circuit_.push_back({"Z_ERROR", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

void cliffordcircuit::add_depolarize1(size_t qindex) {
    circuit_.push_back({"DEPOLARIZE1", {qindex}});
    num_noise_++;
    num_qubit_=std::max(num_qubit_,qindex+1);
}



/*--------------------------------------------1 qubit gate*/


void cliffordcircuit::add_hadamard(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"h", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

void cliffordcircuit::add_phase(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"p", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

void cliffordcircuit::add_pauliX(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"x", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

void cliffordcircuit::add_pauliy(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"y", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

void cliffordcircuit::add_pauliz(size_t qindex){
    add_depolarize1(qindex);
    circuit_.push_back({"z", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}


/*--------------------------------------------2 qubit gate---------------------------*/


void cliffordcircuit::add_cnot(size_t qcontrol, size_t qtarget){
    add_depolarize1(qcontrol);
    add_depolarize1(qtarget);   
    circuit_.push_back({"cnot", {qcontrol,qtarget}});
    num_qubit_=std::max(num_qubit_,qcontrol+1);
    num_qubit_=std::max(num_qubit_,qtarget+1);    
}


/*--------------------------------------------Reset/Measurement gadget---------------*/

void cliffordcircuit::add_reset(size_t qindex){
    //add_depolarize1(qindex);
    circuit_.push_back({"R", {qindex}});
    num_qubit_=std::max(num_qubit_,qindex+1);
}

void cliffordcircuit::add_measurement(size_t qindex){
    add_depolarize1(qindex);
    measureindexList_.push_back(circuit_.size());
    measure_to_parity_index_.emplace_back(parityIndexgroup{{}});
    circuit_.push_back({"M", {qindex}});
    num_meas_++;
    num_qubit_=std::max(num_qubit_,qindex+1);
}


/*---------------------------------------------visualizatation-----------------------*/


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


template <typename Callback>
void for_each_line(std::string_view sv, Callback&& callback){
    while(!sv.empty()){
        auto pos = sv.find_first_of("\n\r");
        std::string_view line=sv.substr(0,pos);
        callback(line);

        if(pos == sv.npos) break;
        sv.remove_prefix(pos+1);

        if(!sv.empty()&&sv.front()=='\n'&& line.back()=='\r')
        sv.remove_prefix(1);
    }
}


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


inline int to_int(std::string_view tok){
    int value{};
    std::from_chars(tok.data(),tok.data()+tok.size(),value);
    return value;
}




//--------------------------------------------------------------------
//  Consume one  rec[<signed_int>]  from the front of `sv`.
//  On success:
//      returns the integer
//      advances `sv` to just after the closing ']'
//  Throws std::runtime_error on malformed input.
//
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
    if (res.ec != std::errc())
        throw std::runtime_error{"bad integer inside rec[]"};

    sv.remove_prefix(end_offset + 1);            // drop "<int>]"
    return value;
}


//--------------------------------------------------------------------
//  Parse *all* rec[...] occurrences in a DETECTOR line.
//  The input `rest` must begin with "(x, y, z)" (already trimmed of
//  the "DETECTOR" token) and may contain any number of rec[...] tokens.
//  Returns a vector<int> of the signed indices in the order found.
//
inline std::vector<int> parse_detector_recs(std::string_view rest)
{
    // 1.  Skip the coordinate block "(…, …, …)"
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



inline std::size_t to_size_t(std::string_view tok){
    std::size_t value{};
    const char* begin = tok.data();
    const char* end= begin + tok.size();

    auto [ptr, ec] = std::from_chars(begin,end,value,10);

    if (ec == std::errc::invalid_argument || ptr != end)
        throw std::invalid_argument{"token is not a non‑negative integer"};
    if (ec == std::errc::result_out_of_range)
        throw std::out_of_range{"integer value exceeds std::size_t range"};

    return value;
}






/*compile from stim string---------------------------------------------------*/
void cliffordcircuit::compile_from_rewrited_stim_string(std::string stim_str){
    
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
                measuregroup.indexlist.push_back((size_t)((int)num_meas_+index));
                measure_to_parity_index_[(int)num_meas_+index].indexlist.emplace_back(num_detectors_);
           }
           detectors_.push_back(measuregroup);
           num_detectors_++; 
        }
        else if(op.substr(0,10)=="OBSERVABLE"){
            std::vector<int> intlist= parse_detector_recs(rest);
            paritygroup measuregroup;
            for(int index: intlist)
                 measuregroup.indexlist.push_back((size_t)((int)num_meas_+index));
            observable_=measuregroup;            
        }
    });


}



}