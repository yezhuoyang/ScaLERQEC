#ifndef QEPG_HPP
#define QEPG_HPP
#pragma once
#include <cstddef>
#include <ostream>
#include <string>
#include <bitset>  
#include <boost/dynamic_bitset.hpp>
#include "clifford.hpp"
#include <iostream>


namespace QEPG{

using Row=boost::dynamic_bitset<>;


template <typename Bitset>
inline std::size_t and_popcount(const Bitset& a, const Bitset&b)
{
    return (a&b).count();
}



template<class BitRow>
void inline print_bit_row(const BitRow& row,
                      char zero = '0', char one='1')
{
    const std::size_t cols= row.size();
    for(std::size_t c=0; c<cols;++c){
        std::cout<<(row.test(c)? one: zero);
    } 
    std::cout<<"\n";
}


template<class BitRow>
inline void print_bit_row(const BitRow& row,
                          std::ostream& out,   // default = console
                          char zero = '0', char one = '1')
{
    const std::size_t cols = row.size();
    for (std::size_t c = 0; c < cols; ++c) {
        out << (row.test(c) ? one : zero);
    }
    out.put('\n');
}





template<class BitRow>
void print_bit_matrix(const std::vector<BitRow>& rows,
                      char zero = '0', char one='1')
{
    if(rows.empty()) return;

    const std::size_t cols= rows.front().size();
    for(const auto& r: rows){
        if(r.size()!=cols){
            std::cerr<<"[print_bit_matrix] row width mismatach\n";
            return;
        }
        for(std::size_t c=0; c<cols;++c){
            std::cout<<(r.test(c)? one: zero);
        }
        std::cout<<"\n";
    }
}



class QEPG{


    public:
    
        QEPG();
        QEPG(clifford::cliffordcircuit othercircuit, size_t total_detectors, size_t total_noise);
        ~QEPG();

        void backward_graph_construction();

        void print_detectorMatrix(char zero = '0', char one='1') const;

        const std::vector<Row>& get_parityPropMatrixTrans() const noexcept;

        const size_t& get_total_noise() const noexcept;

        const size_t& get_total_detector() const noexcept;


    private:

        clifford::cliffordcircuit circuit_;
        std::size_t total_detectors_ = 0;
        std::size_t total_noise_=0;

        std::vector<Row> parityPropMatrixTranspose_;
};
}


#endif




