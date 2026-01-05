#include "QEPG.hpp"
#include <iostream>
#include <algorithm>

namespace QEPG{

/*------------------------------------------ctor----*/

QEPG::QEPG(){

}

QEPG::QEPG(clifford::cliffordcircuit othercircuit, size_t total_detectors, size_t total_noise):
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


void QEPG::print_detectorMatrix(char zero, char one) const{
    std::cout<<"-----------ParitygroupMatrixTranspose:----------------------\n";
    for(const auto& row:parityPropMatrixTranspose_){
        for(std::size_t c=0;c<row.size();++c){
            std::cout<<(row.test(c)? one:zero);
        }
        std::cout<<"\n";
    }
}




/*---------------Construction of the QEPG graph-------------------------------*/





void QEPG::backward_graph_construction(){
    size_t gate_size=circuit_.get_gate_num();

    // Number of detectors (parity checks)
    const size_t num_detectors=circuit_.get_detector_parity_group().size();
    
    std::vector<Row> current_x_parity_prop(circuit_.get_num_qubit(),Row(num_detectors+1));
    std::vector<Row> current_y_parity_prop(circuit_.get_num_qubit(),Row(num_detectors+1));
    std::vector<Row> current_z_parity_prop(circuit_.get_num_qubit(),Row(num_detectors+1));


    size_t current_meas_index=circuit_.get_num_meas()-1;
    size_t current_noise_index=total_noise_-1;

    const clifford::paritygroup& observable=circuit_.get_observable_parity_group();

    for(int t=gate_size-1;t>=0;t--){

        const auto& gate=circuit_.get_gate(t);
        std::string name=gate.name;

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
                    current_x_parity_prop[qindex].set(parityindex);
                    current_y_parity_prop[qindex].set(parityindex);
            }
            /*
            This measurement will flip the observable
            */
            if(std::find(observable.indexlist.begin(), observable.indexlist.end(), current_meas_index) != observable.indexlist.end()){
                    current_x_parity_prop[qindex].set(num_detectors);
                    current_y_parity_prop[qindex].set(num_detectors);
            }
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
        }
    }
} 


const std::vector<Row>& QEPG::get_parityPropMatrixTrans() const noexcept{
    return parityPropMatrixTranspose_;
}


}