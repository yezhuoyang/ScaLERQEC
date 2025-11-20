from ScaLER.clifford import *





'''
Class of quantum error propagation graph
TODO: Optimize the algorithm to construct the QEPG
'''
class QEPGpython:

    def __init__(self,circuit:CliffordCircuit):
        self._circuit = circuit 
        self._total_meas=self._circuit._totalMeas
        self._total_noise=self._circuit._totalnoise
        self._propMatrix=np.zeros((3*self._total_noise,len(self._circuit.get_parityMatchGroup())+1), dtype='uint8')



    def backword_graph_construction(self):
        nqubit=self._circuit._qubit_num
        #Keep track of the effect of X,Y,Z back propagation


        column_size=len(self._circuit.get_parityMatchGroup())+1
        current_x_prop=np.zeros((nqubit,column_size), dtype='uint8')
        current_y_prop=np.zeros((nqubit,column_size), dtype='uint8')
        current_z_prop=np.zeros((nqubit,column_size), dtype='uint8')


        current_noise_index=self._circuit._totalnoise-1
        current_meas_index=self._total_meas-1  


        total_noise=self._total_noise
        T=len(self._circuit._gatelists)

        for t in range(T-1,-1,-1):
            #Update current_x_prop, current_y_prop, current_z_prop based on the current gate and measurement
            gate=self._circuit._gatelists[t]
            '''
            If the gate is a oiginal noise, add edges to the graph based on current propogation
            '''
            if isinstance(gate, pauliNoise):
                noiseindex=current_noise_index 
                #print("Noise!")
                self._propMatrix[noiseindex,:]=current_x_prop[gate._qubitindex,:].copy()
                self._propMatrix[total_noise+noiseindex,:]=current_y_prop[gate._qubitindex,:].copy()
                self._propMatrix[total_noise*2+noiseindex,:]=current_z_prop[gate._qubitindex,:].copy()
                current_noise_index-=1
                continue
            '''
            When there is a measurement, update the current propogation based on the measurement
            We just need to consider the propagation of X and Y because only 
            the X and Y error can be detected by the measurement
            '''
            if isinstance(gate, Measurement):
                measureindex=current_meas_index

                qindex=gate._qubitindex
                if(measureindex in self._circuit.get_observable()):
                    current_x_prop[qindex][column_size-1]=1
                    current_y_prop[qindex][column_size-1]=1


                for parityIdx in self._circuit.get_measIdx_to_parityIdx(measureindex):
                    current_x_prop[qindex][parityIdx]=1
                    current_y_prop[qindex][parityIdx]=1                    

                current_meas_index-=1
                continue

            if isinstance(gate,Reset):
                current_x_prop[gate._qubitindex,:]=0
                current_y_prop[gate._qubitindex,:]=0         
                current_z_prop[gate._qubitindex,:]=0     
                continue

            '''
            Deal with propagation by CNOT gate, we need to consider the propagation of X and Z
            '''
            if gate._name=="CNOT":
                control=gate._control
                target=gate._target
                current_x_prop[control,:]=(current_x_prop[control,:]+current_x_prop[target,:])%2
                current_z_prop[target,:]=(current_z_prop[control,:]+current_z_prop[target,:])%2                
                current_y_prop[control,:]=(current_y_prop[control,:]+current_x_prop[target,:])%2
                current_y_prop[target,:]=(current_y_prop[target,:]+current_z_prop[control,:])%2
                continue
            
            '''
            Deal with propagation by H gate
            If there is a H gate, we need to swap the X and Z propagations
            '''
            if gate._name=="H":
                qubitindex=gate._qubitindex
                tmp_row=current_x_prop[qubitindex,:].copy()
                current_x_prop[qubitindex,:]=current_z_prop[qubitindex,:]
                current_z_prop[qubitindex,:]=tmp_row               
                continue

    '''
    Sample error and compute the detector value(Parity)
    Return a result of the detected value
    '''
    def sample_x_error(self, noise_index):
        return list(self._propMatrix[noise_index,:])

    def sample_y_error(self, noise_index):
        return list(self._propMatrix[self._total_noise+noise_index,:])

    def sample_z_error(self, noise_index):
        return list(self._propMatrix[2*self._total_noise+noise_index,:])


    def sample_noise_vector(self,noise_vector):
        assert len(noise_vector)==3*self._total_noise
        return noise_vector@self._propMatrix  






if __name__ == "__main__":
        stim_str=""
        filepath="C:/Users/username/Documents/Sampling/stimprograms/1cnot"
        with open(filepath, "r", encoding="utf-8") as f:
            stim_str = f.read()
        circuit=CliffordCircuit(3)
        circuit.set_error_rate(0.01)  
        circuit.compile_from_stim_circuit_str(stim_str)    

        graph=QEPGpython(circuit)
        graph.backword_graph_construction()

        print(graph.sample_noise_vector(np.array([1,0,0,0,0,0,0,0,0,0,0,0])))

