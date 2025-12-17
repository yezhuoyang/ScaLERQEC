from scaler.qepg import return_samples, return_samples_many_weights, return_detector_matrix, return_samples_many_weights_separate_obs, return_samples_numpy, compile_QEPG, return_samples_many_weights_separate_obs_with_QEPG
from scaler.clifford import *
from scaler.stimparser import *
from time import time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from scaler.qepg import return_samples_Monte_separate_obs_with_QEPG,return_samples_numpy, compile_QEPG, return_samples_many_weights_separate_obs_with_QEPG


def test_compile_speed(distance):
    print("---------------------------Test distance: ",distance,"---------------------------")
    p=0.001
    circuit=CliffordCircuit(2)
    circuit.set_error_rate(p)
    stim_circuit=stim.Circuit.generated("surface_code:rotated_memory_z",rounds=distance*3,distance=distance).flattened()
    stim_circuit=rewrite_stim_code(str(stim_circuit))
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit) 

    new_stim_circuit=circuit.get_stim_circuit()      
    total_noise=circuit.get_totalnoise()
    string_program=str(new_stim_circuit)
    current_time = time()
    g=compile_QEPG(string_program)
    print("My Time taken for compile_QEPG: ", time()-current_time)

    current_time = time()    
    sampler=new_stim_circuit.compile_sampler()
    print("STIM Time taken for compile_QEPG: ", time()-current_time)




def test_samplerate(filepath):

    print("---------------------------Test file: ",filepath,"---------------------------")
    p=0.001
    stim_str=""
    with open(filepath, "r", encoding="utf-8") as f:
        stim_str = f.read()

    circuit=CliffordCircuit(2)
    stim_circuit=rewrite_stim_code(stim_str)
    circuit.set_stim_str(stim_circuit)
    circuit.compile_from_stim_circuit_str(stim_circuit)           
    new_stim_circuit=circuit.get_stim_circuit()        
    total_noise=circuit.get_totalnoise()

    average_weight=int(total_noise*p)
    if average_weight==0:
        average_weight=1


    current_time = time()
    result=return_samples_numpy(str(new_stim_circuit),average_weight,1000000)
    print("My Time taken for return_samples: ", time()-current_time)


    sampler=new_stim_circuit.compile_sampler()

    current_time = time()
    sampler.sample(shots=1000000)
    print("Stim take {} to sample".format(time()-current_time))



all_files=[
    "repetition/repetition3",
    "repetition/repetition5",
    "repetition/repetition7",
    "repetition/repetition9",
    "repetition/repetition11",
    "repetition/repetition13",
    "repetition/repetition15",
]

all_files=[
    "square/square3",
    "square/square5",
    "square/square7",
    "square/square9",
    "square/square11",
    "square/square13",
    "square/square15",
]


def plot_speed_comparison():

    # Increase all font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 22
    })

    # Data for each code family
    our_times = {
        'Surface':   [0.07723522186279297, 0.12538409233093262, 0.33560729026794434, 0.7937157154083252, 2.4749162197113037, 4.19052243232727, 8.57132077217102],
        'Repetition':[0.07346034049987793, 0.07142376899719238, 0.07645463943481445, 0.09147191047668457, 0.11658215522766113, 0.12368345260620117, 0.1573023796081543],
        'Square':    [0.08960413932800293, 0.1860971450805664, 0.6794803142547607, 1.9798688888549805, 5.654451847076416, 10.722644329071045, 20.0666925907135],
        'Hexagon':   [0.09782648086547852, 0.3812522888183594, 1.507704257965088, 5.400120735168457, 15.202311992645264, 31.231825351715088, 71.94260430335999]
    }

    stim_times = {
        'Surface':   [0.4363071918487549, 2.4868075847625732, 7.275710344314575, 14.408370733261108, 35.194268226623535, 52.63719964027405, 92.57772707939148],
        'Repetition':[0.03874683380126953, 0.28997182846069336, 0.897857666015625, 1.7311599254608154, 2.5915396213531494, 3.4075686931610107, 3.7710063457489014],
        'Square':    [0.9875476360321045, 4.922186613082886, 14.043782949447632, 35.79939651489258, 70.34424996376038, 114.05955696105957, 185.170166015625],
        'Hexagon':   [2.3935494422912598, 11.41777229309082, 33.10367560386658, 76.78172779083252, 160.7542278766632, 314.2602651119232, 580.7379786968231]
    }

    # Distances corresponding to each data point
    distances = [3, 5, 7, 9, 11, 13, 15]
    codes = list(our_times.keys())

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot each code family in its own subplot
    for ax, code in zip(axs.flatten(), codes):
        x = np.array(our_times[code])
        y = np.array(stim_times[code])
        
        # Scatter points in blue
        ax.scatter(x, y, color='C0', marker='o')
        
        # Annotate each point with its distance
        for xi, yi, d in zip(x, y, distances):
            ax.annotate(f'd={d}', (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center')
        
        # Fit linear model
        m, b = np.polyfit(x, y, 1)
        
        # Plot fitted line in red dashed style
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, 'r--')
        
        # Annotate slope
        ax.text(0.05, 0.92, f"Slope = {m:.2f}", transform=ax.transAxes, va='top')
        
        # Titles and labels
        ax.set_title(f'{code} Code')
        ax.set_xlabel('Time (Ours) [s]')
        ax.set_ylabel('Time (STIM) [s]')

    # Layout adjustment
    fig.tight_layout()

    # Save the annotated figure
    output_path = 'sampling_time_comparison_annotated.png'
    fig.savefig(output_path, dpi=300)

    output_path



if __name__ == "__main__":
    #test_samplerate(11)
    #test_compile_speed(13)
    # fileroot="C:/Users/username/Documents/Sampling/stimprograms/"
    # for file in all_files:
    #     filepath=fileroot+file
    #     test_samplerate(filepath)
    plot_speed_comparison()
    # test_samplerate(3)

    # test_samplerate(5)

    # test_samplerate(7)

    # test_samplerate(9)

    # test_samplerate(11)

    # test_samplerate(13)

    # test_samplerate(15)


