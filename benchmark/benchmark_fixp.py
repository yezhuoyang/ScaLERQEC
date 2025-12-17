from scaler.LERcalculator import *
from scaler.monteLER import stimLERcalc
from scaler import stratified_Scurve_LERcalc
from scaler.stratifiedLERcalc import *


p=0.001



def generate_all_repetition_benchmark():

    tmp=stimLERcalc()


    shots3=1000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition3"
    ler=tmp.calculate_LER_from_file(shots3,filepath,p)
    print("Repetition3 LER: ",ler)

    shots5=5000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition5"
    ler=tmp.calculate_LER_from_file(shots5,filepath,p)
    print("Repetition5 LER: ",ler)

    shots7=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition7"
    ler=tmp.calculate_LER_from_file(shots7,filepath,p)
    print("Repetition7 LER: ",ler)


    shots9=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition9"
    ler=tmp.calculate_LER_from_file(shots9,filepath,p)
    print("Repetition9 LER: ",ler)

    shots11=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition11"
    ler=tmp.calculate_LER_from_file(shots11,filepath,p)
    print("Repetition11 LER: ",ler)

    shots13=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition13"
    ler=tmp.calculate_LER_from_file(shots13,filepath,p)
    print("Repetition13 LER: ",ler)

    shots15=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition15"
    ler=tmp.calculate_LER_from_file(shots15,filepath,p)
    print("Repetition15 LER: ",ler)

    shots17=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition17"
    ler=tmp.calculate_LER_from_file(shots17,filepath,p)
    print("Repetition17 LER: ",ler)





def generate_all_surface_benchmark():

    tmp=stimLERcalc()


    shots3=1000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface3"
    ler=tmp.calculate_LER_from_file(shots3,filepath,p)
    print("Surface3 LER: ",ler)

    shots5=5000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface5"
    ler=tmp.calculate_LER_from_file(shots5,filepath,p)
    print("Surface5 LER: ",ler)

    shots7=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface7"
    ler=tmp.calculate_LER_from_file(shots7,filepath,p)
    print("Surface7 LER: ",ler)


    shots9=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface9"
    ler=tmp.calculate_LER_from_file(shots9,filepath,p)
    print("Surface9 LER: ",ler)

    shots11=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface11"
    ler=tmp.calculate_LER_from_file(shots11,filepath,p)
    print("Surface11 LER: ",ler)

    shots13=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface13"
    ler=tmp.calculate_LER_from_file(shots13,filepath,p)
    print("Surface13 LER: ",ler)

    shots15=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface15"
    ler=tmp.calculate_LER_from_file(shots15,filepath,p)
    print("Surface15 LER: ",ler)

    shots17=10000000
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface17"
    ler=tmp.calculate_LER_from_file(shots17,filepath,p)
    print("Surface17 LER: ",ler)


if __name__ == "__main__":
    #generate_all_repetition_benchmark()
    generate_all_surface_benchmark()




