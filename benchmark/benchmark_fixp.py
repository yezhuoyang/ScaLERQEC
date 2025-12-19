from scalerqec.LERcalculator import *
from scalerqec.Monte.monteLER import stimLERcalc
from scalerqec import stratified_Scurve_LERcalc
from scalerqec.Stratified.stratifiedLERcalc import *
from pathlib import Path
home = Path(__file__).resolve().parent.parent

p=0.001

def generate_all_repetition_benchmark():

    tmp=stimLERcalc()

    print("--------------Repetition3--------------")
    shots3=1000000
    filepath = home / "stimprograms" / "repetition" / "repetition3"
    ler=tmp.calculate_LER_from_file(shots3,filepath,p)
    print("Repetition3 LER: ",ler)
    print("---------------------------------------")

    print("--------------Repetition5--------------")
    shots5=5000000
    filepath = home / "stimprograms" / "repetition" / "repetition5"
    ler=tmp.calculate_LER_from_file(shots5,filepath,p)
    print("Repetition5 LER: ",ler)
    print("---------------------------------------")

    print("--------------Repetition7--------------")
    shots7=10000000
    filepath = home / "stimprograms" / "repetition" / "repetition7"
    ler=tmp.calculate_LER_from_file(shots7,filepath,p)
    print("Repetition7 LER: ",ler)
    print("---------------------------------------")

    print("--------------Repetition9--------------")
    shots9=10000000
    filepath = home / "stimprograms" / "repetition" / "repetition9"
    ler=tmp.calculate_LER_from_file(shots9,filepath,p)
    print("Repetition9 LER: ",ler)
    print("---------------------------------------")

    print("--------------Repetition11--------------")
    shots11=10000000
    filepath = home / "stimprograms" / "repetition" / "repetition11"
    ler=tmp.calculate_LER_from_file(shots11,filepath,p)
    print("Repetition11 LER: ",ler)
    print("---------------------------------------")

    print("--------------Repetition13--------------")
    shots13=10000000
    filepath = home / "stimprograms" / "repetition" / "repetition13"
    ler=tmp.calculate_LER_from_file(shots13,filepath,p)
    print("Repetition13 LER: ",ler)
    print("---------------------------------------")

    print("--------------Repetition15--------------")
    shots15=10000000
    filepath = home / "stimprograms" / "repetition" / "repetition15"
    ler=tmp.calculate_LER_from_file(shots15,filepath,p)
    print("Repetition15 LER: ",ler)
    print("---------------------------------------")

    print("--------------Repetition17--------------")
    shots17=10000000
    filepath = home / "stimprograms" / "repetition" / "repetition17"
    ler=tmp.calculate_LER_from_file(shots17,filepath,p)
    print("Repetition17 LER: ",ler)
    print("---------------------------------------")


def generate_all_surface_benchmark():

    tmp=stimLERcalc()

    print("--------------Surface3--------------")
    shots3=1000000
    filepath = home / "stimprograms" / "surface" / "surface3"
    ler=tmp.calculate_LER_from_file(shots3,filepath,p)
    print("Surface3 LER: ",ler)
    print("---------------------------------------")

    print("--------------Surface5--------------")
    shots5=5000000
    filepath = home / "stimprograms" / "surface" / "surface5"
    ler=tmp.calculate_LER_from_file(shots5,filepath,p)
    print("Surface5 LER: ",ler)
    print("---------------------------------------")

    print("--------------Surface7--------------")
    shots7=10000000
    filepath = home / "stimprograms" / "surface" / "surface7"
    ler=tmp.calculate_LER_from_file(shots7,filepath,p)
    print("Surface7 LER: ",ler)
    print("---------------------------------------")

    print("--------------Surface9--------------")
    shots9=10000000
    filepath = home / "stimprograms" / "surface" / "surface9"
    ler=tmp.calculate_LER_from_file(shots9,filepath,p)
    print("Surface9 LER: ",ler)
    print("---------------------------------------")

    print("--------------Surface11--------------")
    shots11=10000000
    filepath = home / "stimprograms" / "surface" / "surface11"
    ler=tmp.calculate_LER_from_file(shots11,filepath,p)
    print("Surface11 LER: ",ler)
    print("---------------------------------------")

    print("--------------Surface13--------------")
    shots13=10000000
    filepath = home / "stimprograms" / "surface" / "surface13"
    ler=tmp.calculate_LER_from_file(shots13,filepath,p)
    print("Surface13 LER: ",ler)
    print("---------------------------------------")

    print("--------------Surface15--------------")
    shots15=10000000
    filepath = home / "stimprograms" / "surface" / "surface15"
    ler=tmp.calculate_LER_from_file(shots15,filepath,p)
    print("Surface15 LER: ",ler)
    print("---------------------------------------")

    print("--------------Surface17--------------")
    shots17=10000000
    filepath = home / "stimprograms" / "surface" / "surface17"
    ler=tmp.calculate_LER_from_file(shots17,filepath,p)
    print("Surface17 LER: ",ler)
    print("---------------------------------------")


if __name__ == "__main__":
    print("========================Repetition Code Benchmarks========================")
    generate_all_repetition_benchmark()
    print("========================Surface Code Benchmarks========================")
    generate_all_surface_benchmark()




