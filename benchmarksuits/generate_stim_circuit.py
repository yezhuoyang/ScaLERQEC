import stim
from scalerqec.stimparser import rewrite_stim_code
from pathlib import Path




def generate_circuit(filepath: str | Path, distance: int = 3) -> Path:
    """
    Build a rotated‑surface‑code memory circuit and write it to `filepath`.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Where to write the circuit text (e.g. 'circuits/my_surface_code.stim').
    distance : int, optional
        Code distance; default is 3.

    Returns
    -------
    pathlib.Path
        Absolute path to the file that was written.
    """
    # 1. Make the circuit and rewrite it
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=3* distance,
        distance=distance
    ).flattened()

    circuit_text = rewrite_stim_code(str(stim_circuit))

    # 2. Resolve the target path and create parent dirs if needed
    filepath = Path(filepath).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 3. Write the text
    with filepath.open("w", encoding="utf-8") as f:
        f.write(circuit_text)

    return filepath



if __name__=="__main__":
    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface37"
    generate_circuit(filepath, distance= 37)

    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface39"
    generate_circuit(filepath, distance= 39)

    filepath="C:/Users/username/Documents/Sampling/stimprograms/surface/surface41"
    generate_circuit(filepath, distance= 41)


    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition7"
    # generate_circuit(filepath, distance= 7)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition9"
    # generate_circuit(filepath, distance= 9)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition11"
    # generate_circuit(filepath, distance= 11)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition13"
    # generate_circuit(filepath, distance= 13)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition13"
    # generate_circuit(filepath, distance= 13)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition15"
    # generate_circuit(filepath, distance= 15)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition17"
    # generate_circuit(filepath, distance= 17)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition19"
    # generate_circuit(filepath, distance= 19)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition21"
    # generate_circuit(filepath, distance= 21)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition23"
    # generate_circuit(filepath, distance= 23)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition25"
    # generate_circuit(filepath, distance= 25)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition27"
    # generate_circuit(filepath, distance= 27)

    # filepath="C:/Users/username/Documents/Sampling/stimprograms/repetition/repetition29"
    # generate_circuit(filepath, distance= 29)