import stim






class stimparser:


    def __init__(self):
        pass

    

    def rewrite_stim_code(self,code: str) -> str:
        """
        Rewrites a Stim program so that each line contains at most one gate or measurement.
        Lines starting with TICK, R, DETECTOR(, and OBSERVABLE_INCLUDE( are kept as-is.
        Multi-target lines for CX, M, and MR are split up.
        """
        lines = code.splitlines()
        output_lines = []

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                # Skip empty lines (optional: you could also preserve them)
                continue

            # Keep lines that we do NOT want to split
            if (stripped_line.startswith("TICK") or
                stripped_line.startswith("DETECTOR(") or
                stripped_line.startswith("QUBIT_COORDS(") or     
                stripped_line.startswith("OBSERVABLE_INCLUDE(")):
                output_lines.append(stripped_line)
                continue
            
            if (stripped_line.startswith("X_ERROR") or
                stripped_line.startswith("DEPOLARIZE1") or
                stripped_line.startswith("DEPOLARIZE2") or
                stripped_line.startswith("SHIFT_COORDS")            
                ):
                continue
                

            tokens = stripped_line.split()
            gate = tokens[0]

            # Handle 2-qubit gate lines like "CX 0 1 2 3 4 5 ..."
            if gate == "CX":
                qubits = tokens[1:]
                # Pair up the qubits [q0, q1, q2, q3, ...] => (q0,q1), (q2,q3), ...
                for i in range(0, len(qubits), 2):
                    q1, q2 = qubits[i], qubits[i + 1]
                    output_lines.append(f"CX {q1} {q2}")

            # Handle multi-qubit measurements "M 1 3 5 ..." => each on its own line
            elif gate == "M":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"M {q}")


            elif gate == "MX":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"H {q}")
                    output_lines.append(f"M {q}")

            elif gate == "MY":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"S {q}")
                    output_lines.append(f"S {q}")
                    output_lines.append(f"S {q}")
                    output_lines.append(f"H {q}")                
                    output_lines.append(f"M {q}")



            elif gate == "H":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"H {q}")

            elif gate == "S":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"S {q}")            

            # Handle multi-qubit measure+reset "MR 1 3 5 ..." => each on its own line
            elif gate == "MR":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"M {q}")
                    output_lines.append(f"R {q}")

            elif gate == "R":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"R {q}")
            
            elif gate == "RX":
                qubits = tokens[1:]
                for q in qubits:
                    output_lines.append(f"R {q}")
                    output_lines.append(f"H {q}")                


            else:
                # If there's some other gate we don't specifically handle,
                # keep it as is, or add more logic if needed.
                output_lines.append(stripped_line)

        return "\n".join(output_lines)





def rewrite_stim_code(code: str) -> str:
    """
    Rewrites a Stim program so that each line contains at most one gate or measurement.
    Lines starting with TICK, R, DETECTOR(, and OBSERVABLE_INCLUDE( are kept as-is.
    Multi-target lines for CX, M, and MR are split up.
    """
    lines = code.splitlines()
    output_lines = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            # Skip empty lines (optional: you could also preserve them)
            continue

        # Keep lines that we do NOT want to split
        if (stripped_line.startswith("TICK") or
            stripped_line.startswith("DETECTOR(") or
            stripped_line.startswith("QUBIT_COORDS(") or     
            stripped_line.startswith("OBSERVABLE_INCLUDE(")):
            output_lines.append(stripped_line)
            continue
        
        if (stripped_line.startswith("X_ERROR") or
            stripped_line.startswith("DEPOLARIZE1") or
            stripped_line.startswith("DEPOLARIZE2") or
            stripped_line.startswith("SHIFT_COORDS")            
            ):
            continue
            

        tokens = stripped_line.split()
        gate = tokens[0]

        # Handle 2-qubit gate lines like "CX 0 1 2 3 4 5 ..."
        if gate == "CX":
            qubits = tokens[1:]
            # Pair up the qubits [q0, q1, q2, q3, ...] => (q0,q1), (q2,q3), ...
            for i in range(0, len(qubits), 2):
                q1, q2 = qubits[i], qubits[i + 1]
                output_lines.append(f"CX {q1} {q2}")

        # Handle multi-qubit measurements "M 1 3 5 ..." => each on its own line
        elif gate == "M":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"M {q}")


        elif gate == "MX":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"H {q}")
                output_lines.append(f"M {q}")

        elif gate == "MY":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"S {q}")
                output_lines.append(f"S {q}")
                output_lines.append(f"S {q}")
                output_lines.append(f"H {q}")                
                output_lines.append(f"M {q}")



        elif gate == "H":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"H {q}")

        elif gate == "S":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"S {q}")            

        # Handle multi-qubit measure+reset "MR 1 3 5 ..." => each on its own line
        elif gate == "MR":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"M {q}")
                output_lines.append(f"R {q}")

        elif gate == "R":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"R {q}")
        
        elif gate == "RX":
            qubits = tokens[1:]
            for q in qubits:
                output_lines.append(f"R {q}")
                output_lines.append(f"H {q}")                


        else:
            # If there's some other gate we don't specifically handle,
            # keep it as is, or add more logic if needed.
            output_lines.append(stripped_line)

    return "\n".join(output_lines)