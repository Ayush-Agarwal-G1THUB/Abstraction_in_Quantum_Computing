import numpy as np
import pennylane as qml

n_qubits = 2
dev = qml.device('default.qubit', wires=n_qubits)

single_gates = [qml.Hadamard, qml.PauliX, qml.PauliY, qml.PauliZ]
multi_gates = [qml.CNOT, qml.CZ]

# this function generates the structure of a random circuit 
# with n_qubits with depth less than or equal to the max_depth
def gen_circuit_structure (
        n_qubits, max_depth, 
        single_gates=single_gates, multi_gates=multi_gates, single_gate_pref=0.5
        ) :
    rng = np.random.default_rng()
    depth = rng.choice(range(1, max_depth+1))

    # don't use multi gates if there is only 1 line
    if n_qubits < 2 : single_gate_pref = 1

    structure = []

    for d in range(depth) :
        # now at each depth level, assign some gates
        if rng.random() < single_gate_pref :
            wire = rng.choice(n_qubits)
            gate = rng.choice(single_gates)
            structure.append([gate, wire])
        else :
            wires = rng.choice(n_qubits, size=2, replace=False).tolist()
            gate = rng.choice(multi_gates)
            structure.append([gate, wires])

    return structure

@qml.qnode(dev)
def exec_circuit (structure) :
    for gate, wires in structure :
        gate(wires=wires)

    return qml.state()

def create_population (pop_size) :
    population = []
    for i in range (pop_size) :
        structure = gen_circuit_structure(n_qubits, max_depth=5)
        population.append(structure)

        print(qml.draw(exec_circuit)(structure))
        U = np.array(qml.matrix(exec_circuit)(structure)) # type: ignore

        print(U)
        print()

    return population

create_population (5)