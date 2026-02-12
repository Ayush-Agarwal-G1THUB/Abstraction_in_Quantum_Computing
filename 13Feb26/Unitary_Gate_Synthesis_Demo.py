import numpy as np
import pennylane as qml

n_qubits = 2
max_depth = 10
dev = qml.device('default.qubit', wires=n_qubits)
TARGET_MATRIX = 0.5 * np.array([
    [1,  1,  1,  1],
    [1,  1j, -1, -1j],
    [1, -1,  1, -1],
    [1, -1j, -1,  1j]
])

single_gates = [qml.Hadamard, qml.PauliX, qml.PauliY, qml.PauliZ, qml.S, qml.T]
multi_gates = [qml.CNOT, qml.CZ]

def gen_circuit_structure (
        n_qubits, max_depth, 
        single_gates=single_gates, multi_gates=multi_gates, single_gate_pref=0.5
        ) :
    """
    This function creates a list of lists with gates and their target wires in order
    
    :param n_qubits         : number of wires
    :param max_depth        : maximum sequential depth of the circuit
    :param single_gates     : list of pennylane single-qubit gates
    :param multi_gates      : list of pennylane multi-qubit gates
    :param single_gate_pref : how much a single-qubit gate is preferred over a multi-qubit gate
    """

    rng = np.random.default_rng()
    depth = rng.choice(range(0, max_depth+1))

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
    """
    Executes the given structure in pennylane
    
    :param structure: list of lists with gates and their target wires in order
    """
    for gate, wires in structure :
        gate(wires=wires)

    return qml.state()

def calc_fitness(U) :
    """
    Calculates the fitness of a given circuit structure compared to a target matrix
    
    :param U: probability density function of the given structure
    """
    V = TARGET_MATRIX
    d = U.shape[0]
    overlap = np.trace(V.conj().T @ U)
    return np.abs(overlap / d)**2

    

def create_population (pop_size) :
    """
    Creates a population of structures
    
    :param pop_size: size of the population
    """
    population = []
    max_fitness = 0
    default_structure = gen_circuit_structure(n_qubits, max_depth=max_depth)
    best_matrix = np.array(qml.matrix(exec_circuit)(default_structure)) # type: ignore
    best_structure = default_structure
    for i in range (pop_size) :
        structure = gen_circuit_structure(n_qubits, max_depth=max_depth)
        population.append(structure)

        # print(qml.draw(exec_circuit)(structure))
        
        U = np.array(qml.matrix(exec_circuit)(structure)) # type: ignore
        # print(U)
        fitness = calc_fitness(U)
        # print(fitness)

        if (fitness > max_fitness) :
            max_fitness = fitness
            best_matrix = U
            best_structure = structure

    print(f"Target matrix")
    print(TARGET_MATRIX)
    print()
    print(f"Best fitness value = {max_fitness}")
    print(best_matrix)
    print(qml.draw(exec_circuit)(best_structure))

    return population


create_population (1000)