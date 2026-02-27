import numpy as np
import pennylane as qml
import copy

N_QUBITS        = 2
MAX_DEPTH       = 20
POP_SIZE        = 2000
NUM_GENERATIONS = 400
DEF_MUT_RATE    = 0.2
DEF_MUT_BOOST_COOLDOWN = 10
STAG_THRESH     = 15

rng = np.random.default_rng()
dev = qml.device('default.qubit', wires=N_QUBITS)


def random_unitary(d, rng=rng):
    Z = (rng.normal(size=(d, d)) +
         1j * rng.normal(size=(d, d))) / np.sqrt(2)

    Q, R = np.linalg.qr(Z)
    phases = np.diag(R) / np.abs(np.diag(R))
    return Q * phases

TARGET_MATRIX = random_unitary(2**N_QUBITS)

single_parametrised_gates = [qml.RX, qml.RY, qml.RZ]
single_gates = [
        qml.Hadamard, 
        qml.PauliX, qml.PauliY, qml.PauliZ, 
        qml.S, qml.T,
        qml.RX, qml.RY, qml.RZ
    ]
multi_gates = [qml.CNOT, qml.CZ]

def gen_circuit_structure (
        single_gates=single_gates, multi_gates=multi_gates, 
        single_gate_pref=0.5
        ) :
    """
    This function creates a list of lists with gates and their target wires in order
    
    :param single_gates     : list of pennylane single-qubit gates
    :param multi_gates      : list of pennylane multi-qubit gates
    :param single_gate_pref : how much a single-qubit gate is preferred over a multi-qubit gate
    """

    depth = rng.choice(range(1, MAX_DEPTH+1))

    # don't use multi gates if there is only 1 line
    if N_QUBITS < 2 : single_gate_pref = 1

    structure = []

    for d in range(depth) :
        # now at each depth level, assign some gates
        theta = rng.uniform(0, 2*np.pi)
        if rng.random() < single_gate_pref :
            wire = rng.choice(N_QUBITS)
            gate = rng.choice(single_gates)
            structure.append([gate, wire, theta]) # theta will be meaningless for non-paramterised gates
        else :
            wires = rng.choice(N_QUBITS, size=2, replace=False).tolist()
            gate = rng.choice(multi_gates)
            structure.append([gate, wires, theta])

    return structure

@qml.qnode(dev)
def exec_circuit (structure) :
    """
    Executes the given structure in pennylane
    
    :param structure: list of lists with gates and their target wires in order
    """
    for gate, wires, theta in structure :
        if gate in single_parametrised_gates :
            gate(theta, wires=wires)
        else :
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
    fitnesses = []
    
    for i in range (pop_size) :
        structure = gen_circuit_structure()
        population.append(structure)
        
        U = np.array(qml.matrix(exec_circuit)(structure)) # type: ignore
        fitness = calc_fitness(U)
        fitnesses.append(fitness)

    return population, fitnesses

def sort_pop (population, fitnesses) :
    # sort fitnesses and population according to descending order of fitnesses
    idx = np.argsort(fitnesses)[::-1]
    fitnesses = np.array(fitnesses)[idx]
    population = np.array(population, dtype=object)[idx]

    return population, fitnesses

def select_parents (population, fitnesses) :
    probs = np.array(fitnesses)
    probs = probs / probs.sum()

    parents_idx = np.random.choice(len(fitnesses), size=2, p=probs, replace=False)
    
    parent1_idx = parents_idx[0]
    parent2_idx = parents_idx[1]

    return population[parent1_idx], population[parent2_idx]

def crossover(parent1, parent2):
    m = len(parent1)
    n = len(parent2)

    max_len = (m + n) // 2
    max_len = max(2, max_len)    

    len1 = rng.integers(1, max_len + 1)
    len2 = rng.integers(1, max_len + 1)

    child1 = parent1[:len1] + parent2[-len2:]
    child2 = parent2[:len1] + parent1[-len2:]

    return copy.deepcopy(child1), copy.deepcopy(child2)

def mutate(ind,
           mutation_rate=DEF_MUT_RATE,
           angle_sigma=0.3,
           p_change_gate=0.2,
           p_change_wire=0.2,
           p_angle_mut=0.6,
           p_insert_delete=0.1):

    ind = copy.deepcopy(ind)

    # skip mutation
    if rng.random() > mutation_rate: return ind

    # angle mutations
    for gene in ind:
        if len(gene) == 3 and gene[0] in single_parametrised_gates:
            if rng.random() < p_angle_mut:
                gene[2] += rng.normal(0, angle_sigma)
                gene[2] %= 2*np.pi   # wrap angle

    # pick random gene for structural mutation
    idx = rng.integers(0, len(ind))

    # change gate
    if rng.random() < p_change_gate:
        if ind[idx][0] in single_gates:
            ind[idx][0] = rng.choice(single_gates)
        else:
            ind[idx][0] = rng.choice(multi_gates)

    # change wires
    if rng.random() < p_change_wire:
        if ind[idx][0] in single_gates:
            ind[idx][1] = rng.integers(N_QUBITS)
        else:
            ind[idx][1] = rng.choice(N_QUBITS, 2, replace=False).tolist()

    # insert/delete gate
    if rng.random() < p_insert_delete:
        if rng.random() < 0.5 and len(ind) > 1:
            del ind[idx]              # delete
        else:
            ind.append(gen_circuit_structure()[0])  # insert random gate

    return ind


def eval_pop (population) :
    fitnesses = []
    for ind in population :
        U = qml.matrix(exec_circuit)(ind) # type: ignore
        fitnesses.append(calc_fitness(U))

    return fitnesses

def remove_global_phase(U, V):
    overlap = np.trace(V.conj().T @ U)
    phase = overlap / np.abs(overlap)   # unit complex number
    return U / phase

def genetic_algorithm (elitism_frac=0.2) :
    print("Target matrix :")
    print(np.round(TARGET_MATRIX, 2)) 

    population, fitnesses = create_population(pop_size=POP_SIZE)
    population, fitnesses = sort_pop(population, fitnesses)

    best_fitnesses = [0]
    mut_rate = DEF_MUT_RATE
    stag_count = 0
    mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN

    generation = 0

    while (generation<NUM_GENERATIONS) and (fitnesses[0]<0.99999) :
        if generation % 10 == 0 :
            print(f"Generation   {generation}")
            print(f"Best fitness {fitnesses[0]}")
        
        if fitnesses[0]==best_fitnesses[-1] : stag_count += 1
        else : 
            stag_count = 0
            if (mut_rate > DEF_MUT_RATE) and (mut_boost_cooldown==0):
                mut_rate *= 0.9
                print("Reducing mutation rate")
                mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN
        mut_boost_cooldown = max(mut_boost_cooldown-1, 0)

        if (stag_count > STAG_THRESH) and (mut_boost_cooldown==0):
            mut_rate *= 1.2
            print("Boosting mutation rate")
            mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN


        best_fitnesses.append(fitnesses[0])

        num_elites = (int)(elitism_frac * POP_SIZE)
        children = [copy.deepcopy(ind) for ind in population[:num_elites]]

        # generate next population
        for _ in range((int)((POP_SIZE-num_elites)/2)) :
            # select two parents
            parent1, parent2 = select_parents(population, fitnesses)

            # perform crossover
            child1, child2 = crossover(parent1, parent2)

            # mutate
            child1 = mutate(child1)
            child2 = mutate(child2)

            children.append(child1)
            children.append(child2)


        population = children
        # evaluate the population
        fitnesses = eval_pop(population)

        # sort population and fitnesses values
        population, fitnesses = sort_pop(population, fitnesses)

        generation += 1


    population, fitnesses = sort_pop(population, fitnesses)

    U = qml.matrix(exec_circuit)(population[0]) # type: ignore
    U_fixed = remove_global_phase(U, TARGET_MATRIX)

    print()
    print(f"{generation} generations completed")
    print(f"Best fitness value : {fitnesses[0]}")
    print("Best matrix :")
    print(np.round(U_fixed, 2))

    print("Target matrix :")
    print(np.round(TARGET_MATRIX, 2))    
    print(qml.draw(exec_circuit)(population[0]))


genetic_algorithm()