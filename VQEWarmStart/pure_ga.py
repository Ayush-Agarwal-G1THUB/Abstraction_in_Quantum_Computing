import numpy as np
import pennylane as qml  # type: ignore
import copy
from core import create_Hamiltonian, ground_state_energy
import json

# --- DEFAULT GA PARAMETERS ---
MAX_DEPTH = 30
POP_SIZE = 200
NUM_GENERATIONS = 50
DEF_MUT_RATE = 0.3
DEF_MUT_BOOST_COOLDOWN = 10
STAG_THRESH = 15

TOURNAMENT = "tournament"
ROULETTE = "roulette"

rng = np.random.default_rng()

# --- ALLOWED GATES ---
single_parametrised_gates = [] #type: ignore
single_gates = [
        qml.Hadamard, 
        qml.PauliX, qml.PauliY, qml.PauliZ, 
        qml.S, qml.T
    ]
multi_gates = [qml.CNOT, qml.CZ, qml.Toffoli, qml.MultiControlledX]

gates = [single_gates, single_parametrised_gates, multi_gates]

# ==========================================
# GENETIC ALGORITHM
# ==========================================

def evaluate(structure):
    energy = float(qnode(structure))
    return -energy - (len(structure) * 0.005)

def print_result(structure):
    print()
    print(f"Circuit diagram:")
    print(qml.draw(qnode)(structure))


def gen_circuit_structure (n_qubits, gates, max_depth, single_gate_pref=0.5):
    depth = rng.choice(range(1, max_depth + 1))

    single_gates = gates[0]
    single_parametrised_gates = gates[1]
    multi_gates = gates[2]

    if n_qubits < 2:
        single_gate_pref = 1
    structure = []

    for d in range(depth):
        theta = rng.uniform(0, 2 * np.pi)
        if rng.random() < single_gate_pref:
            wire = int(rng.choice(n_qubits))
            gate = rng.choice(single_gates)
            structure.append([gate, wire, theta])
        else:
            gate = rng.choice(multi_gates)

            # Use getattr to get num_wires, default to 2
            raw_needed = getattr(gate, "num_wires", 2)

            if not isinstance(raw_needed, int):
                needed_wires = rng.choice(range(2, n_qubits + 1))
            else:
                needed_wires = min(raw_needed, n_qubits)

            wires = rng.choice(n_qubits, size=needed_wires, replace=False).tolist()
            structure.append([gate, wires, theta])

    return structure


def create_population(pop_size, max_depth, n_qubits, gates):
    population = []
    fitnesses = []
    for _ in range(pop_size):
        structure = gen_circuit_structure(n_qubits, gates, max_depth)
        population.append(structure)
        fitness = evaluate(structure)
        fitnesses.append(fitness)
    return population, fitnesses


def sort_pop(population, fitnesses):
    idx = np.argsort(fitnesses)[::-1]
    fitnesses = np.array(fitnesses)[idx]
    population = np.array(population, dtype=object)[idx]
    return population, fitnesses


def select_parents(population, fitnesses, selection_type=TOURNAMENT):
    if selection_type == ROULETTE:
        probs = np.array(fitnesses)
        probs = probs / probs.sum()
        parents_idx = np.random.choice(len(fitnesses), size=2, p=probs, replace=False)
        return population[parents_idx[0]], population[parents_idx[1]]

    if selection_type == TOURNAMENT:
        tournament_size = 5

        def pick_one():
            candidates = rng.choice(len(fitnesses), size=tournament_size)
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            return population[best_idx]

        return pick_one(), pick_one()


def crossover(parent1, parent2):
    m, n = len(parent1), len(parent2)
    max_len = max(2, (m + n) // 2)
    len1 = rng.integers(1, max_len + 1)
    len2 = rng.integers(1, max_len + 1)
    child1 = parent1[:len1] + parent2[-len2:]
    child2 = parent2[:len1] + parent1[-len2:]
    return copy.deepcopy(child1), copy.deepcopy(child2)


def mutate(ind, n_qubits, gates, max_depth, mutation_rate):
    ind = copy.deepcopy(ind)
    if rng.random() > mutation_rate:
        return ind

    # Angle mutation
    for gene in ind:
        if len(gene) == 3 and gene[0] in gates[1]:
            if rng.random() < 0.6:
                gene[2] = (gene[2] + rng.normal(0, 0.3)) % (2 * np.pi)

    if not ind:
        return ind
    idx = rng.integers(0, len(ind))

    # Gate/Wire Mutation
    if rng.random() < 0.3:
        is_single = not isinstance(ind[idx][1], list)
        new_gate = rng.choice(gates[0]) if is_single else rng.choice(gates[2])
        ind[idx][0] = new_gate

        # Robustly determine wires for the new gate
        raw_needed = getattr(new_gate, "num_wires", 1)
        if not isinstance(raw_needed, int):
            needed = rng.choice(range(2, n_qubits + 1))
        else:
            needed = raw_needed

        if needed == 1:
            ind[idx][1] = int(rng.choice(n_qubits))
        else:
            n_to_pick = min(needed, n_qubits)
            ind[idx][1] = rng.choice(n_qubits, size=n_to_pick, replace=False).tolist()

    # Insert/Delete Mutation
    if rng.random() < 0.1:
        if rng.random() < 0.5 and len(ind) > 1:
            del ind[idx]
        else:
            ind.append(gen_circuit_structure(n_qubits, gates, max_depth, single_gate_pref=0.5)[0])

    return ind


def eval_pop(population):
    fitnesses = []
    for ind in population:
        fitnesses.append(evaluate(ind))
    return fitnesses


def genetic_algorithm(
        n_qubits,
        gates,
        n_results:int,
        max_depth=MAX_DEPTH,
        pop_size = POP_SIZE,
        num_generations = NUM_GENERATIONS,
        mut_rate = DEF_MUT_RATE,
        DEF_MUT_RATE = DEF_MUT_RATE,
        mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN,
        DEF_MUT_BOOST_COOLDOWN = DEF_MUT_BOOST_COOLDOWN,
        stag_thresh = STAG_THRESH,
        elitism_frac=0.2
    ):
    population, fitnesses = create_population(max_depth, pop_size, n_qubits, gates)
    population, fitnesses = sort_pop(population, fitnesses)

    best_fitnesses = [0]
    stag_count = 0
    generation = 0

    while generation < num_generations:
        if generation % 10 == 0:
            print(f"  Generation {generation:2} | Best fitness = {fitnesses[0]:.6f}")

        if fitnesses[0] == best_fitnesses[-1]:
            stag_count += 1
        else:
            stag_count = 0
            if (mut_rate > DEF_MUT_RATE) and (mut_boost_cooldown == 0):
                mut_rate *= 0.9
                print("     Reducing mutation rate")
                mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN

        mut_boost_cooldown = max(mut_boost_cooldown - 1, 0)

        if (stag_count > stag_thresh) and (mut_boost_cooldown == 0):
            mut_rate = min(1.2 * mut_rate, 0.8)
            print("     Boosting mutation rate")
            mut_boost_cooldown = DEF_MUT_BOOST_COOLDOWN

        best_fitnesses.append(fitnesses[0])
        num_elites = int(elitism_frac * pop_size)
        children = [copy.deepcopy(ind) for ind in population[:num_elites]]

        for _ in range(int((pop_size - num_elites) / 2)):
            parent1, parent2 = select_parents(population, fitnesses, TOURNAMENT)  # type: ignore
            child1, child2 = crossover(parent1, parent2)
            children.append(mutate(child1, n_qubits, gates, max_depth, mut_rate))
            children.append(mutate(child2, n_qubits, gates, max_depth, mut_rate))

        population = children
        fitnesses = eval_pop(population)  # Pass task here
        population, fitnesses = sort_pop(population, fitnesses)
        generation += 1

    population, fitnesses = sort_pop(population, fitnesses)
    
    best_fitness = -100 * fitnesses[0] / true_ground_energy
    print(f"Best fitness: {best_fitness:.2f}%")

    # print_result(population[0])

    return population[:n_results], fitnesses[:n_results], best_fitnesses



# ==========================================
# EXECUTION
# ==========================================

n_qubits_range = range(5, 10)
all_results = [] # type: ignore

for n_qubits in n_qubits_range :
    H = create_Hamiltonian(n_qubits=n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)
    true_ground_energy = float(ground_state_energy(H))

    dev = qml.device('default.qubit', wires=n_qubits)
    @qml.qnode(dev)
    def _circuit(structure):
        for gate, wires, theta in structure:
            target_wires = wires if isinstance(wires, list) else int(wires)
            if gate in single_parametrised_gates:
                gate(theta, wires=target_wires)
            else: gate(wires=target_wires)
        return qml.expval(H)
    qnode = _circuit

    print("==============================================")
    print(f"NUM QUBITS = {n_qubits}")
    print(f"True ground energy = {true_ground_energy}")

    max_depth_range = range(20, 50, 10)

    for max_depth in max_depth_range :

        num_generations_range = range(50, 60, 10)

        for num_generations in num_generations_range :
            print(f"\nQubits: {n_qubits} | Max depth: {max_depth}")

            results, fitnesses, energy_history = genetic_algorithm (
                n_qubits=n_qubits,
                gates = gates,
                n_results=1,
                max_depth=max_depth,
                num_generations=num_generations
            )

            all_results.append({
                "n_qubits": n_qubits,
                "max_depth": max_depth,
                "num_generations": num_generations,
                "final_energy": energy_history[0],
                "true_energy": true_ground_energy,
                "history": energy_history
            })

    print()

with open("ga_results.json", "w") as f:
    json.dump(all_results, f, indent=4)