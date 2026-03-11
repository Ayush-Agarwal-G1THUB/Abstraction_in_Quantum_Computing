import json
import pennylane as qml #type: ignore
from pennylane import numpy as pnp
from core import create_Hamiltonian, ground_state_energy

n_qubits_range = range(5, 10)
all_results = []

for n_qubits in n_qubits_range:
    threshold_met = False

    H = create_Hamiltonian(n_qubits=n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)
    true_ground_energy = float(ground_state_energy(H))

    print("==============================================")
    print(f"NUM QUBITS = {n_qubits}")
    print(f"True ground energy = {true_ground_energy}")

    @qml.qnode(dev)
    def ansatz_qnode(params):
        qml.StronglyEntanglingLayers(weights=params, wires=range(n_qubits))
        return qml.expval(H)

    for n_layers in range(2, 50, 7):
        if threshold_met : break

        print(f"\nQubits: {n_qubits} | Layers: {n_layers}")
        
        # Initialize params (0 to 2*pi is better for rotations)
        shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        params = pnp.random.uniform(low=0, high=2*pnp.pi, size=shape, requires_grad=True) # type: ignore
        
        opt = qml.AdamOptimizer(stepsize=0.1)
        energy_history = []

        max_steps = 1000

        for step in range(max_steps):
            params, energy = opt.step_and_cost(ansatz_qnode, params)
            energy_history.append(float(energy))
            
            if step % 200 == 0 or step==max_steps-1:
                print(f"  Step {step:3}: Energy = {energy:.6f}")

            if energy/true_ground_energy > 0.995 :
                pass
                break

        fitness = energy / true_ground_energy # type: ignore
        print(f"Fitness = {fitness}")
        
        # Store metadata for visualization
        all_results.append({
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "final_energy": float(energy), # type: ignore
            "true_energy": true_ground_energy,
            "fitness": float(fitness),
            "history": energy_history
        })

        if fitness > 0.995 :
            threshold_met = True
            break

    print()

# Save to JSON for the visualization script
with open("pure_vqe_results.json", "w") as f:
    json.dump(all_results, f, indent=4)