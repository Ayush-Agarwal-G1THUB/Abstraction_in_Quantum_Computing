import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    filename = "vqe_results.json"
else : 
    filename = sys.argv[1]

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

with open(filename, "r") as f:
    data = json.load(f)

# Get unique qubit counts present in the results
unique_qubits = sorted(list(set(run["n_qubits"] for run in data)))

for n in unique_qubits:
    plt.figure(figsize=(10, 6))
    
    # Filter runs for this specific qubit count
    qubit_runs = [run for run in data if run["n_qubits"] == n]
    
    for run in qubit_runs:
        plt.plot(run["history"], label=f"{run['n_layers']} Layers")
    
    # Add a horizontal line for the True Ground Energy
    plt.axhline(y=qubit_runs[0]["true_energy"], color='r', linestyle='--', 
                label=f"Target: {qubit_runs[0]['true_energy']:.4f}")
    
    plt.title(f"VQE Convergence Profile: {n} Qubits")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Energy (Expectation Value)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Best Fitness vs Number of Qubits
plt.figure(figsize=(10, 6))
best_fitness_per_n = []
for n in unique_qubits:
    max_fit = max(run["fitness"] for run in data if run["n_qubits"] == n)
    best_fitness_per_n.append(max_fit * 100)

plt.bar([str(n) for n in unique_qubits], best_fitness_per_n, color='teal')
plt.axhline(y=99.5, color='orange', linestyle='--', label="Threshold (99.5%)")
plt.ylabel("Best Accuracy Achieved (%)")
plt.xlabel("Number of Qubits")
plt.title("VQE Scalability: Best Accuracy per Qubit Count")
plt.ylim(max(0, min(best_fitness_per_n)-5), 105)
plt.legend()
plt.show()