import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    filename = "ga_results.json"
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
        neg_history = [-abs(h) for h in run["history"]] 
        plt.plot(neg_history, label=f"{run['max_depth']} Max depth")
    
    # Add a horizontal line for the True Ground Energy
    plt.axhline(y=qubit_runs[0]["true_energy"], color='r', linestyle='--', 
                label=f"Target: {qubit_runs[0]['true_energy']:.4f}")
    
    plt.title(f"GA Convergence Profile: {n} Qubits")
    plt.xlabel("Generations")
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

plt.show()