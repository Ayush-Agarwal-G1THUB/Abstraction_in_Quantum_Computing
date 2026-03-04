import pennylane as qml
from TaskAgnosticGA import SynthesisTask, genetic_algorithm
import numpy as np

# --- ALLOWED GATES ---
single_parametrised_gates = [qml.RX, qml.RY, qml.RZ]
single_gates = [
        qml.Hadamard, 
        qml.PauliX, qml.PauliY, qml.PauliZ, 
        qml.S, qml.T,
        qml.RX, qml.RY, qml.RZ
    ]
multi_gates = [qml.CNOT, qml.CZ, qml.Toffoli, qml.MultiControlledX]

class VQETask(SynthesisTask):
    def __init__(self, hamiltonian):
        gates = [single_gates, single_parametrised_gates, multi_gates]
        n_qubits = max([max(o.wires) for o in hamiltonian.ops]) + 1
        super().__init__(n_qubits, gates)
        
        self.H = hamiltonian
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(self.dev)
        def _circuit(structure):
            for gate, wires, theta in structure:
                if gate in single_parametrised_gates: gate(theta, wires=wires)
                else: gate(wires=wires)
            return qml.expval(self.H)
        self.qnode = _circuit

    def evaluate(self, structure):
        energy = float(self.qnode(structure))
        return -energy - (len(structure) * 0.005)

    def print_result(self, best_structure):
        print(f"VQE Result for {self.n_qubits} Qubits")
        print(qml.draw(self.qnode)(best_structure))

def create_Hamiltonian(n_qubits=4, connectivity_prob=0.6):
    coeffs = []
    obs = []

    # --- Random pairwise interactions ---
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):

            if np.random.rand() < connectivity_prob:

                Jx = np.random.uniform(-1, 1)
                Jy = np.random.uniform(-1, 1)
                Jz = np.random.uniform(-1, 1)

                coeffs.append(Jx)
                obs.append(qml.PauliX(i) @ qml.PauliX(j))

                coeffs.append(Jy)
                obs.append(qml.PauliY(i) @ qml.PauliY(j))

                coeffs.append(Jz)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    # --- Random local fields ---
    for i in range(n_qubits):

        hx = np.random.uniform(-1, 1)
        hy = np.random.uniform(-1, 1)
        hz = np.random.uniform(-1, 1)

        coeffs.append(hx)
        obs.append(qml.PauliX(i))

        coeffs.append(hy)
        obs.append(qml.PauliY(i))

        coeffs.append(hz)
        obs.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, obs)

def ground_state_energy(H):
    H_mat = np.array(qml.matrix(H))
    eigenvalues = np.linalg.eigvalsh(H_mat)
    return np.min(eigenvalues)

if __name__ == "__main__":
    H = create_Hamiltonian(n_qubits=4)
    print(H)
    print("True ground energy = ", ground_state_energy(H))
    task = VQETask(H)
    genetic_algorithm(task)