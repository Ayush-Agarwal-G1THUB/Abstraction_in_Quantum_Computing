import numpy as np
import pennylane as qml #type: ignore
from abc import ABC, abstractmethod

# ==========================================
# TASK INTERFACE
# ==========================================
class SynthesisTask(ABC):
    """
    Abstract base class for any quantum synthesis problem.
    The GA will only interact with this interface.
    """

    def __init__(self, n_qubits, gates):
        self.n_qubits = n_qubits
        self.gates = gates

    @abstractmethod
    def evaluate(self, structure) -> float:
        pass

    @abstractmethod
    def print_result(self, best_structure):
        pass


# ==========================================
# UNITARY DECOMPOSITION
# ==========================================
class UnitaryDecompositionTask(SynthesisTask):
    """
    Specific implementation for synthesizing a circuit that matches a Target Unitary Matrix.
    """

    def __init__(self, target_matrix, n_qubits, gates):
        self.target = target_matrix
        self.n_qubits = n_qubits
        self.gates = gates
        self.dev = qml.device("default.qubit", wires=n_qubits)

        single_gates = gates[0]
        single_parametrised_gates = gates[1]
        multi_gates = gates[2]

        # We define the exec_circuit QNode inside the task so it binds to the task's device
        @qml.qnode(self.dev)
        def _exec_circuit(structure):
            for gate, wires, theta in structure:
                if gate in single_parametrised_gates:
                    gate(theta, wires=wires)
                else:
                    gate(wires=wires)
            return qml.state()

        self.exec_circuit = _exec_circuit

    def evaluate(self, structure) -> float:
        # Get matrix and calculate overlap
        U = np.array(qml.matrix(self.exec_circuit)(structure))  # type: ignore
        V = self.target
        d = U.shape[0]
        overlap = np.trace(V.conj().T @ U)

        # Calculate raw fidelity
        fidelity = np.abs(overlap / d) ** 2
        return fidelity

    def remove_global_phase(self, U):
        overlap = np.trace(self.target.conj().T @ U)
        phase = overlap / np.abs(overlap)  # unit complex number
        return U / phase

    def print_result(self, best_structure):
        U = qml.matrix(self.exec_circuit)(best_structure)  # type: ignore
        U_fixed = self.remove_global_phase(U)

        print("Best matrix :")
        print(np.round(U_fixed, 2))
        print("\nTarget matrix :")
        print(np.round(self.target, 2))
        print("\nSynthesized Circuit:")
        print(qml.draw(self.exec_circuit)(best_structure))

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