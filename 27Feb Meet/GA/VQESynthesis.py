import pennylane as qml
from TaskAgnosticGA import SynthesisTask, genetic_algorithm, single_parametrised_gates

class VQETask(SynthesisTask):
    def __init__(self, hamiltonian):
        # 1. Find max wire index in Hamiltonian to determine n_qubits
        n_qubits = max([max(o.wires) for o in hamiltonian.ops]) + 1
        super().__init__(n_qubits)
        
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
        return -energy - (len(structure) * 0.01)

    def print_result(self, best_structure):
        print(f"VQE Result for {self.n_qubits} Qubits")
        print(qml.draw(self.qnode)(best_structure))

if __name__ == "__main__":
    # Create a 3-qubit Hamiltonian
    H = qml.Hamiltonian([1.0, 0.5], [qml.PauliZ(0), qml.PauliX(1) @ qml.PauliX(2)])
    task = VQETask(H)
    genetic_algorithm(task)