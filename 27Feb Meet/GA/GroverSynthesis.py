import pennylane as qml
from TaskAgnosticGA import SynthesisTask, genetic_algorithm
import math

# --- ALLOWED GATES ---
single_parametrised_gates = []
single_gates = [
        qml.Hadamard, 
        qml.PauliX, qml.PauliY, qml.PauliZ, 
        qml.S, qml.T
    ]
multi_gates = [qml.CNOT, qml.CZ]
gates = [single_gates, single_parametrised_gates, multi_gates]

class GroverTask(SynthesisTask):
    def __init__(self, target_bitstring="110"):
        n_qubits = len(target_bitstring)
        super().__init__(n_qubits, gates)
        
        self.target_idx = int(target_bitstring, 2)
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(self.dev)
        def _circuit(structure):
            for i in range(self.n_qubits): qml.Hadamard(wires=i)
            
            num_iters = int(math.pi/4 * self.n_qubits)
            for _ in range(num_iters):
                # Oracle
                qml.FlipSign(self.target_idx, wires=range(self.n_qubits))
                # Diffuser
                for gate, wires, theta in structure:
                    if gate in single_parametrised_gates: gate(theta, wires=wires)
                    else: gate(wires=wires)
                    
            return qml.probs(wires=range(self.n_qubits))
            
        self.qnode = _circuit

    def evaluate(self, structure):
        prob = float(self.qnode(structure)[self.target_idx])
        return prob - (len(structure) * 0.001)

    def print_result(self, best_structure):
        print(f"Grover Result for {self.n_qubits} Qubits")
        print(qml.draw(self.qnode)(best_structure))

if __name__ == "__main__":
    target_bitstring = "1011"
    print(target_bitstring)
    task = GroverTask(target_bitstring) 
    genetic_algorithm(task)