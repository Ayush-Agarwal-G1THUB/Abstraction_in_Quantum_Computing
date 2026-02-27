import pennylane as qml
from TaskAgnosticGA import SynthesisTask, genetic_algorithm, single_parametrised_gates

class GroverTask(SynthesisTask):
    def __init__(self, target_bitstring="110"):
        n_qubits = len(target_bitstring)
        super().__init__(n_qubits)
        
        self.target_idx = int(target_bitstring, 2)
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(self.dev)
        def _circuit(structure):
            for i in range(self.n_qubits): qml.Hadamard(wires=i)
            qml.FlipSign(self.target_idx, wires=range(self.n_qubits))
            for gate, wires, theta in structure:
                if gate in single_parametrised_gates: gate(theta, wires=wires)
                else: gate(wires=wires)
            return qml.probs(wires=range(self.n_qubits))
        self.qnode = _circuit

    def evaluate(self, structure):
        prob = float(self.qnode(structure)[self.target_idx])
        return prob - (len(structure) * 0.005)

    def print_result(self, best_structure):
        print(f"Grover Result for {self.n_qubits} Qubits")
        print(qml.draw(self.qnode)(best_structure))

if __name__ == "__main__":
    task = GroverTask(target_bitstring="1010") 
    genetic_algorithm(task)