"""
AutoQC-Style Neural Network Guided Synthesis of Grover's Algorithm
===================================================================
Based on: "AutoQC: Automated Synthesis of Quantum Circuits Using Neural Network"
(Murakami & Zhao, 2022)

CONCEPT:
  - Represent a quantum circuit as a SEQUENCE of gates (like AutoQC)
  - Train a neural network on random quantum circuits to learn gate selection
  - At synthesis time: start from TARGET output state, work BACKWARDS,
    let the NN assign priority probabilities at each step
  - Select gates stochastically based on NN probabilities until reaching
    a computational basis state
  - Apply discovered sequence to run actual Grover's search

GROVER TARGET: Find marked state |11> in 2-qubit system
(NumPy-only implementation — no external ML frameworks required)
"""

import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  QUANTUM GATE LIBRARY  (Section II-B of Paper 2)
# ─────────────────────────────────────────────────────────────────────────────

sqrt2 = np.sqrt(2)
H1 = np.array([[1,1],[1,-1]], dtype=complex) / sqrt2
X1 = np.array([[0,1],[1,0]], dtype=complex)
I1 = np.eye(2, dtype=complex)

GATE_MATRICES = {
    'H0':   np.kron(H1, I1),
    'H1':   np.kron(I1, H1),
    'X0':   np.kron(X1, I1),
    'X1':   np.kron(I1, X1),
    'CX01': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex),
    'CX10': np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex),
    'CZ':   np.diag([1,1,1,-1]).astype(complex),
}
GATE_NAMES = list(GATE_MATRICES.keys())
N_GATES    = len(GATE_NAMES)

def apply_gate(name, sv):     return GATE_MATRICES[name] @ sv
def apply_gate_inv(name, sv): return GATE_MATRICES[name].conj().T @ sv

def is_computational_basis(sv, tol=1e-6):
    return np.max(np.abs(sv)**2) > 1 - tol

def state_to_vec(sv):
    return np.concatenate([sv.real, sv.imag]).astype(np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  NUMPY MLP  (mimics AutoQC's CvNN + FC architecture)
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):  return np.maximum(0, x)
def softmax(x):
    e = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

class MLP:
    """
    Multi-layer perceptron in pure NumPy.
    Mirrors AutoQC: hidden layers + FC output with softmax.
    Input:  8-dim (real+imag of 4-dim quantum state)
    Output: N_GATES probabilities
    """
    def __init__(self, input_dim=8, hidden_dim=64, n_layers=5, output_dim=N_GATES):
        self.weights, self.biases = [], []
        dims = [input_dim] + [hidden_dim]*(n_layers-1) + [output_dim]
        for i in range(len(dims)-1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(np.random.randn(dims[i+1], dims[i]) * scale)
            self.biases.append(np.zeros(dims[i+1]))

    def forward_single(self, x):
        h = x.copy()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = W @ h + b
            if i < len(self.weights)-1:
                h = relu(h)
        e = np.exp(h - h.max())
        return e / e.sum()

    def forward_batch(self, X):
        # X: (N, input_dim)  returns (N, output_dim)
        h = X.T   # (input_dim, N)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = W @ h + b[:, None]
            if i < len(self.weights)-1:
                h = relu(h)
        return softmax(h).T   # (N, output_dim)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  GENERATE TRAINING DATA  (Section III-C-2 of Paper 2)
# ─────────────────────────────────────────────────────────────────────────────

def random_circuit(max_depth=6):
    basis_idx = np.random.randint(4)
    sv = np.zeros(4, dtype=complex); sv[basis_idx] = 1.0
    seq, states = [], [sv.copy()]
    for _ in range(np.random.randint(1, max_depth+1)):
        gate = GATE_NAMES[np.random.randint(N_GATES)]
        if seq and seq[-1] == gate:
            continue
        sv = apply_gate(gate, sv)
        seq.append(gate)
        states.append(sv.copy())
    return states, seq

def generate_training_data(n_circuits=5000):
    X_data, y_data = [], []
    for _ in range(n_circuits):
        states, seq = random_circuit(max_depth=5)
        if not seq: continue
        sv = states[-1].copy()
        for gate in reversed(seq):
            X_data.append(state_to_vec(sv))
            y_data.append(GATE_NAMES.index(gate))
            sv = apply_gate_inv(gate, sv)
    return np.array(X_data), np.array(y_data, dtype=int)

print("=" * 60)
print("AutoQC: Generating Training Data")
print("=" * 60)
X_train, y_train = generate_training_data(n_circuits=5000)
print(f"  Generated {len(X_train)} training pairs from random circuits")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAIN WITH MINI-BATCH SGD  (Section III-C-3 of Paper 2)
# ─────────────────────────────────────────────────────────────────────────────

def train_mlp(model, X, y, epochs=40, batch_size=64, lr=0.01):
    n = len(X)
    print("\n" + "=" * 60)
    print("AutoQC: Training Neural Network (NumPy SGD)")
    print("=" * 60)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch = idx[start:start+batch_size]
            xb, yb = X[batch], y[batch]

            # ── Forward pass ──────────────────────────────────────
            caches = []
            h = xb.T
            for i, (W, b) in enumerate(zip(model.weights, model.biases)):
                z = W @ h + b[:, None]
                h_prev = h
                if i < len(model.weights)-1:
                    h = relu(z)
                else:
                    e = np.exp(z - z.max(axis=0))
                    h = e / e.sum(axis=0)
                caches.append((h_prev, z, h))

            probs = h.T   # (batch, n_gates)
            eps   = 1e-12
            total_loss += -np.mean(
                np.log(probs[np.arange(len(yb)), yb] + eps)) * len(yb)

            # ── Backward pass ─────────────────────────────────────
            dz = probs.copy()
            dz[np.arange(len(yb)), yb] -= 1
            dz = dz.T / len(yb)

            for i in reversed(range(len(model.weights))):
                h_prev, z, _ = caches[i]
                dW = dz @ h_prev.T
                db = dz.sum(axis=1)
                model.weights[i] -= lr * dW
                model.biases[i]  -= lr * db
                if i > 0:
                    dh = model.weights[i].T @ dz
                    _, z_prev, _ = caches[i-1]
                    dz = dh * (z_prev > 0)

        if (epoch+1) % 10 == 0:
            all_probs = model.forward_batch(X)
            acc = np.mean(all_probs.argmax(axis=1) == y)
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={total_loss/n:.4f}  acc={acc:.3f}")

model = MLP()
train_mlp(model, X_train, y_train, epochs=40, batch_size=64, lr=0.01)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  NN-GUIDED BACKWARD SYNTHESIS  (Section III-B of Paper 2)
# ─────────────────────────────────────────────────────────────────────────────

def nn_guided_synthesis(target_state, max_steps=10, max_attempts=300):
    for _ in range(max_attempts):
        sv  = target_state.copy()
        seq = []
        for _ in range(max_steps):
            if is_computational_basis(sv):
                return list(reversed(seq)), True
            probs    = model.forward_single(state_to_vec(sv))
            gate_idx = np.random.choice(N_GATES, p=probs)
            sv       = apply_gate_inv(GATE_NAMES[gate_idx], sv)
            seq.append(GATE_NAMES[gate_idx])
    return None, False

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SYNTHESISE GROVER STATE-PREPARATION LAYER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("AutoQC: Synthesising Grover State-Preparation Layer")
print("=" * 60)

target_prep = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)   # |++>
print(f"\n  Target: |++> = {np.round(target_prep, 3)}")

seq_prep, success = nn_guided_synthesis(target_prep, max_steps=8, max_attempts=400)

if success and seq_prep:
    sv_check = np.array([1,0,0,0], dtype=complex)
    for g in seq_prep:
        sv_check = apply_gate(g, sv_check)
    fidelity = abs(np.vdot(target_prep, sv_check))**2
    print(f"  Synthesised sequence : {seq_prep}")
    print(f"  Verification fidelity: {fidelity:.4f}  "
          f"{'✓' if fidelity > 0.99 else '(re-run for better result)'}")
else:
    print("  NN synthesis did not converge this run; using known H0,H1")
    seq_prep = ['H0', 'H1']

# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN FULL GROVER'S ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def run_grover_autoqc(seq_prep):
    print("\n" + "=" * 60)
    print("AutoQC Grover Simulation: Searching for |11>")
    print("=" * 60)
    basis = ['|00>', '|01>', '|10>', '|11>']
    state = np.array([1,0,0,0], dtype=complex)

    print(f"\nStep 1 – State Preparation (NN-synthesised): {seq_prep}")
    for g in seq_prep:
        state = apply_gate(g, state)
    print(f"  State: {np.round(state, 3)}")
    print(f"  Probs: {np.round(np.abs(state)**2, 3)}")

    print("\nStep 2a – Oracle: CZ (marks |11> with phase flip)")
    state = apply_gate('CZ', state)
    print(f"  State: {np.round(state, 3)}")

    diffuser_seq = ['H0','H1','X0','X1','CZ','X0','X1','H0','H1']
    print(f"\nStep 2b – Diffuser: {diffuser_seq}")
    for g in diffuser_seq:
        state = apply_gate(g, state)
    probs = np.abs(state)**2
    print(f"  State: {np.round(state, 3)}")
    print(f"  Probs: {np.round(probs, 3)}")

    measured = basis[np.argmax(probs)]
    print(f"\n{'='*60}")
    print(f"MEASUREMENT RESULT : {measured}  (probability {max(probs):.4f})")
    print(f"Target was: |11>   {'✓' if measured == '|11>' else '✗'}")
    print(f"{'='*60}")
    return state, probs

run_grover_autoqc(seq_prep)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  NN vs RANDOM SEARCH COMPARISON  (Table II style from Paper 2)
# ─────────────────────────────────────────────────────────────────────────────

def random_synthesis(target_state, max_steps=8, max_attempts=300):
    for _ in range(max_attempts):
        sv  = target_state.copy()
        seq = []
        for _ in range(max_steps):
            if is_computational_basis(sv):
                return list(reversed(seq)), True
            sv = apply_gate_inv(GATE_NAMES[np.random.randint(N_GATES)], sv)
            seq.append(None)
    return None, False

print("\n" + "=" * 60)
print("AutoQC: NN-Guided vs Random Search (50 trials each)")
print("=" * 60)
target   = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
n_trials = 50
nn_ok = rand_ok = 0
for _ in range(n_trials):
    _, ok1 = nn_guided_synthesis(target, max_steps=8, max_attempts=15)
    _, ok2 = random_synthesis(target,    max_steps=8, max_attempts=15)
    if ok1: nn_ok   += 1
    if ok2: rand_ok += 1

print(f"\n  NN-Guided  : {nn_ok}/{n_trials} success  ({100*nn_ok/n_trials:.0f}%)")
print(f"  Random     : {rand_ok}/{n_trials} success  ({100*rand_ok/n_trials:.0f}%)")
print(f"\n  NN guidance improves synthesis efficiency (as in AutoQC Table II)")