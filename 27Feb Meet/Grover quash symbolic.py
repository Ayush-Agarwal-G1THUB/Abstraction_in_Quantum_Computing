"""
QUASH-Style Symbolic Synthesis of Grover's Algorithm
=====================================================
Based on: "Automated Synthesis of Quantum Circuits using Symbolic 
Abstractions and Decision Procedures" (Velasquez et al.)

CONCEPT:
  - Represent quantum states as SYMBOLS (α=|0>, β=|1>, γ=|+>, δ=|-> etc.)
  - Build symbolic I/O tables for each gate (like Table III in the paper)
  - Use a decision-procedure-style search (constraint satisfaction) to find
    the gate sequence that transforms input symbols to desired output symbols
  - Apply the discovered circuit to run actual Grover's search

GROVER TARGET: Find marked state |11> among {|00>,|01>,|10>,|11>}
"""

import numpy as np
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYMBOLIC STATE DEFINITIONS  (Section III-A of Paper 1)
# ─────────────────────────────────────────────────────────────────────────────

# Each symbol maps to a concrete quantum state vector
SYMBOLS = {
    'alpha': np.array([1, 0], dtype=complex),          # |0>
    'beta':  np.array([0, 1], dtype=complex),          # |1>
    'gamma': np.array([1, 1], dtype=complex) / np.sqrt(2),   # |+>
    'delta': np.array([1,-1], dtype=complex) / np.sqrt(2),   # |->
    'bot':   None,   # ⊥ — unknown / don't-care
}

def vec_to_symbol(v):
    """Map a concrete state vector back to its symbol (or 'bot')."""
    if v is None:
        return 'bot'
    for name, sv in SYMBOLS.items():
        if sv is not None and np.allclose(v, sv, atol=1e-9):
            return name
    return 'bot'

def symbol_to_vec(s):
    return SYMBOLS.get(s)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  GATE LIBRARY WITH SYMBOLIC I/O TABLES  (Section III-A)
# ─────────────────────────────────────────────────────────────────────────────

# Each gate is defined by its unitary matrix AND its symbolic I/O table
# symbolic_io[gate][input_symbol] = output_symbol

H = np.array([[1, 1],[1,-1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1],[1, 0]], dtype=complex)
Z = np.array([[1, 0],[0,-1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def apply_gate(U, v):
    return U @ v

def build_symbolic_io(U):
    """Auto-build symbolic I/O table from gate matrix (like paper does automatically)."""
    table = {}
    for name, sv in SYMBOLS.items():
        if sv is not None:
            out = apply_gate(U, sv)
            table[name] = vec_to_symbol(out)
        else:
            table[name] = 'bot'
    return table

# Build tables automatically
sym_H = build_symbolic_io(H)
sym_X = build_symbolic_io(X)
sym_Z = build_symbolic_io(Z)
sym_I = build_symbolic_io(I2)

print("=" * 60)
print("QUASH: Symbolic I/O Tables for Gate Library")
print("=" * 60)
for gate_name, table in [("H", sym_H), ("X", sym_X), ("Z", sym_Z), ("I", sym_I)]:
    print(f"\n  {gate_name} gate:")
    for inp, out in table.items():
        if inp != 'bot':
            print(f"    {gate_name}({inp}) = {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  GROVER ORACLE & DIFFUSER (2-qubit, target |11>)
# ─────────────────────────────────────────────────────────────────────────────
# For 2 qubits the full 4x4 matrices are used for actual simulation

def kron(*gates):
    result = gates[0]
    for g in gates[1:]:
        result = np.kron(result, g)
    return result

CZ = np.diag([1, 1, 1, -1]).astype(complex)   # marks |11>

def grover_oracle():
    """Oracle that flips phase of |11>."""
    return CZ

def grover_diffuser():
    """2-qubit Grover diffuser: 2|s><s| - I  where |s> = |++>."""
    HH = kron(H, H)
    XX = kron(X, X)
    # Apply H⊗H -> X⊗X -> CZ -> X⊗X -> H⊗H
    return HH @ XX @ CZ @ XX @ HH

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SYMBOLIC CIRCUIT SYNTHESIS (Decision-Procedure Style Search)
# ─────────────────────────────────────────────────────────────────────────────
# We define the DESIRED symbolic transformation for Grover's state preparation:
#   Input:  both qubits start as 'alpha' (|0>)
#   Goal:   both qubits become 'gamma' (|+>) after the H⊗H layer
#
# The synthesiser searches over gate sequences to find one whose
# symbolic trace matches the specification — mimicking Z3 constraint solving.

GATE_LIB_1Q = {
    'H': (H, sym_H),
    'X': (X, sym_X),
    'Z': (Z, sym_Z),
    'I': (I2, sym_I),
}

def symbolic_apply_1q(gate_name, sym_in):
    _, table = GATE_LIB_1Q[gate_name]
    return table.get(sym_in, 'bot')

def synthesise_hadamard_layer(target_in='alpha', target_out='gamma', max_depth=3):
    """
    Search for a 1-qubit gate sequence that maps target_in -> target_out
    symbolically.  Returns the first sequence found (like QUASH's SAT check).
    """
    gates = list(GATE_LIB_1Q.keys())
    for depth in range(1, max_depth + 1):
        for seq in product(gates, repeat=depth):
            sym = target_in
            for g in seq:
                sym = symbolic_apply_1q(g, sym)
                if sym == 'bot':
                    break
            if sym == target_out:
                return list(seq)
    return None

print("\n" + "=" * 60)
print("QUASH: Symbolic Search for State-Preparation Layer")
print("=" * 60)
seq = synthesise_hadamard_layer('alpha', 'gamma')
print(f"\n  Spec:  alpha(|0>) ──gates──> gamma(|+>)")
print(f"  Found: {' -> '.join(seq)}")
print(f"  (Applies to each qubit independently)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL GROVER'S ALGORITHM USING THE SYNTHESISED CIRCUIT
# ─────────────────────────────────────────────────────────────────────────────

def build_unitary_from_seq(seq_1q):
    """Convert a 1-qubit gate name sequence to its unitary product."""
    U = I2.copy()
    for g in seq_1q:
        mat, _ = GATE_LIB_1Q[g]
        U = mat @ U
    return U

def run_grover_quash():
    print("\n" + "=" * 60)
    print("QUASH Grover Simulation: Searching for |11>")
    print("=" * 60)

    # Initial state |00>
    state = np.array([1, 0, 0, 0], dtype=complex)

    # Use synthesised gate sequence for state preparation
    U_prep_1q = build_unitary_from_seq(seq)
    U_prep    = kron(U_prep_1q, U_prep_1q)   # apply to both qubits

    print(f"\nStep 1 – State Preparation (synthesised: {seq} ⊗ {seq})")
    state = U_prep @ state
    probs = np.abs(state)**2
    print(f"  State amplitudes: {np.round(state, 3)}")
    print(f"  Probabilities:    {np.round(probs, 3)}")

    oracle   = grover_oracle()
    diffuser = grover_diffuser()
    basis    = ['|00>', '|01>', '|10>', '|11>']

    # Grover iterations  (optimal = 1 for 2 qubits / 4 states)
    n_iter = 1
    for i in range(n_iter):
        print(f"\nStep {i+2}a – Oracle (mark |11>)")
        state = oracle @ state
        print(f"  State amplitudes: {np.round(state, 3)}")

        print(f"Step {i+2}b – Diffusion (amplify marked state)")
        state = diffuser @ state
        probs = np.abs(state)**2
        print(f"  State amplitudes: {np.round(state, 3)}")
        print(f"  Probabilities:    {np.round(probs, 3)}")

    # Measurement
    probs = np.abs(state)**2
    measured = basis[np.argmax(probs)]
    print(f"\n{'='*60}")
    print(f"MEASUREMENT RESULT: {measured}  (probability {max(probs):.4f})")
    print(f"Target was: |11>  ✓" if measured == '|11>' else f"Target was: |11>  ✗")
    print(f"{'='*60}")

    return state, probs

state, probs = run_grover_quash()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  OPTIMALITY CHECK  (mimics QUASH's UNSAT proof)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("QUASH: Optimality / UNSAT Check")
print("=" * 60)
print("\nChecking: can we achieve |0>->|+> in 0 gates? (should be UNSAT)")
result = synthesise_hadamard_layer('alpha', 'gamma', max_depth=0)
print(f"  Depth 0: {'SAT – ' + str(result) if result else 'UNSAT (as expected)'}")
print("\nChecking: can we achieve |0>->|+> in 1 gate? (should be SAT with H)")
result = synthesise_hadamard_layer('alpha', 'gamma', max_depth=1)
print(f"  Depth 1: {'SAT – ' + str(result) if result else 'UNSAT'}")