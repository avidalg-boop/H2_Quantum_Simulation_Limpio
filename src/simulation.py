import numpy as np
from qiskit.quantum_info import Statevector, partial_trace, entropy

class QiskitSimulation:
    def __init__(self, hamiltonian_op):
        self.H = hamiltonian_op
        
    def evolve(self, initial_state, times):
        """
        Evolves the initial state for each time in times.
        Returns a list of Statevectors.
        """
        # Statevector.evolve(H, t) computes exp(-iHt) * state
        # But it might be more efficient to compute unitary for steps if times are uniform.
        # However, for simplicity and exactness, we can just evolve from t=0 or step-by-step.
        # Step-by-step is better to avoid large t errors if any (though exact exponentiation is fine).
        
        # Let's do step-by-step
        dt = times[1] - times[0]
        # Create unitary for one step
        # U_step = exp(-i * H * dt)
        # Qiskit Statevector.evolve allows passing an operator.
        # But SparsePauliOp doesn't support exponentiation directly in all versions efficiently?
        # Actually Statevector.evolve(op) treats op as H? No, it applies op.
        # Wait, Statevector.evolve(other) -> "Evolve statevector by operator."
        # If other is Operator or QuantumCircuit.
        
        # We need exp(-i H dt).
        # We can convert SparsePauliOp to dense matrix (Operator) and exponentiate.
        # Since dim is small (2^7 = 128), this is very fast.
        
        from qiskit.quantum_info import Operator
        from scipy.linalg import expm
        
        H_matrix = self.H.to_matrix()
        # Check hermiticity
        # print(np.linalg.norm(H_matrix - H_matrix.conj().T))
        
        # U_step = expm(-1j * H_matrix * dt)
        # But times might not be perfectly uniform?
        # The main script usually uses linspace.
        
        # Let's just compute exact evolution for each t from t=0 to avoid accumulation errors?
        # No, step-by-step is standard.
        
        U_step = expm(-1j * H_matrix * dt)
        U_step_op = Operator(U_step)
        
        states = []
        current_state = initial_state
        states.append(current_state)
        
        # We assume times[0] = 0.
        for i in range(1, len(times)):
            current_state = current_state.evolve(U_step_op)
            states.append(current_state)
            
        return states

    def compute_entropy(self, state, subsystem_qubits):
        """
        Computes von Neumann entropy of the subsystem.
        """
        # partial_trace requires list of qubits to TRACE OUT.
        # So we need to invert subsystem_qubits.
        all_qubits = set(range(7))
        keep_qubits = set(subsystem_qubits)
        trace_qubits = list(all_qubits - keep_qubits)
        
        rho_reduced = partial_trace(state, trace_qubits)
        return entropy(rho_reduced, base=2)

def get_initial_state():
    """
    Constructs the initial state |Psi_initial> from Eq (12).
    |Psi_initial> = 0.5 * (|Phi0_up Phi0_down> + |Phi1_up Phi0_down> - |Phi0_up Phi1_down> - |Phi1_up Phi1_down>)
    
    Mapping:
    Q0: Photon Up
    Q1: Photon Down
    Q2: Phonon
    Q3: Electron Up (l1)
    Q4: Electron Down (l2)
    Q5: Bond (L)
    Q6: Tunnel (k)
    
    Basis states:
    |Phi0_up Phi0_down> = |0>_ph_up |0>_ph_down |0>_phonon |0>_l1 |0>_l2 |1>_L |1>_k
    Indices:              0          1            2           3       4       5      6
    State string (Q6...Q0): "1100000"
    
    Wait, let's check Eq (13a-d).
    |Phi0_up Phi0_down> (Eq 13a): |0>_OmUp |0>_OmDown |0>_om |0>_l1 |0>_l2 |1>_L |1>_k
    Qubits: Q0=0, Q1=0, Q2=0, Q3=0, Q4=0, Q5=1, Q6=1.
    Ket: |1100000> (ordering Q6 Q5 Q4 Q3 Q2 Q1 Q0)
    
    |Phi1_up Phi0_down> (Eq 13b): ... |1>_l1 ...
    Ket: |1101000> (Q3=1)
    
    |Phi0_up Phi1_down> (Eq 13c): ... |1>_l2 ...
    Ket: |1110000> (Q4=1)
    
    |Phi1_up Phi1_down> (Eq 13d): ... |1>_l1 |1>_l2 ...
    Ket: |1111000> (Q3=1, Q4=1)
    
    Psi = 0.5 * (|1100000> + |1101000> - |1110000> - |1111000>)
    """
    
    # Qiskit Statevector.from_label uses string "Qn...Q0"
    
    psi1 = Statevector.from_label("1100000")
    psi2 = Statevector.from_label("1101000")
    psi3 = Statevector.from_label("1110000")
    psi4 = Statevector.from_label("1111000")
    
    psi = 0.5 * (psi1 + psi2 - psi3 - psi4)
    return psi
