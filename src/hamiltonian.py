import numpy as np
from qiskit.quantum_info import SparsePauliOp

class H2Hamiltonian:
    """
    Hamiltonian for the Neutral Hydrogen Molecule Association-Dissociation Model.
    
    Qubit Mapping (7 qubits):
    Q0: Photon Up (Omega_up)
    Q1: Photon Down (Omega_down)
    Q2: Phonon (omega)
    Q3: Electron Up (l1)
    Q4: Electron Down (l2)
    Q5: Bond (L)
    Q6: Tunnel (k)
    """
    
    def __init__(self, 
                 omega_up=1e9, 
                 omega_down=1e9, 
                 omega_phonon=1e8,
                 hbar=1.0):
        self.omega_up = omega_up
        self.omega_down = omega_down
        self.omega_phonon = omega_phonon
        self.hbar = hbar
        
        # Define basic operators for a single qubit
        # a = 0.5 * (X + iY) = |0><1| (annihilation: 1 -> 0)
        # a_dag = 0.5 * (X - iY) = |1><0| (creation: 0 -> 1)
        # n = 0.5 * (I - Z) = |1><1| (number operator)
        # sigma = |0><1| (same as a for 2-level system)
        
        self.I = SparsePauliOp("I")
        self.X = SparsePauliOp("X")
        self.Y = SparsePauliOp("Y")
        self.Z = SparsePauliOp("Z")
        
        self.a = 0.5 * (self.X + 1j * self.Y)
        self.a_dag = 0.5 * (self.X - 1j * self.Y)
        self.n = 0.5 * (self.I - self.Z)
        
    def _op(self, op_dict):
        """
        Helper to create a 7-qubit operator from a dictionary {qubit_index: single_qubit_op}.
        """
        full_op = SparsePauliOp("I" * 7)
        for q_idx, op in op_dict.items():
            # Construct the operator string list and coeffs
            # But SparsePauliOp doesn't support easy tensor product by index directly like this
            # So we build the tensor product manually
            
            # We need to apply 'op' at position q_idx (0 is rightmost in Qiskit usually, 
            # but let's define 0 as leftmost for clarity in code, then reverse for Qiskit if needed.
            # Qiskit Little Endian: q0 is rightmost.
            # Let's stick to Qiskit convention: q0 is rightmost char in string.
            
            # If we want q0 to be Photon Up, it's the rightmost character.
            
            current_op = op
            # We need to identity on all other qubits
            # This is inefficient to do repeatedly. 
            # Better approach: Create the full list of Paulis.
            pass
        
        # Let's use a simpler approach: compose the full operator directly
        # q0 is rightmost
        ops_list = [self.I] * 7
        for q, o in op_dict.items():
            ops_list[q] = o
            
        # Tensor product: op[6] ^ ... ^ op[0]
        res = ops_list[6]
        for i in range(5, -1, -1):
            res = res.tensor(ops_list[i])
        return res

    def get_hamiltonian_op(self, g_omega, g_phonon, zeta):
        """
        Constructs the full Hamiltonian operator.
        """
        # 1. Free energy terms
        # H_free = hbar * (omega_up * n0 + omega_down * n1 + omega_phonon * n2 + ...)
        
        # Operators for each mode
        # Q0: Photon Up
        n_up = self._op({0: self.n})
        a_up = self._op({0: self.a})
        a_up_dag = self._op({0: self.a_dag})
        
        # Q1: Photon Down
        n_down = self._op({1: self.n})
        a_down = self._op({1: self.a})
        a_down_dag = self._op({1: self.a_dag})
        
        # Q2: Phonon
        n_phonon = self._op({2: self.n})
        a_phonon = self._op({2: self.a})
        a_phonon_dag = self._op({2: self.a_dag})
        
        # Q3: Electron Up (sigma_up)
        n_sigma_up = self._op({3: self.n})
        sigma_up = self._op({3: self.a}) # |0><1|
        sigma_up_dag = self._op({3: self.a_dag}) # |1><0|
        
        # Q4: Electron Down (sigma_down)
        n_sigma_down = self._op({4: self.n})
        sigma_down = self._op({4: self.a})
        sigma_down_dag = self._op({4: self.a_dag})
        
        # Q5: Bond (sigma_bond)
        # L=0 (bond formed) -> |0>. L=1 (bond broken) -> |1>.
        # Wait, paper says: L=0 formation, L=1 breaking.
        # Operators: sigma_bond |0>_cb = 0, sigma_bond |1>_cb = |0>_cb.
        # So sigma_bond is lowering operator |0><1|.
        # n_sigma_bond = sigma_bond_dag * sigma_bond = |1><1| (counts if bond is broken L=1)
        n_sigma_bond = self._op({5: self.n})
        sigma_bond = self._op({5: self.a})
        sigma_bond_dag = self._op({5: self.a_dag})
        
        # Q6: Tunnel (sigma_tunnel)
        # k=0 together, k=1 apart.
        # sigma_tunnel |1> = |0>.
        sigma_tunnel = self._op({6: self.a})
        sigma_tunnel_dag = self._op({6: self.a_dag})
        
        
        H = SparsePauliOp("I" * 7, coeffs=[0.0])
        
        # Free Energy
        H += self.hbar * self.omega_up * n_up
        H += self.hbar * self.omega_down * n_down
        H += self.hbar * self.omega_phonon * n_phonon
        H += self.hbar * self.omega_up * n_sigma_up
        H += self.hbar * self.omega_down * n_sigma_down
        H += self.hbar * self.omega_phonon * n_sigma_bond
        
        # Interaction Terms
        
        # Bond formed operator: sigma_bond * sigma_bond_dag = |0><1| * |1><0| = |0><0|
        # This projects onto L=0 (bond formed)
        # Note: self.n is |1><1|. So |0><0| is I - n.
        bond_formed = self._op({5: self.I}) - n_sigma_bond
        
        # Nuclei Together Projector (k=0)
        # Q6 is Tunnel (k). k=0 is together.
        # P_together = |0><0|_k = 0.5 * (I + Z)
        P_together = self._op({6: 0.5 * (self.I + self.Z)})
        
        # Jaynes-Cummings Up: g_omega * (a_up_dag * sigma_up + a_up * sigma_up_dag) * bond_formed
        # Paper says hybridization (and thus bond formation) only possible when atoms together.
        # Once bond is formed, does it imply atoms are together? 
        # Usually yes. But let's assume the gating is primarily on the FORMATION process (g_phonon).
        # However, if the bond is formed, the electrons can interact with photons.
        # Let's leave JC terms as is (dependent on bond_formed).
        jc_up = (a_up_dag @ sigma_up) + (a_up @ sigma_up_dag)
        H += g_omega * (jc_up @ bond_formed)
        
        # Jaynes-Cummings Down: g_omega * (a_down_dag * sigma_down + a_down * sigma_down_dag) * bond_formed
        jc_down = (a_down_dag @ sigma_down) + (a_down @ sigma_down_dag)
        H += g_omega * (jc_down @ bond_formed)
        
        # Phonon-Bond Interaction: g_phonon * (a_phonon_dag * sigma_bond + a_phonon * sigma_bond_dag)
        # This creates/breaks the bond.
        # Constraint 1: Hybridization (bond formation) only possible if atoms are together (k=0).
        bond_phonon = (a_phonon_dag @ sigma_bond) + (a_phonon @ sigma_bond_dag)
        H += g_phonon * (bond_phonon @ P_together)
        
        # Tunneling: zeta * (sigma_tunnel_dag * sigma_tunnel + sigma_tunnel * sigma_tunnel_dag)
        # Constraint 2: Tunneling (scattering to k=1) only possible if bond is broken (L=1).
        # If bonded (L=0), atoms are held together.
        # P_broken = |1><1|_L = n_sigma_bond
        P_broken = n_sigma_bond
        
        # Tunneling operator (X on Q6)
        tunnel_op = self._op({6: self.X})
        H += zeta * (tunnel_op @ P_broken)
        
        return H.simplify()

