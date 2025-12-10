"""
Bosonic Simulation for H2 Neutral Molecule Association-Dissociation.

This module implements the hybrid boson-fermion Hamiltonian from the paper
using truncated bosonic modes (dim > 2) instead of qubits for photons/phonons.

Uses qiskit.quantum_info for exact linear algebra simulation.
"""

import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Operator, Statevector, partial_trace, entropy


# =============================================================================
# 1. HILBERT SPACE DIMENSIONS
# =============================================================================

class HilbertSpaceConfig:
    """
    Configuration for the hybrid Hilbert space.
    
    Structure (following Eq. 3 of the paper):
    |Ψ⟩_C = |p1⟩ ⊗ |p2⟩ ⊗ |m⟩ ⊗ |l1⟩ ⊗ |l2⟩ ⊗ |L⟩ ⊗ |k⟩
    
    Where:
    - p1, p2: Photon modes (Ω↑, Ω↓) - Bosonic, truncated at dim_photon
    - m: Phonon mode (ω) - Bosonic, truncated at dim_phonon  
    - l1, l2: Electron orbital states - Fermionic (2-level)
    - L: Bond state - Fermionic (2-level)
    - k: Tunneling state - Fermionic (2-level)
    """
    
    def __init__(self, dim_photon: int = 5, dim_phonon: int = 5):
        self.dim_photon = dim_photon  # Truncation for p1, p2
        self.dim_phonon = dim_phonon  # Truncation for m
        self.dim_fermion = 2          # l1, l2, L, k are 2-level
        
        # Subsystem dimensions in order: [p1, p2, m, l1, l2, L, k]
        self.dims = [
            dim_photon,    # 0: Photon Up (Ω↑)
            dim_photon,    # 1: Photon Down (Ω↓)
            dim_phonon,    # 2: Phonon (ω)
            2,             # 3: Electron Up (l1)
            2,             # 4: Electron Down (l2)
            2,             # 5: Bond (L)
            2,             # 6: Tunnel (k)
        ]
        
        self.total_dim = np.prod(self.dims)
        self.n_subsystems = len(self.dims)
        
    def __repr__(self):
        return (f"HilbertSpaceConfig(dim_photon={self.dim_photon}, "
                f"dim_phonon={self.dim_phonon}, total_dim={self.total_dim})")


# =============================================================================
# 2. BOSONIC AND FERMIONIC OPERATORS
# =============================================================================

def annihilation_operator(dim: int) -> np.ndarray:
    """
    Create bosonic annihilation operator a for d-dimensional space.
    a|n⟩ = √n |n-1⟩
    """
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n-1, n] = np.sqrt(n)
    return a


def creation_operator(dim: int) -> np.ndarray:
    """
    Create bosonic creation operator a† for d-dimensional space.
    a†|n⟩ = √(n+1) |n+1⟩
    """
    return annihilation_operator(dim).conj().T


def number_operator(dim: int) -> np.ndarray:
    """
    Create number operator n = a†a for d-dimensional space.
    n|n⟩ = n |n⟩
    """
    return np.diag(np.arange(dim, dtype=complex))


def sigma_lower() -> np.ndarray:
    """
    Fermionic lowering operator σ = |0⟩⟨1|
    σ|1⟩ = |0⟩, σ|0⟩ = 0
    """
    return np.array([[0, 1], [0, 0]], dtype=complex)


def sigma_raise() -> np.ndarray:
    """
    Fermionic raising operator σ† = |1⟩⟨0|
    σ†|0⟩ = |1⟩, σ†|1⟩ = 0
    """
    return np.array([[0, 0], [1, 0]], dtype=complex)


def sigma_number() -> np.ndarray:
    """
    Fermionic number operator n = σ†σ = |1⟩⟨1|
    """
    return np.array([[0, 0], [0, 1]], dtype=complex)


def identity(dim: int) -> np.ndarray:
    """Identity matrix of given dimension."""
    return np.eye(dim, dtype=complex)


# =============================================================================
# 3. TENSOR PRODUCT UTILITIES
# =============================================================================

def tensor_chain(*matrices: np.ndarray) -> np.ndarray:
    """
    Compute tensor product A1 ⊗ A2 ⊗ ... ⊗ An
    """
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def embed_operator(config: HilbertSpaceConfig, 
                   subsystem_idx: int, 
                   op: np.ndarray) -> np.ndarray:
    """
    Embed a single-subsystem operator into the full Hilbert space.
    
    Places 'op' at position 'subsystem_idx' and identity elsewhere.
    """
    ops = []
    for i, dim in enumerate(config.dims):
        if i == subsystem_idx:
            ops.append(op)
        else:
            ops.append(identity(dim))
    return tensor_chain(*ops)


def embed_two_operators(config: HilbertSpaceConfig,
                        idx1: int, op1: np.ndarray,
                        idx2: int, op2: np.ndarray) -> np.ndarray:
    """
    Embed two operators at different positions.
    Returns: I ⊗ ... ⊗ op1 ⊗ ... ⊗ op2 ⊗ ... ⊗ I
    """
    ops = []
    for i, dim in enumerate(config.dims):
        if i == idx1:
            ops.append(op1)
        elif i == idx2:
            ops.append(op2)
        else:
            ops.append(identity(dim))
    return tensor_chain(*ops)


def embed_operators(config: HilbertSpaceConfig,
                    op_dict: dict) -> np.ndarray:
    """
    Embed multiple operators at specified positions.
    
    Args:
        config: Hilbert space configuration
        op_dict: Dictionary {subsystem_index: operator}
    
    Returns:
        Full tensor product operator
    """
    ops = []
    for i, dim in enumerate(config.dims):
        if i in op_dict:
            ops.append(op_dict[i])
        else:
            ops.append(identity(dim))
    return tensor_chain(*ops)


# =============================================================================
# 4. HAMILTONIAN CONSTRUCTION (Eq. 4 of the paper)
# =============================================================================

class BosonicH2Hamiltonian:
    """
    Full Hamiltonian for the H2 molecule simulation using bosonic modes.
    
    Implements Eq. (4) from the paper with proper tensor products.
    
    Parameters from paper:
    - Ω↑ = Ω↓ = 10^9 Hz (photon frequencies)
    - ω = 10^8 Hz (phonon frequency)
    - g = 10^7 (base coupling)
    - g_Ω varies: g, 1.5g, 2g, 4g for different simulations
    """
    
    def __init__(self, 
                 config: HilbertSpaceConfig,
                 omega_photon_up: float = 1e9,
                 omega_photon_down: float = 1e9,
                 omega_phonon: float = 1e8,
                 hbar: float = 1.0):
        
        self.config = config
        self.omega_up = omega_photon_up
        self.omega_down = omega_photon_down
        self.omega_phonon = omega_phonon
        self.hbar = hbar
        
        # Pre-compute basic operators for each subsystem
        self._build_basic_operators()
        
    def _build_basic_operators(self):
        """Pre-compute commonly used operators."""
        dim_p = self.config.dim_photon
        dim_m = self.config.dim_phonon
        
        # Bosonic operators for photons and phonon
        self.a_p = annihilation_operator(dim_p)
        self.a_p_dag = creation_operator(dim_p)
        self.n_p = number_operator(dim_p)
        
        self.a_m = annihilation_operator(dim_m)
        self.a_m_dag = creation_operator(dim_m)
        self.n_m = number_operator(dim_m)
        
        # Fermionic operators
        self.sigma = sigma_lower()
        self.sigma_dag = sigma_raise()
        self.n_f = sigma_number()
        
        # Identity operators
        self.I_p = identity(dim_p)
        self.I_m = identity(dim_m)
        self.I_f = identity(2)
        
    def build(self, g_omega: float, g_phonon: float, zeta: float) -> np.ndarray:
        """
        Build the full Hamiltonian matrix.
        
        Args:
            g_omega: Photon-electron coupling strength
            g_phonon: Phonon-bond coupling strength
            zeta: Tunneling intensity
            
        Returns:
            Hamiltonian matrix as numpy array
        """
        cfg = self.config
        H = np.zeros((cfg.total_dim, cfg.total_dim), dtype=complex)
        
        # =====================================================================
        # FREE ENERGY TERMS: H_free = ℏΩ↑ n_{p1} + ℏΩ↓ n_{p2} + ℏω n_m + ...
        # =====================================================================
        
        # Photon Up energy: ℏΩ↑ a†_{p1} a_{p1}
        H += self.hbar * self.omega_up * embed_operator(cfg, 0, self.n_p)
        
        # Photon Down energy: ℏΩ↓ a†_{p2} a_{p2}
        H += self.hbar * self.omega_down * embed_operator(cfg, 1, self.n_p)
        
        # Phonon energy: ℏω a†_m a_m
        H += self.hbar * self.omega_phonon * embed_operator(cfg, 2, self.n_m)
        
        # Orbital energies (using photon frequencies for orbitals as per paper)
        H += self.hbar * self.omega_up * embed_operator(cfg, 3, self.n_f)
        H += self.hbar * self.omega_down * embed_operator(cfg, 4, self.n_f)
        
        # Bond energy (using phonon frequency)
        H += self.hbar * self.omega_phonon * embed_operator(cfg, 5, self.n_f)
        
        # =====================================================================
        # INTERACTION TERMS
        # =====================================================================
        
        # Projector onto bond formed state (L=0): |0⟩⟨0|_L = I - n_L
        P_bond_formed = self.I_f - self.n_f  # |0⟩⟨0| projector
        
        # Projector onto nuclei together (k=0): |0⟩⟨0|_k
        P_together = self.I_f - self.n_f
        
        # ---------------------------------------------------------------------
        # Jaynes-Cummings Up: g_Ω (a†_{p1} σ_{l1} + a_{p1} σ†_{l1}) × P_bond_formed
        # ---------------------------------------------------------------------
        # Term: a†_{p1} σ_{l1}
        jc_up_1 = embed_operators(cfg, {
            0: self.a_p_dag,
            3: self.sigma,
            5: P_bond_formed
        })
        # Term: a_{p1} σ†_{l1}
        jc_up_2 = embed_operators(cfg, {
            0: self.a_p,
            3: self.sigma_dag,
            5: P_bond_formed
        })
        H += g_omega * (jc_up_1 + jc_up_2)
        
        # ---------------------------------------------------------------------
        # Jaynes-Cummings Down: g_Ω (a†_{p2} σ_{l2} + a_{p2} σ†_{l2}) × P_bond_formed
        # ---------------------------------------------------------------------
        jc_down_1 = embed_operators(cfg, {
            1: self.a_p_dag,
            4: self.sigma,
            5: P_bond_formed
        })
        jc_down_2 = embed_operators(cfg, {
            1: self.a_p,
            4: self.sigma_dag,
            5: P_bond_formed
        })
        H += g_omega * (jc_down_1 + jc_down_2)
        
        # ---------------------------------------------------------------------
        # Phonon-Bond Interaction: g_ω (a†_m σ_L + a_m σ†_L) × P_together
        # PHYSICAL CONSTRAINT: Bonding only occurs when nuclei are together (k=0)
        # ---------------------------------------------------------------------
        phonon_bond_1 = embed_operators(cfg, {
            2: self.a_m_dag,
            5: self.sigma,
            6: P_together  # Constraint: k=0
        })
        phonon_bond_2 = embed_operators(cfg, {
            2: self.a_m,
            5: self.sigma_dag,
            6: P_together  # Constraint: k=0
        })
        H += g_phonon * (phonon_bond_1 + phonon_bond_2)
        
        # ---------------------------------------------------------------------
        # Tunneling: ζ (σ†_k σ_k + σ_k σ†_k) = ζ × X_k (Pauli X on tunnel qubit)
        # This represents hopping between nuclei together/apart states
        # ---------------------------------------------------------------------
        X_tunnel = self.sigma + self.sigma_dag  # Pauli X = σ + σ†
        tunnel_term = embed_operator(cfg, 6, X_tunnel)
        H += zeta * tunnel_term
        
        return H
    
    def get_operator(self, g_omega: float, g_phonon: float, zeta: float) -> Operator:
        """Return Hamiltonian as Qiskit Operator."""
        return Operator(self.build(g_omega, g_phonon, zeta))


# =============================================================================
# 5. INITIAL STATE (Eq. 12 and 13 of the paper)
# =============================================================================

def get_initial_state_bosonic(config: HilbertSpaceConfig) -> np.ndarray:
    """
    Construct the initial state |Ψ_initial⟩ from Eq. (12) of the paper.
    
    |Ψ_initial⟩ = 0.5 (|Φ0↑ Φ0↓⟩ + |Φ1↑ Φ0↓⟩ - |Φ0↑ Φ1↓⟩ - |Φ1↑ Φ1↓⟩)
    
    Where (Eq. 13a-d):
    - |Φ0↑ Φ0↓⟩ = |0⟩_{p1} |0⟩_{p2} |0⟩_m |0⟩_{l1} |0⟩_{l2} |1⟩_L |1⟩_k
    - |Φ1↑ Φ0↓⟩ = |0⟩_{p1} |0⟩_{p2} |0⟩_m |1⟩_{l1} |0⟩_{l2} |1⟩_L |1⟩_k
    - |Φ0↑ Φ1↓⟩ = |0⟩_{p1} |0⟩_{p2} |0⟩_m |0⟩_{l1} |1⟩_{l2} |1⟩_L |1⟩_k
    - |Φ1↑ Φ1↓⟩ = |0⟩_{p1} |0⟩_{p2} |0⟩_m |1⟩_{l1} |1⟩_{l2} |1⟩_L |1⟩_k
    
    Initial condition: Photons and phonon in ground state |0⟩.
    L=1: Bond is broken (dissociated)
    k=1: Nuclei are apart (separated atoms)
    """
    dims = config.dims
    total_dim = config.total_dim
    
    def basis_vector(occupation: list) -> np.ndarray:
        """
        Create basis vector from occupation numbers.
        occupation = [n_p1, n_p2, n_m, n_l1, n_l2, n_L, n_k]
        """
        vec = np.zeros(total_dim, dtype=complex)
        
        # Compute linear index from occupation numbers
        # Index = n_p1 + d_p1 * (n_p2 + d_p2 * (n_m + d_m * (...)))
        idx = 0
        multiplier = 1
        for i, (n, d) in enumerate(zip(occupation, dims)):
            idx += n * multiplier
            multiplier *= d
        
        vec[idx] = 1.0
        return vec
    
    # Define the four basis states from Eq. (13)
    # Order: [p1, p2, m, l1, l2, L, k]
    # L=1 (bond broken), k=1 (apart)
    
    phi_00 = basis_vector([0, 0, 0, 0, 0, 1, 1])  # |Φ0↑ Φ0↓⟩
    phi_10 = basis_vector([0, 0, 0, 1, 0, 1, 1])  # |Φ1↑ Φ0↓⟩
    phi_01 = basis_vector([0, 0, 0, 0, 1, 1, 1])  # |Φ0↑ Φ1↓⟩
    phi_11 = basis_vector([0, 0, 0, 1, 1, 1, 1])  # |Φ1↑ Φ1↓⟩
    
    # Eq. (12): Superposition
    psi_initial = 0.5 * (phi_00 + phi_10 - phi_01 - phi_11)
    
    return psi_initial


def get_initial_statevector(config: HilbertSpaceConfig) -> Statevector:
    """Return initial state as Qiskit Statevector."""
    return Statevector(get_initial_state_bosonic(config))


# =============================================================================
# 6. TIME EVOLUTION (PTSIM equivalent)
# =============================================================================

class BosonicSimulation:
    """
    Time evolution simulation using exact matrix exponentiation.
    
    Equivalent to PTSIM method: U(t) = exp(-iHt/ℏ)
    """
    
    def __init__(self, H_matrix: np.ndarray, config: HilbertSpaceConfig):
        """
        Args:
            H_matrix: Hamiltonian as numpy array
            config: Hilbert space configuration (for partial trace)
        """
        self.H = H_matrix
        self.config = config
        
    def evolve_state(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """
        Evolve state from t=0 to time t.
        
        |ψ(t)⟩ = exp(-iHt) |ψ(0)⟩
        """
        U = expm(-1j * self.H * t)
        return U @ psi0
    
    def evolve_trajectory(self, psi0: np.ndarray, 
                          times: np.ndarray) -> list[np.ndarray]:
        """
        Compute state trajectory at multiple time points.
        
        For efficiency, uses stepwise evolution with uniform dt.
        """
        if len(times) < 2:
            return [psi0.copy()]
        
        dt = times[1] - times[0]
        U_step = expm(-1j * self.H * dt)
        
        states = [psi0.copy()]
        current = psi0.copy()
        
        for _ in range(1, len(times)):
            current = U_step @ current
            states.append(current)
            
        return states
    
    def compute_entropy(self, psi: np.ndarray, 
                        keep_subsystems: list[int]) -> float:
        """
        Compute von Neumann entropy S(ρ_A) for a subsystem.
        
        Uses custom partial trace for arbitrary dimension Hilbert spaces.
        
        Args:
            psi: Full system state vector
            keep_subsystems: Indices of subsystems to keep (trace out the rest)
            
        Returns:
            von Neumann entropy in bits (log base 2)
        """
        # Get reduced density matrix via custom partial trace
        rho_reduced = self._partial_trace(psi, keep_subsystems)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_reduced)
        
        # Filter out zeros and compute entropy: S = -Σ p_i log2(p_i)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0:
            return 0.0
        
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def _partial_trace(self, psi: np.ndarray, 
                       keep_subsystems: list[int]) -> np.ndarray:
        """
        Compute partial trace for arbitrary dimension subsystems.
        
        Args:
            psi: State vector of full system
            keep_subsystems: Indices of subsystems to keep
            
        Returns:
            Reduced density matrix as numpy array
        """
        dims = self.config.dims
        n = len(dims)
        
        # Reshape state vector into tensor with one index per subsystem
        psi_tensor = psi.reshape(dims)
        
        # Create density matrix ρ = |ψ⟩⟨ψ|
        # ρ[i1,i2,...,in, j1,j2,...,jn] = ψ[i1,...,in] * ψ*[j1,...,jn]
        rho_tensor = np.outer(psi, psi.conj()).reshape(dims + dims)
        
        # Determine which subsystems to trace out
        trace_out = sorted(set(range(n)) - set(keep_subsystems))
        keep = sorted(keep_subsystems)
        
        # Trace out subsystems by summing over diagonal elements
        # We need to contract indices: for subsystem k to trace out,
        # set index k = index (k + n) and sum
        
        # Work from highest index to lowest to avoid index shifting
        for k in reversed(trace_out):
            # Current tensor has shape [..., d_k, ..., d_k, ...]
            # where first d_k is at position k, second at position k + n_remaining
            n_remaining = rho_tensor.ndim // 2
            
            # Use np.trace which traces over two axes
            rho_tensor = np.trace(rho_tensor, axis1=k, axis2=k + n_remaining)
        
        # Remaining tensor has shape [d_keep1, ..., d_keepN, d_keep1, ..., d_keepN]
        # Reshape to matrix
        keep_dims = [dims[k] for k in keep]
        dim_A = int(np.prod(keep_dims))
        
        return rho_tensor.reshape(dim_A, dim_A)


# =============================================================================
# 7. CONVENIENCE FUNCTIONS FOR REPLICATING FIGURES
# =============================================================================

def run_simulation(g_omega: float, g_phonon: float, zeta: float,
                   times: np.ndarray,
                   subsystem_A: list[int] = [0, 1],
                   config: HilbertSpaceConfig = None) -> tuple[list, list]:
    """
    Run a complete simulation and return entropy trajectory.
    
    Args:
        g_omega: Photon-electron coupling
        g_phonon: Phonon-bond coupling
        zeta: Tunneling intensity
        times: Time points array
        subsystem_A: Subsystem indices for entropy (default: photons [0,1])
        config: Hilbert space config (default: dim_photon=5, dim_phonon=5)
        
    Returns:
        (times, entropies)
    """
    if config is None:
        config = HilbertSpaceConfig(dim_photon=5, dim_phonon=5)
    
    # Build Hamiltonian
    ham = BosonicH2Hamiltonian(config)
    H_matrix = ham.build(g_omega, g_phonon, zeta)
    
    # Initial state
    psi0 = get_initial_state_bosonic(config)
    
    # Simulation
    sim = BosonicSimulation(H_matrix, config)
    states = sim.evolve_trajectory(psi0, times)
    
    # Compute entropies
    entropies = [sim.compute_entropy(state, subsystem_A) for state in states]
    
    return list(times), entropies


# =============================================================================
# 8. EXAMPLE USAGE - Replicating Figure 2(a)
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Bosonic Simulation of H2 Molecule")
    print("=" * 60)
    
    # Configuration
    config = HilbertSpaceConfig(dim_photon=5, dim_phonon=5)
    print(f"\n{config}")
    print(f"Total Hilbert space dimension: {config.total_dim}")
    
    # Parameters from paper (Fig. 2a)
    G_BASE = 1e7
    g_phonon = 0.1 * G_BASE
    zeta = G_BASE
    
    t_max = 2e-6
    n_steps = 200
    times = np.linspace(0, t_max, n_steps)
    
    # Values of g_Omega to compare
    g_omega_values = [G_BASE, 1.5*G_BASE, 2*G_BASE, 4*G_BASE]
    colors = ['magenta', 'green', 'red', 'blue']
    labels = [r'$g_\Omega = g$', r'$g_\Omega = 1.5g$', 
              r'$g_\Omega = 2g$', r'$g_\Omega = 4g$']
    
    print("\nRunning simulations for Figure 2(a)...")
    
    plt.figure(figsize=(10, 6))
    
    for g_omega, color, label in zip(g_omega_values, colors, labels):
        print(f"  Computing g_Omega = {g_omega/G_BASE:.1f}g ...")
        
        _, entropies = run_simulation(
            g_omega=g_omega,
            g_phonon=g_phonon,
            zeta=zeta,
            times=times,
            subsystem_A=[0, 1],  # Photonic subsystem
            config=config
        )
        
        plt.plot(times * 1e6, entropies, color=color, label=label, linewidth=2)
    
    plt.xlabel('Time (μs)', fontsize=12)
    plt.ylabel('Entropy $S_\\Omega$ (bits)', fontsize=12)
    plt.title('Figure 2(a): Photonic Entropy vs Time\n'
              '(Bosonic simulation with truncated Fock space)', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)
    plt.ylim(0, 1.2)
    
    import os
    
    # Ensure output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure_2a_bosonic.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nDone! Output saved to: {output_path}")
