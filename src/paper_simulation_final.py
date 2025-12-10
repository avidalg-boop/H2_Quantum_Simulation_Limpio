"""
IMPLEMENTACIÓN COMPLETA DEL PAPER - Siguiendo la guía del usuario

Referencia: arXiv:2405.05696v1

Espacio de Hilbert (Eq. 3):
|Ψ⟩ = |p1⟩_Ω↑ ⊗ |p2⟩_Ω↓ ⊗ |m⟩_ω ⊗ |l1⟩_Φ1↑ ⊗ |l2⟩_Φ1↓ ⊗ |L⟩_cb ⊗ |k⟩_n

Modos bosónicos truncados a dimensión d=4 (2 qubits cada uno).
Sistemas de 2 niveles como qubits.

Total: 3 modos bosónicos (d=4) + 4 qubits = 4³ × 2⁴ = 64 × 16 = 1024 dimensiones
"""

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# =============================================================================
# CONFIGURACIÓN DEL ESPACIO DE HILBERT
# =============================================================================

class H2SystemConfig:
    def __init__(self, dim_boson=4):
        """
        dim_boson: Dimensión de truncamiento para modos bosónicos (d=4 o d=8)
        """
        self.d = dim_boson  # Fock space truncation
        
        # Orden: [p1, p2, m, l1, l2, L, k]
        self.dims = [dim_boson, dim_boson, dim_boson, 2, 2, 2, 2]
        self.n_subsystems = 7
        self.total_dim = int(np.prod(self.dims))
        
        # Índices de subsistemas
        self.IDX_P1 = 0   # Fotón Ω↑
        self.IDX_P2 = 1   # Fotón Ω↓
        self.IDX_M = 2    # Fonón ω
        self.IDX_L1 = 3   # Orbital electrón ↑
        self.IDX_L2 = 4   # Orbital electrón ↓
        self.IDX_L = 5    # Enlace covalente (L)
        self.IDX_K = 6    # Núcleos tunelamiento (k)
        
    def __repr__(self):
        return f"H2SystemConfig(d_boson={self.d}, total_dim={self.total_dim})"


# =============================================================================
# OPERADORES BÁSICOS
# =============================================================================

def bosonic_annihilation(d):
    """Operador aniquilación bosónico truncado a[n⟩ = √n |n-1⟩"""
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n-1, n] = np.sqrt(n)
    return a

def bosonic_creation(d):
    """Operador creación bosónico a†|n⟩ = √(n+1) |n+1⟩"""
    return bosonic_annihilation(d).conj().T

def bosonic_number(d):
    """Operador número n = a†a"""
    return np.diag(np.arange(d, dtype=complex))

def fermionic_sigma():
    """σ = |0⟩⟨1| = (X + iY)/2 - Operador de relajación"""
    return np.array([[0, 1], [0, 0]], dtype=complex)

def fermionic_sigma_dag():
    """σ† = |1⟩⟨0| = (X - iY)/2 - Operador de excitación"""
    return np.array([[0, 0], [1, 0]], dtype=complex)

def fermionic_number():
    """σ†σ = |1⟩⟨1| = (I - Z)/2"""
    return np.array([[0, 0], [0, 1]], dtype=complex)

def pauli_x():
    """Pauli X = σ + σ†"""
    return np.array([[0, 1], [1, 0]], dtype=complex)

def identity(d):
    return np.eye(d, dtype=complex)


# =============================================================================
# PRODUCTO TENSORIAL
# =============================================================================

def tensor_chain(*matrices):
    """Producto tensorial: A ⊗ B ⊗ C ⊗ ..."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def embed_operator(cfg, subsystem_idx, op):
    """Embebe operador en espacio de Hilbert completo."""
    ops = []
    for i, dim in enumerate(cfg.dims):
        if i == subsystem_idx:
            ops.append(op)
        else:
            ops.append(identity(dim))
    return tensor_chain(*ops)

def embed_operators(cfg, op_dict):
    """Embebe múltiples operadores."""
    ops = []
    for i, dim in enumerate(cfg.dims):
        if i in op_dict:
            ops.append(op_dict[i])
        else:
            ops.append(identity(dim))
    return tensor_chain(*ops)


# =============================================================================
# CONSTRUCCIÓN DEL HAMILTONIANO (Eq. 4)
# =============================================================================

def build_hamiltonian(cfg, g_omega_up, g_omega_down, g_phonon, zeta):
    """
    Construye el Hamiltoniano completo de la Eq. 4.
    
    Parámetros del paper (Sec. IV):
    - hbar = 1
    - Ω↑ = Ω↓ = 10^9
    - ω = 10^8
    - g = 10^7 (base)
    """
    d = cfg.d
    total_dim = cfg.total_dim
    
    # Constantes
    hbar = 1.0
    Omega_up = 1e9
    Omega_down = 1e9
    omega_phonon = 1e8
    
    H = np.zeros((total_dim, total_dim), dtype=complex)
    
    # Operadores bosónicos
    a = bosonic_annihilation(d)
    a_dag = bosonic_creation(d)
    n_bos = bosonic_number(d)
    I_bos = identity(d)
    
    # Operadores fermiónicos
    sigma = fermionic_sigma()
    sigma_dag = fermionic_sigma_dag()
    n_ferm = fermionic_number()
    I_ferm = identity(2)
    X = pauli_x()
    
    # Proyectores
    P0 = I_ferm - n_ferm  # |0⟩⟨0| = proyecta sobre L=0 (enlace formado)
    P1 = n_ferm           # |1⟩⟨1| = proyecta sobre L=1 (enlace roto)
    
    # =========================================================================
    # LÍNEA 1: Energía libre de fotones y fonón
    # hbar*Ω↑ a†_Ω↑ a_Ω↑ + hbar*Ω↓ a†_Ω↓ a_Ω↓ + hbar*ω a†_ω a_ω
    # =========================================================================
    
    H += hbar * Omega_up * embed_operator(cfg, cfg.IDX_P1, n_bos)
    H += hbar * Omega_down * embed_operator(cfg, cfg.IDX_P2, n_bos)
    H += hbar * omega_phonon * embed_operator(cfg, cfg.IDX_M, n_bos)
    
    # =========================================================================
    # LÍNEA 2: Energía de los estados fermiónicos
    # hbar*Ω↑ σ†_Ω↑ σ_Ω↑ + hbar*Ω↓ σ†_Ω↓ σ_Ω↓ + hbar*ω σ†_ω σ_ω
    # =========================================================================
    
    H += hbar * Omega_up * embed_operator(cfg, cfg.IDX_L1, n_ferm)
    H += hbar * Omega_down * embed_operator(cfg, cfg.IDX_L2, n_ferm)
    H += hbar * omega_phonon * embed_operator(cfg, cfg.IDX_L, n_ferm)
    
    # =========================================================================
    # LÍNEA 3-4: Interacción Jaynes-Cummings con restricción
    # g_Ω↑ (a†_Ω↑ σ_Ω↑ + a_Ω↑ σ†_Ω↑) × σ_ω σ†_ω
    # 
    # NOTA: σ_ω σ†_ω = |0⟩⟨0|_L = P0 (proyecta sobre enlace FORMADO)
    # Esto significa que JC solo actúa cuando hay enlace molecular
    # =========================================================================
    
    # JC para Ω↑ (fotón p1, electrón l1)
    jc_up_1 = embed_operators(cfg, {
        cfg.IDX_P1: a_dag,   # Crea fotón
        cfg.IDX_L1: sigma,   # Relaja electrón (1→0)
        cfg.IDX_L: P0        # Solo si enlace formado (L=0)
    })
    jc_up_2 = embed_operators(cfg, {
        cfg.IDX_P1: a,       # Destruye fotón
        cfg.IDX_L1: sigma_dag,  # Excita electrón (0→1)
        cfg.IDX_L: P0
    })
    H += g_omega_up * (jc_up_1 + jc_up_2)
    
    # JC para Ω↓ (fotón p2, electrón l2)
    jc_down_1 = embed_operators(cfg, {
        cfg.IDX_P2: a_dag,
        cfg.IDX_L2: sigma,
        cfg.IDX_L: P0
    })
    jc_down_2 = embed_operators(cfg, {
        cfg.IDX_P2: a,
        cfg.IDX_L2: sigma_dag,
        cfg.IDX_L: P0
    })
    H += g_omega_down * (jc_down_1 + jc_down_2)
    
    # =========================================================================
    # LÍNEA 5: Interacción fonón-enlace (formación/ruptura de enlace covalente)
    # g_ω (a†_ω σ_ω + a_ω σ†_ω)
    # 
    # - a†_ω σ_ω: Crea fonón cuando enlace se forma (L: 1→0)
    # - a_ω σ†_ω: Destruye fonón cuando enlace se rompe (L: 0→1)
    # 
    # NOTA: La Eq. 4 NO muestra restricción explícita. Seguimos literal.
    # =========================================================================
    
    phonon_bond_1 = embed_operators(cfg, {
        cfg.IDX_M: a_dag,    # Crea fonón
        cfg.IDX_L: sigma     # Forma enlace (L: 1→0)
    })
    phonon_bond_2 = embed_operators(cfg, {
        cfg.IDX_M: a,        # Destruye fonón
        cfg.IDX_L: sigma_dag # Rompe enlace (L: 0→1)
    })
    H += g_phonon * (phonon_bond_1 + phonon_bond_2)
    
    # =========================================================================
    # LÍNEA 6: Tunelamiento
    # ζ (σ†_n σ_n + σ_n σ†_n)
    # 
    # NOTA: σ†σ + σσ† = |1⟩⟨1| + |0⟩⟨0| = I (identidad)
    # Esto es matemáticamente un desplazamiento de energía constante.
    # 
    # INTERPRETACIÓN FÍSICA: El tunelamiento debería ser σ + σ† = X
    # que permite transiciones k: 0 ↔ 1
    # 
    # Implementamos X como es físicamente correcto para tunelamiento.
    # =========================================================================
    
    tunnel_op = embed_operator(cfg, cfg.IDX_K, X)
    H += zeta * tunnel_op
    
    return H


# =============================================================================
# ESTADO INICIAL (Eq. 12, 13)
# =============================================================================

def get_initial_state(cfg):
    """
    Construye |Ψ_initial⟩ según Ec. 12:
    
    |Ψ_initial⟩ = (1/2)(|Φ0↑Φ0↓⟩ + |Φ1↑Φ0↓⟩ - |Φ0↑Φ1↓⟩ - |Φ1↑Φ1↓⟩)
    
    Donde (Ec. 13):
    - Todos los bosones en vacío |0⟩
    - L=1 (enlace roto), k=1 (núcleos separados)
    - l1, l2 varían según el estado
    """
    dims = cfg.dims
    total_dim = cfg.total_dim
    d = cfg.d
    
    def get_index(occupation):
        """
        Calcula índice lineal desde números de ocupación.
        occupation = [p1, p2, m, l1, l2, L, k]
        """
        idx = 0
        multiplier = 1
        for i, (n, dim) in enumerate(zip(occupation, dims)):
            idx += n * multiplier
            multiplier *= dim
        return idx
    
    psi = np.zeros(total_dim, dtype=complex)
    
    # Ec. 13a: |Φ0↑Φ0↓⟩ = |0,0,0,0,0,1,1⟩
    psi[get_index([0, 0, 0, 0, 0, 1, 1])] = 0.5
    
    # Ec. 13b: |Φ1↑Φ0↓⟩ = |0,0,0,1,0,1,1⟩
    psi[get_index([0, 0, 0, 1, 0, 1, 1])] = 0.5
    
    # Ec. 13c: |Φ0↑Φ1↓⟩ = |0,0,0,0,1,1,1⟩
    psi[get_index([0, 0, 0, 0, 1, 1, 1])] = -0.5
    
    # Ec. 13d: |Φ1↑Φ1↓⟩ = |0,0,0,1,1,1,1⟩
    psi[get_index([0, 0, 0, 1, 1, 1, 1])] = -0.5
    
    # Verificar normalización
    assert np.abs(np.linalg.norm(psi) - 1.0) < 1e-10, "Estado no normalizado"
    
    return psi


# =============================================================================
# TRAZA PARCIAL Y ENTROPÍA
# =============================================================================

def partial_trace(psi, cfg, keep_subsystems):
    """
    Calcula la traza parcial para obtener ρ_A.
    
    psi: vector de estado
    keep_subsystems: lista de índices de subsistemas a mantener
    """
    dims = cfg.dims
    n = len(dims)
    
    # Reshape a tensor
    psi_tensor = psi.reshape(dims)
    
    # Matriz de densidad como tensor
    rho_tensor = np.outer(psi, psi.conj()).reshape(dims + dims)
    
    # Determinar qué subsistemas trazar
    trace_out = sorted(set(range(n)) - set(keep_subsystems))
    keep = sorted(keep_subsystems)
    
    # Trazar desde el índice más alto hacia abajo
    for k in reversed(trace_out):
        n_remaining = rho_tensor.ndim // 2
        rho_tensor = np.trace(rho_tensor, axis1=k, axis2=k + n_remaining)
    
    # Reshape a matriz
    keep_dims = [dims[k] for k in keep]
    dim_A = int(np.prod(keep_dims))
    
    return rho_tensor.reshape(dim_A, dim_A)


def von_neumann_entropy(rho, base=2):
    """Calcula entropía de von Neumann S(ρ) = -Tr(ρ log ρ)."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    if base == 2:
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    else:
        return -np.sum(eigenvalues * np.log(eigenvalues))


# =============================================================================
# SIMULACIÓN COMPLETA
# =============================================================================

def run_figure2_simulation():
    """
    Replica la Figura 2 del paper.
    
    Parámetros:
    - g_ω = 0.1g
    - ζ = g
    - g_Ω ∈ {g, 1.5g, 2g, 4g}
    """
    print("=" * 70)
    print("SIMULACION FIGURA 2: Efecto de g_Omega en entropia")
    print("=" * 70)
    
    # Configuración
    cfg = H2SystemConfig(dim_boson=4)  # Truncamiento a 4 niveles de Fock
    print(f"\n{cfg}")
    
    # Parámetros base
    G_BASE = 1e7
    g_phonon = 0.1 * G_BASE  # g_ω = 0.1g
    zeta = G_BASE            # ζ = g
    
    # Tiempo: 0 a 2μs
    t_max = 2e-6
    n_steps = 200
    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]
    
    # Valores de g_Ω a probar
    g_omega_values = [G_BASE, 1.5*G_BASE, 2*G_BASE, 4*G_BASE]
    colors = ['magenta', 'green', 'red', 'blue']
    labels = [r'$g_\Omega = g$', r'$g_\Omega = 1.5g$', 
              r'$g_\Omega = 2g$', r'$g_\Omega = 4g$']
    
    # Estado inicial
    psi0 = get_initial_state(cfg)
    
    plt.figure(figsize=(10, 6))
    
    for g_omega, color, label in zip(g_omega_values, colors, labels):
        print(f"\nSimulando {label}...")
        
        # Construir Hamiltoniano
        H = build_hamiltonian(cfg, g_omega, g_omega, g_phonon, zeta)
        
        # Verificar hermiticidad
        herm_err = np.linalg.norm(H - H.conj().T)
        print(f"  Hermiticidad: {herm_err:.2e}")
        
        # Operador de evolución temporal
        U_step = expm(-1j * H * dt)
        
        # Evolución
        psi = psi0.copy()
        entropies = []
        
        for t in times:
            # Calcular entropía del subsistema fotónico (p1, p2)
            rho_photons = partial_trace(psi, cfg, [cfg.IDX_P1, cfg.IDX_P2])
            S = von_neumann_entropy(rho_photons, base=2)
            entropies.append(S)
            
            # Evolucionar
            psi = U_step @ psi
        
        plt.plot(times * 1e6, entropies, color=color, label=label, linewidth=2)
        print(f"  Peak entropy: {max(entropies):.4f}")
    
    plt.xlabel('Time (us)', fontsize=12)
    plt.ylabel(r'Entropy $S_{\Omega,\omega}$ (bits)', fontsize=12)
    plt.title(r'Figure 2: Effect of $g_\Omega$ on Entropy' + '\n' +
              r'$g_\omega = 0.1g$, $\zeta = g$', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)
    plt.ylim(0, 2.5)  # Ajustado para el caso bosónico
    
    # Guardar
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure2_final.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nGuardado en: {output_path}")


if __name__ == "__main__":
    run_figure2_simulation()
