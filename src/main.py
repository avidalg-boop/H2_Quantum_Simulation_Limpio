import numpy as np
import matplotlib.pyplot as plt
import os
from hamiltonian import H2Hamiltonian
from simulation import QiskitSimulation, get_initial_state

# Constants
G_BASE = 1e7

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def simulate_figure_2(outdir):
    print("Simulating Figure 2...")
    t_max = 2e-6
    n_steps = 200
    times = np.linspace(0, t_max, n_steps)
    
    g_phonon = 0.1 * G_BASE
    zeta = G_BASE
    g_omega_values = [G_BASE, 1.5*G_BASE, 2*G_BASE, 4*G_BASE]
    colors = ['magenta', 'green', 'red', 'blue']
    labels = [r'$g_\Omega = g$', r'$g_\Omega = 1.5g$', r'$g_\Omega = 2g$', r'$g_\Omega = 4g$']
    
    plt.figure(figsize=(8, 6)) # Single plot
    
    for g_omega, color, label in zip(g_omega_values, colors, labels):
        ham = H2Hamiltonian()
        H_op = ham.get_hamiltonian_op(g_omega, g_phonon, zeta)
        
        sim = QiskitSimulation(H_op)
        psi0 = get_initial_state()
        
        states = sim.evolve(psi0, times)
        entropies = [sim.compute_entropy(state, [0, 1]) for state in states]
        
        plt.plot(times, entropies, color=color, label=label)
        
    plt.xlabel('Time (s)')
    plt.ylabel('Entropy')
    plt.title('Figure 2: Effect of $g_\Omega$')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    ticks = np.arange(0, 2.01e-6, 0.25e-6)
    plt.xticks(ticks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figure_2_qiskit.png'))
    plt.close()

def simulate_figure_3(outdir):
    print("Simulating Figure 3...")
    t_max = 1e-4
    n_steps = 1000 
    times = np.linspace(0, t_max, n_steps)
    
    g_omega = G_BASE
    zeta = G_BASE
    g_phonon_values = [G_BASE/100.0, G_BASE/20.0, G_BASE/10.0, G_BASE/5.0]
    colors = ['magenta', 'green', 'red', 'blue']
    labels = [r'$g_\omega = g/100$', r'$g_\omega = g/20$', r'$g_\omega = g/10$', r'$g_\omega = g/5$']
    panel_labels = ['(b)', '(c)', '(d)', '(e)']
    
    plt.figure(figsize=(8, 10)) # Vertical stack
    
    for i, (g_phonon, color, label) in enumerate(zip(g_phonon_values, colors, labels)):
        ham = H2Hamiltonian()
        H_op = ham.get_hamiltonian_op(g_omega, g_phonon, zeta)
        
        sim = QiskitSimulation(H_op)
        psi0 = get_initial_state()
        
        states = sim.evolve(psi0, times)
        entropies = [sim.compute_entropy(state, [0, 1]) for state in states]
        
        plt.subplot(4, 1, i+1)
        plt.plot(times, entropies, color=color, label=label)
        
        plt.text(0.02, 0.85, panel_labels[i], transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
        
        if i == 3:
            plt.xlabel('Time (s)')
            ticks = np.arange(0, 1.01e-4, 0.2e-4)
            plt.xticks(ticks)
        else:
            plt.xticks([]) 
            
        plt.ylabel('Entropy')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figure_3_qiskit.png'))
    plt.close()

def simulate_figure_4(outdir):
    print("Simulating Figure 4...")
    t_max = 1e-6
    n_steps = 200
    times = np.linspace(0, t_max, n_steps)
    
    g_omega = G_BASE
    g_phonon = G_BASE / 10.0
    zeta_values = [G_BASE/10.0, G_BASE/2.0, G_BASE, 1.5 * G_BASE]
    colors = ['magenta', 'green', 'red', 'blue']
    labels = [r'$\zeta = g/10$', r'$\zeta = g/2$', r'$\zeta = g$', r'$\zeta = 3g/2$']
    
    plt.figure(figsize=(12, 8))
    
    for i, (zeta, color, label) in enumerate(zip(zeta_values, colors, labels)):
        ham = H2Hamiltonian()
        H_op = ham.get_hamiltonian_op(g_omega, g_phonon, zeta)
        
        sim = QiskitSimulation(H_op)
        psi0 = get_initial_state()
        
        states = sim.evolve(psi0, times)
        entropies = [sim.compute_entropy(state, [0, 1]) for state in states]
        
        plt.subplot(2, 2, i+1)
        plt.plot(times, entropies, color=color, label=label)
        plt.xlabel('Time (s)')
        plt.ylabel('Entropy')
        plt.title(f'Fig 4: {label}')
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figure_4_qiskit.png'))
    plt.close()

def simulate_figure_6_and_7(outdir):
    print("Simulating Figures 6 and 7...")
    t_max = 1e-5 # From uploaded image 0 (Figure 6 x-axis goes to 1.0 1e-5)
    n_steps = 1000
    times = np.linspace(0, t_max, n_steps)
    
    g_phonon = G_BASE / 10.0
    
    # Figure 6: g_omega = zeta = g, 2g, 3g, 4g
    g_values = [G_BASE, 2*G_BASE, 3*G_BASE, 4*G_BASE]
    colors = ['magenta', 'green', 'red', 'blue']
    labels = ['g', '2g', '3g', '4g']
    
    plt.figure(figsize=(12, 8))
    for i, (g_val, color, label) in enumerate(zip(g_values, colors, labels)):
        g_omega = g_val
        zeta = g_val
        
        ham = H2Hamiltonian()
        H_op = ham.get_hamiltonian_op(g_omega, g_phonon, zeta)
        sim = QiskitSimulation(H_op)
        psi0 = get_initial_state()
        states = sim.evolve(psi0, times)
        entropies = [sim.compute_entropy(state, [0, 1]) for state in states]
        
        plt.subplot(2, 2, i+1)
        plt.plot(times, entropies, color=color, label=f'$g_\Omega=\zeta={label}$')
        plt.title(f'Fig 6: $g_\Omega=\zeta={label}$')
        plt.ylim(0, 2)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figure_6_qiskit.png'))
    plt.close()
    
    # Figure 7: g_omega vs zeta with 2x relation
    # (0.5g, g), (g, 2g), (1.5g, 3g), (2g, 4g)
    params = [(0.5*G_BASE, 1.0*G_BASE), (1.0*G_BASE, 2.0*G_BASE),
              (1.5*G_BASE, 3.0*G_BASE), (2.0*G_BASE, 4.0*G_BASE)]
    
    plt.figure(figsize=(12, 8))
    for i, ((g_omega, zeta), color) in enumerate(zip(params, colors)):
        ham = H2Hamiltonian()
        H_op = ham.get_hamiltonian_op(g_omega, g_phonon, zeta)
        sim = QiskitSimulation(H_op)
        psi0 = get_initial_state()
        states = sim.evolve(psi0, times)
        entropies = [sim.compute_entropy(state, [0, 1]) for state in states]
        
        plt.subplot(2, 2, i+1)
        plt.plot(times, entropies, color=color)
        plt.title(f'Fig 7: $g_\Omega={g_omega/G_BASE}g, \zeta={zeta/G_BASE}g$')
        plt.ylim(0, 2)
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figure_7_qiskit.png'))
    plt.close()

def simulate_figure_8(outdir):
    print("Simulating Figure 8...")
    t_max = 2e-6
    n_steps = 200
    times = np.linspace(0, t_max, n_steps)
    
    g_omega = G_BASE
    g_phonon = 0.1 * G_BASE
    zeta = G_BASE
    
    ham = H2Hamiltonian()
    H_op = ham.get_hamiltonian_op(g_omega, g_phonon, zeta)
    sim = QiskitSimulation(H_op)
    psi0 = get_initial_state()
    states = sim.evolve(psi0, times)
    
    # S_Omega_up (Q0), S_Omega_down (Q1), S_Omega (Q0, Q1)
    s_up = [sim.compute_entropy(state, [0]) for state in states]
    s_down = [sim.compute_entropy(state, [1]) for state in states]
    s_total = [sim.compute_entropy(state, [0, 1]) for state in states]
    s_sum = [u + d for u, d in zip(s_up, s_down)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, s_total, 'b-', label=r'$S_\Omega$')
    plt.plot(times, s_up, 'm-', label=r'$S_{\Omega^\uparrow}, S_{\Omega^\downarrow}$')
    plt.plot(times, s_sum, 'r-', label=r'$S_{\Omega^\uparrow} + S_{\Omega^\downarrow}$')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Entropy')
    plt.title('Figure 8: Comparison of Photonic Entropies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, 'figure_8_qiskit.png'))
    plt.close()

def simulate_figure_9(outdir):
    print("Simulating Figure 9...")
    t_max = 6e-6 # From uploaded image 1 (goes to 6e-6)
    n_steps = 600
    times = np.linspace(0, t_max, n_steps)
    
    g_omega = G_BASE
    g_phonon = 0.1 * G_BASE
    zeta = G_BASE
    
    ham = H2Hamiltonian()
    H_op = ham.get_hamiltonian_op(g_omega, g_phonon, zeta)
    sim = QiskitSimulation(H_op)
    psi0 = get_initial_state()
    states = sim.evolve(psi0, times)
    
    # S_Omega (Q0, Q1) vs S_omega (Q2 - Phonon)
    s_photons = [sim.compute_entropy(state, [0, 1]) for state in states]
    s_phonon = [sim.compute_entropy(state, [2]) for state in states]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, s_photons, 'b-', label=r'$S_\Omega$')
    plt.plot(times, s_phonon, 'g-', label=r'$S_\omega$')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Entropy')
    plt.title('Figure 9: Photons vs Phonon Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, 'figure_9_qiskit.png'))
    plt.close()

def simulate_figure_5(outdir):
    print("Simulating Figure 5...")
    # Figure 5: Max entropy vs g_Omega and zeta.
    # We need to scan a range of g_Omega and zeta.
    # Paper says "linear relationship... g_Omega = zeta... g_Omega = 2*zeta".
    # Let's scan a grid.
    
    t_max = 2e-6 # Sufficient time to reach peak
    n_steps = 100
    times = np.linspace(0, t_max, n_steps)
    
    g_phonon = G_BASE / 10.0
    
    # Grid
    # g_Omega from 0 to 4g
    # zeta from 0 to 4g
    # Resolution: 20x20?
    
    n_grid = 20
    g_omega_vals = np.linspace(0, 4*G_BASE, n_grid)
    zeta_vals = np.linspace(0, 4*G_BASE, n_grid)
    
    max_entropies = np.zeros((n_grid, n_grid))
    
    for i, g_om in enumerate(g_omega_vals):
        for j, z in enumerate(zeta_vals):
            ham = H2Hamiltonian()
            H_op = ham.get_hamiltonian_op(g_om, g_phonon, z)
            sim = QiskitSimulation(H_op)
            psi0 = get_initial_state()
            
            # We don't need all states, just max entropy.
            # But evolve returns all.
            states = sim.evolve(psi0, times)
            entropies = [sim.compute_entropy(state, [0, 1]) for state in states]
            max_entropies[i, j] = np.max(entropies)
            
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(zeta_vals/G_BASE, g_omega_vals/G_BASE)
    plt.pcolormesh(X, Y, max_entropies, shading='auto', cmap='viridis')
    plt.colorbar(label='Max Entropy')
    plt.xlabel(r'$\zeta / g$')
    plt.ylabel(r'$g_\Omega / g$')
    plt.title('Figure 5: Max Entropy vs Interaction Strengths')
    
    # Add lines for ridges if needed, but heatmap shows them.
    # Ridge 1: g_Omega = zeta
    plt.plot([0, 4], [0, 4], 'w--', alpha=0.5, label=r'$g_\Omega = \zeta$')
    # Ridge 2: g_Omega = 2*zeta
    plt.plot([0, 2], [0, 4], 'w:', alpha=0.5, label=r'$g_\Omega = 2\zeta$')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figure_5_qiskit.png'))
    plt.close()

if __name__ == "__main__":
    outdir = "output"
    ensure_dir(outdir)
    
    simulate_figure_2(outdir)
    simulate_figure_3(outdir)
    simulate_figure_4(outdir)
    simulate_figure_5(outdir)
    simulate_figure_6_and_7(outdir)
    simulate_figure_8(outdir)
    simulate_figure_9(outdir)
