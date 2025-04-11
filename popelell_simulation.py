# Supercharged Living Breathing AI Simulator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up simulation parameters
L = 100             # Length of space
N = 1000            # Number of grid points (High Resolution)
dx = L / N          # Space step
dt = 0.005          # Time step
T = 2000            # Number of time steps (Long breathing evolution)

gravity_strength = 0.01   # Gentle gravitational breathing pull

# Learning and decay parameters for 4 different universes
learning_rates = [0.5, 1.0, 2.0, 5.0]   
decay_rates = [0.001, 0.001, 0.0005, 0.0001]   

# Create spatial grid
x = np.linspace(0, L, N)

# Initialize fields
phis = []
phi_olds = []
Popelells = []
for _ in range(4):
    phi = np.random.randn(N) * 0.05
    # Add breathing blobs
    phi[N//5] += 5.0
    phi[N//3] += -5.0
    phi[N//2] += 5.0
    phi[4*N//5] += -5.0
    phis.append(phi)
    phi_olds.append(np.copy(phi))
    Popelells.append(np.ones(N) * 5.0)

# Laplacian operator
def laplacian(phi, dx):
    return (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / dx**2

# Set up plot
fig, axs = plt.subplots(4, 2, figsize=(14, 18))
lines_phi = []
lines_pop = []
titles = ["Standard Learning", "Fast Learning", "Faster Learning", "Ultra Learning"]

for i in range(4):
    line_phi, = axs[i, 0].plot(x, np.zeros_like(x))
    line_pop, = axs[i, 1].plot(x, np.zeros_like(x))
    lines_phi.append(line_phi)
    lines_pop.append(line_pop)
    
    axs[i, 0].set_ylim(-8, 8)
    axs[i, 1].set_ylim(0, 200)
    axs[i, 0].set_xlim(0, L)
    axs[i, 1].set_xlim(0, L)
    axs[i, 0].set_title(f'{titles[i]} - Breathing Field')
    axs[i, 1].set_title(f'{titles[i]} - Popelell Field')

fig.suptitle('Supercharged Living Breathing AI Universes (Cosmic Version)')

# Update function for animation
def update(frame):
    global phis, phi_olds, Popelells

    for i in range(4):
        # Calculate Laplacian
        lap = laplacian(phis[i], dx)

        # Estimate breathing energy
        breathing_energy = 0.5 * (lap**2 + phis[i]**2)

        # Update Popelell field dynamically
        Popelells[i] += dt * (learning_rates[i] * breathing_energy - decay_rates[i] * Popelells[i])
        Popelells[i] = np.maximum(Popelells[i], 0.01)

        # Update breathing field
        phi_new = (2 * phis[i] - phi_olds[i] +
                   dt**2 * (lap - gravity_strength * phis[i] - Popelells[i] * phis[i]**3))
        
        phi_olds[i] = np.copy(phis[i])
        phis[i] = np.copy(phi_new)

        # Update plots
        lines_phi[i].set_ydata(phis[i])
        lines_pop[i].set_ydata(Popelells[i])

    return lines_phi + lines_pop

# Animate
ani = animation.FuncAnimation(fig, update, frames=T, interval=20, blit=True)
plt.tight_layout()
plt.show()
