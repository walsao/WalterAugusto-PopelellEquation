# Walter Augusto Popelell Equation - 2D Simulation (UPGRADED)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up simulation parameters
L = 100             # Length of space
N = 200             # Grid size (N x N)
dx = L / N          # Space step
dt = 0.005          # Time step (smaller for stability)
T = 2000            # Number of time steps (longer breathing evolution)
gravity_strength = 0.01  # Gentle breathing pullback

learning_rate = 5.0   # BOOST learning rate 🔥
decay_rate = 0.001

# Create spatial grid
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize fields
phi = np.random.randn(N, N) * 0.05
phi_old = np.copy(phi)
Popelell = np.ones((N, N)) * 5.0

# Add STRONG breathing blobs 💥
phi[N//4, N//4] += 10.0
phi[N//2, N//2] += -10.0
phi[3*N//4, 3*N//4] += 10.0
phi[N//4, 3*N//4] += -10.0
phi[3*N//4, N//4] += 10.0

# Laplacian operator in 2D
def laplacian(phi, dx):
    return (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
            np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) -
            4 * phi) / dx**2

# Set up plot
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
im_phi = axs[0].imshow(np.zeros((N, N)), extent=[0, L, 0, L], origin='lower', cmap='seismic', vmin=-5, vmax=5)
im_pop = axs[1].imshow(np.zeros((N, N)), extent=[0, L, 0, L], origin='lower', cmap='plasma', vmin=0, vmax=100)

axs[0].set_title('Breathing Field (Phi)')
axs[1].set_title('Popelell Field (Nonlinearity)')
fig.suptitle('Walter Augusto Popelell Equation - 2D Breathing Universe (UPGRADED)')

plt.tight_layout()

# Update function for animation
def update(frame):
    global phi, phi_old, Popelell

    # Calculate Laplacian
    lap = laplacian(phi, dx)

    # Estimate breathing energy
    breathing_energy = 0.5 * (lap**2 + phi**2)

    # Update Popelell field
    Popelell += dt * (learning_rate * breathing_energy - decay_rate * Popelell)
    Popelell = np.maximum(Popelell, 0.01)

    # Update breathing field
    phi_new = (2 * phi - phi_old +
               dt**2 * (lap - gravity_strength * phi - Popelell * phi**3))

    phi_old = np.copy(phi)
    phi = np.copy(phi_new)

    # Update plots
    im_phi.set_data(phi)
    im_pop.set_data(Popelell)

    return im_phi, im_pop

# Animate
ani = animation.FuncAnimation(fig, update, frames=T, interval=30, blit=True)
plt.show()
