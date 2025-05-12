import torch

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Parameters
L = 1.0          # Length of the domain
T = 5.0          # Total time
c = 1.0          # Wave speed
nx = 100         # Number of spatial points
nt = 500         # Number of time steps
dx = L / (nx - 1)
dt = T / nt
gamma = 0.0001     # Control effort weight
delta = 50 

# Stability condition
assert c * dt / dx <= 1, "Stability condition violated!"

# Spatial and temporal grids
x = torch.linspace(0, L, nx, device=device)
t = torch.linspace(0, T, nt, device=device)

# Initial conditions
u0 = torch.sin(torch.pi * x).to(device)
v0 = torch.zeros(nx, device=device)

# Desired state (e.g., zero displacement)
u_desired = torch.zeros((nt, nx), device=device)

# Control input (can be optimized or predefined)
u_in = torch.ones(nt, device=device, requires_grad=True)

# PyTorch learning parameters
r = torch.tensor(0.2, device=device, requires_grad=True)
num_epochs = 200
learning_rate = 0.4
momentum=0.9

num_samples_for_surrogate = 100
num_epochs_for_surrogate = 200
learning_rate_for_surrogate = 0.6
momentum_for_surrogate = 0.9
