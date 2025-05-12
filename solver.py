import torch
from parameters import *

# Simulation function
def simulate_wave(r, u_in):
    u = torch.zeros((nt, nx), device=device)
    u[0] = u0
    u[1] = u0 + dt * v0

    # Create actuator influence vector
    actuator = torch.exp(-((x - r) ** 2) / (2 * (dx ** 2)))
    actuator = delta * actuator / actuator.sum()  # Normalize

    for n in range(1, nt - 1):
        u[n + 1, 1:-1] = (2 * u[n, 1:-1] - u[n - 1, 1:-1] +
                          (c * dt / dx) ** 2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]) +
                          dt ** 2 * u_in[n] * actuator[1:-1])
    return u
