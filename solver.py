import torch
from parameters import *

# Simulation function
def simulate_wave(r, u_in, u0, v0):
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


# Surrogate model using a multi-layer dense network
class WaveSurrogateModel(torch.nn.Module):
    def __init__(self, nx, nt):
        super(WaveSurrogateModel, self).__init__()
        N = 128
        M = 256
        self.fc1 = torch.nn.Linear(2*nx, N)  # Input layer (r + u_in)
        self.fc2 = torch.nn.Linear(nt + 1, N)  # Hidden layer
        self.fc2n = torch.nn.Linear(N, N)  # Hidden layer
        self.fc3 = torch.nn.Linear(2*N, M)  # Hidden layer
        self.fc4 = torch.nn.Linear(M, nt * nx)  # Hidden layer

    def forward(self, r, u_in, u0, v0):
        # Concatenate u0 and v0, and pass through a dense layer
        uv_combined = torch.cat([u0, v0], dim=0)
        uv_out = self.fc1(uv_combined)

        # Concatenate u_in and r, and pass through a dense layer
        r_expanded = r.unsqueeze(0)  # Expand r to match u_in dimensions
        ur_combined = torch.cat([r_expanded, u_in], dim=0)
        ur_out = torch.tanh(self.fc2(ur_combined))
        ur_out = self.fc2n(ur_out)

        # Sum the outputs of the two layers and pass through another dense layer
        combined_out = torch.tanh(self.fc3(torch.cat([uv_out, ur_out])))
        x = self.fc4(combined_out)

        # Reshape to (nt, nx)
        x = x.view(nt, nx)
        return x

    # Function to train the surrogate model
    def train_on_synthetic_data(self, num_samples, num_epochs, learning_rate):
        # Generate training data using simulate_wave
        r_samples = torch.rand(num_samples, device=device) * L
        frequencies = torch.linspace(1, 10, num_samples, device=device)  # Frequencies from 1 to 10
        time = torch.linspace(0, T, nt, device=device)  # Time vector
        space = torch.linspace(0, L, nx, device=device)  # Space vector
        u_in_samples = torch.stack([torch.sin(torch.pi * f * time) for f in frequencies])
        u0_samples = torch.stack([torch.sin(torch.pi * f * space) for f in frequencies])
        v0_samples = torch.stack([torch.cos(torch.pi * f * space) for f in frequencies])
        wave_solutions = torch.stack([simulate_wave(r_samples[i], u_in_samples[i], u0_samples[i], v0_samples[i]) for i in range(num_samples)])

        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        print("Starting surrogate model training...")
        # Training loop
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            # Forward pass
            predictions = torch.stack([self(r_samples[i], u_in_samples[i], u0_samples[i], v0_samples[i]) for i in range(num_samples)])
            loss = criterion(predictions, wave_solutions)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Training complete.")