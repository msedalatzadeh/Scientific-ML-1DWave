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


# Surrogate model using a multi-layer dense network
class WaveSurrogateModel(torch.nn.Module):
    def __init__(self, nx, nt):
        super(WaveSurrogateModel, self).__init__()
        self.fc1 = torch.nn.Linear(nt + 1, 128)  # Input layer (r + u_in)
        self.fc2 = torch.nn.Linear(128, 256)  # Hidden layer
        self.fc3 = torch.nn.Linear(256, nt * nx)  # Hidden layer

    def forward(self, r, u_in):
        # Combine r and u_in into a single input tensor
        r_expanded = r.unsqueeze(0)  # Expand r to match the time dimension
        r_u_in = torch.cat([r_expanded, u_in], dim=0)  # Combine along feature dimension
        x = torch.relu(self.fc1(r_u_in))  # Pass through first dense layer
        x = torch.relu(self.fc2(x))  # Pass through second dense layer
        x = self.fc3(x)  # Output layer
        x = x.view(nt, nx)  # Reshape to (nt, nx)
        return x

    
    # Function to train the surrogate model
    def train_on_synthetic_data(self, num_samples, num_epochs, learning_rate):
        # Generate training data using simulate_wave
        r_samples = torch.rand(num_samples, device=device) * L
        u_in_samples = torch.rand((num_samples, nt), device=device) * 10 - 5  # Scale to [-1, 1]
        u_in_samples = torch.nn.functional.avg_pool1d(u_in_samples.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)  # Smooth over time
        wave_solutions = torch.stack([simulate_wave(r_samples[i], u_in_samples[i]) for i in range(num_samples)])

        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        print("Starting surrogate model training...")
        # Training loop
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            # Forward pass
            predictions = torch.stack([self(r_samples[i], u_in_samples[i]) for i in range(num_samples)])
            loss = criterion(predictions, wave_solutions)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Training complete.")