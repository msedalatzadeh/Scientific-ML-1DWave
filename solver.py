import torch
from parameters import *

# Simulation function
def simulate_wave(r, u_in, u0, v0):
    u = torch.zeros((nt, nx), device=device)
    u[0] = u0
    u[1] = v0

    # Create actuator influence vector
    actuator = torch.exp(-((x - r) ** 2) / (2 * (dx ** 2)))
    actuator = delta * actuator / actuator.sum()  # Normalize

    for n in range(1, nt - 1):
        u[n + 1, 1:-1] = (2 * u[n, 1:-1] - u[n - 1, 1:-1] +
                        (c * dt / dx) ** 2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]) +
                        dt ** 2 * u_in[n-1] * actuator[1:-1])
    return u


# Surrogate model using a multi-layer dense network
class WaveSurrogateModel(torch.nn.Module):
    def __init__(self, nx, nt):
        super(WaveSurrogateModel, self).__init__()
        N = nt + nx
        self.fc1 = torch.nn.Linear(2*nx, N)  # Input layer (r + u_in)
        self.fc2 = torch.nn.Linear(nt + 1, N)  # Hidden layer
        self.fc2n = torch.nn.Linear(N, N)  # Hidden layer
        self.fc3 = torch.nn.Linear(2*N, N)  # Hidden layer
        self.fc4 = torch.nn.Linear(N, nt * nx)  # Hidden layer

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
        combined_out = torch.sin(self.fc3(torch.cat([uv_out, ur_out])))
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


class WaveSurrogateSingleStepModel(torch.nn.Module):
    def __init__(self, nx):
        super(WaveSurrogateSingleStepModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 4, kernel_size=3, padding=1)  # Convolution layer for u0 and v0
        self.fc1 = torch.nn.Linear(4 * nx + 1, nx)  # Fully connected layer after convolution

    def forward(self, r, u_in_t, u0, v0):
        # Concatenate u0 and v0 along the channel dimension and apply convolution
        uv_combined = torch.stack([u0, v0], dim=0).unsqueeze(0)  # Shape: [1, 2, nx]
        uv_out = self.conv1(uv_combined).squeeze(0).view(-1)  # Shape: [4 * nx]

        # Multiply u_in_t and r, and concatenate with uv_out
        u_in_r_combined = (u_in_t * r).unsqueeze(0)  # Shape: [1]
        input_combined = torch.cat([uv_out, u_in_r_combined], dim=0)  # Shape: [4 * nx + 1]

        # Pass through fully connected layers
        next_step = self.fc1(input_combined)
        return next_step

    def predict_full_wave(self, r, u_in, u0, v0):
        # Initialize the wave solution tensor
        u = torch.zeros((nt, nx), device=device)
        u[0] = u0
        u[1] = v0

        # Predict each time step iteratively
        for n in range(0, nt - 2):
            u[n + 2] = self(r, u_in[n], u[n], u[n + 1])
        return u

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

            # Predict the full wave for all samples
            predictions = torch.stack([
                self.predict_full_wave(r_samples[i], u_in_samples[i], u0_samples[i], v0_samples[i])
                for i in range(num_samples)
            ])

            # Compute loss for the entire batch
            loss = criterion(predictions, wave_solutions)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Training complete.")

class WaveSurrogateCauchyModel(torch.nn.Module):
    def __init__(self, nx, nt):
        super(WaveSurrogateCauchyModel, self).__init__()
        N = nt + nx
        self.fc1 = torch.nn.Linear(2*nx, nx * nt)
        self.fc2 = torch.nn.Linear(nt + 1, nx * nt)
        self.fc3 = torch.nn.Linear(nt, 4*N)
        self.fc4 = torch.nn.Linear(4*N, nt * nx)

    def forward(self, r, u_in, u0, v0):
        # We implement cauchy formula: u(t) = T(t)[u0, v0] + \int_0^t T(t-tau)B(r)u_in(tau)dtau
        uv_combined = torch.cat([u0, v0], dim=0)
        semigroup_term = self.fc1(uv_combined) # --> T(t)[u0, v0]
        semigroup_term = semigroup_term.view(nx, nt)
        
        r_expanded = r.unsqueeze(0)
        ur_combined = torch.cat([r_expanded, u_in], dim=0)
        B_ru = torch.tanh(self.fc2(ur_combined)) # --> B(r)u_in(tau)
        B_ru = B_ru.view(nx, nt)

        B_ru_reshaped = B_ru.view(1, nx, nt)  # shape: [nx, 1, nt]
        semigroup_kernel = semigroup_term.flip(-1).view(nx, 1, nt)  # [nx, 1, nt] = one kernel per x
        convolution_term = torch.nn.functional.conv1d(
            B_ru_reshaped,
            semigroup_kernel,
            padding=nt - 1,
            groups=nx
        )  # shape: [1, nx, 2*nt - 1]
        convolution_term = convolution_term[0, :, :nt]  # truncate to [nx, nt]

        overall_evolution = semigroup_term + convolution_term

        return overall_evolution.view(nt, nx)

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


class WaveSurrogateFourierModel(torch.nn.Module):
    def __init__(self, nx, nt, N):
        super(WaveSurrogateFourierModel, self).__init__()
        self.N = N
        # Generate basis functions sin(t)cos(x) with various frequencies
        self.time = torch.linspace(0, T, nt, device=device).view(1, 1, nt) 
        self.space = torch.linspace(0, L, nx, device=device).view(1, nx, 1)
        self.frequencies = torch.arange(1, self.N + 1, device=device).view(-1, 1, 1)  # Shape: [N, 1, 1]

        self.fc1 = torch.nn.Linear(2 * nx, nx)  # Fully connected layer after convolution
        self.fc2 = torch.nn.Linear(nx, 2 * N * nx)  # Hidden layer
        self.fc3 = torch.nn.Linear(nt + 1, nt)  # Hidden layer
        self.fc4 = torch.nn.Linear(nt, 2 * N * nt)  # Hidden layer

    def forward(self, r, u_in, u0, v0):
        u0v0_combined = torch.cat([u0, v0], dim=0)
        initial_condition_influence = torch.relu(self.fc1(u0v0_combined))
        initial_condition_influence =  self.fc2(initial_condition_influence)

        r_expanded = r.unsqueeze(0)  # Expand r to match u_in dimensions
        inputs_combined = torch.cat([r_expanded, u_in], dim=0)
        inputs_influence = torch.relu(self.fc3(inputs_combined))
        inputs_influence = self.fc4(inputs_influence)

        
        basis_sin_cos = torch.cat([
                torch.sin(torch.pi * c * self.frequencies * self.time) * torch.sin(torch.pi * self.frequencies * self.space) / (self.frequencies ** 2),
                torch.cos(torch.pi * c * self.frequencies * self.time) * torch.sin(torch.pi * self.frequencies * self.space) / (self.frequencies ** 2),
            ], dim=0)  # Shape: [4 * N, nx, nt]
        
        semigroup_term = torch.einsum('nx,nxt->tx', initial_condition_influence.view(2 * self.N, nx), basis_sin_cos)
        convolution_term = torch.einsum('nt,ntx->tx', inputs_influence.view(2 * self.N, nt), basis_sin_cos.permute(0, 2, 1))
        u = semigroup_term + convolution_term
        
        return u

    # Function to train the surrogate model
    def train_on_synthetic_data(self, num_samples, num_epochs, learning_rate):
        # Generate training data using simulate_wave
        r_samples = torch.rand(num_samples, device=device) * L
        frequencies = torch.arange(1, num_samples + 1, device=device)
        time = torch.linspace(0, T, nt, device=device)  # Time vector
        space = torch.linspace(0, L, nx, device=device)  # Space vector
        u_in_samples = torch.stack([torch.sin(5 * torch.pi * c * f * time) / f for f in frequencies])
        u0_samples = torch.stack([torch.sin(torch.pi * f * space) / f for f in frequencies])
        v0_samples = torch.stack([-torch.sin(torch.pi * f * space) / f for f in frequencies])

        # Shuffle all samples
        indices = torch.randperm(num_samples, device=device)
        u_in_samples = u_in_samples[indices]
        u0_samples = u0_samples[indices]
        v0_samples = v0_samples[indices]
        
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