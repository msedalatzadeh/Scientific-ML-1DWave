import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from parameters import *
from solver import *

surrogate_model = WaveSurrogateFourierModel(nx, nt, nfourier).to(device)
# Check if a saved model exists, load it; otherwise, train and save the model
model_path = 'results/surrogate_model.pth'
try:
    surrogate_model.load_state_dict(torch.load(model_path))
    surrogate_model.eval()
    print(f"Loaded surrogate model from {model_path}")
except FileNotFoundError:
    print("No saved surrogate model found. Training a new model...")
    surrogate_model.train_on_synthetic_data(num_samples=num_samples_for_surrogate, num_epochs=num_epochs_for_surrogate, learning_rate=learning_rate_for_surrogate)
    torch.save(surrogate_model.state_dict(), model_path)
    print(f"Surrogate model saved to {model_path}")

# Optimization settings
optimizer = torch.optim.AdamW([r, u_in], lr=learning_rate)
# Initialize lists to store epoch data
epoch_data = []
# Training loop
print('Starting Control and Actuator Location Optimization...')
for epoch in range(num_epochs):
    optimizer.zero_grad()
    u = surrogate_model(r, u_in, u0, v0)
    state_cost = torch.sum((u - u_desired) ** 2) * dx * dt
    control_cost = gamma * torch.sum(u_in ** 2) * dt
    loss = state_cost + control_cost
    loss.backward()
    optimizer.step()

    # Clamp r to [0, L]
    with torch.no_grad():
        r.clamp_(0.0, L)

    # Save epoch data
    epoch_data.append((epoch + 1, loss.item(), r.item()))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, r: {r.item():.4f}')
print('Optimization Finished.')

# Plot loss and r over epochs in one figure
epochs, losses, r_values = zip(*epoch_data)

fig, ax1 = plt.subplots(figsize=(8, 4))

# Plot loss on the left y-axis
ax1.plot(epochs, losses, 'b-', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(0, 1000)
ax1.grid()

# Plot r on the right y-axis
ax2 = ax1.twinx()
ax2.plot(epochs, r_values, 'r-', label='r (Optimal Location)')
ax2.set_ylabel('r (Optimal Location)', color='r')
ax2.set_ylim(0, L)
ax2.set_xlim(0, num_epochs)
ax2.tick_params(axis='y', labelcolor='r')

# Add title and save the figure
plt.title('Cost Function and r Over Epochs')
fig.tight_layout()
plt.savefig('results/loss_and_r_over_epochs.png', dpi=300)
plt.close()

# Save final displacement plot
u_final = simulate_wave(r, u_in, u0, v0).detach().cpu().numpy()
plt.figure(figsize=(8, 4))
plt.imshow(u_final.T, extent=[0, T, 0, L], aspect='auto', origin='lower')
plt.colorbar(label='Displacement')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Finite Difference Model under Optimal Control and Actuator Location')
plt.savefig('results/wave_propagation.png', dpi=300)
plt.close()

# Save final displacement plot using the surrogate model
u_final = surrogate_model(r, u_in, u0, v0).detach().cpu().numpy()
plt.figure(figsize=(8, 4))
plt.imshow(u_final.T, extent=[0, T, 0, L], aspect='auto', origin='lower')
plt.colorbar(label='Displacement')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Fourier Surrogate Model under Optimal Control and Actuator Location')
plt.savefig('results/wave_propagation_surrogate.png', dpi=300)
plt.close()

# Save control input plot
plt.figure(figsize=(8, 4))
plt.plot(t.cpu().numpy(), u_in.detach().cpu().numpy(), label="Control Input (u_in)")
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.xlim(0, T)
plt.title('Optimal Control Input Over Time')
plt.legend()
plt.grid()
plt.savefig('results/control_input.png', dpi=300)
plt.close()