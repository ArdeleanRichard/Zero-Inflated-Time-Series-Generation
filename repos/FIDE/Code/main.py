"""
Main training script for FIDE (Frequency-enhanced Implicit Diffusion for Extreme event generation)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from constants import device, is_fit_AR, diffusion_steps, n_epochs, batch_size, is_regularizer, gp_sigma
from data_process import fit_AR_model, load_iot_data, data_enhance_frequency, load_m5_data
from general_utilities import (fitting_gev_and_sampling, plot_kde, KS_Test, CMD, KL_JS_divergence, CRPS)
from model import TransformerModel
from train_utilities import (get_betas, get_loss, plot_losses, sample)


# ========== DATA PROCESSING ==========
print("Loading and processing data...")
# t, real_data, seq_len, n_condition = load_iot_data()
t, real_data, seq_len, n_condition = load_m5_data()
print(t.shape)
print(real_data.shape)

print(f"Real Data: Mean: {torch.mean(real_data):.4f}, Std: {torch.std(real_data):.4f}")

if is_fit_AR:
    fit_AR_model(real_data, true_coeffs=[0.5], order=1, seq_len=seq_len)

# Extract block maxima
block_maxima_real_data_value, block_maxima_real_data_pos = torch.max(real_data, dim=1)
block_maxima_real_data_value = block_maxima_real_data_value.reshape(-1, 1, 1)
block_maxima_real_data_pos = block_maxima_real_data_pos.reshape(-1, 1, 1)
block_maxima_real_data = block_maxima_real_data_value

print(f"Shape of real data, time steps (t), block maxima: {real_data.shape}, {t.shape}, {block_maxima_real_data.shape}")
num_samples = block_maxima_real_data.shape[0]
print(f"Number of samples: {num_samples}")

# ========== GEV FITTING AND METRICS ==========
print("\nFitting GEV distribution...")

block_maxima_real_data_value = block_maxima_real_data_value.cpu().numpy().reshape(-1)
block_maxima_real_data_pos = block_maxima_real_data_pos.cpu().numpy().reshape(-1)
bm_samples_gev, gev_model = fitting_gev_and_sampling(block_maxima_real_data_value, num_samples)

print("\nEvaluating GEV fit:")
plot_kde(block_maxima_real_data_value, bm_samples_gev, 
         x_axis_label="Max Value", 
         title="KDE Density Plot of Max Values (Real vs GEV Fitted)")
KS_Test(block_maxima_real_data_value, bm_samples_gev)
CMD(block_maxima_real_data_value, bm_samples_gev)
KL_JS_divergence(block_maxima_real_data_value, bm_samples_gev)
CRPS(block_maxima_real_data_value, bm_samples_gev)

# ========== FREQUENCY ENHANCEMENT ==========
real_data = data_enhance_frequency(real_data)

# ========== DIFFUSION SETUP ==========
print("\nInitializing diffusion model...")

betas = get_betas(diffusion_steps, device)
alphas = torch.cumprod(1 - betas, dim=0)

# Reconstruct block maxima tensor
block_maxima_real_data = torch.from_numpy(block_maxima_real_data_value.reshape(-1, 1, 1)).to(device).float()

# ========== MODEL INITIALIZATION ==========
print("\nInitializing model...")

model = TransformerModel(
    dim=1, 
    hidden_dim=64, 
    max_i=diffusion_steps,
    seq_len=seq_len,
    n_condition=n_condition
).to(device)

optim = torch.optim.Adam(model.parameters())

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ========== TRAINING ==========
print(f"\nStarting training for {n_epochs} epochs...")

training_loss_history = np.array([])
train_loader = DataLoader(
    TensorDataset(real_data, t, block_maxima_real_data), 
    batch_size=batch_size, 
    shuffle=False, 
    drop_last=True
)

for epoch in tqdm(range(n_epochs)):
    for i, (input_data, time, bm) in enumerate(train_loader):
        optim.zero_grad()
        loss, ddpm_loss, reg_loss = get_loss(
            input_data, time, bm, model, diffusion_steps, 
            alphas, gev_model, is_regularizer, gp_sigma, device
        )
        loss.backward()
        optim.step()
    
    if epoch < 25 or epoch % 20 == 0:
        if is_regularizer:
            print(f"Epoch {epoch} --- Loss: {loss.cpu().item():.6f}, "
                  f"DDPM Loss: {ddpm_loss.cpu().item():.6f}, "
                  f"Reg Loss: {reg_loss.cpu().item():.6f}")
        else:
            print(f"Epoch {epoch} --- Loss: {loss.cpu().item():.6f}")
    
    training_loss_history = np.append(training_loss_history, loss.item())

print("\nTraining completed!")
plot_losses(training_loss_history)

# ========== SAMPLING ==========
print("\nGenerating samples...")

# Prepare conditional samples
bm_samples_conditional = torch.tensor(bm_samples_gev, dtype=torch.float32).to(device)
bm_samples_conditional = bm_samples_conditional.reshape(-1, 1, 1)

# Generate time grid
t_grid = torch.linspace(0, seq_len, seq_len).view(1, -1, 1).to(device)

# Sample from the model
samples_ddpm = sample(
    t_grid.repeat(num_samples, 1, 1), 
    bm_samples_conditional,
    model,
    diffusion_steps,
    betas,
    alphas,
    gp_sigma,
    device
)

# Visualize some samples
print("\nVisualizing generated samples...")
for i in range(10):
    plt.plot(
        t_grid.squeeze().detach().cpu().numpy(), 
        samples_ddpm[i].squeeze().detach().cpu().numpy(),
        color='C0', 
        alpha=1 / (i + 1)
    )
plt.title('10 new realizations')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig("./figs/generated_samples.png")
plt.close()

gen_data = samples_ddpm.detach().cpu().numpy()
np.savez_compressed('./model/fide_generated_data.npz', data=gen_data)
num_zero_rows = np.sum(np.all(gen_data.squeeze() == 0, axis=1))
print("Generated data - count full 0s", num_zero_rows)

# Extract generated block maxima
bm_samples_ddpm_value, bm_samples_ddpm_pos = torch.max(samples_ddpm, dim=1)
bm_samples_ddpm_value = bm_samples_ddpm_value.cpu().numpy().reshape(-1)
bm_samples_ddpm = samples_ddpm.cpu().numpy().reshape(-1)

print(f"\nShape (block_maxima_real_data, bm_samples_conditional, samples_ddpm): "
      f"{block_maxima_real_data_value.shape}, {bm_samples_gev.shape}, {samples_ddpm.shape}")

# ========== EVALUATION ==========
print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)

print("\nBlock Maxima Evaluation:")
plot_kde(block_maxima_real_data_value, bm_samples_ddpm_value, 
         x_axis_label="Max Value", 
         title="KDE Density Plot of Max Values (Real vs DDPM Generated)")
KS_Test(block_maxima_real_data_value, bm_samples_ddpm_value)
CMD(block_maxima_real_data_value, bm_samples_ddpm_value)
KL_JS_divergence(block_maxima_real_data_value, bm_samples_ddpm_value)
CRPS(block_maxima_real_data_value, bm_samples_ddpm_value)

print("\nAll Values Evaluation:")
real_data_cpu = real_data.cpu().numpy().reshape(-1, 1)
plot_kde(real_data_cpu.reshape(-1), bm_samples_ddpm, 
         x_axis_label="All Values", 
         title="KDE Density Plot of All Values (Real vs Generated (DDPM))")
KS_Test(real_data_cpu.reshape(-1), bm_samples_ddpm)
CMD(real_data_cpu.reshape(-1), bm_samples_ddpm)
KL_JS_divergence(real_data_cpu.reshape(-1), bm_samples_ddpm)
CRPS(real_data_cpu.reshape(-1), bm_samples_ddpm)

if is_fit_AR:
    fit_AR_model(
        torch.tensor(bm_samples_ddpm.reshape(-1, seq_len)).to(device),
        true_coeffs=[0.5], 
        order=1,
        seq_len=seq_len
    )

# Save model
print("\nSaving model...")
torch.save(model.state_dict(), './model/diffusion_model.pth')
print("Model saved as './model/diffusion_model.pth'")
