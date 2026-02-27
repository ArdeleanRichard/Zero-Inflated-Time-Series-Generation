"""
Training utilities for diffusion model
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_betas(steps, device):
    """
    Get beta schedule for diffusion process
    
    Args:
        steps: Number of diffusion steps
        device: Torch device (cuda/cpu)
        
    Returns:
        Beta values for each diffusion step
    """
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind


def get_gp_covariance(t, gp_sigma=0.05):
    """
    Get Gaussian Process covariance matrix
    
    Args:
        t: Time points
        gp_sigma: GP kernel parameter
        
    Returns:
        Covariance matrix
    """
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5  # for numerical stability
    return torch.exp(-torch.square(s / gp_sigma)) + diag


def add_noise(x, t, i, alphas, gp_sigma=0.05):
    """
    Add noise to data sample based on diffusion step
    
    Args:
        x: Clean data sample, shape [B, S, D]
        t: Times of observations, shape [B, S, 1]
        i: Diffusion step, shape [B, S, 1]
        alphas: Cumulative product of (1-beta)
        gp_sigma: GP kernel parameter
        
    Returns:
        x_noisy: Noisy data
        noise: Added noise
    """
    noise_gaussian = torch.randn_like(x)

    cov = get_gp_covariance(t, gp_sigma)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian

    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

    return x_noisy, noise


def linear_decay(input_tensor, diffusion_steps):
    """
    Calculate linear decay weights for regularization
    
    Args:
        input_tensor: Diffusion step indices
        diffusion_steps: Total number of diffusion steps
        
    Returns:
        Decay weights
    """
    # Set the decay starting point and ending point
    start_index = 0
    end_index = int(0.66 * diffusion_steps)

    # Create an output tensor with the same size as the input tensor
    output_tensor = torch.zeros_like(input_tensor)

    # Apply linear decay to the values based on their indices
    for i in range(input_tensor.size(0)):
        index_value = input_tensor[i, 0].item()

        if index_value == start_index:
            output_tensor[i, 0] = 1.0
        elif start_index < index_value < end_index:
            # Linear decay function: f(x) = 1 - (x - start_index) / (end_index - start_index)
            output_tensor[i, 0] = 1.0 - (index_value - start_index) / (end_index - start_index)
        else:
            output_tensor[i, 0] = 0.0

    return output_tensor


def get_loss(x, t, bm, model, diffusion_steps, alphas, gev_model, is_regularizer=True, 
             gp_sigma=0.05, device='cuda'):
    """
    Calculate training loss
    
    Args:
        x: Input data
        t: Time points
        bm: Block maxima (conditional information)
        model: Diffusion model
        diffusion_steps: Number of diffusion steps
        alphas: Cumulative product of (1-beta)
        gev_model: Fitted GEV distribution model
        is_regularizer: Whether to use GEV regularization
        gp_sigma: GP kernel parameter
        device: Torch device
        
    Returns:
        loss: Total loss
        ddpm_loss: DDPM loss component
        reg_loss: Regularization loss component
    """
    i = torch.randint(0, diffusion_steps, size=(x.shape[0],))
    i = i.view(-1, 1, 1).expand_as(x[..., :1]).to(x)

    x_noisy, noise = add_noise(x, t, i, alphas, gp_sigma)
    pred_noise = model(x_noisy, t, i, bm)
    ddpm_loss = torch.sqrt(torch.mean((pred_noise - noise)**2))

    if is_regularizer:
        lambda_1 = linear_decay(i[:, 0, :].reshape(-1, 1), diffusion_steps)
        pred_0 = x_noisy - pred_noise
        bm_pred, _ = torch.max(pred_0, dim=1)
        reg_loss = np.mean(gev_model.logpdf(bm_pred.detach().cpu().numpy()))
        reg_loss = -0.05 * torch.tensor(reg_loss, dtype=torch.float32).to(device)

        loss = ddpm_loss + reg_loss
    else:
        reg_loss = torch.tensor(0.0).to(device)
        loss = ddpm_loss
        
    return loss, ddpm_loss, reg_loss


def plot_losses(train_history, ylim_low=0, ylim_high=0.05):
    """
    Plot training loss history
    
    Args:
        train_history: Array of loss values
        ylim_low: Lower y-axis limit
        ylim_high: Upper y-axis limit
    """
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plotting the first subplot (linear scale)
    x_values = np.arange(len(train_history)) * 10
    axes[0].plot(x_values, train_history, label="Training loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plotting the second subplot (logarithmic scale)
    axes[1].plot(x_values, train_history, label="Training loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_yscale("log")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("./figs/losses.png")
    plt.close()


@torch.no_grad()
def sample(t, bm_sample, model, diffusion_steps, betas, alphas, gp_sigma=0.05, device='cuda'):
    """
    Generate samples using the trained diffusion model
    
    Args:
        t: Time grid for generation
        bm_sample: Block maxima conditional information
        model: Trained diffusion model
        diffusion_steps: Number of diffusion steps
        betas: Beta schedule
        alphas: Alpha schedule
        gp_sigma: GP kernel parameter
        device: Torch device
        
    Returns:
        Generated samples
    """
    cov = get_gp_covariance(t, gp_sigma)
    L = torch.linalg.cholesky(cov)

    x = L @ torch.randn_like(t)

    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]

        z = L @ torch.randn_like(t)

        i = torch.Tensor([diff_step]).expand_as(x[..., :1]).to(device)
        pred_noise = model(x, t, i, bm_sample)

        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z
    
    return x

