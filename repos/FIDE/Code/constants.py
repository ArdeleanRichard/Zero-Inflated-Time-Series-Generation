import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== CONFIGURATION ==========
# time_series_seq_len = 365
# seq_len = time_series_seq_len
# n_condition = 1
is_fit_AR = False
is_regularizer = True

batch_size = 2000
n_epochs = 10


# Global variables for diffusion configuration
diffusion_steps = 100
gp_sigma = 0.05