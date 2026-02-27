import numpy as np
import torch

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


DATA_FOLDER = "../../data/"
FIGS_FOLDER = "./figs/"
MODS_FOLDER = "./models/"
RES_FOLDER = "./m5_results/"