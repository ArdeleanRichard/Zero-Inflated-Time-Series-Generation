import os
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_FOLDER      = "./data/"
OUT_FOLDER       = "./out_m5/"
SECONDS_PER_HOUR = 3600.0
MAX_HOURS        = 24.0   # hard physical upper bound

os.makedirs(OUT_FOLDER, exist_ok=True)


MAP_MODEL_NAMES = {
    "zits-gan": "ZITS-GAN",
    "zits-vae": "ZITS-VAE",
    "timegan": "TimeGAN",
    "transfusion": "TransFusion",
    "fide": "FIDE",
    "chronogan": "ChronoGAN",
    "tsgm_timegan": "TSGM-TimeGAN",
    "tsgm_vae": "TSGM-TimeVAE",
    "vae_dense": "VAE-Dense",
    "vae_conv": "VAE-Conv",
    "timeVAE": "TimeVAE",
}