# FIDE: Frequency-enhanced Implicit Diffusion for Extreme Event Generation

This project implements a diffusion-based model for generating time series data with extreme events, using GEV (Generalized Extreme Value) distribution conditioning.

## Project Structure

```
.
├── main.py                    # Main training script
├── model.py                   # Transformer-based diffusion model
├── train_utilities.py         # Training utilities and diffusion functions
├── data_process.py           # Data processing and AR model utilities
├── general_utilities.py      # GEV fitting and statistical tests
├── evaluation.py             # Model evaluation metrics
├── evaluation_runner.py      # Comprehensive evaluation script
├── requirements.txt          # Python dependencies
└── Data/                     # Data directory (create this)
    └── temperature_raw.csv   # Your temperature data
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The code expects a CSV file with the following format:
- Column 1: Date (datetime format)
- Column 2: Temperature values (numeric)

Example `temperature_raw.csv`:
```
Date,Temperature
2020-01-01,15.5
2020-01-02,16.2
...
```

Place your data file in a `Data/` directory.

## Usage

### Training

Run the main training script:
```bash
python main.py
```

This will:
1. Load and process the temperature data
2. Fit a GEV distribution to block maxima
3. Apply frequency enhancement
4. Train the diffusion model
5. Generate samples
6. Evaluate the generated samples
7. Save the trained model as `diffusion_model.pth`

### Configuration

You can modify the following parameters in `main.py`:

```python
# Model configuration
time_series_seq_len = 30    # Sequence length
batch_size = 2000            # Training batch size
n_epochs = 400               # Number of training epochs
is_regularizer = True        # Use GEV regularization

# Frequency enhancement
c = 1.1                      # Enhancement factor
percentage_of_freq_enhanced = 20  # Percentage of frequencies to enhance

# Diffusion parameters (in train_utilities.py)
diffusion_steps = 100        # Number of diffusion steps
gp_sigma = 0.05             # GP kernel parameter
```

### Evaluation

The training script automatically runs evaluation. For separate evaluation:

```python
from evaluation_runner import run_comprehensive_evaluation

run_comprehensive_evaluation(
    real_data=real_data,
    samples_ddpm=samples_ddpm,
    seq_len=30,
    n_seq=1,
    block_maxima_real_data_value=block_maxima_values,
    bm_samples_ddpm_value=generated_maxima_values,
    is_fit_AR=False
)
```

## Model Architecture

The model uses a **Transformer-based architecture** with:
- Positional encoding for time and diffusion steps
- Conditional generation based on GEV-sampled block maxima
- Multi-head self-attention layers
- Feed-forward networks

## Key Features

1. **GEV Conditioning**: Conditions generation on extreme value distributions
2. **Frequency Enhancement**: Enhances high-frequency components before training
3. **Gaussian Process Noise**: Uses GP-correlated noise in the diffusion process
4. **Regularization**: Optional GEV-based regularization during training

## Evaluation Metrics

The code provides comprehensive evaluation:

1. **Predictive Score**: RNN regression on real vs synthetic data
2. **Discriminative Score**: LSTM classifier distinguishing real from synthetic
3. **Statistical Tests**:
   - Kolmogorov-Smirnov test
   - Cramér-von Mises distance
   - KL/JS divergence
   - CRPS (Continuous Ranked Probability Score)
4. **Visualizations**:
   - KDE density plots
   - PCA projection
   - t-SNE visualization

## Output

The training script produces:
- Training loss plots (linear and log scale)
- Generated sample visualizations
- Statistical comparison metrics
- Saved model: `diffusion_model.pth`

## Notes

- The code uses GPU if available (CUDA)
- Training time depends on dataset size and number of epochs
- Adjust `batch_size` based on available GPU memory
- For best results, tune hyperparameters for your specific dataset

## Citation

If you use this code, please cite the original FIDE paper (add citation here).

## License

(Add your license information here)
