# Quick Start Guide

## Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Data

### Option A: Use Your Own Data
Place your temperature CSV file in the `Data/` directory with format:
```
Date,Temperature
2020-01-01,15.5
2020-01-02,16.2
...
```

## Step 3: Run Training

```bash
python main.py
```

This will:
- Load and process the data
- Train the diffusion model
- Generate samples
- Run evaluation metrics
- Save the trained model

**Expected output:**
- Training progress with loss values
- Multiple plots showing:
  - GEV fit quality
  - Training loss curves
  - Generated samples
  - KDE comparisons
  - Statistical test results

**Training time:** Approximately 10-30 minutes depending on your hardware.

## Step 4: Monitor Results

Watch for these key metrics in the output:

1. **GEV Fit Quality** (should have high p-value)
2. **Training Loss** (should decrease over epochs)
3. **K-S Statistic** (lower is better, close to 0)
4. **CRPS** (lower is better)
5. **Predictive Score (MAE)** (lower is better)
6. **Discriminative Score** (closer to 0 means harder to distinguish, indicating better quality)

## Step 5: Adjust Parameters (Optional)

If results are not satisfactory, try adjusting in `main.py`:

```python
# Increase epochs for better convergence
n_epochs = 600

# Adjust frequency enhancement
c = 1.2  # Higher value = more enhancement
percentage_of_freq_enhanced = 30

# Change model capacity
hidden_dim = 128  # Larger model (in TransformerModel initialization)

# Adjust regularization
is_regularizer = False  # Disable if GEV regularization causes issues
```

## Common Issues

### Issue: Out of Memory
**Solution:** Reduce batch_size in `main.py`:
```python
batch_size = 1000  # or even lower
```

### Issue: Training is too slow
**Solution:** 
1. Reduce number of epochs
2. Use smaller model (hidden_dim=32)
3. Ensure you're using GPU if available

### Issue: Generated samples don't match real data well
**Solution:**
1. Increase number of epochs
2. Adjust frequency enhancement parameters
3. Enable/disable regularization
4. Check if your data has sufficient samples (need at least 1000+)

### Issue: Import errors
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

## File Descriptions

- **main.py**: Main training script - run this first
- **model.py**: Neural network architecture
- **train_utilities.py**: Diffusion process implementation
- **data_process.py**: Data preprocessing functions
- **general_utilities.py**: Statistical tests and GEV fitting
- **evaluation.py**: Evaluation metrics (RNN, LSTM, PCA, t-SNE)
- **evaluation_runner.py**: Comprehensive evaluation wrapper
- **generate_example_data.py**: Create synthetic test data

## Next Steps

After successful training:

1. **Load and use the model:**

```python
import torch
from model import TransformerModel

# Load trained model
model = TransformerModel(dim=1, hidden_dim=64, max_i=100, seq_len=30, n_condition=1)
model.load_state_dict(torch.load('model/diffusion_model.pth'))
model.eval()
```

2. **Generate new samples:**
```python
from train_utilities import sample, get_betas

# Setup (use same values as training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
betas = get_betas(100, device)
alphas = torch.cumprod(1 - betas, dim=0)

# Generate
new_samples = sample(t_grid, bm_conditional, model, 100, betas, alphas, device=device)
```

3. **Run comprehensive evaluation:**
```python
from evaluation_runner import run_comprehensive_evaluation

run_comprehensive_evaluation(
    real_data=your_real_data,
    samples_ddpm=generated_samples,
    seq_len=30,
    n_seq=1,
    block_maxima_real_data_value=real_maxima,
    bm_samples_ddpm_value=generated_maxima
)
```

## Tips for Best Results

1. **Data Quality**: Ensure your temperature data has sufficient length (preferably 5+ years)
2. **Preprocessing**: The code automatically standardizes by monthly mean/std - appropriate for temperature data
3. **Hyperparameters**: Start with defaults, then tune based on results
4. **Evaluation**: Don't just look at one metric - consider all statistical tests together
5. **GPU**: Training is much faster on GPU - use CUDA if available

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify your data format matches the expected format
3. Try with the generated example data first
4. Reduce complexity (smaller model, fewer epochs) to isolate issues
5. Check that all dependencies are correctly installed

## Performance Benchmarks

Typical performance on example data (10 years, daily):
- **Training Time**: ~15 minutes (GPU) / ~2 hours (CPU)
- **Memory Usage**: ~4GB GPU / ~8GB RAM
- **Sample Generation**: ~30 seconds for 1000 samples
- **KS p-value**: > 0.05 (good fit)
- **CRPS**: < 0.1 (good quality)
- **Discriminative Score**: < 0.1 (hard to distinguish)
