import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split

from constants import DATA_FOLDER, SECONDS_PER_HOUR, MAX_HOURS


class DataPreprocessor:
    """
    Fit log10(hours+1) / log_scale transform on training data.

    Parameters
    ----------
    None — all hyperparameters are derived from the data during fit_transform.
    """

    def __init__(self):
        self.log_scale: float = float(np.log10(MAX_HOURS + 1.0))  # overwritten at fit
        self.max_seconds: float = MAX_HOURS * SECONDS_PER_HOUR
        self.stats: dict = {}

    # ------------------------------------------------------------------
    # Fit + transform
    # ------------------------------------------------------------------

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit on data and return normalised array.
        Call once on the full dataset before any train/val split.
        """
        data = np.asarray(data, dtype=np.float64)
        nz   = data[data > 0]

        max_hours = float(np.max(data)) / SECONDS_PER_HOUR
        # Anchor log_scale to the larger of the data max and the physical max
        # so the scale is stable and the physical ceiling maps to ≤ 1.
        self.log_scale   = float(np.log10(max(max_hours, MAX_HOURS) + 1.0))
        self.max_seconds = float(np.max(data))

        self.stats = {
            'log_scale':    self.log_scale,
            'max_seconds':  self.max_seconds,
            'max_hours':    max_hours,
            'zero_ratio':   float(np.mean(data == 0)),
            'min_nonzero':  float(np.min(nz)) if len(nz) else 0.0,
            'mean_nonzero': float(np.mean(nz)) if len(nz) else 0.0,
        }
        return self._apply(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform new data with already-fitted parameters."""
        if not self.stats:
            raise RuntimeError("Call fit_transform before transform.")
        return self._apply(np.asarray(data, dtype=np.float64))

    def _apply(self, data: np.ndarray) -> np.ndarray:
        """log10(hours + 1) / log_scale  →  [0, 1]."""
        return (np.log10(data / SECONDS_PER_HOUR + 1.0) / self.log_scale).astype(np.float32)

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Map normalised values back to seconds.
        Zeros in → zeros out (exact, no threshold needed).
        """
        x = np.asarray(data, dtype=np.float64)
        # Undo normalisation: norm → log-space hours → hours → seconds
        x = (10.0 ** (x * self.log_scale) - 1.0) * SECONDS_PER_HOUR
        # Clip to valid physical range
        x = np.clip(x, 0.0, self.stats.get('max_seconds', MAX_HOURS * SECONDS_PER_HOUR))
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({'log_scale': self.log_scale,
                       'max_seconds': self.max_seconds,
                       'stats': self.stats}, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            state = json.load(f)
        self.log_scale   = state['log_scale']
        self.max_seconds = state['max_seconds']
        self.stats       = state['stats']


class CountDataPreprocessor:
    """
    Fit log10(count+1) / log_scale transform on count data.
    quantile-based scaling for robustness to outliers.

    Parameters
    ----------
    quantile : float, default=0.99
        Quantile to use for scaling (0.95, 0.99, or 0.999).
        Higher = more compression, lower = more overflow for extremes.
    clip_strategy : str, default='none'
        How to handle extreme values:
        - 'none': Allow values > 1.0 (recommended for heavy-tailed data)
        - 'soft': Apply gentle sigmoid squashing above 1.0
        - 'hard': Hard clip at some multiple of the scale
    max_count : float or None, default=None
        Optional hard ceiling for inverse transform (e.g., 10000)
        If None, uses observed maximum during fit
    """

    def __init__(self, quantile: float = 0.99, clip_strategy: str = 'none',
                 max_count: float = None):
        self.quantile = quantile
        self.clip_strategy = clip_strategy
        self.max_count = max_count

        # Fitted parameters
        self.log_scale: float = 1.0
        self.fitted_max: float = 0.0
        self.fitted_quantile: float = 0.0
        self.stats: dict = {}

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit on data and return normalised array.
        Call once on the full dataset before any train/val split.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, seq_length), non-negative counts

        Returns
        -------
        np.ndarray
            Normalised data in approximately [0, 1] (may slightly exceed 1.0)
        """
        data = np.asarray(data, dtype=np.float64)

        # Validate input
        if np.any(data < 0):
            raise ValueError("Count data must be non-negative")

        # Compute statistics
        nz = data[data > 0]
        self.fitted_max = float(np.max(data))
        self.fitted_quantile = float(np.quantile(data[data > 0], self.quantile)) if len(nz) > 0 else 1.0

        # Set max_count if not provided
        if self.max_count is None:
            self.max_count = self.fitted_max

        # Compute log scale from quantile (not max, for robustness)
        self.log_scale = float(np.log10(self.fitted_quantile + 1.0))

        # Handle edge case where all values are zero
        if self.log_scale == 0:
            self.log_scale = 1.0

        # Store statistics
        self.stats = {
            'log_scale': self.log_scale,
            'max_count': self.max_count,
            'fitted_max': self.fitted_max,
            'fitted_quantile': self.fitted_quantile,
            'quantile': self.quantile,
            'zero_ratio': float(np.mean(data == 0)),
            'min_nonzero': float(np.min(nz)) if len(nz) > 0 else 0.0,
            'mean_nonzero': float(np.mean(nz)) if len(nz) > 0 else 0.0,
            'median_nonzero': float(np.median(nz)) if len(nz) > 0 else 0.0,
            'p95_nonzero': float(np.quantile(nz, 0.95)) if len(nz) > 0 else 0.0,
            'p99_nonzero': float(np.quantile(nz, 0.99)) if len(nz) > 0 else 0.0,
        }

        return self._apply(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform new data with already-fitted parameters."""
        if not self.stats:
            raise RuntimeError("Call fit_transform before transform.")
        return self._apply(np.asarray(data, dtype=np.float64))

    def _apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply log10(x + 1) / log_scale transformation.

        Returns values primarily in [0, 1], but allows overflow for extreme values.
        """
        # Basic log transform
        x_norm = np.log10(data + 1.0) / self.log_scale

        # Apply clipping strategy
        if self.clip_strategy == 'hard':
            # Hard clip at 1.5x (allows some overflow but prevents extremes)
            x_norm = np.clip(x_norm, 0.0, 1.5)
        elif self.clip_strategy == 'soft':
            # Soft squashing using sigmoid for values > 1.0
            mask = x_norm > 1.0
            x_norm[mask] = 1.0 + 0.5 * np.tanh(x_norm[mask] - 1.0)
        # else: 'none' - no clipping, allow natural overflow

        return x_norm.astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Map normalised values back to counts.

        For count data, we round to nearest integer.
        Zeros in → zeros out (exact, no threshold needed).

        Parameters
        ----------
        data : np.ndarray
            Normalised data

        Returns
        -------
        np.ndarray
            Count data (non-negative integers)
        """
        x = np.asarray(data, dtype=np.float64)

        # Handle soft clipping inverse (approximate)
        if self.clip_strategy == 'soft':
            mask = x > 1.0
            # Approximate inverse of tanh squashing
            x[mask] = 1.0 + np.arctanh(np.clip((x[mask] - 1.0) / 0.5, -0.99, 0.99))

        # Undo normalisation: norm → log-space → counts
        x = 10.0 ** (x * self.log_scale) - 1.0

        # Clip to valid range
        x = np.clip(x, 0.0, self.max_count)

        # Round to integers for count data
        x = np.round(x)

        return x.astype(np.float32)

    def save(self, path: str) -> None:
        """Save preprocessor state to JSON file."""
        state = {
            'log_scale': self.log_scale,
            'max_count': self.max_count,
            'fitted_max': self.fitted_max,
            'fitted_quantile': self.fitted_quantile,
            'quantile': self.quantile,
            'clip_strategy': self.clip_strategy,
            'stats': self.stats
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load preprocessor state from JSON file."""
        with open(path) as f:
            state = json.load(f)
        self.log_scale = state['log_scale']
        self.max_count = state['max_count']
        self.fitted_max = state['fitted_max']
        self.fitted_quantile = state['fitted_quantile']
        self.quantile = state['quantile']
        self.clip_strategy = state['clip_strategy']
        self.stats = state['stats']


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Wraps a 2-D numpy array (n_samples × seq_length) as a PyTorch Dataset."""

    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(dataset: Dataset, train_ratio: float = 0.8, batch_size: int = 64):
    """Split into train/val DataLoaders. drop_last=True on train avoids BatchNorm issues."""
    train_size = int(train_ratio * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Raw-data helpers
# ---------------------------------------------------------------------------

def read_iot_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FOLDER + 'iot_durations_2021.csv', index_col=0)
    return df.transpose()   # (n_devices, 365)


def load_iot_data(verbose: bool = True) -> np.ndarray:
    data = read_iot_data().to_numpy().astype(np.float32)
    if verbose:
        nz = data[data > 0]
        print(f"Dataset shape:    {data.shape}")
        print(f"Zero ratio:       {np.mean(data == 0):.2%}")
        print(f"Max value:        {np.max(data):.1f}s  ({np.max(data)/3600:.2f}h)")
        print(f"Min non-zero:     {np.min(nz):.1f}s")
        print(f"Mean (non-zero):  {np.mean(nz):.1f}s  ({np.mean(nz)/3600:.2f}h)")
    return data

def load_m5_data(verbose: bool = True) -> np.ndarray:
    data = np.load("./data/m5/m5_X_365.npy").astype(np.float32)
    print("alo", np.min(data), np.max(data))
    if verbose:
        nz = data[data > 0]
        print(f"Dataset shape:    {data.shape}")
        print(f"Zero ratio:       {np.mean(data == 0):.2%}")
        print(f"Max value:        {np.max(data):.1f}s  ({np.max(data)/3600:.2f}h)")
        print(f"Min non-zero:     {np.min(nz):.1f}s")
        print(f"Mean (non-zero):  {np.mean(nz):.1f}s  ({np.mean(nz)/3600:.2f}h)")
    return data

# ---------------------------------------------------------------------------
# Visual sanity-check
# ---------------------------------------------------------------------------

def plot_samples(normalize: bool = False, n_examples: int = 5) -> None:
    data_df = read_iot_data()
    data = DataPreprocessor().fit_transform(data_df.to_numpy()) if normalize else data_df.to_numpy()
    fig, axs = plt.subplots(n_examples, 1, figsize=(16, 12))
    for i in range(n_examples):
        axs[i].plot(data[i + 5])
        axs[i].set_title(f"Sample {i + 6}")
        ymax = float(np.max(data[i + 5]))
        axs[i].set_ylim(0, ymax * 1.1 if ymax > 0 else 0.1)
    plt.tight_layout()
    plt.savefig(f"./data_examples{'_norm' if normalize else ''}.png")
    plt.close()


if __name__ == "__main__":
    # Roundtrip smoke-test
    raw  = np.array([[0, 1800, 3600, 36000, 79200, 0]], dtype=np.float32)
    pp   = DataPreprocessor()
    norm = pp.fit_transform(raw)
    back = pp.inverse_transform(norm)
