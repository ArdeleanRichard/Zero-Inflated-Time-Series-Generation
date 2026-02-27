import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')

from constants import DATA_FOLDER, FIGS_FOLDER

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Utility functions for data preparation and evaluation

def prepare_time_series_data(df, scaler=None):
    """
    Prepare time series data for VAE/GAN training

    Args:
        df: Pandas DataFrame with time series data
        scaler: Optional scaler for data normalization

    Returns:
        PyTorch DataLoader objects, scaler
    """
    # Convert to numpy array
    data = df.values

    if len(data.shape) == 2:
        # Reshape to (n_samples, n_timesteps, 1)
        data = data.reshape(data.shape[0], data.shape[1], 1)

    # Create or use provided scaler
    if scaler is None:
        scaler = MinMaxScaler()
        # Flatten, scale, and reshape back
        orig_shape = data.shape
        flattened = data.reshape(-1, 1)
        scaled_flat = scaler.fit_transform(flattened)
        data = scaled_flat.reshape(orig_shape)
    else:
        # Apply existing scaler
        orig_shape = data.shape
        flattened = data.reshape(-1, 1)
        scaled_flat = scaler.transform(flattened)
        data = scaled_flat.reshape(orig_shape)

    return data, scaler


def create_dataloaders(data, train_ratio=0.8, batch_size=32):
    """
    Create PyTorch DataLoaders from numpy data

    Args:
        data: Numpy array of shape (n_samples, n_timesteps, 1)
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for DataLoaders

    Returns:
        train_loader, val_loader
    """
    # Split into train and validation
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    val_data = data[n_train:]

    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)

    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def read_iot_data():
    df = pd.read_csv(DATA_FOLDER + 'iot_durations_2021.csv', index_col=0)
    data_df = df.transpose()

    return data_df


def data_convert():
    # Load CSV with pandas
    df = pd.read_csv("./data/iot_durations_2021.csv")  # replace with your filename

    # Convert to NumPy array
    arr = df.values  # shape = (x, y)
    arr = arr.T

    # Reshape to (x, y, 1)
    arr_reshaped = arr.reshape(arr.shape[0], arr.shape[1], 1)
    print(arr.shape)
    print(arr_reshaped.shape)

    # Save as NPZ
    np.savez("./data/iot_durations_2021.npz", data=arr_reshaped)


if __name__ == "__main__":
    data_convert()
    # plot_samples()