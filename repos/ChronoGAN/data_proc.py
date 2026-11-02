import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')

from constants import DATA_FOLDER, FIGS_FOLDER


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




def read_iot_data():
    df = pd.read_csv(DATA_FOLDER + 'iot_durations_2021.csv', index_col=0)
    data_df = df.transpose()

    return data_df



def plot_samples(normalize=False):
    data_df = read_iot_data()
    print(f"Preparing data with shape: {data_df.shape}")

    if normalize:
        data, scaler = prepare_time_series_data(data_df)
        data = data[:, :, 0]
    else:
        data = data_df.to_numpy()
    print(f"Loaded data shape: {data.shape}")


    nr_examples = 5
    fig, axs = plt.subplots(nr_examples, 1, figsize=(16, 12))

    # Plot original samples
    for i in range(nr_examples):
        axs[i].plot(data[i+5])
        axs[i].set_title(f"Original Sample {i + 1}")
        axs[i].set_ylim(0, max(data[i]) * 1.1 if max(data[i]) > 0 else 0.1)


    plt.tight_layout()
    plt.savefig(f"./data_examples{'' if normalize == False else '_normalized'}.png")
    plt.close()



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