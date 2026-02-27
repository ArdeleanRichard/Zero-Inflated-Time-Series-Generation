"""
Data processing utilities and AR model fitting
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg

from constants import device


def process_data(ori_data, time_series_seq_len=30):
    """
    Process and normalize time series data into sequences
    
    Args:
        ori_data: Original time series data
        time_series_seq_len: Length of each sequence
        
    Returns:
        Torch tensor of processed sequences
    """
    # Normalize the data
    scaler = StandardScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - time_series_seq_len):
        _x = ori_data[i:i + time_series_seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.arange(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    
    return torch.from_numpy(np.array(data))


def fit_AR_model(data, true_coeffs, order=2, seq_len=30):
    """
    Fit an AR (AutoRegressive) model to the data and compare with true coefficients
    
    Args:
        data: Time series data
        true_coeffs: True AR coefficients for comparison
        order: Order of the AR model
        seq_len: Sequence length
    """
    time_series_data = np.array(data.cpu()).reshape(-1, seq_len)

    print("True AR Coefficients:", true_coeffs)

    # Estimate AR coefficients from the generated data
    estimated_ar_coefficients = []
    for series in time_series_data:
        model = AutoReg(series, lags=order)
        result = model.fit()
        estimated_ar_coefficients.append(result.params[1:])

    # Convert the estimated coefficients to a numpy array
    estimated_ar_coefficients = np.array(estimated_ar_coefficients)

    # Print the estimated AR coefficients (mean across all samples)
    print("Estimated AR Coefficients:", np.mean(estimated_ar_coefficients, axis=0))

    mse = np.mean((true_coeffs - np.mean(estimated_ar_coefficients, axis=0))**2)
    mae = np.mean(np.abs(true_coeffs - np.mean(estimated_ar_coefficients, axis=0)))
    print("Mean squared error of the estimated coefficients, MSE:", mse)
    print("Mean absolute error of the estimated coefficients, MAE:", mae)



def read_iot_data():
    df = pd.read_csv('../../../data/iot_durations_2021.csv', index_col=0)
    data_df = df.transpose()

    return data_df


def load_iot_data():
    data_df = read_iot_data()
    # real_data = data_df.to_numpy()[:20000]
    real_data = data_df.to_numpy()

    real_data = real_data[..., np.newaxis].astype(np.float32)

    seq_len = real_data.shape[1]
    t = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(real_data.shape[0], seq_len, 1).to(device)
    real_data = torch.from_numpy(real_data)

    return t, real_data, real_data.shape[1], real_data.shape[2]




def load_m5_data():
    data = np.load("../../../data/m5/m5_X_365.npy")
    # real_data = data[:20000]
    real_data = data

    real_data = real_data[..., np.newaxis].astype(np.float32)

    seq_len = real_data.shape[1]
    t = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(real_data.shape[0], seq_len, 1).to(device)
    real_data = torch.from_numpy(real_data)

    return t, real_data, real_data.shape[1], real_data.shape[2]


def data_enhance_frequency(real_data):
    from scipy.fft import rfft, rfftfreq, irfft
    # Frequency enhancement parameters
    c = 1.1  # Enhancement factor
    percentage_of_freq_enhanced = 20

    print("\nApplying frequency enhancement...")
    if not isinstance(real_data, np.ndarray):
        real_data = real_data.cpu().numpy()

    real_data_fft = rfft(real_data, axis=1)
    real_data_freq = rfftfreq(real_data.shape[1])
    print(f"Frequency shape: {real_data_freq.shape}")

    top_freq_enhanced = int((real_data_fft.shape[1] * percentage_of_freq_enhanced) / 100)
    high_freq_enhanced_fft_result = real_data_fft.copy()
    top_indices = np.argsort(real_data_freq)[-top_freq_enhanced:]

    # Enhance high frequencies
    for i in range(real_data_fft.shape[0]):
        high_freq_enhanced_fft_result[i, :, 0][top_indices] *= c

    # Convert back to time domain
    real_data = irfft(high_freq_enhanced_fft_result.reshape(-1, real_data_freq.shape[0]), n=real_data.shape[1])
    real_data = torch.from_numpy(real_data.reshape(real_data.shape[0], real_data.shape[1], 1)).to(device)

    print(f"Enhanced data shape: {real_data.shape}")
    return real_data