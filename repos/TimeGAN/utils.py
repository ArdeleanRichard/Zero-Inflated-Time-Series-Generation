"""Time-series Generative Adversarial Networks (TimeGAN) Utilities - PyTorch

Utility functions for TimeGAN model.
"""

import numpy as np
from typing import List, Tuple


def train_test_divide(data_x: List[np.ndarray],
                      data_x_hat: List[np.ndarray],
                      data_t: List[int],
                      data_t_hat: List[int],
                      train_rate: float = 0.8) -> Tuple:
    """
    Divide train and test data for both original and synthetic data.

    Args:
        data_x: Original data
        data_x_hat: Generated data
        data_t: Original time
        data_t_hat: Generated time
        train_rate: Ratio of training data from the original data

    Returns:
        Tuple of (train_x, train_x_hat, test_x, test_x_hat,
                  train_t, train_t_hat, test_t, test_t_hat)
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data: List[np.ndarray]) -> Tuple[List[int], int]:
    """
    Returns maximum sequence length and each sequence length.

    Args:
        data: Original data (list of arrays)

    Returns:
        time: Extracted time information (list of sequence lengths)
        max_seq_len: Maximum sequence length
    """
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        seq_len = len(data[i])
        max_seq_len = max(max_seq_len, seq_len)
        time.append(seq_len)

    return time, max_seq_len


def random_generator(batch_size: int,
                     z_dim: int,
                     T_mb: List[int],
                     max_seq_len: int) -> np.ndarray:
    """
    Random vector generation.

    Args:
        batch_size: Size of the random vector
        z_dim: Dimension of random vector
        T_mb: Time information for the random vector
        max_seq_len: Maximum sequence length

    Returns:
        Z_mb: Generated random vector array
    """
    Z_mb = []
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp)
    return np.array(Z_mb)


def batch_generator(data: np.ndarray,
                    time: List[int],
                    batch_size: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Mini-batch generator.

    Args:
        data: Time-series data
        time: Time information
        batch_size: The number of samples in each batch

    Returns:
        X_mb: Time-series data in each batch
        T_mb: Time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb