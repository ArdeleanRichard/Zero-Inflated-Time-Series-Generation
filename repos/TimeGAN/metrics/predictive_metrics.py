"""
Predictive Metric (TF2 / Keras)
Converted from TF1 TimeGAN code to TensorFlow 2.

Trains a post-hoc GRU predictor on synthetic data to predict one-step-ahead
(the last feature column). Evaluates MAE on original data.

Assumptions:
- ori_data and generated_data: numpy arrays of shape (N, seq_len, dim) padded with zeros.
- extract_time(data) returns (time_list, max_seq_len) where time_list[i] is the
  true length of sequence i, and max_seq_len is the maximum length across dataset.
- Sequences are padded at the end (post-padding).
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from TimeSeriesGenerator.repos.TimeGAN.utils import extract_time


class Predictor(tf.keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        # return_sequences=True to predict at each time-step
        self.gru = tf.keras.layers.GRU(hidden_dim, activation="tanh", return_sequences=True)
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))  # predict scalar next-step feature

    def call(self, x, mask=None, training=False):
        h = self.gru(x, mask=mask, training=training)  # (batch, time, hidden)
        logits = self.out(h)  # (batch, time, 1)
        # original used sigmoid after logits — to preserve identical scaling use sigmoid
        y_hat = tf.sigmoid(logits)
        return y_hat


def _pad_batch(seqs, max_time, feature_dim):
    """
    seqs: list/iterable of arrays with shape (time_i, feature_dim)
    returns: ndarray (batch, max_time, feature_dim) padded with zeros
    """
    batch_size = len(seqs)
    out = np.zeros((batch_size, max_time, feature_dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        t = s.shape[0]
        out[i, :t, :] = s
    return out


def predictive_score_metrics(ori_data,
                             generated_data,
                             iterations=5000,
                             batch_size=128,
                             hidden_dim=None,
                             learning_rate=1e-3,
                             verbose=False):
    """
    Train a post-hoc RNN predictor (on generated_data) to predict the last feature one-step-ahead,
    then evaluate MAE on original data.

    Returns:
      predictive_score: float (mean MAE across original sequences)
    """
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)

    no, seq_len, dim = ori_data.shape

    # Extract sequence lengths and max lengths
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)  # FIXED: use generated_data
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    # input timesteps to predictor = max_seq_len - 1 (we use x[:-1] -> predict y = next-step last feature)
    input_time_steps = max_seq_len - 1
    input_dim = dim - 1  # predictor uses all features except the last one as input

    if hidden_dim is None:
        hidden_dim = max(1, dim // 2)

    # Build model and optimizer
    predictor = Predictor(hidden_dim=hidden_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Helper: compute masked MAE loss (mean absolute error over valid timesteps)
    def masked_mae(y_true, y_pred, seq_lengths):
        # y_true, y_pred: (batch, time, 1)
        # seq_lengths: (batch,) lengths (<= time)
        mask = tf.sequence_mask(seq_lengths, maxlen=tf.shape(y_true)[1], dtype=tf.float32)  # (batch, time)
        mask = tf.expand_dims(mask, axis=-1)  # (batch, time, 1)
        abs_diff = tf.abs(y_true - y_pred) * mask
        # avoid division by zero: sum mask elements
        denom = tf.reduce_sum(mask)
        # if denom==0, return 0 loss
        return tf.reduce_sum(abs_diff) / tf.maximum(denom, 1.0)

    # Training loop on generated data
    gen_N = len(generated_data)
    for itt in range(1, iterations + 1):
        # Sample a random minibatch of indices
        idx = np.random.permutation(gen_N)[:batch_size]
        # Build X_mb, T_mb, Y_mb lists then pad
        X_seq_list = []
        Y_seq_list = []
        T_mb = []
        for i in idx:
            seq = generated_data[i]
            seq_len_i = generated_time[i]
            # drop last timestep for inputs, drop first for targets (next-step)
            x_i = seq[:seq_len_i - 1, :input_dim]    # shape (seq_len_i-1, input_dim)
            y_i = seq[1:seq_len_i, input_dim:input_dim + 1]  # last feature at next timestep -> shape (seq_len_i-1, 1)
            X_seq_list.append(x_i)
            Y_seq_list.append(y_i)
            T_mb.append(seq_len_i - 1)

        # pad to input_time_steps
        X_mb_padded = _pad_batch(X_seq_list, input_time_steps, input_dim)   # (batch, time, input_dim)
        Y_mb_padded = _pad_batch(Y_seq_list, input_time_steps, 1)           # (batch, time, 1)
        T_mb = np.asarray(T_mb, dtype=np.int32)

        # convert to tensors
        X_tensor = tf.convert_to_tensor(X_mb_padded, dtype=tf.float32)
        Y_tensor = tf.convert_to_tensor(Y_mb_padded, dtype=tf.float32)
        seq_lens_tensor = tf.convert_to_tensor(T_mb, dtype=tf.int32)

        with tf.GradientTape() as tape:
            # create mask for Keras GRU
            mask = tf.sequence_mask(seq_lens_tensor, maxlen=input_time_steps)
            y_pred = predictor(X_tensor, mask=mask, training=True)  # (batch, time, 1)
            loss = masked_mae(Y_tensor, y_pred, seq_lens_tensor)

        grads = tape.gradient(loss, predictor.trainable_variables)
        optimizer.apply_gradients(zip(grads, predictor.trainable_variables))

        if verbose and (itt == 1 or itt % 1000 == 0):
            tf.print("Iteration", itt, "train_masked_MAE:", loss)

    # Evaluation on original data (use all original examples)
    N_ori = len(ori_data)
    X_seq_list = []
    Y_seq_list = []
    T_test = []
    for i in range(N_ori):
        seq = ori_data[i]
        seq_len_i = ori_time[i]
        x_i = seq[:seq_len_i - 1, :input_dim]
        y_i = seq[1:seq_len_i, input_dim:input_dim + 1]
        X_seq_list.append(x_i)
        Y_seq_list.append(y_i)
        T_test.append(seq_len_i - 1)

    X_test_padded = _pad_batch(X_seq_list, input_time_steps, input_dim)
    Y_test_padded = _pad_batch(Y_seq_list, input_time_steps, 1)
    T_test = np.asarray(T_test, dtype=np.int32)

    # Predict
    mask_test = tf.sequence_mask(T_test, maxlen=input_time_steps)
    y_pred_test = predictor(tf.convert_to_tensor(X_test_padded, dtype=tf.float32), mask=mask_test, training=False)
    y_pred_np = y_pred_test.numpy()  # (N_ori, time, 1)

    # Compute MAE per sequence using true lengths, then average across sequences
    MAE_total = 0.0
    for i in range(N_ori):
        L = T_test[i]
        if L <= 0:
            # if sequence has no valid timesteps (edge case), skip contribution
            continue
        y_true_seq = Y_test_padded[i, :L, 0]
        y_pred_seq = y_pred_np[i, :L, 0]
        MAE_total += mean_absolute_error(y_true_seq, y_pred_seq)

    predictive_score = MAE_total / float(N_ori)
    return predictive_score
