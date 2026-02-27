"""
Discriminative metric (TF2)
Post-hoc RNN discriminator that classifies real vs synthetic sequences.
Returns discriminative_score = |accuracy - 0.5|

This implementation is robust to train/test outputs being lists (variable-length
sequences) or numpy arrays (fixed-length).
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from TimeSeriesGenerator.repos.TimeGAN.utils import train_test_divide, extract_time

# Small helper: pad a list of sequences (each shape (T_i, D)) into (batch, T, D)
def _pad_sequences(seqs, max_len=None, dtype=np.float32):
    if len(seqs) == 0:
        return np.zeros((0, 0, 0), dtype=dtype)
    # seqs can be np.ndarray of shape (N, T, D) or list of arrays
    if isinstance(seqs, np.ndarray):
        # already an array - nothing to do (but ensure dtype)
        return seqs.astype(dtype)
    # it's a list
    lengths = [s.shape[0] for s in seqs]
    D = seqs[0].shape[1] if seqs[0].ndim > 1 else 1
    if max_len is None:
        max_len = max(lengths)
    batch = np.zeros((len(seqs), max_len, D), dtype=dtype)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        if L == 0:
            continue
        batch[i, :L, :] = s
    return batch

# Sample minibatch from variable-length dataset (seqs: list/array, times: list/array)
def _sample_minibatch(seqs, times, batch_size, max_len=None):
    N = len(seqs)
    if N == 0:
        raise ValueError("Empty dataset for minibatch sampling")
    idx = np.random.randint(0, N, size=batch_size)
    sampled_seqs = [seqs[i] for i in idx]
    sampled_times = [times[i] for i in idx]
    X_mb = _pad_sequences(sampled_seqs, max_len=max_len)
    T_mb = np.array(sampled_times, dtype=np.int32)
    return X_mb, T_mb

class Discriminator(tf.keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = tf.keras.layers.GRU(hidden_dim, activation="tanh", return_sequences=False)
        self.fc = tf.keras.layers.Dense(1)  # logits

    def call(self, x, seq_lengths=None, training=False):
        # x: (batch, time, dim)
        mask = None
        if seq_lengths is not None:
            mask = tf.sequence_mask(seq_lengths, maxlen=tf.shape(x)[1])
        h = self.gru(x, mask=mask, training=training)
        logits = self.fc(h)
        probs = tf.sigmoid(logits)
        return logits, probs

def discriminative_score_metrics(ori_data,
                                 generated_data,
                                 iterations=2000,
                                 batch_size=128,
                                 hidden_dim=None,
                                 learning_rate=1e-3,
                                 verbose=False):
    """
    Trains a GRU discriminator to distinguish ori_data vs generated_data.
    Returns: discriminative_score = |accuracy - 0.5|
    """
    # ensure lists/arrays as-is (do NOT force .astype on lists)
    ori_data = ori_data
    generated_data = generated_data

    # Basic shapes - handle list or array
    # If ori_data is numpy array, infer dim from it; else from first element
    if isinstance(ori_data, np.ndarray):
        no, seq_len, dim = ori_data.shape
    else:
        no = len(ori_data)
        if no == 0:
            raise ValueError("ori_data is empty")
        dim = ori_data[0].shape[1]

    # Extract time info (fixed bug: use generated_data for generated_time)
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    if hidden_dim is None:
        hidden_dim = max(1, dim // 2)

    # Build model + optimizer
    disc = Discriminator(hidden_dim=hidden_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Prepare train/test splits using the provided utility
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # If train_x/test_x are numpy arrays of shape (N,T,D) we can use them directly.
    # Otherwise they are lists and we'll use our _sample_minibatch helper.

    # Training loop
    for itt in range(1, iterations + 1):
        # Sample real minibatch
        if isinstance(train_x, np.ndarray):
            # sample indices
            N = train_x.shape[0]
            idx = np.random.randint(0, N, size=batch_size)
            X_mb = train_x[idx]
            T_mb = np.array(train_t)[idx] if not isinstance(train_t, np.ndarray) else train_t[idx]
        else:
            X_mb, T_mb = _sample_minibatch(train_x, train_t, batch_size, max_len=max_seq_len)

        # Sample fake minibatch
        if isinstance(train_x_hat, np.ndarray):
            Nf = train_x_hat.shape[0]
            idxf = np.random.randint(0, Nf, size=batch_size)
            X_hat_mb = train_x_hat[idxf]
            T_hat_mb = np.array(train_t_hat)[idxf] if not isinstance(train_t_hat, np.ndarray) else train_t_hat[idxf]
        else:
            X_hat_mb, T_hat_mb = _sample_minibatch(train_x_hat, train_t_hat, batch_size, max_len=max_seq_len)

        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb = tf.convert_to_tensor(T_mb, dtype=tf.int32)
        X_hat_mb = tf.convert_to_tensor(X_hat_mb, dtype=tf.float32)
        T_hat_mb = tf.convert_to_tensor(T_hat_mb, dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits_real, _ = disc(X_mb, seq_lengths=T_mb, training=True)
            logits_fake, _ = disc(X_hat_mb, seq_lengths=T_hat_mb, training=True)

            labels_real = tf.ones_like(logits_real)
            labels_fake = tf.zeros_like(logits_fake)

            loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=labels_real))
            loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=labels_fake))
            loss = loss_real + loss_fake

        grads = tape.gradient(loss, disc.trainable_variables)
        optimizer.apply_gradients(zip(grads, disc.trainable_variables))

        if verbose and (itt == 1 or itt % (iterations//10) == 0):
            tf.print("Iteration", itt, "d_loss:", loss)

    # Prepare test sets (pad to max_seq_len)
    X_test = _pad_sequences(test_x, max_len=max_seq_len)
    X_hat_test = _pad_sequences(test_x_hat, max_len=max_seq_len)
    T_test = np.array(test_t, dtype=np.int32)
    T_hat_test = np.array(test_t_hat, dtype=np.int32)

    logits_real_test, probs_real_test = disc(tf.convert_to_tensor(X_test, dtype=tf.float32),
                                            seq_lengths=tf.convert_to_tensor(T_test, dtype=tf.int32),
                                            training=False)
    logits_fake_test, probs_fake_test = disc(tf.convert_to_tensor(X_hat_test, dtype=tf.float32),
                                            seq_lengths=tf.convert_to_tensor(T_hat_test, dtype=tf.int32),
                                            training=False)

    probs_real = probs_real_test.numpy().reshape(-1)
    probs_fake = probs_fake_test.numpy().reshape(-1)

    y_pred_final = np.concatenate([probs_real, probs_fake], axis=0)
    y_label_final = np.concatenate([np.ones(len(probs_real)), np.zeros(len(probs_fake))], axis=0)

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5).astype(int))
    discriminative_score = float(np.abs(0.5 - acc))
    return discriminative_score
