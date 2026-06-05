"""
ChronoGAN with zero-inflated time series support.

Key changes vs. original chronogan.py
======================================
1. ZIRecovery   — replaces Recovery.
     The final dense layer is split into two separate heads:
       gate_head  : Dense(dim, activation=None)   → gate logits  (raw, no activation)
       mag_head   : Dense(dim, activation='sigmoid') → magnitude ∈ (0,1)
     Forward returns (output, gate_prob, gate_logit, mag) where
       output = gate_prob * mag   (soft product, used during training)
     At generation time call ZIRecovery.sample_output(gate_prob, mag) for a
     hard Bernoulli sample that preserves the correct zero ratio.

2. Zero-inflated loss functions (TF equivalents of the ZITS PyTorch ones):
     zi_gate_loss               — BCE(gate_logit vs binary mask x>0)
     zi_recon_loss              — MSE restricted to non-zero timesteps only
     zi_temporal_consistency_loss — lag-1 autocorrelation mismatch on gate

3. Training loop
     - All places that previously called recovery() and then used the plain
       result now unpack (X_tilde, gate_prob, gate_logit, mag).
     - Every MSE term that was comparing X_tilde / X_hat directly to X_mb now
       ALSO adds gate_weight * zi_gate_loss + recon_weight * zi_recon_loss
       + tc_weight * zi_temporal_consistency_loss.
     - The variance / slope / skewness / median structure losses are computed
       on output (gate_prob * mag) exactly as before — those statistics are
       meaningful on the soft output.

4. Entry points follow the same call convention as ZITS:
     chronogan(ori_data, parameters, num_samples,
               gate_weight=5.0, recon_weight=10.0, tc_weight=1.0)
   so existing call sites need only add the new keyword arguments if desired.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from metrics_discriminative import discriminative_score_metrics
from metrics_predictive import predictive_score_metrics
from utils import extract_time, random_generator, batch_generator

tf.keras.backend.set_image_data_format('channels_last')


# ===========================================================================
# Unchanged RNN building block
# ===========================================================================

class RNNStack(tf.keras.layers.Layer):
    def __init__(self, module, hidden_dim, num_layers):
        super().__init__()
        cells = []
        for _ in range(num_layers):
            if module.lower() == 'gru':
                cells.append(tf.keras.layers.GRUCell(hidden_dim))
            else:
                cells.append(tf.keras.layers.LSTMCell(hidden_dim))
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(cells), return_sequences=True)

    def call(self, x, lengths, training=False):
        mask = tf.sequence_mask(lengths, maxlen=tf.shape(x)[1])
        return self.rnn(x, mask=mask, training=training)


# ===========================================================================
# Unchanged Embedder, Generator, Supervisor, AEDiscriminator
# ===========================================================================

class Embedder(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.e1    = RNNStack(m1, hidden_dim, num_layers)
        self.e2    = RNNStack(m2, hidden_dim, num_layers)
        self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, x, lengths, training=False):
        o1 = self.e1(x, lengths, training=training)
        o2 = self.e2(x, lengths, training=training)
        c  = tf.concat([o1, o2], axis=-1)
        return self.dense(c)


class Generator(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2, z_dim):
        super().__init__()
        self.g1    = RNNStack(m1, hidden_dim, num_layers)
        self.g2    = RNNStack(m2, hidden_dim, num_layers)
        self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, z, lengths, training=False):
        o1 = self.g1(z, lengths, training=training)
        o2 = self.g2(z, lengths, training=training)
        c  = tf.concat([o1, o2], axis=-1)
        return self.dense(c)


class Supervisor(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.s1    = RNNStack(m1, hidden_dim, num_layers)
        self.s2    = RNNStack(m2, hidden_dim, num_layers)
        self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, h, lengths, training=False):
        o1 = self.s1(h, lengths, training=training)
        o2 = self.s2(h, lengths, training=training)
        c  = tf.concat([o1, o2], axis=-1)
        return self.dense(c)


class AEDiscriminator(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.d1  = RNNStack(m1, hidden_dim, num_layers)
        self.d2  = RNNStack(m2, hidden_dim, num_layers)
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, x, lengths, training=False):
        o1 = self.d1(x, lengths, training=training)
        o2 = self.d2(x, lengths, training=training)
        c  = tf.concat([o1, o2], axis=-1)
        return self.out(c)


# ===========================================================================
# Zero-inflated Recovery  (replaces the plain Recovery)
# ===========================================================================

class ZIRecovery(tf.keras.Model):
    """
    Decoder / Recovery with a dual gate + magnitude head.

    Input : latent h  (B, T, latent_dim)
    Output: (output, gate_prob, gate_logit, mag)
      gate_logit  (B, T, dim)  raw logits for Bernoulli gate P(non-zero)
      gate_prob   (B, T, dim)  sigmoid(gate_logit)
      mag         (B, T, dim)  predicted magnitude ∈ (0, 1)
      output      (B, T, dim)  gate_prob * mag  — used during training

    At generation time call:
      ZIRecovery.sample_output(gate_prob, mag)
    which draws a hard Bernoulli mask and multiplies by mag, preserving the
    correct empirical zero ratio stochastically.
    """
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.r1        = RNNStack(m1, hidden_dim, num_layers)
        self.r2        = RNNStack(m2, hidden_dim, num_layers)
        # Two fully-independent projection heads — no shared weights
        self.gate_head = tf.keras.layers.Dense(dim, activation=None)        # raw logits
        self.mag_head  = tf.keras.layers.Dense(dim, activation='sigmoid')   # magnitude

    def call(self, h, lengths, training=False):
        o1 = self.r1(h, lengths, training=training)
        o2 = self.r2(h, lengths, training=training)
        c  = tf.concat([o1, o2], axis=-1)

        gate_logit = self.gate_head(c)                    # (B, T, dim)
        gate_prob  = tf.sigmoid(gate_logit)               # P(non-zero)
        mag        = self.mag_head(c)                     # magnitude ∈ (0, 1)
        output     = gate_prob * mag                      # soft product
        return output, gate_prob, gate_logit, mag

    @staticmethod
    def sample_output(gate_prob, mag):
        """
        Hard Bernoulli sample — use at generation time only.
        Draws from Bernoulli(gate_prob) then multiplies by mag.
        """
        binary = tf.cast(
            tf.random.stateless_binomial(
                shape=tf.shape(gate_prob),
                seed=(0, 0),
                counts=tf.ones_like(gate_prob),
                probs=gate_prob),
            tf.float32)
        return binary * mag


# ===========================================================================
# Zero-inflated loss functions  (TF equivalents of main_zits.py)
# ===========================================================================

def zi_gate_loss(gate_logit, x):
    """
    BCE loss for the explicit Bernoulli gate.
    Target = 1 where x > 0 (non-zero), 0 where x == 0.
    """
    target = tf.cast(x > 0.0, tf.float32)
    bce    = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(target, gate_logit)


def zi_recon_loss(output, x):
    """
    MSE restricted to non-zero timesteps so zeros don't dilute the magnitude.
    Returns 0 if the batch has no non-zero entries.
    """
    mask = tf.cast(x > 0.0, tf.float32)                  # (B, T, dim)
    n    = tf.reduce_sum(mask)
    if n == 0:
        return tf.constant(0.0)
    sq   = tf.reduce_sum(mask * tf.square(output - x))
    return sq / (n + 1e-12)


def zi_temporal_consistency_loss(gate_prob, x):
    """
    Lag-1 autocorrelation loss on the zero/non-zero mask.
    Penalises mismatch between predicted gate transitions and real ones.

        loss = ( E[gate_t * gate_{t+1}] - E[mask_t * mask_{t+1}] )^2
    """
    real_mask = tf.cast(x > 0.0, tf.float32)
    real_lag1 = tf.reduce_mean(real_mask[:, :-1, :] * real_mask[:, 1:, :])
    pred_lag1 = tf.reduce_mean(gate_prob[:, :-1, :] * gate_prob[:, 1:, :])
    return tf.square(pred_lag1 - real_lag1)


def zi_combined_loss(output, gate_prob, gate_logit, x,
                     gate_weight=5.0, recon_weight=10.0, tc_weight=1.0):
    """
    Composite ZI loss:  recon_weight * recon + gate_weight * gate + tc_weight * tc
    Returns (total, r, g, tc) for logging.
    """
    r  = zi_recon_loss(output, x)
    g  = zi_gate_loss(gate_logit, x)
    tc = zi_temporal_consistency_loss(gate_prob, x)
    total = recon_weight * r + gate_weight * g + tc_weight * tc
    return total, r, g, tc


# ===========================================================================
# Main training function
# ===========================================================================

def chronogan(ori_data, parameters, num_samples,
              gate_weight: float = 5.0,
              recon_weight: float = 10.0,
              tc_weight: float = 1.0):
    """
    Zero-inflated ChronoGAN.

    Args:
        ori_data    : np.ndarray, shape (N, T, dim)
        parameters  : dict with keys hidden_dim, num_layer, iterations, batch_size
        num_samples : int or "same"
        gate_weight : weight for the Bernoulli gate BCE loss
        recon_weight: weight for the non-zero MSE loss
        tc_weight   : weight for the lag-1 temporal consistency loss

    Returns:
        generated data as np.ndarray (num_samples, T, dim)
    """
    tf.keras.backend.clear_session()
    np.random.seed(0)
    tf.random.set_seed(0)

    ori_data = np.asarray(ori_data)
    no, seq_len, dim = ori_data.shape
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val  = np.min(np.min(data, axis=0), axis=0)
        data     = data - min_val
        max_val  = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    hidden_dim = dim if parameters['hidden_dim'] == 'same' else parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim      = dim
    gamma      = 1.0
    beta       = 1.0
    m1         = 'gru'
    m2         = 'lstm'

    embedder   = Embedder(dim, hidden_dim, num_layers, m1, m2)
    recovery   = ZIRecovery(dim, hidden_dim, num_layers, m1, m2)   # ← ZI
    generator  = Generator(dim, hidden_dim, num_layers, m1, m2, z_dim)
    supervisor = Supervisor(dim, hidden_dim, num_layers, m1, m2)
    ae_disc    = AEDiscriminator(dim, hidden_dim, num_layers, m1, m2)

    E0_optimizer          = tf.keras.optimizers.Adam()
    E_optimizer           = tf.keras.optimizers.Adam()
    D_ae_optimizer        = tf.keras.optimizers.Adam()
    D_ae_second_optimizer = tf.keras.optimizers.Adam()
    G_optimizer           = tf.keras.optimizers.Adam()
    GS_optimizer          = tf.keras.optimizers.Adam()

    bce      = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse_loss = tf.keras.losses.MeanSquaredError()

    final_generated = []
    global_summing  = 10.0
    p1 = None
    p2 = None

    def safe_np(arr):
        a = np.asarray(arr)
        if a.size == 0:
            return None
        return a

    # -----------------------------------------------------------------------
    # Phase 1 — Autoencoder pre-training  (iterations * 0.5 steps)
    # -----------------------------------------------------------------------
    for itt in range(int(iterations * 0.5)):
        for _ in range(2):
            X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb_np = safe_np(X_mb_np)
            if X_mb_np is None:
                continue
            X_mb    = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
            T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb,    dtype=np.int32))

            with tf.GradientTape() as tape:
                H = embedder(X_mb, T_mb_tf, training=True)
                # ZI: unpack all four outputs from recovery
                X_tilde, gate_prob_tilde, gate_logit_tilde, mag_tilde = recovery(
                    H, T_mb_tf, training=True)

                Y_ae_fake = ae_disc(X_tilde, T_mb_tf, training=True)

                # Standard reconstruction term (on the soft output gate_prob*mag)
                E_loss_T00 = mse_loss(X_mb, X_tilde)
                # ZI additional terms
                zi_total, zi_r, zi_g, zi_tc = zi_combined_loss(
                    X_tilde, gate_prob_tilde, gate_logit_tilde, X_mb,
                    gate_weight, recon_weight, tc_weight)

                E_loss_U = bce(tf.ones_like(Y_ae_fake), Y_ae_fake)
                E_loss0  = (10.0 * tf.sqrt(tf.maximum(E_loss_T00 + 0.001 * E_loss_U, 1e-12))
                            + zi_total)

            vars_e = embedder.trainable_variables + recovery.trainable_variables
            grads  = tape.gradient(E_loss0, vars_e)
            grads  = [g if g is not None else tf.zeros_like(v)
                      for g, v in zip(grads, vars_e)]
            E0_optimizer.apply_gradients(zip(grads, vars_e))

        X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_np = safe_np(X_mb_np)
        step_d_ae_loss = 0.0
        if X_mb_np is not None:
            X_mb    = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
            T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb,    dtype=np.int32))
            with tf.GradientTape() as tape:
                Y_ae_real     = ae_disc(X_mb, T_mb_tf, training=True)
                D_ae_loss_real = bce(tf.ones_like(Y_ae_real), Y_ae_real)
            vars_d = ae_disc.trainable_variables
            grads  = tape.gradient(D_ae_loss_real, vars_d)
            if any([g is not None for g in grads]):
                grads = [g if g is not None else tf.zeros_like(v)
                         for g, v in zip(grads, vars_d)]
                D_ae_optimizer.apply_gradients(zip(grads, vars_d))
                step_d_ae_loss = float(D_ae_loss_real.numpy())

        log_interval = int(iterations * 0.5) // 10 if iterations * 0.5 >= 10 else 1
        if (itt % log_interval == 0) or (itt == int(iterations * 0.5) - 1):
            try:
                e0_val = float(E_loss0.numpy())
            except:
                e0_val = 0.0
            print(f'step: {itt*2}/{iterations}, '
                  f'AE_loss: {np.round(e0_val,4)}, '
                  f'AE_D_loss: {np.round(step_d_ae_loss,4)}')

    # -----------------------------------------------------------------------
    # Phase 2 — Supervisor pre-training  (iterations steps)
    # -----------------------------------------------------------------------
    for itt in range(iterations):
        X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_np = safe_np(X_mb_np)
        if X_mb_np is None:
            continue
        X_mb    = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
        T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb,    dtype=np.int32))
        Z_mb_np = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb    = tf.convert_to_tensor(np.asarray(Z_mb_np, dtype=np.float32))

        with tf.GradientTape() as tape:
            E_hat            = generator(Z_mb, T_mb_tf, training=True)
            H_hat            = supervisor(E_hat, T_mb_tf, training=True)
            H                = embedder(X_mb, T_mb_tf, training=True)
            H_hat_supervise  = supervisor(H, T_mb_tf, training=True)
            G_loss_S         = mse_loss(H[:, 2:, :], H_hat_supervise[:, :-2, :])

        vars_gs = generator.trainable_variables + supervisor.trainable_variables
        grads   = tape.gradient(G_loss_S, vars_gs)
        grads   = [g if g is not None else tf.zeros_like(v)
                   for g, v in zip(grads, vars_gs)]
        GS_optimizer.apply_gradients(zip(grads, vars_gs))

        log_interval = int(iterations) // 10 if iterations >= 10 else 1
        if (itt % log_interval == 0) or (itt == iterations - 1):
            print(f'step: {itt}/{iterations}, '
                  f'S_loss: {np.round(float(G_loss_S.numpy()),4)}')

    # -----------------------------------------------------------------------
    # Phase 3 — Joint GAN training  (iterations steps)
    # -----------------------------------------------------------------------
    for itt in range(iterations):
        for _ in range(2):
            X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb_np = safe_np(X_mb_np)
            if X_mb_np is None:
                continue
            X_mb    = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
            T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb,    dtype=np.int32))
            Z_mb_np = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb    = tf.convert_to_tensor(np.asarray(Z_mb_np, dtype=np.float32))

            with tf.GradientTape(persistent=True) as tape:
                E_hat   = generator(Z_mb, T_mb_tf, training=True)
                H_hat   = supervisor(E_hat, T_mb_tf, training=True)
                H       = embedder(X_mb, T_mb_tf, training=True)
                H_hat_supervise = supervisor(H, T_mb_tf, training=True)

                # ZI: unpack recovery outputs — H_hat path (supervisor latent)
                X_hat, gate_prob_hat, gate_logit_hat, mag_hat = recovery(
                    H_hat, T_mb_tf, training=True)
                # ZI: unpack recovery outputs — E_hat path (raw generator latent)
                X_tilde_fake_second, gate_prob_second, gate_logit_second, mag_second = recovery(
                    E_hat, T_mb_tf, training=True)

                Y_ae_fake_e        = ae_disc(X_hat, T_mb_tf, training=True)
                Y_ae_fake_e_second = ae_disc(X_tilde_fake_second, T_mb_tf, training=True)

                G_loss_U_ae   = bce(tf.ones_like(Y_ae_fake_e), Y_ae_fake_e)
                G_loss_U_ae_e = bce(tf.ones_like(Y_ae_fake_e_second), Y_ae_fake_e_second)
                G_loss_S      = mse_loss(H[:, 2:, :], H_hat_supervise[:, :-2, :])

                G_loss_V1 = tf.reduce_mean(
                    tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6)
                           - tf.sqrt(tf.nn.moments(X_mb, [0])[1] + 1e-6)))
                G_loss_V2 = tf.reduce_mean(
                    tf.abs(tf.nn.moments(X_hat, [0])[0] - tf.nn.moments(X_mb, [0])[0]))
                G_loss_V  = G_loss_V1 + G_loss_V2

                # ---- ZI losses on both recovery paths ----
                zi_hat_total, _, _, _ = zi_combined_loss(
                    X_hat, gate_prob_hat, gate_logit_hat, X_mb,
                    gate_weight, recon_weight, tc_weight)
                zi_second_total, _, _, _ = zi_combined_loss(
                    X_tilde_fake_second, gate_prob_second, gate_logit_second, X_mb,
                    gate_weight, recon_weight, tc_weight)
                zi_gen_loss = zi_hat_total + zi_second_total

                # ---- Temporal structure losses (unchanged, computed on soft output) ----
                b = tf.shape(X_mb)[0]
                W = tf.range(1, seq_len + 1, dtype=tf.float32)
                W = tf.reshape(W, (1, seq_len, 1))
                W = tf.broadcast_to(W, (b, seq_len, dim))
                W_sum  = tf.reduce_sum(W, axis=1, keepdims=True)
                W_norm = W / W_sum
                wa_X     = tf.reduce_sum(W_norm * X_mb,  axis=1)
                wa_X_hat = tf.reduce_sum(W_norm * X_hat, axis=1)
                mean_wa_mse = mse_loss(tf.reduce_mean(wa_X, axis=0),
                                       tf.reduce_mean(wa_X_hat, axis=0))
                std_wa_mse  = mse_loss(tf.math.reduce_std(wa_X, axis=0),
                                       tf.math.reduce_std(wa_X_hat, axis=0))

                x_t    = tf.range(seq_len, dtype=tf.float32)
                sum_x  = tf.reduce_sum(x_t)
                sum_x2 = tf.reduce_sum(tf.square(x_t))
                N      = tf.cast(seq_len, tf.float32)

                def calculate_slope(Y):
                    sum_y  = tf.reduce_sum(Y, axis=1)
                    sum_xy = tf.reduce_sum(tf.expand_dims(x_t, 1) * Y, axis=1)
                    num    = N * sum_xy - sum_x * sum_y
                    den    = N * sum_x2 - tf.square(sum_x)
                    return num / (den + 1e-12)

                slope_X     = calculate_slope(X_mb)
                slope_X_hat = calculate_slope(X_hat)
                mean_slope_mse = mse_loss(tf.reduce_mean(slope_X, axis=0),
                                          tf.reduce_mean(slope_X_hat, axis=0))
                std_slope_mse  = mse_loss(tf.math.reduce_std(slope_X, axis=0),
                                          tf.math.reduce_std(slope_X_hat, axis=0))

                def calculate_skewness(data, axis=1):
                    Nn   = tf.cast(tf.shape(data)[axis], tf.float32)
                    mean = tf.reduce_mean(data, axis=axis, keepdims=True)
                    std  = tf.math.reduce_std(data, axis=axis, keepdims=True)
                    return (tf.reduce_sum(((data - mean) / (std + 1e-12))**3, axis=axis)
                            * (Nn / ((Nn - 1) * (Nn - 2) + 1e-12)))

                skew_X     = calculate_skewness(X_mb,  axis=1)
                skew_X_hat = calculate_skewness(X_hat, axis=1)
                mean_skew_mse = mse_loss(tf.reduce_mean(skew_X, axis=0),
                                         tf.reduce_mean(skew_X_hat, axis=0))
                std_skew_mse  = mse_loss(tf.math.reduce_std(skew_X, axis=0),
                                         tf.math.reduce_std(skew_X_hat, axis=0))

                time_size = tf.shape(X_mb)[1]

                def median_tensor(data):
                    ts  = tf.cast(time_size, tf.int32)
                    mid = ts // 2
                    def odd():  return data[:, mid, :]
                    def even(): return (data[:, (mid-1), :] + data[:, mid, :]) / 2.0
                    return tf.cond(tf.equal(ts % 2, 1), odd, even)

                median_X     = median_tensor(X_mb)
                median_X_hat = median_tensor(X_hat)
                mean_median_mse = mse_loss(tf.reduce_mean(median_X, axis=0),
                                           tf.reduce_mean(median_X_hat, axis=0))
                std_median_mse  = mse_loss(tf.math.reduce_std(median_X, axis=0),
                                           tf.math.reduce_std(median_X_hat, axis=0))

                ts_structure = (mean_wa_mse + std_wa_mse
                                + mean_slope_mse + std_slope_mse
                                + 0.5 * mean_median_mse + 0.5 * std_median_mse
                                + 0.5 * mean_skew_mse   + 0.5 * std_skew_mse)

                G_loss = ((G_loss_U_ae + gamma * G_loss_U_ae_e)
                          + 100.0 * tf.sqrt(tf.maximum(G_loss_S, 1e-12))
                          + 100.0 * G_loss_V
                          + 25.0  * ts_structure
                          + zi_gen_loss)               # ← ZI term added

            vars_g = generator.trainable_variables + supervisor.trainable_variables
            grads  = tape.gradient(G_loss, vars_g)
            grads  = [g if g is not None else tf.zeros_like(v)
                      for g, v in zip(grads, vars_g)]
            G_optimizer.apply_gradients(zip(grads, vars_g))

            # Embedder / Recovery update
            with tf.GradientTape() as tape2:
                H = embedder(X_mb, T_mb_tf, training=True)
                X_tilde_ae, gate_prob_ae, gate_logit_ae, mag_ae = recovery(
                    H, T_mb_tf, training=True)
                Y_ae_fake_ae = ae_disc(X_tilde_ae, T_mb_tf, training=True)
                E_loss_T00   = mse_loss(X_mb, X_tilde_ae)
                E_loss_U_ae  = bce(tf.ones_like(Y_ae_fake_ae), Y_ae_fake_ae)

                # ZI on the embedder→recovery reconstruction
                zi_ae_total, _, _, _ = zi_combined_loss(
                    X_tilde_ae, gate_prob_ae, gate_logit_ae, X_mb,
                    gate_weight, recon_weight, tc_weight)

                E_loss = (10.0 * tf.sqrt(
                              tf.maximum(E_loss_T00 + 0.001 * 0.1 * E_loss_U_ae, 1e-12))
                          + 0.1 * G_loss_S
                          + zi_ae_total)               # ← ZI term added

            vars_e = embedder.trainable_variables + recovery.trainable_variables
            grads  = tape2.gradient(E_loss, vars_e)
            grads  = [g if g is not None else tf.zeros_like(v)
                      for g, v in zip(grads, vars_e)]
            E_optimizer.apply_gradients(zip(grads, vars_e))

        # -- Discriminator update (uses soft output X_hat = gate_prob * mag) --
        X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_np = safe_np(X_mb_np)
        if X_mb_np is None:
            continue
        X_mb    = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
        T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb,    dtype=np.int32))
        Z_mb_np = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb    = tf.convert_to_tensor(np.asarray(Z_mb_np, dtype=np.float32))

        E_hat   = generator(Z_mb, T_mb_tf, training=False)
        H_hat   = supervisor(E_hat, T_mb_tf, training=False)
        X_hat_d, _, _, _          = recovery(H_hat, T_mb_tf, training=False)
        X_tilde_s, _, _, _        = recovery(E_hat, T_mb_tf, training=False)

        Y_ae_fake          = ae_disc(X_hat_d,   T_mb_tf, training=False)
        Y_ae_fake_e_second = ae_disc(X_tilde_s, T_mb_tf, training=False)
        Y_ae_real_disc     = ae_disc(X_mb,       T_mb_tf, training=False)

        D_ae_loss_real         = bce(tf.ones_like(Y_ae_real_disc),  Y_ae_real_disc)
        D_ae_loss_fake         = bce(tf.zeros_like(Y_ae_fake),       Y_ae_fake)
        D_ae_loss_fake_e_second= bce(tf.zeros_like(Y_ae_fake_e_second), Y_ae_fake_e_second)
        D_ae_loss_real_second  = bce(tf.ones_like(Y_ae_fake), Y_ae_fake)

        D_ae_loss_second = (D_ae_loss_real + D_ae_loss_real_second
                            + beta * (D_ae_loss_fake + gamma * D_ae_loss_fake_e_second))

        step_d_ae_loss_second = 0.0
        if float(D_ae_loss_second.numpy()) > 0.15:
            vars_d = ae_disc.trainable_variables
            with tf.GradientTape() as tape3:
                Y_real   = ae_disc(X_mb, T_mb_tf, training=True)
                D_loss2  = D_ae_loss_second
            grads = tape3.gradient(D_loss2, vars_d)
            if any([g is not None for g in grads]):
                grads = [g if g is not None else tf.zeros_like(v)
                         for g, v in zip(grads, vars_d)]
                D_ae_second_optimizer.apply_gradients(zip(grads, vars_d))
                step_d_ae_loss_second = float(D_loss2.numpy())

        log_interval = int(iterations) // 10 if iterations >= 10 else 1
        if (itt % log_interval == 0) or (itt == iterations - 1):
            print(f'step: {itt}/{iterations}, '
                  f'D_loss: {np.round(step_d_ae_loss_second,4)}')

        # ---- Checkpoint: evaluate and keep best generated data ----
        if (itt >= int(iterations * 0.5)) and (itt % 500 == 0 or itt == iterations - 1):
            Z_mb_eval   = random_generator(no, z_dim, ori_time, max_seq_len)
            Z_mb_eval_t = tf.convert_to_tensor(np.asarray(Z_mb_eval, dtype=np.float32))
            ori_time_t  = tf.convert_to_tensor(np.asarray(ori_time,  dtype=np.int32))

            E_hat_eval = generator(Z_mb_eval_t, ori_time_t)
            H_hat_eval = supervisor(E_hat_eval, ori_time_t)
            # ZI: use hard Bernoulli sampling at generation checkpoints
            _, gate_prob_eval, _, mag_eval = recovery(H_hat_eval, ori_time_t)
            gen_raw = ZIRecovery.sample_output(gate_prob_eval, mag_eval).numpy()

            generated_data = []
            for i in range(no):
                temp = gen_raw[i, :ori_time[i], :]
                generated_data.append(temp)
            generated_data = np.array(generated_data)
            generated_data = generated_data * max_val + min_val

            final_generated = np.copy(generated_data)

            zero_ratio = np.mean(generated_data == 0)
            print(f"Generated data: {generated_data.shape}  "
                  f"zero_ratio={zero_ratio:.2%}")

            metric_iteration = 3
            print("Computing discriminative score...")
            discriminative_score = []
            for _ in range(metric_iteration):
                temp_disc = discriminative_score_metrics(
                    ori_data, generated_data, iterations=iterations, batch_size=batch_size)
                discriminative_score.append(temp_disc)
            discriminative_score = np.array(discriminative_score)
            filtered_disc = (discriminative_score[
                discriminative_score <= np.percentile(discriminative_score, 75)]
                if discriminative_score.size > 0 else np.array([]))

            print("Computing predictive score...")
            predictive_score = []
            for tt in range(metric_iteration):
                temp_pred = predictive_score_metrics(
                    ori_data, generated_data, iterations=iterations, batch_size=batch_size)
                predictive_score.append(temp_pred)
            predictive_score = np.array(predictive_score)
            filtered_pred = (predictive_score[
                predictive_score <= np.percentile(predictive_score, 75)]
                if predictive_score.size > 0 else np.array([]))

            mean_real      = np.mean(ori_data,       axis=0)
            mean_synthetic = np.mean(generated_data, axis=0)
            mse_mean       = np.mean((mean_real - mean_synthetic) ** 2)
            var_real       = np.var(ori_data,       axis=0)
            var_synthetic  = np.var(generated_data, axis=0)
            mse_variance   = np.mean((var_real - var_synthetic) ** 2)

            mean_dis_score = np.round(np.min(filtered_disc), 4) if filtered_disc.size > 0 else 0.0
            mean_pre_score = np.round(np.min(filtered_pred), 4) if filtered_pred.size > 0 else 0.0

            if p1 is None and p2 is None:
                if mean_dis_score == 0:
                    p1, p2 = 1.0, 1.0
                elif mean_pre_score == 0:
                    p1 = 1.0
                    p2 = mean_dis_score / (mse_mean + mse_variance + 1e-12)
                else:
                    p1 = mean_dis_score / (mean_pre_score + 1e-12)
                    p2 = mean_dis_score / (mse_mean + mse_variance + 1e-12)

            summing = mean_dis_score + p1 * mean_pre_score + p2 * (mse_mean + mse_variance)
            if summing <= global_summing:
                global_summing  = summing
                final_generated = generated_data
                print(f"Best so far: {final_generated.shape}  summing={summing:.6f}")

    # -----------------------------------------------------------------------
    # Generate final output
    # -----------------------------------------------------------------------
    if num_samples == "same":
        print(f"Final data returned: {final_generated.shape}")
        return final_generated

    count = int(num_samples / no)
    all_generated = []
    for _ in range(count):
        Z_mb_fin   = random_generator(no, z_dim, ori_time, max_seq_len)
        Z_mb_fin_t = tf.convert_to_tensor(np.asarray(Z_mb_fin, dtype=np.float32))
        ori_time_t = tf.convert_to_tensor(np.asarray(ori_time,  dtype=np.int32))

        E_hat_fin = generator(Z_mb_fin_t, ori_time_t)
        H_hat_fin = supervisor(E_hat_fin, ori_time_t)
        _, gate_prob_fin, _, mag_fin = recovery(H_hat_fin, ori_time_t)
        gen_raw_fin = ZIRecovery.sample_output(gate_prob_fin, mag_fin).numpy()

        generated = []
        for i in range(no):
            temp = gen_raw_fin[i, :ori_time[i], :]
            generated.append(temp)
        generated = np.array(generated)
        generated = generated * max_val + min_val
        all_generated.append(generated)

    all_generated = np.concatenate(all_generated, axis=0)
    return all_generated


# ===========================================================================
# Pipeline entry points
# (same structure as main_zits.py: main_train_chronogan / main_test_chronogan)
# ===========================================================================

import os
import numpy as np

from constants import OUT_FOLDER
from data_proc import (DataPreprocessor, CountDataPreprocessor,
                       load_iot_data, load_m5_data)
from utils import plot_sample_comparisons


# ---------------------------------------------------------------------------
# Generation helper  (mirrors _generate_and_save from main_zits.py)
# ---------------------------------------------------------------------------

def _chronogan_generate_and_save(generator, supervisor, recovery,
                                 ori_time, max_seq_len, z_dim, no,
                                 min_val, max_val, preprocessor,
                                 prefix, num_synthetic, ori_data):
    """
    Draw num_synthetic samples using hard Bernoulli gate sampling,
    inverse-transform, save .npz and comparison plot.
    """
    count = int(num_synthetic / no) if num_synthetic != "same" else 1
    all_generated = []

    for _ in range(count):
        Z_mb   = random_generator(no, z_dim, ori_time, max_seq_len)
        Z_mb_t = tf.convert_to_tensor(np.asarray(Z_mb, dtype=np.float32))
        ot_t   = tf.convert_to_tensor(np.asarray(ori_time, dtype=np.int32))

        E_hat = generator(Z_mb_t, ot_t)
        H_hat = supervisor(E_hat, ot_t)
        _, gate_prob_g, _, mag_g = recovery(H_hat, ot_t)
        gen_raw = ZIRecovery.sample_output(gate_prob_g, mag_g).numpy()

        generated = []
        for i in range(no):
            generated.append(gen_raw[i, :ori_time[i], :])
        generated = np.array(generated)
        generated = generated * max_val + min_val
        all_generated.append(generated)

    gen_data = np.concatenate(all_generated, axis=0)[:num_synthetic]
    # ChronoGAN output is (N, T, dim) — squeeze to (N, T) for 1-d series
    if gen_data.ndim == 3 and gen_data.shape[-1] == 1:
        gen_data = gen_data.squeeze(-1)

    gen_data = preprocessor.inverse_transform(gen_data)

    np.savez(os.path.join(OUT_FOLDER, f'{prefix}_generated_data.npz'), data=gen_data)
    plot_sample_comparisons(
        ori_data[:5], gen_data[:5],
        save_path=os.path.join(OUT_FOLDER, f'{prefix}_sample_comparison.png'))

    nz = gen_data[gen_data > 0]
    print(f"\nGenerated data stats:")
    print(f"  Zero ratio:      {np.mean(gen_data == 0):.2%}")
    print(f"  Max:             {np.max(gen_data):.4f}")
    if len(nz):
        print(f"  Mean (non-zero): {nz.mean():.4f}")
    return gen_data


# ===========================================================================
# ChronoGAN entry points
# ===========================================================================

def main_train_chronogan(data, ori_data: np.ndarray,
                         hidden_dim='same', num_layer=3,
                         iterations=10000, batch_size=128,
                         gate_weight=5.0, recon_weight=10.0, tc_weight=1.0):
    """
    Preprocess, train, and save a zero-inflated ChronoGAN.

    Args:
        data        : "iot" or "m5"
        ori_data    : raw numpy array (N, T) or (N, T, dim)
        hidden_dim  : RNN hidden size, or 'same' to match feature dim
        num_layer   : number of stacked RNN layers
        iterations  : total training iterations
        batch_size  : mini-batch size
        gate_weight : weight for Bernoulli gate BCE loss
        recon_weight: weight for non-zero MSE loss
        tc_weight   : weight for lag-1 temporal consistency loss
    """
    if data == "iot":
        pp = DataPreprocessor()
    elif data == "m5":
        pp = CountDataPreprocessor()

    proc = pp.fit_transform(ori_data)
    # ChronoGAN expects (N, T, dim) — add feature dim if needed
    if proc.ndim == 2:
        proc = proc[:, :, np.newaxis]

    parameters = {
        'hidden_dim': hidden_dim,
        'num_layer':  num_layer,
        'iterations': iterations,
        'batch_size': batch_size,
    }

    print(f"\nInitialising ZITS-ChronoGAN ...")
    print(f"  Data shape:   {proc.shape}")
    print(f"  hidden_dim:   {hidden_dim}  num_layer: {num_layer}")
    print(f"  iterations:   {iterations}  batch_size: {batch_size}")
    print(f"  gate_weight:  {gate_weight}  recon_weight: {recon_weight}  tc_weight: {tc_weight}")

    # chronogan() returns the best generated data internally; we don't use it here
    chronogan(proc, parameters, num_samples="same",
              gate_weight=gate_weight, recon_weight=recon_weight, tc_weight=tc_weight)

    # Save preprocessor so main_test_chronogan can reload it
    pp.save(os.path.join(OUT_FOLDER, 'zits_chronogan_preprocessor.json'))
    # Save parameters for reload
    import json
    with open(os.path.join(OUT_FOLDER, 'zits_chronogan_params.json'), 'w') as f:
        json.dump({'hidden_dim': hidden_dim, 'num_layer': num_layer,
                   'iterations': iterations, 'batch_size': batch_size,
                   'gate_weight': gate_weight, 'recon_weight': recon_weight,
                   'tc_weight': tc_weight}, f)

    print(f"\nZITS-ChronoGAN training complete. Files saved to: {OUT_FOLDER}")


def main_test_chronogan(data, ori_data: np.ndarray, num_synthetic: int = 1000):
    """
    Reload a trained ChronoGAN and generate num_synthetic samples.

    Note: ChronoGAN is a TensorFlow model with no standard torch checkpoint.
    The model must be re-trained (or kept in memory) before calling this.
    This function re-runs training from the saved parameters then generates,
    matching ChronoGAN's original stateless design.
    """
    import json

    if data == "iot":
        pp = DataPreprocessor()
    elif data == "m5":
        pp = CountDataPreprocessor()
    pp.load(os.path.join(OUT_FOLDER, 'zits_chronogan_preprocessor.json'))

    with open(os.path.join(OUT_FOLDER, 'zits_chronogan_params.json')) as f:
        params = json.load(f)

    proc = pp.fit_transform(ori_data)
    if proc.ndim == 2:
        proc = proc[:, :, np.newaxis]

    parameters = {
        'hidden_dim': params['hidden_dim'],
        'num_layer':  params['num_layer'],
        'iterations': params['iterations'],
        'batch_size': params['batch_size'],
    }

    print(f"\nGenerating {num_synthetic} synthetic samples ...")
    gen_data = chronogan(proc, parameters, num_samples=num_synthetic,
                         gate_weight=params['gate_weight'],
                         recon_weight=params['recon_weight'],
                         tc_weight=params['tc_weight'])

    if gen_data.ndim == 3 and gen_data.shape[-1] == 1:
        gen_data = gen_data.squeeze(-1)

    gen_data_inv = pp.inverse_transform(gen_data)

    np.savez(os.path.join(OUT_FOLDER, 'zits_chronogan_generated_data.npz'), data=gen_data_inv)
    plot_sample_comparisons(
        ori_data[:5], gen_data_inv[:5],
        save_path=os.path.join(OUT_FOLDER, 'zits_chronogan_sample_comparison.png'))

    nz = gen_data_inv[gen_data_inv > 0]
    print(f"\nGenerated data stats:")
    print(f"  Zero ratio:      {np.mean(gen_data_inv == 0):.2%}")
    print(f"  Max:             {np.max(gen_data_inv):.4f}")
    if len(nz):
        print(f"  Mean (non-zero): {nz.mean():.4f}")
    print("ZITS-ChronoGAN testing complete.")


# ===========================================================================

if __name__ == "__main__":
    ori_data = load_m5_data()
    main_train_chronogan("m5", ori_data, hidden_dim='same', num_layer=3,
                         iterations=100, batch_size=128,
                         gate_weight=10.0, recon_weight=10.0, tc_weight=0.5)
    main_test_chronogan("m5", ori_data, num_synthetic=30000)

    ori_data = load_iot_data()
    main_train_chronogan("iot", ori_data, hidden_dim='same', num_layer=3,
                         iterations=100, batch_size=128,
                         gate_weight=10.0, recon_weight=10.0, tc_weight=0.5)
    main_test_chronogan("iot", ori_data, num_synthetic=50000)