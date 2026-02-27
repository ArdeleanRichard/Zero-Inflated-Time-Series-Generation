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

class RNNStack(tf.keras.layers.Layer):
    def __init__(self, module, hidden_dim, num_layers):
        super().__init__()
        cells = []
        for _ in range(num_layers):
            if module.lower() == 'gru':
                cells.append(tf.keras.layers.GRUCell(hidden_dim))
            else:
                cells.append(tf.keras.layers.LSTMCell(hidden_dim))
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(cells), return_sequences=True)

    def call(self, x, lengths, training=False):
        mask = tf.sequence_mask(lengths, maxlen=tf.shape(x)[1])
        return self.rnn(x, mask=mask, training=training)

class Embedder(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.e1 = RNNStack(m1, hidden_dim, num_layers)
        self.e2 = RNNStack(m2, hidden_dim, num_layers)
        self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, x, lengths, training=False):
        o1 = self.e1(x, lengths, training=training)
        o2 = self.e2(x, lengths, training=training)
        c = tf.concat([o1, o2], axis=-1)
        return self.dense(c)

class Recovery(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.r1 = RNNStack(m1, hidden_dim, num_layers)
        self.r2 = RNNStack(m2, hidden_dim, num_layers)
        self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, h, lengths, training=False):
        o1 = self.r1(h, lengths, training=training)
        o2 = self.r2(h, lengths, training=training)
        c = tf.concat([o1, o2], axis=-1)
        return self.dense(c)

class Generator(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2, z_dim):
        super().__init__()
        self.g1 = RNNStack(m1, hidden_dim, num_layers)
        self.g2 = RNNStack(m2, hidden_dim, num_layers)
        self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, z, lengths, training=False):
        o1 = self.g1(z, lengths, training=training)
        o2 = self.g2(z, lengths, training=training)
        c = tf.concat([o1, o2], axis=-1)
        return self.dense(c)

class Supervisor(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.s1 = RNNStack(m1, hidden_dim, num_layers)
        self.s2 = RNNStack(m2, hidden_dim, num_layers)
        self.dense = tf.keras.layers.Dense(dim, activation='sigmoid')

    def call(self, h, lengths, training=False):
        o1 = self.s1(h, lengths, training=training)
        o2 = self.s2(h, lengths, training=training)
        c = tf.concat([o1, o2], axis=-1)
        return self.dense(c)

class AEDiscriminator(tf.keras.Model):
    def __init__(self, dim, hidden_dim, num_layers, m1, m2):
        super().__init__()
        self.d1 = RNNStack(m1, hidden_dim, num_layers)
        self.d2 = RNNStack(m2, hidden_dim, num_layers)
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, x, lengths, training=False):
        o1 = self.d1(x, lengths, training=training)
        o2 = self.d2(x, lengths, training=training)
        c = tf.concat([o1, o2], axis=-1)
        return self.out(c)

def chronogan(ori_data, parameters, num_samples):
    tf.keras.backend.clear_session()
    np.random.seed(0)
    tf.random.set_seed(0)

    ori_data = np.asarray(ori_data)
    no, seq_len, dim = ori_data.shape
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    hidden_dim = dim if parameters['hidden_dim'] == 'same' else parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = dim
    gamma = 1.0
    beta = 1.0
    m1 = 'gru'
    m2 = 'lstm'

    embedder = Embedder(dim, hidden_dim, num_layers, m1, m2)
    recovery = Recovery(dim, hidden_dim, num_layers, m1, m2)
    generator = Generator(dim, hidden_dim, num_layers, m1, m2, z_dim)
    supervisor = Supervisor(dim, hidden_dim, num_layers, m1, m2)
    ae_disc = AEDiscriminator(dim, hidden_dim, num_layers, m1, m2)

    E0_optimizer = tf.keras.optimizers.Adam()
    E_optimizer = tf.keras.optimizers.Adam()
    D_ae_optimizer = tf.keras.optimizers.Adam()
    D_ae_second_optimizer = tf.keras.optimizers.Adam()
    G_optimizer = tf.keras.optimizers.Adam()
    GS_optimizer = tf.keras.optimizers.Adam()

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse_loss = tf.keras.losses.MeanSquaredError()

    final_generated = []
    global_summing = 10.0
    p1 = None
    p2 = None

    def safe_np(arr):
        a = np.asarray(arr)
        if a.size == 0:
            return None
        return a

    for itt in range(int(iterations * 0.5)):
        for _ in range(2):
            X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb_np = safe_np(X_mb_np)
            if X_mb_np is None:
                continue
            X_mb = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
            T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb, dtype=np.int32))
            with tf.GradientTape() as tape:
                H = embedder(X_mb, T_mb_tf, training=True)
                X_tilde = recovery(H, T_mb_tf, training=True)
                Y_ae_fake = ae_disc(X_tilde, T_mb_tf, training=True)
                E_loss_T00 = mse_loss(X_mb, X_tilde)
                E_loss_U = bce(tf.ones_like(Y_ae_fake), Y_ae_fake)
                E_loss0 = 10.0 * tf.sqrt(tf.maximum(E_loss_T00 + 0.001 * E_loss_U, 1e-12))
            vars_e = embedder.trainable_variables + recovery.trainable_variables
            grads = tape.gradient(E_loss0, vars_e)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, vars_e)]
            E0_optimizer.apply_gradients(zip(grads, vars_e))

        X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_np = safe_np(X_mb_np)
        step_d_ae_loss = 0.0
        if X_mb_np is not None:
            X_mb = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
            T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb, dtype=np.int32))
            with tf.GradientTape() as tape:
                Y_ae_real = ae_disc(X_mb, T_mb_tf, training=True)
                D_ae_loss_real = bce(tf.ones_like(Y_ae_real), Y_ae_real)
            vars_d = ae_disc.trainable_variables
            grads = tape.gradient(D_ae_loss_real, vars_d)
            if any([g is not None for g in grads]):
                grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, vars_d)]
                D_ae_optimizer.apply_gradients(zip(grads, vars_d))
                step_d_ae_loss = float(D_ae_loss_real.numpy())

        if (itt % (int(iterations * .5) //10 if iterations*0.5>=10 else 1) == 0) or (itt == int(iterations * 0.5) - 1):
            try:
                e0_val = float(E_loss0.numpy())
            except:
                e0_val = 0.0
            print('step: '+ str(itt*2) + '/' + str(iterations) + ', AE_loss: ' + str(np.round(e0_val,4)) + ', AE_D_loss: ' + str(np.round(step_d_ae_loss,4)))

    for itt in range(iterations):
        X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_np = safe_np(X_mb_np)
        if X_mb_np is None:
            continue
        X_mb = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
        T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb, dtype=np.int32))
        Z_mb_np = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb = tf.convert_to_tensor(np.asarray(Z_mb_np, dtype=np.float32))
        with tf.GradientTape() as tape:
            E_hat = generator(Z_mb, T_mb_tf, training=True)
            H_hat = supervisor(E_hat, T_mb_tf, training=True)
            H = embedder(X_mb, T_mb_tf, training=True)
            H_hat_supervise = supervisor(H, T_mb_tf, training=True)
            G_loss_S = mse_loss(H[:,2:,:], H_hat_supervise[:,:-2,:])
        vars_gs = generator.trainable_variables + supervisor.trainable_variables
        grads = tape.gradient(G_loss_S, vars_gs)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, vars_gs)]
        GS_optimizer.apply_gradients(zip(grads, vars_gs))
        if (itt % (int(iterations)//10 if iterations>=10 else 1) == 0) or (itt == iterations - 1):
            print('step: '+ str(itt)  + '/' + str(iterations) +', S_loss: ' + str(np.round(float(G_loss_S.numpy()),4)) )

    for itt in range(iterations):
        for _ in range(2):
            X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb_np = safe_np(X_mb_np)
            if X_mb_np is None:
                continue
            X_mb = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
            T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb, dtype=np.int32))
            Z_mb_np = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb = tf.convert_to_tensor(np.asarray(Z_mb_np, dtype=np.float32))

            with tf.GradientTape(persistent=True) as tape:
                E_hat = generator(Z_mb, T_mb_tf, training=True)
                H_hat = supervisor(E_hat, T_mb_tf, training=True)
                H = embedder(X_mb, T_mb_tf, training=True)
                H_hat_supervise = supervisor(H, T_mb_tf, training=True)
                X_hat = recovery(H_hat, T_mb_tf, training=True)
                Y_ae_fake_e = ae_disc(X_hat, T_mb_tf, training=True)
                X_tilde_fake_second = recovery(E_hat, T_mb_tf, training=True)
                Y_ae_fake_e_second = ae_disc(X_tilde_fake_second, T_mb_tf, training=True)

                G_loss_U_ae = bce(tf.ones_like(Y_ae_fake_e), Y_ae_fake_e)
                G_loss_U_ae_e = bce(tf.ones_like(Y_ae_fake_e_second), Y_ae_fake_e_second)
                G_loss_S = mse_loss(H[:,2:,:], H_hat_supervise[:,:-2,:])
                G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X_mb,[0])[1] + 1e-6)))
                G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X_mb,[0])[0])))
                G_loss_V = G_loss_V1 + G_loss_V2

                b = tf.shape(X_mb)[0]
                W = tf.range(1, seq_len + 1, dtype=tf.float32)
                W = tf.reshape(W, (1, seq_len, 1))
                W = tf.broadcast_to(W, (b, seq_len, dim))
                W_sum = tf.reduce_sum(W, axis=1, keepdims=True)
                W_normalized = W / W_sum
                weighted_average_X = tf.reduce_sum(W_normalized * X_mb, axis=1)
                weighted_average_X_hat = tf.reduce_sum(W_normalized * X_hat, axis=1)
                mean_weighted_average_X = tf.reduce_mean(weighted_average_X, axis=0)
                mean_weighted_average_X_hat = tf.reduce_mean(weighted_average_X_hat, axis=0)
                std_weighted_average_X = tf.math.reduce_std(weighted_average_X, axis=0)
                std_weighted_average_X_hat = tf.math.reduce_std(weighted_average_X_hat, axis=0)
                mean_weighted_average_mse = mse_loss(mean_weighted_average_X, mean_weighted_average_X_hat)
                std_weighted_average_mse = mse_loss(std_weighted_average_X, std_weighted_average_X_hat)

                x = tf.range(seq_len, dtype=tf.float32)
                sum_x = tf.reduce_sum(x)
                sum_x2 = tf.reduce_sum(tf.square(x))
                N = tf.cast(seq_len, tf.float32)
                def calculate_slope(Y):
                    sum_y = tf.reduce_sum(Y, axis=1)
                    sum_xy = tf.reduce_sum(tf.expand_dims(x,1) * Y, axis=1)
                    numerator = N * sum_xy - sum_x * sum_y
                    denominator = N * sum_x2 - tf.square(sum_x)
                    slope = numerator / (denominator + 1e-12)
                    return slope

                slope_X = calculate_slope(X_mb)
                slope_X_hat = calculate_slope(X_hat)
                mean_slope_X = tf.reduce_mean(slope_X, axis=0)
                mean_slope_X_hat = tf.reduce_mean(slope_X_hat, axis=0)
                std_slope_X = tf.math.reduce_std(slope_X, axis=0)
                std_slope_X_hat = tf.math.reduce_std(slope_X_hat, axis=0)
                mean_slope_mse = mse_loss(mean_slope_X, mean_slope_X_hat)
                std_slope_mse = mse_loss(std_slope_X, std_slope_X_hat)

                def calculate_skewness(data, axis=1):
                    Nn = tf.cast(tf.shape(data)[axis], tf.float32)
                    mean = tf.reduce_mean(data, axis=axis, keepdims=True)
                    std_dev = tf.math.reduce_std(data, axis=axis, keepdims=True)
                    skewness = tf.reduce_sum(((data - mean) / (std_dev + 1e-12))**3, axis=axis) * (Nn / ((Nn - 1) * (Nn - 2) + 1e-12))
                    return skewness

                skew_X = calculate_skewness(X_mb, axis=1)
                skew_X_hat = calculate_skewness(X_hat, axis=1)
                mean_skew_X = tf.reduce_mean(skew_X, axis=0)
                mean_skew_X_hat = tf.reduce_mean(skew_X_hat, axis=0)
                std_skew_X = tf.math.reduce_std(skew_X, axis=0)
                std_skew_X_hat = tf.math.reduce_std(skew_X_hat, axis=0)
                mean_skew_mse = mse_loss(mean_skew_X, mean_skew_X_hat)
                std_skew_mse = mse_loss(std_skew_X, std_skew_X_hat)

                time_size = tf.shape(X_mb)[1]
                def median_tensor(data):
                    ts = tf.cast(time_size, tf.int32)
                    mid = ts // 2
                    def odd():
                        return data[:, mid, :]
                    def even():
                        return (data[:, (mid-1), :] + data[:, mid, :]) / 2.0
                    return tf.cond(tf.equal(ts % 2, 1), odd, even)

                median_X = median_tensor(X_mb)
                median_X_hat = median_tensor(X_hat)
                mean_median_X = tf.reduce_mean(median_X, axis=0)
                mean_median_X_hat = tf.reduce_mean(median_X_hat, axis=0)
                std_median_X = tf.math.reduce_std(median_X, axis=0)
                std_median_X_hat = tf.math.reduce_std(median_X_hat, axis=0)
                mean_median_mse = mse_loss(mean_median_X, mean_median_X_hat)
                std_median_mse = mse_loss(std_median_X, std_median_X_hat)

                ts_structure = mean_weighted_average_mse + std_weighted_average_mse + mean_slope_mse + std_slope_mse + 0.5*mean_median_mse + 0.5*std_median_mse + 0.5*mean_skew_mse + 0.5*std_skew_mse
                G_loss = (G_loss_U_ae + gamma * G_loss_U_ae_e) + 100.0 * tf.sqrt(tf.maximum(G_loss_S, 1e-12)) + 100.0*G_loss_V + 25.0 * ts_structure

            vars_g = generator.trainable_variables + supervisor.trainable_variables
            grads = tape.gradient(G_loss, vars_g)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, vars_g)]
            G_optimizer.apply_gradients(zip(grads, vars_g))

            with tf.GradientTape() as tape2:
                H = embedder(X_mb, T_mb_tf, training=True)
                X_tilde = recovery(H, T_mb_tf, training=True)
                Y_ae_fake = ae_disc(X_tilde, T_mb_tf, training=True)
                E_loss_T00 = mse_loss(X_mb, X_tilde)
                E_loss_U = bce(tf.ones_like(Y_ae_fake), Y_ae_fake)
                E_loss = 10.0 * tf.sqrt(tf.maximum(E_loss_T00 + 0.001 * 0.1 * E_loss_U, 1e-12)) + 0.1 * G_loss_S
            vars_e = embedder.trainable_variables + recovery.trainable_variables
            grads = tape2.gradient(E_loss, vars_e)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, vars_e)]
            E_optimizer.apply_gradients(zip(grads, vars_e))

        X_mb_np, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_np = safe_np(X_mb_np)
        if X_mb_np is None:
            continue
        X_mb = tf.convert_to_tensor(np.asarray(X_mb_np, dtype=np.float32))
        T_mb_tf = tf.convert_to_tensor(np.asarray(T_mb, dtype=np.int32))
        Z_mb_np = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb = tf.convert_to_tensor(np.asarray(Z_mb_np, dtype=np.float32))

        E_hat = generator(Z_mb, T_mb_tf, training=False)
        H_hat = supervisor(E_hat, T_mb_tf, training=False)
        X_hat = recovery(H_hat, T_mb_tf, training=False)
        X_tilde_fake_second = recovery(E_hat, T_mb_tf, training=False)
        Y_ae_fake = ae_disc(X_hat, T_mb_tf, training=False)
        Y_ae_fake_e = ae_disc(X_hat, T_mb_tf, training=False)
        Y_ae_fake_e_second = ae_disc(X_tilde_fake_second, T_mb_tf, training=False)

        D_ae_loss_real = bce(tf.ones_like(ae_disc(X_mb, T_mb_tf, training=False)), ae_disc(X_mb, T_mb_tf, training=False))
        D_ae_loss_fake = bce(tf.zeros_like(Y_ae_fake), Y_ae_fake)
        D_ae_loss_fake_e = bce(tf.zeros_like(Y_ae_fake_e), Y_ae_fake_e)
        D_ae_loss_fake_e_second = bce(tf.zeros_like(Y_ae_fake_e_second), Y_ae_fake_e_second)
        D_ae_loss = D_ae_loss_real + D_ae_loss_fake
        D_ae_loss_real_second = bce(tf.ones_like(Y_ae_fake), Y_ae_fake)
        D_ae_loss_second = D_ae_loss_real + D_ae_loss_real_second + beta * (D_ae_loss_fake_e + gamma * D_ae_loss_fake_e_second)

        step_d_ae_loss_second = 0.0
        if float(D_ae_loss_second.numpy()) > 0.15:
            vars_d = ae_disc.trainable_variables
            with tf.GradientTape() as tape3:
                # recompute for tape3
                Y_real = ae_disc(X_mb, T_mb_tf, training=True)
                # use previously computed D_ae_loss_second as objective surrogate
                D_loss2 = D_ae_loss_second
            grads = tape3.gradient(D_loss2, vars_d)
            if any([g is not None for g in grads]):
                grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, vars_d)]
                D_ae_second_optimizer.apply_gradients(zip(grads, vars_d))
                step_d_ae_loss_second = float(D_loss2.numpy())

        if (itt % (int(iterations)//10 if iterations>=10 else 1) == 0) or (itt == iterations - 1):
            print('step: '+ str(itt) + '/' + str(iterations) + ', D_loss: ' + str(np.round(step_d_ae_loss_second,4)))

        if (itt >= int(iterations*0.5)) and (itt % 500 == 0 or itt==iterations-1):
            Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
            Z_mb_t = tf.convert_to_tensor(np.asarray(Z_mb, dtype=np.float32))
            generated_data_curr = recovery(supervisor(generator(Z_mb_t, tf.convert_to_tensor(np.asarray(ori_time,dtype=np.int32))), tf.convert_to_tensor(np.asarray(ori_time,dtype=np.int32))), tf.convert_to_tensor(np.asarray(ori_time,dtype=np.int32))).numpy()
            generated_data = []
            for i in range(no):
                temp = generated_data_curr[i,:ori_time[i],:]
                generated_data.append(temp)
            generated_data = np.array(generated_data)
            generated_data = generated_data * max_val
            generated_data = generated_data + min_val

            final_generated = np.copy(generated_data)
            print(f"Generated data: {generated_data.shape}")

            metric_iteration = 3
            print("Computing discriminative score...")
            discriminative_score = []
            for _ in range(metric_iteration):
                temp_disc = discriminative_score_metrics(ori_data, generated_data, iterations=iterations, batch_size=batch_size)
                discriminative_score.append(temp_disc)
            discriminative_score = np.array(discriminative_score)
            filtered_discriminative_score = discriminative_score[(discriminative_score <= np.percentile(discriminative_score,75))] if discriminative_score.size>0 else np.array([])

            print("Computing predictive score...")
            predictive_score = []
            for tt in range(metric_iteration):
                temp_pred = predictive_score_metrics(ori_data, generated_data, iterations=iterations, batch_size=batch_size)
                predictive_score.append(temp_pred)
            predictive_score = np.array(predictive_score)
            filtered_predictive_score = predictive_score[(predictive_score <= np.percentile(predictive_score,75))] if predictive_score.size>0 else np.array([])

            mean_real = np.mean(ori_data, axis=0)
            mean_synthetic = np.mean(generated_data, axis=0)
            mse_mean = np.mean((mean_real - mean_synthetic) ** 2)
            variance_real = np.var(ori_data, axis=0)
            variance_synthetic = np.var(generated_data, axis=0)
            mse_variance = np.mean((variance_real - variance_synthetic) ** 2)

            mean_dis_score = np.round(np.min(filtered_discriminative_score), 4) if filtered_discriminative_score.size>0 else 0.0
            mean_pre_score = np.round(np.min(filtered_predictive_score), 4) if filtered_predictive_score.size>0 else 0.0

            if p1 is None and p2 is None:
                if mean_dis_score == 0:
                    p1 = 1.0
                    p2 = 1.0
                elif mean_pre_score == 0:
                    p1 = 1.0
                    p2 = mean_dis_score / (mse_mean + mse_variance + 1e-12)
                else:
                    p1 = mean_dis_score / (mean_pre_score + 1e-12)
                    p2 = mean_dis_score / (mse_mean + mse_variance + 1e-12)

            summing = mean_dis_score + p1 * mean_pre_score + p2 * ( mse_mean + mse_variance )
            if summing <= global_summing:
                global_summing = summing
                final_generated = generated_data
                print(f"Generated data: {final_generated.shape}")

    if num_samples == "same":
        print(f"Final data returned: {final_generated.shape}")
        return final_generated
    else:
        count = int(num_samples / no)
        all_generated_data = []
        for _ in range(count):
            Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
            Z_mb_t = tf.convert_to_tensor(np.asarray(Z_mb, dtype=np.float32))
            generated_data_curr = recovery(supervisor(generator(Z_mb_t, tf.convert_to_tensor(np.asarray(ori_time,dtype=np.int32))), tf.convert_to_tensor(np.asarray(ori_time,dtype=np.int32))), tf.convert_to_tensor(np.asarray(ori_time,dtype=np.int32))).numpy()
            generated_data = []
            for i in range(no):
                temp = generated_data_curr[i,:ori_time[i],:]
                generated_data.append(temp)
            generated_data = np.array(generated_data)
            generated_data = generated_data * max_val
            generated_data = generated_data + min_val
            all_generated_data.append(generated_data)
        all_generated_data = np.concatenate(all_generated_data, axis=0)
        return all_generated_data
