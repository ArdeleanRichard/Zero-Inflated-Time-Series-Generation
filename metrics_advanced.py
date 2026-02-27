import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torch import nn, Tensor
from tqdm import tqdm
from tqdm import trange
from sklearn.metrics import accuracy_score
import math

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # Pack padded sequence for efficient GRU processing
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        # Unpack sequence
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Apply output layer to all timesteps
        logits = self.out(h)  # (batch, time, 1)
        # Use sigmoid to match TF implementation
        y_hat = torch.sigmoid(logits)
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

    print(ori_data.shape)
    print(generated_data.shape)

    no, seq_len, dim = ori_data.shape

    # Extract sequence lengths and max lengths
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    # input timesteps to predictor = max_seq_len - 1
    input_time_steps = max_seq_len - 1
    # For multivariate data: use all features except the last as input, predict the last.
    # For univariate data (dim==1): use the current timestep value as input, predict next timestep.
    if dim > 1:
        input_dim = dim - 1
        target_start = input_dim
    else:
        input_dim = dim  # = 1
        target_start = 0  # predict the same (only) feature, one step ahead

    if hidden_dim is None:
        hidden_dim = max(1, dim // 2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and optimizer
    predictor = Predictor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

    # Helper: compute masked MAE loss
    def masked_mae(y_true, y_pred, seq_lengths):
        # y_true, y_pred: (batch, time, 1)
        # seq_lengths: (batch,) lengths (<= time)
        batch_size = y_true.size(0)
        max_len = y_true.size(1)

        # Create mask
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch, time, 1)

        abs_diff = torch.abs(y_true - y_pred) * mask
        denom = mask.sum()
        return abs_diff.sum() / torch.clamp(denom, min=1.0)

    # Training loop on generated data
    gen_N = len(generated_data)
    predictor.train()

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
            # skip sequences too short to produce any input/target pair
            if seq_len_i <= 1:
                continue
            # drop last timestep for inputs, drop first for targets (next-step)
            x_i = seq[:seq_len_i - 1, :input_dim]
            y_i = seq[1:seq_len_i, target_start:target_start + 1]
            X_seq_list.append(x_i)
            Y_seq_list.append(y_i)
            T_mb.append(seq_len_i - 1)

        # If the entire minibatch was filtered out, skip this iteration
        if len(T_mb) == 0:
            continue

        # pad to input_time_steps
        X_mb_padded = _pad_batch(X_seq_list, input_time_steps, input_dim)
        Y_mb_padded = _pad_batch(Y_seq_list, input_time_steps, 1)
        T_mb = np.asarray(T_mb, dtype=np.int64)

        # convert to tensors
        X_tensor = torch.tensor(X_mb_padded, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(Y_mb_padded, dtype=torch.float32, device=device)
        seq_lens_tensor = torch.tensor(T_mb, dtype=torch.int64, device=device)

        optimizer.zero_grad()
        y_pred = predictor(X_tensor, seq_lens_tensor)
        loss = masked_mae(Y_tensor, y_pred, seq_lens_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (itt == 1 or itt % 1000 == 0):
            print(f"Iteration {itt} train_masked_MAE: {loss.item():.6f}")

    # Evaluation on original data
    predictor.eval()
    N_ori = len(ori_data)
    X_seq_list = []
    Y_seq_list = []
    T_test = []
    valid_eval_count = 0
    for i in range(N_ori):
        seq = ori_data[i]
        seq_len_i = ori_time[i]
        # skip sequences too short to produce any input/target pair
        if seq_len_i <= 1:
            continue
        x_i = seq[:seq_len_i - 1, :input_dim]
        y_i = seq[1:seq_len_i, target_start:target_start + 1]
        X_seq_list.append(x_i)
        Y_seq_list.append(y_i)
        T_test.append(seq_len_i - 1)
        valid_eval_count += 1

    X_test_padded = _pad_batch(X_seq_list, input_time_steps, input_dim)
    Y_test_padded = _pad_batch(Y_seq_list, input_time_steps, 1)
    T_test = np.asarray(T_test, dtype=np.int64)

    # Predict
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_padded, dtype=torch.float32, device=device)
        T_test_tensor = torch.tensor(T_test, dtype=torch.int64, device=device)
        y_pred_test = predictor(X_test_tensor, T_test_tensor)
        y_pred_np = y_pred_test.cpu().numpy()  # (N_ori, time, 1)

    # Compute MAE per sequence using true lengths, then average across sequences
    MAE_total = 0.0
    for i in range(valid_eval_count):
        L = T_test[i]
        if L <= 0:
            continue
        y_true_seq = Y_test_padded[i, :L, 0]
        y_pred_seq = y_pred_np[i, :L, 0]
        MAE_total += mean_absolute_error(y_true_seq, y_pred_seq)

    predictive_score = MAE_total / float(max(valid_eval_count, 1))
    return predictive_score



# Small helper: pad a list of sequences (each shape (T_i, D)) into (batch, T, D)
def _pad_sequences(seqs, max_len=None, dtype=np.float32):
    if len(seqs) == 0:
        return np.zeros((0, 0, 0), dtype=dtype)
    # seqs can be np.ndarray of shape (N, T, D) or list of arrays
    if isinstance(seqs, np.ndarray):
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


# Sample minibatch from variable-length dataset
def _sample_minibatch(seqs, times, batch_size, max_len=None):
    N = len(seqs)
    if N == 0:
        raise ValueError("Empty dataset for minibatch sampling")
    idx = np.random.randint(0, N, size=batch_size)
    sampled_seqs = [seqs[i] for i in idx]
    sampled_times = [times[i] for i in idx]
    X_mb = _pad_sequences(sampled_seqs, max_len=max_len)
    T_mb = np.array(sampled_times, dtype=np.int64)
    return X_mb, T_mb


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, seq_lengths):
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.gru(packed)
        # Use final hidden state (last layer, all sequences)
        h = h_n.squeeze(0)  # (batch, hidden_dim)
        logits = self.fc(h)
        probs = torch.sigmoid(logits)
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
    # Basic shapes - handle list or array
    if isinstance(ori_data, np.ndarray):
        no, seq_len, dim = ori_data.shape
    else:
        no = len(ori_data)
        if no == 0:
            raise ValueError("ori_data is empty")
        dim = ori_data[0].shape[1]

    # Extract time info
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    if hidden_dim is None:
        hidden_dim = max(1, dim // 2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model + optimizer
    disc = Discriminator(input_dim=dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(disc.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare train/test splits
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Training loop
    disc.train()
    for itt in range(1, iterations + 1):
        # Sample real minibatch
        if isinstance(train_x, np.ndarray):
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

        # Convert to tensors
        X_mb = torch.tensor(X_mb, dtype=torch.float32, device=device)
        T_mb = torch.tensor(T_mb, dtype=torch.int64, device=device)
        X_hat_mb = torch.tensor(X_hat_mb, dtype=torch.float32, device=device)
        T_hat_mb = torch.tensor(T_hat_mb, dtype=torch.int64, device=device)

        optimizer.zero_grad()

        logits_real, _ = disc(X_mb, T_mb)
        logits_fake, _ = disc(X_hat_mb, T_hat_mb)

        labels_real = torch.ones_like(logits_real)
        labels_fake = torch.zeros_like(logits_fake)

        loss_real = criterion(logits_real, labels_real)
        loss_fake = criterion(logits_fake, labels_fake)
        loss = loss_real + loss_fake

        loss.backward()
        optimizer.step()

        if verbose and (itt == 1 or itt % 500 == 0):
            print(f"Iteration {itt} d_loss: {loss.item():.6f}")

    # Evaluation
    disc.eval()

    # Prepare test sets
    X_test = _pad_sequences(test_x, max_len=max_seq_len)
    X_hat_test = _pad_sequences(test_x_hat, max_len=max_seq_len)
    T_test = np.array(test_t, dtype=np.int64)
    T_hat_test = np.array(test_t_hat, dtype=np.int64)

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        T_test_tensor = torch.tensor(T_test, dtype=torch.int64, device=device)
        logits_real_test, probs_real_test = disc(X_test_tensor, T_test_tensor)

        X_hat_test_tensor = torch.tensor(X_hat_test, dtype=torch.float32, device=device)
        T_hat_test_tensor = torch.tensor(T_hat_test, dtype=torch.int64, device=device)
        logits_fake_test, probs_fake_test = disc(X_hat_test_tensor, T_hat_test_tensor)

    probs_real = probs_real_test.cpu().numpy().reshape(-1)
    probs_fake = probs_fake_test.cpu().numpy().reshape(-1)

    y_pred_final = np.concatenate([probs_real, probs_fake], axis=0)
    y_label_final = np.concatenate([np.ones(len(probs_real)), np.zeros(len(probs_fake))], axis=0)

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5).astype(int))
    discriminative_score = float(np.abs(0.5 - acc))
    return discriminative_score


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
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


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def train_test_divide_torch(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = np.take(data_x, train_idx, 0)
    test_x = np.take(data_x, test_idx, 0)

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = np.take(data_x_hat, train_idx, 0)
    test_x_hat = np.take(data_x_hat, test_idx, 0)

    return train_x, train_x_hat, test_x, test_x_hat


def long_discriminative_score_metrics(ori_data, generated_data,
    iterations = 100,
    batch_size = 128,

):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)

    # Network parameters
    hidden_dim = max((int(dim / 2), 1))

    class Discriminator_GRU(nn.Module):
        def __init__(self, dim, hidden_dim):
            super(Discriminator_GRU, self).__init__()
            self.p_cell = nn.GRU(dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, d_last_states = self.p_cell(x)
            y_hat_logit = self.fc(d_last_states)
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat

    class Discriminator_Trans(nn.Module):
        def __init__(self, num_tokens, feature_dim, hidden_dim=3, nhead=8, num_layers=1):
            super(Discriminator_Trans, self).__init__()
            self.projection = nn.Linear(feature_dim, hidden_dim)
            self.positional_encoding = PositionalEncodingLDS(d_model=hidden_dim, dropout=0, max_len=num_tokens + 10)
            encoder_block = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=False)
            self.transformer_encoder = nn.TransformerEncoder(encoder_block, num_layers=num_layers)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.projection(x)
            x = x.permute(1, 0, 2)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)
            x = torch.mean(x, dim=1, keepdim=True).permute(1, 0, 2)
            y_hat_logit = self.fc(x)
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat

    # model
    discriminator = Discriminator_Trans(num_tokens=seq_len, feature_dim=dim, hidden_dim=8).to(device)

    # optimizer
    d_optimizer = torch.optim.Adam(discriminator.parameters())

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat = \
        train_test_divide_torch(ori_data, generated_data, ori_time, generated_time)

    loss = nn.functional.binary_cross_entropy_with_logits

    # Training step
    for itt in tqdm(range(iterations)):
        d_optimizer.zero_grad()

        # Batch setting
        no = len(train_x)
        idx = torch.randperm(no)
        train_idx = idx[:batch_size]
        X_mb = torch.index_select(train_x, 0, train_idx)
        no = len(train_x_hat)
        idx = torch.randperm(no)
        train_idx = idx[:batch_size]
        X_hat_mb = torch.index_select(train_x_hat, 0, train_idx)

        X_mb = X_mb.to(device)
        X_hat_mb = X_hat_mb.to(device)

        # model inference
        y_logit_real, y_pred_real = discriminator(X_mb)
        y_logit_fake, y_pred_fake = discriminator(X_hat_mb)

        # loss calculation
        d_loss_real = loss(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = loss(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        d_optimizer.step()

    test_x = test_x.to(device)

    test_x_hat = test_x_hat.to(device)

    with torch.no_grad():
        _, y_pred_real_curr = discriminator(test_x)
        y_pred_real_curr = y_pred_real_curr.squeeze()
        _, y_pred_fake_curr = discriminator(test_x_hat)
        y_pred_fake_curr = y_pred_fake_curr.squeeze()

    y_pred_final = torch.cat((y_pred_real_curr, y_pred_fake_curr), dim=0)
    y_label_final = torch.cat((torch.ones([len(y_pred_real_curr), ]), torch.zeros([len(y_pred_fake_curr), ])), dim=0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final.cpu().numpy(), (y_pred_final.cpu().numpy() > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score


class PositionalEncodingLDS(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def long_predictive_score_metrics(ori_data, generated_data,
    iterations=100,
    batch_size=128,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    ## Builde a post-hoc RNN predictive network
    # Network parameters
    hidden_dim = max((int(dim / 2), 1))


    class PredictorTransformer(nn.Module):
        def __init__(self, in_features_dim, hidden_dim, seq_len, nhead=8):
            super().__init__()
            self.projection = nn.Linear(in_features_dim - 1, hidden_dim)
            self.pos_encoding = PositionalEncodingLPS(
                d_model=hidden_dim, dropout=0, max_len=seq_len + 1
            )  # TODO: reduce seq+N
            encoder_block = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=hidden_dim, batch_first=False
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_block, num_layers=1
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.projection(x)
            x = x.permute(
                1, 0, 2
            )
            x = self.pos_encoding(x)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)
            y_hat_logit = self.fc(x)
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat


    model = PredictorTransformer(
        in_features_dim=dim, hidden_dim=hidden_dim, seq_len=seq_len
    ).to(device)

    # Loss for the predictor
    p_loss = nn.L1Loss()
    # optimizer
    p_optimizer = torch.optim.Adam(model.parameters())

    batch_size = min((batch_size, len(generated_data)))

    # Training
    for itt in trange(iterations):
        p_optimizer.zero_grad()
        # Set mini-batch
        idx = torch.randperm(len(generated_data))
        train_idx = idx[:batch_size]

        # selection of  batch with pytorch approach
        X_mb = torch.index_select(generated_data[:, :-1, : (dim - 1)], 0, train_idx)
        Y_mb = torch.index_select(generated_data[:, 1:, (dim - 1)], 0, train_idx)
        Y_mb = Y_mb.reshape(batch_size, seq_len - 1, 1)

        X_mb = X_mb.to(device)
        Y_mb = Y_mb.to(device)

        y_pred_mb = model(X_mb)
        loss = p_loss(Y_mb, y_pred_mb)
        loss.backward()
        p_optimizer.step()



    ## Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb = list(ori_data[i][:-1, : (dim - 1)] for i in train_idx)
    Y_mb = list(
        np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1])
        for i in train_idx
    )

    model = model.to("cpu")
    MAE_temp = 0
    for i in range(no):
        with torch.no_grad():

            pred_Y_curr = model(X_mb[i].unsqueeze(0)).squeeze(0)
            MAE_temp = MAE_temp + mean_absolute_error(
                Y_mb[i].detach(), pred_Y_curr.detach()
            )

    predictive_score = MAE_temp / no

    return predictive_score


class PositionalEncodingLPS(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        d_model = d_model if (d_model % 2) == 0 else d_model + 1
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        seq_len, _, emb_dim = x.shape
        x = x + self.pe[:seq_len, :, :emb_dim]
        return self.dropout(x)