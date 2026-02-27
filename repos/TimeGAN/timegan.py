"""Time-series Generative Adversarial Networks (TimeGAN) - PyTorch Implementation

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNModule(nn.Module):
    """RNN module supporting GRU, LSTM, and LayerNorm LSTM."""

    def __init__(self, module_name: str, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.module_name = module_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if module_name == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstmLN':
            # Layer normalized LSTM
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown module: {module_name}")

    def forward(self, x, seq_lengths):
        # Pack padded sequence for efficient RNN processing
        packed = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.module_name == 'lstmLN':
            # Apply layer normalization
            output = self.layer_norms[-1](output)

        return output


class Embedder(nn.Module):
    """Embedding network: original feature space to latent space."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int, module_name: str):
        super().__init__()
        self.rnn = RNNModule(module_name, dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_lengths):
        rnn_out = self.rnn(x, seq_lengths)
        h = self.sigmoid(self.fc(rnn_out))
        return h


class Recovery(nn.Module):
    """Recovery network: latent space to original space."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int, module_name: str):
        super().__init__()
        self.rnn = RNNModule(module_name, hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, seq_lengths):
        rnn_out = self.rnn(h, seq_lengths)
        x_tilde = self.sigmoid(self.fc(rnn_out))
        return x_tilde


class Generator(nn.Module):
    """Generator: generate time-series data in latent space."""

    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int, module_name: str):
        super().__init__()
        self.rnn = RNNModule(module_name, z_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, seq_lengths):
        rnn_out = self.rnn(z, seq_lengths)
        e = self.sigmoid(self.fc(rnn_out))
        return e


class Supervisor(nn.Module):
    """Supervisor: generate next sequence using previous sequence."""

    def __init__(self, hidden_dim: int, num_layers: int, module_name: str):
        super().__init__()
        # Note: supervisor uses num_layers - 1
        self.rnn = RNNModule(module_name, hidden_dim, hidden_dim, max(1, num_layers - 1))
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, seq_lengths):
        rnn_out = self.rnn(h, seq_lengths)
        s = self.sigmoid(self.fc(rnn_out))
        return s


class Discriminator(nn.Module):
    """Discriminator: discriminate original vs synthetic time-series."""

    def __init__(self, hidden_dim: int, num_layers: int, module_name: str):
        super().__init__()
        self.rnn = RNNModule(module_name, hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h, seq_lengths):
        rnn_out = self.rnn(h, seq_lengths)
        y_hat = self.fc(rnn_out)
        return y_hat


class TimeGAN:
    """TimeGAN model for time-series generation."""

    def __init__(self, parameters: Dict, device: str = None):
        """
        Args:
            parameters: Dictionary containing:
                - hidden_dim: Hidden dimension size
                - num_layer: Number of RNN layers
                - iterations: Training iterations
                - batch_size: Batch size
                - module: RNN type ('gru', 'lstm', 'lstmLN')
        """
        self.params = parameters
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_dim = parameters['hidden_dim']
        self.num_layers = parameters['num_layer']
        self.iterations = parameters['iterations']
        self.batch_size = parameters['batch_size']
        self.module_name = parameters['module']

        # Will be set during training
        self.dim = None
        self.z_dim = None
        self.max_seq_len = None

        # Networks
        self.embedder = None
        self.recovery = None
        self.generator = None
        self.supervisor = None
        self.discriminator = None

        # Optimizers
        self.e_optimizer = None
        self.r_optimizer = None
        self.g_optimizer = None
        self.s_optimizer = None
        self.d_optimizer = None

        # Normalization parameters
        self.min_val = None
        self.max_val = None

    def _initialize_networks(self, dim: int, z_dim: int):
        """Initialize all networks."""
        self.dim = dim
        self.z_dim = z_dim

        self.embedder = Embedder(dim, self.hidden_dim, self.num_layers, self.module_name).to(self.device)
        self.recovery = Recovery(dim, self.hidden_dim, self.num_layers, self.module_name).to(self.device)
        self.generator = Generator(z_dim, self.hidden_dim, self.num_layers, self.module_name).to(self.device)
        self.supervisor = Supervisor(self.hidden_dim, self.num_layers, self.module_name).to(self.device)
        self.discriminator = Discriminator(self.hidden_dim, self.num_layers, self.module_name).to(self.device)

        # Optimizers
        self.e_optimizer = optim.Adam(self.embedder.parameters())
        self.r_optimizer = optim.Adam(self.recovery.parameters())
        self.g_optimizer = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()))
        self.d_optimizer = optim.Adam(self.discriminator.parameters())

    def _min_max_scale(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Min-Max normalization."""
        min_val = np.min(data, axis=(0, 1))
        data = data - min_val

        max_val = np.max(data, axis=(0, 1))
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    def _extract_time(self, data: List[np.ndarray]) -> Tuple[List[int], int]:
        """Extract time information."""
        time = [len(seq) for seq in data]
        max_seq_len = max(time)
        return time, max_seq_len

    def _batch_generator(self, data: np.ndarray, time: List[int], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate mini-batch."""
        no = len(data)
        idx = np.random.permutation(no)[:batch_size]

        X_mb = [data[i] for i in idx]
        T_mb = [time[i] for i in idx]

        # Pad sequences
        max_len = max(T_mb)
        X_padded = np.zeros((batch_size, max_len, self.dim))
        for i, (x, t) in enumerate(zip(X_mb, T_mb)):
            X_padded[i, :t, :] = x

        return torch.FloatTensor(X_padded).to(self.device), torch.LongTensor(T_mb).to(self.device)

    def _random_generator(self, batch_size: int, T_mb: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """Generate random vectors."""
        Z_mb = torch.zeros(batch_size, max_seq_len, self.z_dim).to(self.device)
        for i in range(batch_size):
            t = T_mb[i].item()
            Z_mb[i, :t, :] = torch.rand(t, self.z_dim)
        return Z_mb

    def train(self, ori_data: List[np.ndarray]) -> np.ndarray:
        """
        Train TimeGAN and generate synthetic data.

        Args:
            ori_data: List of original time-series data arrays

        Returns:
            generated_data: Generated time-series data
        """
        # Convert to numpy array and get dimensions
        ori_data_array = np.array([np.array(seq) for seq in ori_data], dtype=object)
        no = len(ori_data)

        # Extract time information
        ori_time, self.max_seq_len = self._extract_time(ori_data)

        # Pad sequences for normalization
        dim = ori_data[0].shape[1]
        padded_data = np.zeros((no, self.max_seq_len, dim))
        for i, (seq, t) in enumerate(zip(ori_data, ori_time)):
            padded_data[i, :t, :] = seq

        # Normalization
        ori_data_norm, self.min_val, self.max_val = self._min_max_scale(padded_data)

        # Initialize networks
        self._initialize_networks(dim, dim)

        gamma = 1.0

        print('Start Embedding Network Training')
        # Phase 1: Embedding network training
        for itt in range(self.iterations):
            X_mb, T_mb = self._batch_generator(ori_data_norm, ori_time, self.batch_size)

            # Forward pass
            H = self.embedder(X_mb, T_mb)
            X_tilde = self.recovery(H, T_mb)

            # Loss
            E_loss_T0 = nn.MSELoss()(X_tilde, X_mb)
            E_loss0 = 10 * torch.sqrt(E_loss_T0)

            # Backward pass
            self.e_optimizer.zero_grad()
            self.r_optimizer.zero_grad()
            E_loss0.backward()
            self.e_optimizer.step()
            self.r_optimizer.step()

            if itt % (self.iterations//10) == 0:
                print(f'step: {itt}/{self.iterations}, e_loss: {np.sqrt(E_loss_T0.item()):.4f}')

        print('Finish Embedding Network Training')

        print('Start Training with Supervised Loss Only')
        # Phase 2: Supervised training
        for itt in range(self.iterations):
            X_mb, T_mb = self._batch_generator(ori_data_norm, ori_time, self.batch_size)
            Z_mb = self._random_generator(self.batch_size, T_mb, self.max_seq_len)

            # Forward pass
            H = self.embedder(X_mb, T_mb)
            E_hat = self.generator(Z_mb, T_mb)
            H_hat_supervise = self.supervisor(H, T_mb)

            # Supervised loss
            G_loss_S = nn.MSELoss()(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            # Backward pass
            self.g_optimizer.zero_grad()
            G_loss_S.backward()
            self.g_optimizer.step()

            if itt % (self.iterations//10) == 0:
                print(f'step: {itt}/{self.iterations}, s_loss: {np.sqrt(G_loss_S.item()):.4f}')

        print('Finish Training with Supervised Loss Only')

        print('Start Joint Training')
        # Phase 3: Joint training
        for itt in range(self.iterations):
            # Train generator twice per discriminator update
            for _ in range(2):
                X_mb, T_mb = self._batch_generator(ori_data_norm, ori_time, self.batch_size)
                Z_mb = self._random_generator(self.batch_size, T_mb, self.max_seq_len)

                # Forward pass
                H = self.embedder(X_mb, T_mb)
                E_hat = self.generator(Z_mb, T_mb)
                H_hat = self.supervisor(E_hat, T_mb)
                H_hat_supervise = self.supervisor(H, T_mb)
                X_hat = self.recovery(H_hat, T_mb)

                Y_fake = self.discriminator(H_hat, T_mb)
                Y_fake_e = self.discriminator(E_hat, T_mb)

                # Generator losses
                G_loss_U = nn.BCEWithLogitsLoss()(Y_fake, torch.ones_like(Y_fake))
                G_loss_U_e = nn.BCEWithLogitsLoss()(Y_fake_e, torch.ones_like(Y_fake_e))
                G_loss_S = nn.MSELoss()(H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # Moment matching losses
                X_hat_mean, X_hat_var = torch.mean(X_hat, dim=0), torch.var(X_hat, dim=0)
                X_mean, X_var = torch.mean(X_mb, dim=0), torch.var(X_mb, dim=0)
                G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat_var + 1e-6) - torch.sqrt(X_var + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs(X_hat_mean - X_mean))
                G_loss_V = G_loss_V1 + G_loss_V2

                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

                # Update generator
                self.g_optimizer.zero_grad()
                G_loss.backward()
                self.g_optimizer.step()

                # Update embedder
                H = self.embedder(X_mb, T_mb)
                H_hat_supervise = self.supervisor(H, T_mb)
                X_tilde = self.recovery(H, T_mb)

                E_loss_T0 = nn.MSELoss()(X_tilde, X_mb)
                E_loss0 = 10 * torch.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1 * nn.MSELoss()(H[:, 1:, :], H_hat_supervise[:, :-1, :])

                self.e_optimizer.zero_grad()
                self.r_optimizer.zero_grad()
                E_loss.backward()
                self.e_optimizer.step()
                self.r_optimizer.step()

            # Train discriminator
            X_mb, T_mb = self._batch_generator(ori_data_norm, ori_time, self.batch_size)
            Z_mb = self._random_generator(self.batch_size, T_mb, self.max_seq_len)

            H = self.embedder(X_mb, T_mb)
            E_hat = self.generator(Z_mb, T_mb)
            H_hat = self.supervisor(E_hat, T_mb)

            Y_real = self.discriminator(H, T_mb)
            Y_fake = self.discriminator(H_hat, T_mb)
            Y_fake_e = self.discriminator(E_hat, T_mb)

            D_loss_real = nn.BCEWithLogitsLoss()(Y_real, torch.ones_like(Y_real))
            D_loss_fake = nn.BCEWithLogitsLoss()(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = nn.BCEWithLogitsLoss()(Y_fake_e, torch.zeros_like(Y_fake_e))
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

            # Only update if discriminator is not too good
            if D_loss.item() > 0.15:
                self.d_optimizer.zero_grad()
                D_loss.backward()
                self.d_optimizer.step()

            if itt % (self.iterations//10) == 0:
                print(f'step: {itt}/{self.iterations}, d_loss: {D_loss.item():.4f}, '
                      f'g_loss_u: {G_loss_U.item():.4f}, '
                      f'g_loss_s: {np.sqrt(G_loss_S.item()):.4f}, '
                      f'g_loss_v: {G_loss_V.item():.4f}, '
                      f'e_loss_t0: {np.sqrt(E_loss_T0.item()):.4f}')

        print('Finish Joint Training')

        # Generate synthetic data
        Z_mb = self._random_generator(no, torch.LongTensor(ori_time).to(self.device), self.max_seq_len)
        X_mb = torch.FloatTensor(ori_data_norm).to(self.device)
        T_mb = torch.LongTensor(ori_time).to(self.device)

        with torch.no_grad():
            E_hat = self.generator(Z_mb, T_mb)
            H_hat = self.supervisor(E_hat, T_mb)
            X_hat = self.recovery(H_hat, T_mb)

        generated_data_curr = X_hat.cpu().numpy()

        # Extract actual sequences
        generated_data = []
        for i in range(no):
            temp = generated_data_curr[i, :ori_time[i], :]
            generated_data.append(temp)

        # Denormalize
        generated_data = np.array(generated_data, dtype=object)
        for i in range(len(generated_data)):
            generated_data[i] = generated_data[i] * self.max_val + self.min_val

        return generated_data


def timegan(ori_data: List[np.ndarray], parameters: Dict) -> List[np.ndarray]:
    """
    TimeGAN wrapper function for compatibility with original API.

    Args:
        ori_data: List of original time-series data arrays
        parameters: TimeGAN network parameters

    Returns:
        generated_data: Generated time-series data
    """
    model = TimeGAN(parameters)
    return model.train(ori_data)