"""
Transformer-based diffusion model architecture with zero-inflated time series support.

Key additions vs. original fide_model.py:
  - ZeroInflatedHead  : dual gate/magnitude head replacing the plain output projection.
  - gate_loss         : BCE on binary non-zero mask (same design as ZITS).
  - recon_loss        : MSE restricted to non-zero timesteps only.
  - temporal_consistency_loss : lag-1 autocorrelation match on the gate.
  - TransformerModel.forward  now returns (output, gate_prob, gate_logit, mag)
    so callers can compute the full ZI loss.
  - TransformerModel.sample_output  hard Bernoulli generation (same as ZITS).
"""

import math
from typing import List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Unchanged building blocks
# ===========================================================================

class ScalarEmbedding(nn.Module):
    """Embed scalar values into higher dimensional space"""
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(ScalarEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embedding_layer_1 = nn.Linear(input_dim, seq_len)
        self.embedding_layer_2 = nn.Linear(seq_len, seq_len * hidden_dim)

    def forward(self, x):
        x = self.embedding_layer_1(x.float())
        x = self.embedding_layer_2(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for time and diffusion step"""
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        self.scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        self.shift = torch.zeros(periodic_dim)
        self.shift[::2] = 0.5 * math.pi

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / self.max_value)
        return torch.cat([linear, periodic], -1)


class FeedForward(nn.Module):
    """Feed-forward network with configurable layers"""
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 activation: Callable = nn.ReLU(), final_activation: Callable = None):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Zero-inflated output head
# ===========================================================================

class ZeroInflatedHead(nn.Module):
    """
    Replaces the plain output projection of TransformerModel.

    Input : hidden representation  (B, T, hidden_dim)
    Output:
      gate_logit  (B, T, dim)  — raw logits for Bernoulli gate P(non-zero)
      gate_prob   (B, T, dim)  — sigmoid(gate_logit)
      mag         (B, T, dim)  — magnitude in (0, 1), via sigmoid
      output      (B, T, dim)  — gate_prob * mag  (soft product, used during training)

    Two separate linear projections share no weights, matching the ZITS design
    where gate and magnitude are learned independently.
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, out_dim)   # raw logits
        self.mag_proj  = nn.Linear(hidden_dim, out_dim)   # sigmoid-activated magnitude

    def forward(self, h: torch.Tensor):
        # h: (B, T, hidden_dim)
        gate_logit = self.gate_proj(h)                    # (B, T, dim)
        gate_prob  = torch.sigmoid(gate_logit)            # P(non-zero)
        mag        = torch.sigmoid(self.mag_proj(h))      # magnitude ∈ (0, 1)
        output     = gate_prob * mag                      # soft reconstruction
        return output, gate_prob, gate_logit, mag

    @staticmethod
    def sample_output(gate_prob: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        """
        Hard Bernoulli sample — preserves the correct zero ratio stochastically.
        Use at generation / inference time only (not during training).
        """
        binary = torch.bernoulli(gate_prob)
        return binary * mag


# ===========================================================================
# Zero-inflated loss functions  (same API as main_zits.py)
# ===========================================================================

def gate_loss(gate_logit: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for the explicit Bernoulli gate.
    Target = 1 where x > 0 (non-zero timestep), 0 where x == 0.
    Works for arbitrary trailing dims: (B, T, dim) or (B, T).
    """
    target = (x > 0.0).float()
    return F.binary_cross_entropy_with_logits(gate_logit, target)


def recon_loss(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    MSE restricted to non-zero entries so zeros don't dilute the magnitude signal.
    Returns 0 if the batch has no non-zero entries.
    """
    mask = x > 0.0
    if not mask.any():
        return torch.tensor(0.0, device=x.device)
    return F.mse_loss(output[mask], x[mask])


def temporal_consistency_loss(gate_prob: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Lag-1 autocorrelation loss on the zero/non-zero mask (along the TIME axis).

    Penalises mismatch between the predicted gate transitions and the real ones,
    teaching the model that zeros tend to cluster (run-length statistics).

        loss = ( E[gate_t * gate_{t+1}] - E[mask_t * mask_{t+1}] )^2

    Supports (B, T) and (B, T, dim) tensors.
    The time axis is always dim=1 (index 1), regardless of trailing feature dims.
    """
    real_mask = (x > 0.0).float()                     # (B, T) or (B, T, dim)
    # Slice along dim=1 (time) — works for both 2D and 3D tensors
    real_lag1 = (real_mask[:, :-1] * real_mask[:, 1:]).mean()
    pred_lag1 = (gate_prob[:, :-1] * gate_prob[:, 1:]).mean()
    return (pred_lag1 - real_lag1) ** 2


def zi_diffusion_loss(output, gate_prob, gate_logit, mag, x,
                      gate_weight: float = 5.0,
                      recon_weight: float = 10.0,
                      tc_weight: float = 1.0):
    """
    Composite zero-inflated loss for the diffusion model.

    Returns (total, r, g, tc) for logging.

    recon_weight * MSE(non-zeros)
    + gate_weight  * BCE(gate vs binary mask)
    + tc_weight    * lag-1 autocorrelation mismatch
    """
    r  = recon_loss(output, x)
    g  = gate_loss(gate_logit, x)
    tc = temporal_consistency_loss(gate_prob, x)
    total = recon_weight * r + gate_weight * g + tc_weight * tc
    return total, r, g, tc


# ===========================================================================
# Transformer model — zero-inflated variant
# ===========================================================================

class TransformerModel(nn.Module):
    """
    Transformer-based diffusion model with zero-inflated conditional generation.

    Changes vs. original:
      - output_proj replaced by ZeroInflatedHead (gate + magnitude).
      - forward() returns (output, gate_prob, gate_logit, mag, x_reconstructed)
        where x_reconstructed == output (gate_prob * mag, soft product).
      - sample_output() is exposed as a static method for inference.

    The model is intentionally backward-compatible: callers that only read
    output[0] from forward() will still get the reconstructed series.
    """
    def __init__(self, dim, hidden_dim, max_i, seq_len, n_condition=1, num_layers=8, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len    = seq_len

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj      = FeedForward(dim, [], hidden_dim)
        self.conditional_proj = ScalarEmbedding(n_condition, hidden_dim, seq_len)

        self.proj = FeedForward(4 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())

        self.enc_att = []
        self.i_proj  = []
        for _ in range(num_layers):
            self.enc_att.append(nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True))
            self.i_proj.append(nn.Linear(3 * hidden_dim, hidden_dim))
        self.enc_att = nn.ModuleList(self.enc_att)
        self.i_proj  = nn.ModuleList(self.i_proj)

        # --- Zero-inflated output head (replaces plain FeedForward output_proj) ---
        self.output_proj = ZeroInflatedHead(hidden_dim, dim)

    def forward(self, x, t, i, bm):
        """
        Args:
            x   : (B, T, dim) or (..., T, dim) noisy input
            t   : (..., T, 1) diffusion time in [0, 1]
            i   : (..., T, 1) diffusion step index
            bm  : (..., 1)    scalar conditional (e.g. building mean)

        Returns:
            output      : (B, T, dim)  gate_prob * mag  (soft, for training)
            gate_prob   : (B, T, dim)  P(non-zero per feature per timestep)
            gate_logit  : (B, T, dim)  raw gate logits (feed into gate_loss)
            mag         : (B, T, dim)  predicted magnitude ∈ (0, 1)
        """
        shape = x.shape
        x  = x.view(-1, *shape[-2:])
        t  = t.view(-1, shape[-2], 1)
        i  = i.view(-1, shape[-2], 1)

        x  = self.input_proj(x)
        t  = self.t_enc(t)
        i  = self.i_enc(i)
        bm = self.conditional_proj(bm.view(-1, 1)).view(-1, self.seq_len, self.hidden_dim)
        x  = self.proj(torch.cat([x, t, i, bm], -1))

        for att_layer, i_proj in zip(self.enc_att, self.i_proj):
            y, _ = att_layer(query=x, key=x, value=x)
            x    = x + torch.relu(y)

        # Zero-inflated head — produces four tensors
        output, gate_prob, gate_logit, mag = self.output_proj(x)  # all (B*..., T, dim)

        # Restore leading batch dimensions
        output     = output.view(*shape)
        gate_prob  = gate_prob.view(*shape)
        gate_logit = gate_logit.view(*shape)
        mag        = mag.view(*shape)
        return output, gate_prob, gate_logit, mag

    @staticmethod
    def sample_output(gate_prob: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        """
        Hard Bernoulli sample for generation.
        Draws a binary mask from Bernoulli(gate_prob) then multiplies by mag.
        Use only at inference time — not during training.
        """
        return ZeroInflatedHead.sample_output(gate_prob, mag)


# ===========================================================================
# Pipeline entry points
# (same structure as main_zits.py: main_train_fide / main_test_fide)
# ===========================================================================

import os
import numpy as np
import torch.optim as optim

from constants import device, OUT_FOLDER
from data_proc import (DataPreprocessor, CountDataPreprocessor,
                       TimeSeriesDataset, load_iot_data, make_dataloaders, load_m5_data)
from utils import plot_training_history, plot_sample_comparisons


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fide(model, train_loader, val_loader, optimizer,
               num_epochs: int = 100,
               gate_weight: float = 5.0,
               recon_weight: float = 10.0,
               tc_weight: float = 1.0):
    """
    Training loop for the zero-inflated TransformerModel (FIDE).

    At each step the model receives the *noisy* input x_noisy and is trained
    to predict the denoised output. The ZI loss is computed against the
    *clean* batch x_clean so the gate and magnitude heads learn the true
    sparsity structure.

    For simplicity we use a single-step noise schedule (add Gaussian noise
    with sigma sampled uniformly in [0, 1]) so the model can be exercised
    without a full external diffusion harness. Replace with your own
    scheduler if needed.

    History keys match main_zits.py conventions so plot_training_history works:
      train_loss, val_loss, train_recon_loss, train_kl_loss (= gate here),
      train_sparsity_loss (= tc here).
    """
    history = {k: [] for k in ('train_loss', 'val_loss',
                                'train_recon_loss', 'train_kl_loss',
                                'train_sparsity_loss')}
    best_val         = float('inf')
    patience_counter = 0
    patience         = 20

    seq_len = model.seq_len
    max_i   = model.i_enc.max_value          # reuse the stored max diffusion step

    for epoch in range(num_epochs):
        model.train()
        t_loss = t_r = t_g = t_tc = 0.0

        for batch in train_loader:
            # batch: (B, T) from DataPreprocessor — unsqueeze to (B, T, 1)
            x_clean = batch.to(device).unsqueeze(-1)          # (B, T, 1)
            B, T, D = x_clean.shape

            # Simple noise schedule: sigma ~ U(0,1), i ~ U(0, max_i)
            sigma   = torch.rand(B, 1, 1, device=device)
            noise   = torch.randn_like(x_clean) * sigma
            x_noisy = (x_clean + noise).clamp(0.0, 1.0)

            t_val   = sigma.expand(B, T, 1)                   # (B, T, 1)  diffusion time proxy
            i_val   = torch.randint(0, int(max_i), (B,), device=device)
            i_val   = i_val[:, None, None].float().expand(B, T, 1)

            # Conditional: per-sample mean of clean series (building mean analogue)
            bm = x_clean.mean(dim=1)                          # (B, 1, 1) → (B, 1)
            bm = bm.squeeze(-1)

            optimizer.zero_grad()
            output, gate_prob, gate_logit, mag = model(x_noisy, t_val, i_val, bm)
            loss, r, g, tc = zi_diffusion_loss(
                output, gate_prob, gate_logit, mag, x_clean,
                gate_weight=gate_weight, recon_weight=recon_weight, tc_weight=tc_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss += loss.item(); t_r += r.item()
            t_g    += g.item();    t_tc += tc.item()

        n = len(train_loader)
        t_loss /= n; t_r /= n; t_g /= n; t_tc /= n

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_clean = batch.to(device).unsqueeze(-1)
                B, T, D = x_clean.shape
                sigma   = torch.rand(B, 1, 1, device=device)
                noise   = torch.randn_like(x_clean) * sigma
                x_noisy = (x_clean + noise).clamp(0.0, 1.0)
                t_val   = sigma.expand(B, T, 1)
                i_val   = torch.randint(0, int(max_i), (B,), device=device)
                i_val   = i_val[:, None, None].float().expand(B, T, 1)
                bm      = x_clean.mean(dim=1).squeeze(-1)
                output, gate_prob, gate_logit, mag = model(x_noisy, t_val, i_val, bm)
                loss, *_ = zi_diffusion_loss(
                    output, gate_prob, gate_logit, mag, x_clean,
                    gate_weight=gate_weight, recon_weight=recon_weight, tc_weight=tc_weight)
                v_loss += loss.item()
        v_loss /= len(val_loader)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_recon_loss'].append(t_r)
        history['train_kl_loss'].append(t_g)        # gate BCE — reuse 'kl' slot for plot compat
        history['train_sparsity_loss'].append(t_tc) # tc — reuse 'sparsity' slot

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"train={t_loss:.4f} "
                  f"(recon={t_r:.4f} gate={t_g:.4f} tc={t_tc:.4f}) "
                  f"val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss; patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


# ---------------------------------------------------------------------------
# Sampling helper (mirrors ZITS _generate_and_save but for FIDE)
# ---------------------------------------------------------------------------

def _fide_generate_and_save(model, preprocessor, prefix, num_synthetic, ori_data):
    """
    Draw num_synthetic samples from the trained FIDE model using hard
    Bernoulli gate sampling, inverse-transform, save .npz and comparison plot.
    """
    model.eval()
    seq_len = model.seq_len
    max_i   = model.i_enc.max_value

    all_samples = []
    batch_sz    = 512
    remaining   = num_synthetic

    with torch.no_grad():
        while remaining > 0:
            bs      = min(batch_sz, remaining)
            # Start from pure noise
            x_noise = torch.rand(bs, seq_len, 1, device=device)
            t_val   = torch.ones(bs, seq_len, 1, device=device)   # t=1 → pure noise
            i_val   = torch.zeros(bs, seq_len, 1, device=device)  # step 0
            bm      = x_noise.mean(dim=1).squeeze(-1)

            _, gate_prob, _, mag = model(x_noise, t_val, i_val, bm)
            samples = TransformerModel.sample_output(gate_prob, mag)   # hard Bernoulli
            all_samples.append(samples.squeeze(-1).cpu().numpy())      # (bs, T)
            remaining -= bs

    norm_samples = np.concatenate(all_samples, axis=0)[:num_synthetic]
    gen_data     = preprocessor.inverse_transform(norm_samples)

    np.savez(os.path.join(OUT_FOLDER, f'{prefix}_generated_data.npz'), data=gen_data)
    plot_sample_comparisons(
        ori_data[:5], gen_data[:5],
        save_path=os.path.join(OUT_FOLDER, f'{prefix}_sample_comparison.png'))

    nz = gen_data[gen_data > 0]
    print(f"\nGenerated data stats:")
    print(f"  Zero ratio:      {np.mean(gen_data == 0):.2%}")
    print(f"  Max:             {np.max(gen_data):.1f}s  ({np.max(gen_data)/3600:.2f}h)")
    if len(nz):
        print(f"  Mean (non-zero): {nz.mean():.1f}s  ({nz.mean()/3600:.2f}h)")
    return gen_data


def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _load_and_preprocess(data, raw: np.ndarray):
    if data == "iot":
        pp = DataPreprocessor()
    elif data == "m5":
        pp = CountDataPreprocessor()
    proc = pp.fit_transform(raw)
    return raw, proc, pp


def _make_loaders(proc: np.ndarray, batch_size: int = 64):
    return make_dataloaders(TimeSeriesDataset(proc), batch_size=batch_size)



def main_train_fide(data, ori_data: np.ndarray,
                    hidden_dim=64, num_layers=8, max_i=1000,
                    num_epochs=100, lr=1e-3,
                    gate_weight=5.0, recon_weight=10.0, tc_weight=1.0):
    raw, proc, pp = _load_and_preprocess(data, ori_data)
    seq_len       = proc.shape[1]
    train_loader, val_loader = _make_loaders(proc)

    print("\nInitialising ZITS-FIDE ...")
    model     = TransformerModel(dim=1, hidden_dim=hidden_dim, max_i=max_i,
                                 seq_len=seq_len, n_condition=1,
                                 num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Parameters: {_n_params(model):,}")

    history = train_fide(model, train_loader, val_loader, optimizer,
                         num_epochs=num_epochs,
                         gate_weight=gate_weight, recon_weight=recon_weight,
                         tc_weight=tc_weight)

    torch.save({'model_state_dict': model.state_dict(),
                'seq_length': seq_len, 'hidden_dim': hidden_dim,
                'num_layers': num_layers, 'max_i': max_i},
               os.path.join(OUT_FOLDER, 'zits_fide_model.pth'))
    pp.save(os.path.join(OUT_FOLDER, 'zits_fide_preprocessor.json'))
    plot_training_history(history,
                          save_path=os.path.join(OUT_FOLDER, 'zits_fide_training_history.png'),
                          model_type='vae')
    print(f"\nZITS-FIDE training complete. Files saved to: {OUT_FOLDER}")


def main_test_fide(data, ori_data: np.ndarray, num_synthetic: int = 1000):
    if data == "iot":
        pp = DataPreprocessor()
    elif data == "m5":
        pp = CountDataPreprocessor()
    pp.load(os.path.join(OUT_FOLDER, 'zits_fide_preprocessor.json'))

    ckpt  = torch.load(os.path.join(OUT_FOLDER, 'zits_fide_model.pth'), map_location=device)
    model = TransformerModel(dim=1,
                             hidden_dim=ckpt['hidden_dim'],
                             max_i=ckpt['max_i'],
                             seq_len=ckpt['seq_length'],
                             n_condition=1,
                             num_layers=ckpt['num_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"ZITS-FIDE parameters: {_n_params(model):,}")
    print(f"\nGenerating {num_synthetic} synthetic samples ...")
    _fide_generate_and_save(model, pp, 'zits_fide', num_synthetic, ori_data)
    print("ZITS-FIDE testing complete.")


# ===========================================================================

if __name__ == "__main__":
    ori_data = load_m5_data()
    main_train_fide("m5", ori_data, hidden_dim=64, num_layers=8, max_i=1000,
                    num_epochs=100, lr=1e-3,
                    gate_weight=10.0, recon_weight=10.0, tc_weight=0.5)
    main_test_fide("m5", ori_data, num_synthetic=30000)


    ori_data = load_iot_data()
    main_train_fide("iot", ori_data,
                    hidden_dim=64, num_layers=8, max_i=1000,
                    num_epochs=100, lr=1e-3,
                    gate_weight=10.0, recon_weight=10.0, tc_weight=0.5)
    main_test_fide("iot", ori_data, num_synthetic=50000)