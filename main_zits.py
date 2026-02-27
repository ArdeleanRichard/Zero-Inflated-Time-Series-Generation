import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from constants import device, OUT_FOLDER
from data_proc import (DataPreprocessor, CountDataPreprocessor, TimeSeriesDataset, load_iot_data, make_dataloaders, load_m5_data)
from utils import plot_training_history, plot_sample_comparisons


# ===========================================================================
# Shared building blocks
# ===========================================================================

def _dilated_block(in_ch: int, out_ch: int, dilation: int,
                   dropout: float = 0.0, spectral: bool = False) -> nn.Sequential:
    """
    Dilated Conv1d, kernel=3, stride=1, padding=dilation → L_out == L_in.
    BatchNorm + LeakyReLU + optional Dropout.
    spectral=True wraps the conv with spectral normalisation (used in WGAN-GP discriminator).
    """
    conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
    if spectral:
        conv = nn.utils.spectral_norm(conv)

    layers: list[nn.Module] = [conv, nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.2, inplace=True)]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualDilatedBlock(nn.Module):
    """
    Two dilated conv layers with a residual skip connection.
    If in_ch != out_ch a 1×1 conv adapts the residual branch.
    """
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity())
        self.act     = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.block(x) + self.skip(x)))


# ---------------------------------------------------------------------------
# Shared Encoder
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """
    (B, T) → (B, latent_dim * 2)  [mu and logvar concatenated]
    Used by both VAE and (optionally) the GAN encoder.
    Dilated conv stack: d=1,2,4 with residual blocks.
    """
    def __init__(self, seq_length: int, latent_dim: int, hidden_ch: int = 64):
        super().__init__()
        mid_ch = hidden_ch // 2
        self.conv = nn.Sequential(
            _dilated_block(1, hidden_ch, dilation=1, dropout=0.2),
            ResidualDilatedBlock(hidden_ch, hidden_ch, dilation=2, dropout=0.2),
            _dilated_block(hidden_ch, mid_ch, dilation=4),
        )
        enc_flat = mid_ch * seq_length
        self.fc_mu     = nn.Linear(enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T) → unsqueeze → (B, 1, T)
        h = self.conv(x.unsqueeze(1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------------------------------------------------------------------------
# Shared Zero-Inflated Decoder
# ---------------------------------------------------------------------------

class ZeroInflatedDecoder(nn.Module):
    """
    z ∈ R^latent_dim → (gate_prob, mag, output)

    Two SEPARATE heads (Jose et al. 2024 design):
      gate_prob ∈ (0,1)^T  — P(non-zero) per timestep, trained with BCE
      mag       ∈ (0,1)^T  — normalised magnitude,   trained with MSE on non-zeros

    output = gate_prob * mag  (used as the reconstructed/generated sample)

    At generation time call .sample_output(gate_prob, mag) which draws from
    Bernoulli(gate_prob) rather than using a fixed threshold.
    """
    def __init__(self, seq_length: int, latent_dim: int, hidden_ch: int = 64):
        super().__init__()
        mid_ch = hidden_ch // 2
        self.seq_length = seq_length
        self.mid_ch     = mid_ch

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, mid_ch * seq_length),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = nn.Sequential(
            ResidualDilatedBlock(mid_ch, hidden_ch, dilation=4, dropout=0.2),
            ResidualDilatedBlock(hidden_ch, hidden_ch, dilation=2, dropout=0.2),
            _dilated_block(hidden_ch, mid_ch, dilation=1),
        )
        flat = mid_ch * seq_length
        # Explicit Bernoulli gate head (outputs logits, apply sigmoid externally)
        self.gate_head = nn.Linear(flat, seq_length)
        # Magnitude head
        self.mag_head  = nn.Sequential(nn.Linear(flat, seq_length), nn.Sigmoid())

    def forward(self, z: torch.Tensor):
        h          = self.fc(z).view(-1, self.mid_ch, self.seq_length)
        h          = self.conv(h).flatten(1)
        gate_logit = self.gate_head(h)                    # raw logits
        gate_prob  = torch.sigmoid(gate_logit)            # P(non-zero)
        mag        = self.mag_head(h)                     # magnitude ∈ (0,1)
        output     = gate_prob * mag                      # soft product (used for training)
        return output, gate_prob, gate_logit, mag

    @staticmethod
    def sample_output(gate_prob: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        """
        Hard Bernoulli sample — preserves the correct zero ratio stochastically.
        Used only at generation time (not during training).
        """
        binary = torch.bernoulli(gate_prob)
        return binary * mag


# ===========================================================================
# Shared loss functions
# ===========================================================================

def gate_loss(gate_logit: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for the explicit Bernoulli gate.
    Target = 1 where x > 0 (machine ran), 0 where x == 0.
    """
    target = (x > 0.0).float()
    return F.binary_cross_entropy_with_logits(gate_logit, target)


def recon_loss(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    MSE restricted to non-zero timesteps.
    Zeros not dilute the magnitude signal.
    Returns 0 if the batch has no non-zero entries.
    """
    mask = x > 0.0
    if not mask.any():
        return torch.tensor(0.0, device=x.device)
    return F.mse_loss(output[mask], x[mask])


def temporal_consistency_loss(gate_prob: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Lag-1 autocorrelation loss on the zero/non-zero mask.
    Penalises mismatch between the predicted gate transitions and the real ones.
    Encourages the model to learn that zeros tend to cluster (run-length statistics).

    For a sequence b_t ∈ {0,1} the lag-1 autocorrelation is:
        E[b_t * b_{t+1}]  (unnormalised co-occurrence of consecutive non-zeros)

    We match this expectation in expectation:
        loss = ( E[gate_{t} * gate_{t+1}] - E[mask_{t} * mask_{t+1}] )^2
    """
    real_mask  = (x > 0.0).float()                     # (B, T)
    real_lag1  = (real_mask[:, :-1] * real_mask[:, 1:]).mean()
    pred_lag1  = (gate_prob[:, :-1] * gate_prob[:, 1:]).mean()
    return (pred_lag1 - real_lag1) ** 2


# ===========================================================================
# VAE
# ===========================================================================

class TimeSeriesVAE(nn.Module):
    """
    VAE with shared ConvEncoder + ZeroInflatedDecoder.

    Encoder: (B,T) → mu, logvar  ∈ R^latent_dim
    Decoder: z     → (output, gate_prob, gate_logit, mag)
    Sample:  z ~ N(0,I) → Bernoulli(gate_prob) * mag
    """
    def __init__(self, seq_length: int, latent_dim: int = 64, hidden_ch: int = 64):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.encoder    = ConvEncoder(seq_length, latent_dim, hidden_ch)
        self.decoder    = ZeroInflatedDecoder(seq_length, latent_dim, hidden_ch)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor):
        mu, logvar                        = self.encoder(x)
        z                                 = self.reparameterize(mu, logvar)
        output, gate_prob, gate_logit, mag = self.decoder(z)
        return output, gate_prob, gate_logit, mag, mu, logvar

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw num_samples synthetic series using Bernoulli gate sampling."""
        self.eval()
        with torch.no_grad():
            z                                  = torch.randn(num_samples, self.latent_dim, device=device)
            _, gate_prob, _, mag               = self.decoder(z)
            out                                = ZeroInflatedDecoder.sample_output(gate_prob, mag)
        return out


def vae_loss(output, gate_prob, gate_logit, mag, x, mu, logvar,
             beta: float = 0.5,
             gate_weight: float = 5.0,
             recon_weight: float = 10.0,
             tc_weight: float = 1.0):
    """
    Total VAE loss:
      recon_weight * MSE(non-zeros)
    + gate_weight  * BCE(gate vs binary mask)
    + tc_weight    * lag-1 autocorrelation mismatch
    + beta         * KL divergence
    """
    r  = recon_loss(output, x)
    g  = gate_loss(gate_logit, x)
    tc = temporal_consistency_loss(gate_prob, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_weight * r + gate_weight * g + tc_weight * tc + beta * kl
    return total, r, g, tc, kl


def train_vae(model, train_loader, val_loader, optimizer,
              num_epochs: int = 100, beta: float = 0.3,
              gate_weight: float = 5.0, recon_weight: float = 10.0,
              tc_weight: float = 1.0):
    history = {k: [] for k in ('train_loss', 'val_loss', 'train_recon_loss', 'train_kl_loss', 'train_sparsity_loss')}
    best_val         = float('inf')
    patience_counter = 0
    patience         = 20

    for epoch in range(num_epochs):
        model.train()
        t_loss = t_r = t_g = t_tc = t_kl = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output, gate_prob, gate_logit, mag, mu, logvar = model(batch)
            loss, r, g, tc, kl = vae_loss(
                output, gate_prob, gate_logit, mag, batch,
                mu, logvar, beta, gate_weight, recon_weight, tc_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item(); t_r += r.item(); t_g += g.item()
            t_tc   += tc.item();   t_kl += kl.item()

        n = len(train_loader)
        t_loss /= n; t_r /= n; t_g /= n; t_tc /= n; t_kl /= n

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output, gate_prob, gate_logit, mag, mu, logvar = model(batch)
                loss, *_ = vae_loss(
                    output, gate_prob, gate_logit, mag, batch,
                    mu, logvar, beta, gate_weight, recon_weight, tc_weight)
                v_loss += loss.item()
        v_loss /= len(val_loader)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_recon_loss'].append(t_r)
        history['train_sparsity_loss'].append(t_g)   # gate (BCE) stored as "sparsity" for utils compat
        history['train_kl_loss'].append(t_kl)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"train={t_loss:.4f} "
                  f"(recon={t_r:.4f} gate={t_g:.4f} tc={t_tc:.4f} kl={t_kl:.4f}) "
                  f"val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss; patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


# ===========================================================================
# GAN — Generator
# ===========================================================================

class TimeSeriesGenerator(nn.Module):
    """
    Generator: z → (output, gate_prob, gate_logit, mag)
    Reuses ZeroInflatedDecoder directly.
    """
    def __init__(self, seq_length: int, latent_dim: int = 64, hidden_ch: int = 64):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.decoder    = ZeroInflatedDecoder(seq_length, latent_dim, hidden_ch)

    def forward(self, z: torch.Tensor):
        return self.decoder(z)

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw num_samples synthetic series using Bernoulli gate sampling."""
        self.eval()
        with torch.no_grad():
            z                        = torch.randn(num_samples, self.latent_dim, device=device)
            _, gate_prob, _, mag     = self.decoder(z)
            out                      = ZeroInflatedDecoder.sample_output(gate_prob, mag)
        return out


# ===========================================================================
# GAN — Discriminator (WGAN-GP compatible, spectral norm, no BN)
# ===========================================================================

class TimeSeriesDiscriminator(nn.Module):
    """
    Discriminator for WGAN-GP.

    Key:
      - Spectral normalisation on all conv layers for Lipschitz stability.
      - No BatchNorm (incompatible with WGAN-GP gradient penalty).
      - Global average pool over time → fc → scalar critic score.
      - Returns mid-level features for optional feature-matching loss.
    """
    def __init__(self, seq_length: int, hidden_ch: int = 64):
        super().__init__()
        # No BN anywhere; spectral norm replaces it for stability.
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(1, hidden_ch // 2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(hidden_ch // 2, hidden_ch, kernel_size=3,
                          stride=1, padding=2, dilation=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(hidden_ch, hidden_ch * 2, kernel_size=3,
                          stride=1, padding=4, dilation=4)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.utils.spectral_norm(nn.Linear(hidden_ch * 2, 1))

    def forward(self, x: torch.Tensor):
        # x: (B, T) → (B, 1, T)
        h1       = self.conv1(x.unsqueeze(1))
        h2       = self.conv2(h1)
        h3       = self.conv3(h2)
        score    = self.fc(h3.mean(dim=2))        # (B, 1) — WGAN critic score
        features = h2.mean(dim=2)                 # (B, hidden_ch) — for FM loss
        return score, features


# ===========================================================================
# GAN — WGAN-GP loss functions
# ===========================================================================

def gradient_penalty(discriminator, real: torch.Tensor, fake: torch.Tensor, lambda_gp: float = 10.0) -> torch.Tensor:
    """
    WGAN-GP gradient penalty (Gulrajani et al. 2017).
    Penalises |∇D(x̂)|_2 deviating from 1, where x̂ = ε·real + (1-ε)·fake.
    """
    bs    = real.size(0)
    eps   = torch.rand(bs, 1, device=real.device)               # (B, 1) broadcast over T
    x_hat = eps * real + (1.0 - eps) * fake.detach()
    x_hat.requires_grad_(True)
    scores, _ = discriminator(x_hat)
    grads = torch.autograd.grad(
        outputs=scores, inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True)[0]
    return lambda_gp * ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def wgan_discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """WGAN critic loss: maximise E[D(real)] - E[D(fake)]  → minimise negative."""
    return fake_scores.mean() - real_scores.mean()


def wgan_generator_loss(fake_scores, fake_output, gate_prob, gate_logit, real_x,
                        real_features, fake_features,
                        gate_weight: float = 5.0,
                        recon_weight: float = 10.0,
                        fm_weight: float = 1.0,
                        tc_weight: float = 1.0):
    """
    Generator loss:
      WGAN adversarial + gate BCE + magnitude MSE + feature matching + temporal consistency
    """
    adv  = -fake_scores.mean()
    r    = recon_loss(fake_output, real_x)
    g    = gate_loss(gate_logit, real_x)
    fm   = F.mse_loss(fake_features, real_features.detach())
    tc   = temporal_consistency_loss(gate_prob, real_x)
    total = adv + recon_weight * r + gate_weight * g + fm_weight * fm + tc_weight * tc
    return total, r, g, fm


def train_gan(generator, discriminator, train_loader,
              g_optimizer, d_optimizer,
              num_epochs: int = 100,
              gate_weight: float = 5.0,
              recon_weight: float = 10.0,
              fm_weight: float = 1.0,
              tc_weight: float = 1.0,
              n_critic: int = 5,
              lambda_gp: float = 10.0):
    """
    WGAN-GP training loop.
    n_critic: number of discriminator steps per generator step (standard WGAN-GP 5).
    """
    history = {k: [] for k in ('d_loss', 'g_loss',
                                'g_recon_loss', 'g_sparsity_loss', 'g_fm_loss')}
    best_g           = float('inf')
    patience_counter = 0
    patience         = 20

    for epoch in range(num_epochs):
        generator.train(); discriminator.train()
        ed = eg = eg_r = eg_s = eg_fm = 0.0
        nd = ng = 0

        for step, real_batch in enumerate(train_loader):
            real_batch = real_batch.to(device)
            bs         = real_batch.size(0)

            # ------------------------------------------------------------------
            # Discriminator / Critic step (n_critic times per generator step)
            # ------------------------------------------------------------------
            d_optimizer.zero_grad()
            z              = torch.randn(bs, generator.latent_dim, device=device)
            fake_out, _, _, _ = generator(z)
            real_scores, _ = discriminator(real_batch)
            fake_scores_d, _ = discriminator(fake_out.detach())
            gp             = gradient_penalty(discriminator, real_batch, fake_out, lambda_gp)
            d_loss         = wgan_discriminator_loss(real_scores, fake_scores_d) + gp
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            d_optimizer.step()
            ed += d_loss.item(); nd += 1

            # ------------------------------------------------------------------
            # Generator step (every n_critic discriminator steps)
            # ------------------------------------------------------------------
            if (step + 1) % n_critic == 0:
                g_optimizer.zero_grad()
                z                               = torch.randn(bs, generator.latent_dim, device=device)
                fake_out, fake_gate_prob, fake_gate_logit, _ = generator(z)
                fake_scores_g, fake_features    = discriminator(fake_out)
                _, real_features                = discriminator(real_batch)
                g_loss, g_r, g_s, g_fm          = wgan_generator_loss(
                    fake_scores_g, fake_out, fake_gate_prob, fake_gate_logit,
                    real_batch, real_features, fake_features,
                    gate_weight, recon_weight, fm_weight, tc_weight)
                g_loss.backward()
                nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                g_optimizer.step()
                eg += g_loss.item(); eg_r += g_r.item()
                eg_s += g_s.item();  eg_fm += g_fm.item(); ng += 1

        ed /= max(nd, 1); eg /= max(ng, 1)
        eg_r /= max(ng, 1); eg_s /= max(ng, 1); eg_fm /= max(ng, 1)

        history['d_loss'].append(ed)
        history['g_loss'].append(eg)
        history['g_recon_loss'].append(eg_r)
        history['g_sparsity_loss'].append(eg_s)
        history['g_fm_loss'].append(eg_fm)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"D={ed:.4f}  G={eg:.4f} "
                  f"(recon={eg_r:.4f} gate={eg_s:.4f} fm={eg_fm:.4f})")

        if eg < best_g:
            best_g = eg; patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


# ===========================================================================
# Shared pipeline helpers
# ===========================================================================

def _load_and_preprocess(data, raw: np.ndarray):
    if data == "iot":
        pp   = DataPreprocessor()
    elif data == "m5":
        pp = CountDataPreprocessor()

    proc = pp.fit_transform(raw)
    # print(f"Preprocessor: zero_ratio={pp.stats['zero_ratio']:.2%}  "
    #       f"max={pp.max_seconds:.0f}s  norm_max={proc.max():.4f}")
    return raw, proc, pp


def _make_loaders(proc: np.ndarray, batch_size: int = 64):
    return make_dataloaders(TimeSeriesDataset(proc), batch_size=batch_size)


def _generate_and_save(model, preprocessor: DataPreprocessor,
                       prefix: str, num_synthetic: int, ori_data: np.ndarray):
    model.eval()
    norm_samples = model.sample(num_synthetic).cpu().numpy()
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


def _save_checkpoint(state_dict, path: str, **meta):
    torch.save({'model_state_dict': state_dict, **meta}, path)


def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ===========================================================================
# VAE entry points
# ===========================================================================

def main_train_vae(data, ori_data: np.ndarray):
    raw, proc, pp = _load_and_preprocess(data, ori_data)
    seq_len       = proc.shape[1]
    train_loader, val_loader = _make_loaders(proc)

    print("\nInitialising VAE ...")
    model     = TimeSeriesVAE(seq_length=seq_len, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Parameters: {_n_params(model):,}")

    history = train_vae(model, train_loader, val_loader, optimizer,
                        num_epochs=100, beta=0.3,
                        gate_weight=5.0, recon_weight=10.0, tc_weight=1.0)

    _save_checkpoint(model.state_dict(),
                     os.path.join(OUT_FOLDER, 'vae_model.pth'),
                     seq_length=seq_len, latent_dim=64)
    pp.save(os.path.join(OUT_FOLDER, 'vae_preprocessor.json'))
    plot_training_history(history,
                          save_path=os.path.join(OUT_FOLDER, 'vae_training_history.png'),
                          model_type='vae')
    print(f"\nVAE training complete. Files saved to: {OUT_FOLDER}")


def main_test_vae(data, ori_data: np.ndarray, num_synthetic: int = 1000):
    if data == "iot":
        pp = DataPreprocessor()
    elif data == "m5":
        pp = CountDataPreprocessor()
    pp.load(os.path.join(OUT_FOLDER, 'vae_preprocessor.json'))

    ckpt  = torch.load(os.path.join(OUT_FOLDER, 'vae_model.pth'), map_location=device)
    model = TimeSeriesVAE(seq_length=ckpt['seq_length'],
                          latent_dim=ckpt['latent_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"VAE parameters: {_n_params(model):,}")
    print(f"\nGenerating {num_synthetic} synthetic samples ...")
    _generate_and_save(model, pp, 'vae', num_synthetic, ori_data)
    print("VAE testing complete.")


# ===========================================================================
# GAN entry points
# ===========================================================================

def main_train_gan(data, ori_data: np.ndarray):
    raw, proc, pp = _load_and_preprocess(data, ori_data)
    seq_len       = proc.shape[1]
    train_loader, val_loader = _make_loaders(proc)

    print("\nInitialising GAN (WGAN-GP) ...")
    generator     = TimeSeriesGenerator(seq_length=seq_len, latent_dim=64).to(device)
    discriminator = TimeSeriesDiscriminator(seq_length=seq_len).to(device)
    # WGAN-GP recommends lower lr and no momentum (betas=(0, 0.9))
    g_opt = optim.Adam(generator.parameters(),     lr=1e-4, betas=(0.0, 0.9))
    d_opt = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    print(f"Generator:     {_n_params(generator):,}  params")
    print(f"Discriminator: {_n_params(discriminator):,}  params")

    history = train_gan(generator, discriminator, train_loader, g_opt, d_opt,
                        num_epochs=100,
                        gate_weight=5.0, recon_weight=10.0,
                        fm_weight=1.0, tc_weight=1.0,
                        n_critic=5, lambda_gp=10.0)

    _save_checkpoint(generator.state_dict(),
                     os.path.join(OUT_FOLDER, 'gan_generator.pth'),
                     seq_length=seq_len, latent_dim=64)
    _save_checkpoint(discriminator.state_dict(),
                     os.path.join(OUT_FOLDER, 'gan_discriminator.pth'),
                     seq_length=seq_len)
    pp.save(os.path.join(OUT_FOLDER, 'gan_preprocessor.json'))
    plot_training_history(history,
                          save_path=os.path.join(OUT_FOLDER, 'gan_training_history.png'),
                          model_type='gan')
    print(f"\nGAN training complete. Files saved to: {OUT_FOLDER}")


def main_test_gan(data, ori_data: np.ndarray, num_synthetic: int = 1000):
    if data == "iot":
        pp = DataPreprocessor()
    elif data == "m5":
        pp = CountDataPreprocessor()
    pp.load(os.path.join(OUT_FOLDER, 'gan_preprocessor.json'))

    ckpt  = torch.load(os.path.join(OUT_FOLDER, 'gan_generator.pth'), map_location=device)
    model = TimeSeriesGenerator(seq_length=ckpt['seq_length'], latent_dim=ckpt['latent_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Generator parameters: {_n_params(model):,}")
    print(f"\nGenerating {num_synthetic} synthetic samples ...")
    _generate_and_save(model, pp, 'gan', num_synthetic, ori_data)
    print("GAN testing complete.")


# ===========================================================================

if __name__ == "__main__":
    # ori_data = load_iot_data()
    # main_train_vae("iot", ori_data)
    # main_test_vae("iot", ori_data, num_synthetic=50000)
    # main_train_gan("iot", ori_data)
    # main_test_gan("iot", ori_data, num_synthetic=50000)

    ori_data = load_m5_data()
    main_train_vae("m5", ori_data)
    main_test_vae("m5", ori_data, num_synthetic=30000)
    main_train_gan("m5", ori_data)
    main_test_gan("m5", ori_data, num_synthetic=30000)
