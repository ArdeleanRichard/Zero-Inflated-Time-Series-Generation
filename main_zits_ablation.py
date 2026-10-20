import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from constants import device
from data_proc import CountDataPreprocessor, TimeSeriesDataset, load_m5_data, make_dataloaders
from metrics import calculate_evaluation_metrics
from utils import plot_sample_comparisons, plot_training_history


# ---------------------------------------------------------------------------
# Fast, reusable building blocks for the ablation study.
# ---------------------------------------------------------------------------

def _dilated_block(in_ch: int, out_ch: int, dilation: int,
                   dropout: float = 0.0, spectral: bool = False) -> nn.Sequential:
    conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
    if spectral:
        conv = nn.utils.spectral_norm(conv)
    layers: list[nn.Module] = [conv, nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.2, inplace=True)]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.block(x) + self.skip(x)))


class StandardConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.block(x) + self.skip(x)))


class AblationConvEncoder(nn.Module):
    def __init__(self, seq_length: int, latent_dim: int, hidden_ch: int = 32, use_dilated: bool = True):
        super().__init__()
        mid_ch = hidden_ch // 2
        if use_dilated:
            self.conv = nn.Sequential(
                _dilated_block(1, hidden_ch, dilation=1, dropout=0.1),
                ResidualDilatedBlock(hidden_ch, hidden_ch, dilation=2, dropout=0.1),
                _dilated_block(hidden_ch, mid_ch, dilation=4),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(1, hidden_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(hidden_ch),
                nn.LeakyReLU(0.2, inplace=True),
                StandardConvBlock(hidden_ch, hidden_ch, dropout=0.1),
                nn.Conv1d(hidden_ch, mid_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(mid_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        enc_flat = mid_ch * seq_length
        self.fc_mu = nn.Linear(enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x.unsqueeze(1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class AblationZeroInflatedDecoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        latent_dim: int,
        hidden_ch: int = 32,
        use_gate_head: bool = True,
        use_mag_head: bool = True,
        use_dilated: bool = True,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.mid_ch = hidden_ch // 2
        self.use_gate_head = use_gate_head
        self.use_mag_head = use_mag_head

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.mid_ch * seq_length),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if use_dilated:
            self.conv = nn.Sequential(
                ResidualDilatedBlock(self.mid_ch, hidden_ch, dilation=4, dropout=0.1),
                ResidualDilatedBlock(hidden_ch, hidden_ch, dilation=2, dropout=0.1),
                _dilated_block(hidden_ch, self.mid_ch, dilation=1),
            )
        else:
            self.conv = nn.Sequential(
                StandardConvBlock(self.mid_ch, hidden_ch, dropout=0.1),
                StandardConvBlock(hidden_ch, hidden_ch, dropout=0.1),
                nn.Conv1d(hidden_ch, self.mid_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(self.mid_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        flat = self.mid_ch * seq_length
        self.gate_head = nn.Linear(flat, seq_length) if use_gate_head else None
        self.mag_head = nn.Sequential(nn.Linear(flat, seq_length), nn.Sigmoid()) if use_mag_head else None

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(-1, self.mid_ch, self.seq_length)
        h = self.conv(h).flatten(1)

        if self.use_gate_head:
            gate_logit = self.gate_head(h)
            gate_prob = torch.sigmoid(gate_logit)
        else:
            gate_logit = None
            gate_prob = torch.ones(h.shape[0], self.seq_length, device=h.device)

        if self.use_mag_head:
            mag = self.mag_head(h)
        else:
            mag = torch.ones(h.shape[0], self.seq_length, device=h.device)

        if self.use_gate_head and self.use_mag_head:
            output = gate_prob * mag
        elif self.use_gate_head:
            output = gate_prob
        elif self.use_mag_head:
            output = mag
        else:
            output = torch.zeros_like(mag)

        return output, gate_prob, gate_logit, mag

    @staticmethod
    def sample_output(gate_prob: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        binary = torch.bernoulli(gate_prob)
        return binary * mag


class AblationZITS_VAE(nn.Module):
    def __init__(
        self,
        seq_length: int,
        latent_dim: int = 32,
        hidden_ch: int = 32,
        use_gate_head: bool = True,
        use_mag_head: bool = True,
        use_dilated: bool = True,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.encoder = AblationConvEncoder(seq_length, latent_dim, hidden_ch=hidden_ch, use_dilated=use_dilated)
        self.decoder = AblationZeroInflatedDecoder(
            seq_length,
            latent_dim,
            hidden_ch=hidden_ch,
            use_gate_head=use_gate_head,
            use_mag_head=use_mag_head,
            use_dilated=use_dilated,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        output, gate_prob, gate_logit, mag = self.decoder(z)
        return output, gate_prob, gate_logit, mag, mu, logvar

    def sample(self, num_samples: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            _, gate_prob, _, mag = self.decoder(z)
            return AblationZeroInflatedDecoder.sample_output(gate_prob, mag)


# ---------------------------------------------------------------------------
# Losses used by the ablation study.
# ---------------------------------------------------------------------------

def gate_loss(gate_logit: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    if gate_logit is None:
        return torch.tensor(0.0, device=x.device)
    target = (x > 0.0).float()
    return F.binary_cross_entropy_with_logits(gate_logit, target)


def recon_loss(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mask = x > 0.0
    if not mask.any():
        return torch.tensor(0.0, device=x.device)
    return F.mse_loss(output[mask], x[mask])


def temporal_consistency_loss(gate_prob: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    real_mask = (x > 0.0).float()
    real_lag1 = (real_mask[:, :-1] * real_mask[:, 1:]).mean()
    pred_lag1 = (gate_prob[:, :-1] * gate_prob[:, 1:]).mean()
    return (pred_lag1 - real_lag1) ** 2


def ablation_vae_loss(
    output,
    gate_prob,
    gate_logit,
    mag,
    x,
    mu,
    logvar,
    beta: float = 0.3,
    gate_weight: float = 5.0,
    recon_weight: float = 10.0,
    tc_weight: float = 1.0,
):
    r = recon_loss(output, x)
    g = gate_loss(gate_logit, x)
    tc = temporal_consistency_loss(gate_prob, x) if tc_weight > 0.0 else torch.tensor(0.0, device=x.device)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_weight * r + gate_weight * g + tc_weight * tc + beta * kl
    return total, r, g, tc, kl


def train_ablation_vae(
    model: AblationZITS_VAE,
    train_loader,
    val_loader,
    optimizer,
    num_epochs: int = 30,
    beta: float = 0.3,
    gate_weight: float = 5.0,
    recon_weight: float = 10.0,
    tc_weight: float = 1.0,
):
    history = {k: [] for k in ("train_loss", "val_loss", "train_recon_loss", "train_kl_loss", "train_sparsity_loss")}
    best_val = float("inf")
    patience = 12
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_gate = 0.0
        running_tc = 0.0
        running_kl = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output, gate_prob, gate_logit, mag, mu, logvar = model(batch)
            loss, r, g, tc, kl = ablation_vae_loss(
                output, gate_prob, gate_logit, mag, batch, mu, logvar,
                beta=beta, gate_weight=gate_weight, recon_weight=recon_weight, tc_weight=tc_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_recon += r.item()
            running_gate += g.item()
            running_tc += tc.item()
            running_kl += kl.item()

        n = max(1, len(train_loader))
        running_loss /= n
        running_recon /= n
        running_gate /= n
        running_tc /= n
        running_kl /= n

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output, gate_prob, gate_logit, mag, mu, logvar = model(batch)
                loss, *_ = ablation_vae_loss(
                    output, gate_prob, gate_logit, mag, batch, mu, logvar,
                    beta=beta, gate_weight=gate_weight, recon_weight=recon_weight, tc_weight=tc_weight,
                )
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        history["train_loss"].append(running_loss)
        history["val_loss"].append(val_loss)
        history["train_recon_loss"].append(running_recon)
        history["train_kl_loss"].append(running_kl)
        history["train_sparsity_loss"].append(running_gate)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs}: "
                f"train={running_loss:.4f} val={val_loss:.4f} "
                f"(recon={running_recon:.4f} gate={running_gate:.4f} tc={running_tc:.4f} kl={running_kl:.4f})"
            )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


def _generate_and_save(model, preprocessor: CountDataPreprocessor, out_dir: Path, prefix: str, num_synthetic: int, ori_data: np.ndarray):
    model.eval()
    with torch.no_grad():
        norm_samples = model.sample(num_synthetic).cpu().numpy()
    gen_data = preprocessor.inverse_transform(norm_samples)

    np.savez(out_dir / f"{prefix}_generated_data.npz", data=gen_data)
    plot_sample_comparisons(
        np.squeeze(np.nan_to_num(ori_data[:5])).astype(np.float64),
        np.squeeze(np.nan_to_num(gen_data[:5])).astype(np.float64),
        save_path=str(out_dir / f"{prefix}_sample_comparison.png"),
    )

    nz = gen_data[gen_data > 0]
    print(f"Generated data stats: zero ratio={np.mean(gen_data == 0):.2%}, max={np.max(gen_data):.1f}, nonzero mean={nz.mean() if len(nz) else 0.0:.1f}")
    return gen_data


def _save_checkpoint(model: nn.Module, path: Path, **meta):
    torch.save({"model_state_dict": model.state_dict(), **meta}, path)


def build_ablation_cases() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("full", {"use_gate_head": True, "use_mag_head": True, "use_dilated": True, "tc_weight": 1.0}),
        ("no_gate_head", {"use_gate_head": False, "use_mag_head": True, "use_dilated": True, "tc_weight": 1.0}),
        ("no_mag_head", {"use_gate_head": True, "use_mag_head": False, "use_dilated": True, "tc_weight": 1.0}),
        ("no_temporal_consistency", {"use_gate_head": True, "use_mag_head": True, "use_dilated": True, "tc_weight": 0.0}),
        ("no_dilated_conv", {"use_gate_head": True, "use_mag_head": True, "use_dilated": False, "tc_weight": 1.0}),
    ]


def run_single_ablation(
    data: np.ndarray,
    ablation_name: str,
    run_idx: int,
    run_dir: Path,
   
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    gate_weight: float,
    recon_weight: float,
    num_synthetic: int,
    seed: int,
):
    # Reproducible per-run seeding.
    np.random.seed(seed + run_idx)
    torch.manual_seed(seed + run_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + run_idx)

    # M5 uses the count preprocessor.
    pp = CountDataPreprocessor()
    processed = pp.fit_transform(data)
    seq_len = processed.shape[1]
    train_loader, val_loader = make_dataloaders(TimeSeriesDataset(processed), batch_size=batch_size)

    config = None
    for key, value in build_ablation_cases():
        if key == ablation_name:
            config = value
            break
    if config is None:
        raise ValueError(f"Unknown ablation case: {ablation_name}")

    model = AblationZITS_VAE(
        seq_length=seq_len,
        latent_dim=latent_dim,
        hidden_ch=32,
        use_gate_head=config["use_gate_head"],
        use_mag_head=config["use_mag_head"],
        use_dilated=config["use_dilated"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"\n=== {ablation_name} | run {run_idx:02d} | params={sum(p.numel() for p in model.parameters()):,} ===")

    history = train_ablation_vae(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=epochs,
        beta=beta,
        gate_weight=gate_weight,
        recon_weight=recon_weight,
        tc_weight=config["tc_weight"],
    )

    # Save per-run artifacts.
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, save_path=str(run_dir / "training_history.png"), model_type="vae")
    pp.save(str(run_dir / "preprocessor.json"))
    _save_checkpoint(model, run_dir / "model.pth", seq_length=seq_len, latent_dim=latent_dim, ablation=ablation_name)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"ablation": ablation_name, "seed": seed + run_idx, "latent_dim": latent_dim, "epochs": epochs, "batch_size": batch_size, "lr": lr, "beta": beta, "gate_weight": gate_weight, "recon_weight": recon_weight, "tc_weight": config["tc_weight"], "use_gate_head": config["use_gate_head"], "use_mag_head": config["use_mag_head"], "use_dilated": config["use_dilated"]}, f, indent=2)

    gen_data = _generate_and_save(model, pp, run_dir, prefix="synthetic", num_synthetic=num_synthetic, ori_data=data)
    metrics = calculate_evaluation_metrics(np.squeeze(np.nan_to_num(data)).astype(np.float64), np.squeeze(np.nan_to_num(gen_data)).astype(np.float64))
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    return metrics


def aggregate_mean_metrics(metrics_by_run: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    all_metrics = []
    for run_metrics in metrics_by_run:
        all_metrics.append(run_metrics)

    if not all_metrics:
        return {}

    metric_keys = sorted({k for metrics in all_metrics for k in metrics.keys()})
    means: Dict[str, float] = {}
    for key in metric_keys:
        values = [
            float(metrics[key])
            for metrics in all_metrics
            if key in metrics and metrics[key] is not None and np.isfinite(float(metrics[key]))
        ]
        if values:
            means[key] = float(np.mean(values))
    return means


def recompute_saved_metrics(
    data: np.ndarray,
    output_dir: str,
    n_runs: int,
    iterations: int = 100,
    batch_size: int = 128,
):
    """Recompute metrics from saved synthetic data without loading models or training."""
    root = Path(output_dir)
    raw = np.squeeze(np.nan_to_num(data)).astype(np.float64)
    case_table = []

    for ablation_name, _ in build_ablation_cases():
        case_dir = root / ablation_name
        run_metrics = []

        for run_idx in range(n_runs):
            run_dir = case_dir / f"run_{run_idx:02d}"
            generated_path = run_dir / "synthetic_generated_data.npz"
            if not generated_path.is_file():
                raise FileNotFoundError(
                    f"Missing saved generated data for {ablation_name} run {run_idx}: {generated_path}"
                )

            with np.load(generated_path) as generated_file:
                if "data" not in generated_file:
                    raise KeyError(f"No 'data' array found in {generated_path}")
                generated = np.squeeze(np.nan_to_num(generated_file["data"])).astype(np.float64)

            metrics = calculate_evaluation_metrics(
                raw, generated, iterations=iterations, batch_size=batch_size
            )
            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=float)
            run_metrics.append(metrics)
            print(f"Recomputed metrics: {ablation_name} run {run_idx:02d}")

        mean_metrics = aggregate_mean_metrics(run_metrics)
        case_table.append({"ablation": ablation_name, **mean_metrics})
        with open(case_dir / "mean_metrics.json", "w", encoding="utf-8") as f:
            json.dump(mean_metrics, f, indent=2, default=float)

    summary_df = pd.DataFrame(case_table)
    summary_df.to_csv(root / "ablation_summary_means.csv", index=False)
    with open(root / "ablation_summary_means.json", "w", encoding="utf-8") as f:
        json.dump(case_table, f, indent=2, default=float)

    print(f"\nMetrics-only run complete. Mean summary saved in {root}")
    columns = [
        key for key in ("ablation", "zero_ratio_diff", "mean_diff", "mmd", "autocorr_mae", "run_length_wasserstein", "ps", "ds", "lps", "lds")
        if key in summary_df.columns
    ]
    print(summary_df[columns].to_string(index=False))


def run_ablation_study(
    output_dir: str,
    dataset_name: str = "m5",
    n_runs: int = 3,
    latent_dim: int = 32,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 0.3,
    gate_weight: float = 5.0,
    recon_weight: float = 10.0,
    num_synthetic: int = 2000,
    seed: int = 42,
    metrics_only: bool = False,
    metric_iterations: int = 100,
    metric_batch_size: int = 128,
):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    data = load_m5_data(verbose=True)
    raw = np.squeeze(np.nan_to_num(data)).astype(np.float64)

    if metrics_only:
        recompute_saved_metrics(
            raw,
            output_dir=output_dir,
            n_runs=n_runs,
            iterations=metric_iterations,
            batch_size=metric_batch_size,
        )
        return

    case_table = []
    for ablation_name, _ in build_ablation_cases():
        run_metrics = []
        case_dir = root / ablation_name
        case_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(n_runs):
            run_dir = case_dir / f"run_{run_idx:02d}"
            metrics = run_single_ablation(
                data=raw,
                ablation_name=ablation_name,
                run_idx=run_idx,
                run_dir=run_dir,
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                beta=beta,
                gate_weight=gate_weight,
                recon_weight=recon_weight,
                num_synthetic=num_synthetic,
                seed=seed,
            )
            run_metrics.append(metrics)

        mean_metrics = aggregate_mean_metrics(run_metrics)
        case_table.append({"ablation": ablation_name, **mean_metrics})

        summary_path = case_dir / "mean_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(mean_metrics, f, indent=2, default=float)

        print(f"\nMean metrics for {ablation_name} saved to {summary_path}")

    summary_df = pd.DataFrame(case_table)
    output_csv = root / "ablation_summary_means.csv"
    summary_df.to_csv(output_csv, index=False)
    print(f"\nAblation summary saved to {output_csv}\n")
    print(summary_df[["ablation", "zero_ratio_diff", "mean_diff", "mmd", "autocorr_mae", "ps", "ds", "lps", "lds"]].to_string(index=False))

    # Save a compact JSON summary as well.
    with open(root / "ablation_summary_means.json", "w", encoding="utf-8") as f:
        json.dump(case_table, f, indent=2, default=float)


# ---------------------------------------------------------------------------
# Direct config variables: easier to edit and run quickly in a notebook/script.
# ---------------------------------------------------------------------------
OUTPUT_DIR = './results_m5_ablation'
METRICS_ONLY = True
N_RUNS = 3
LATENT_DIM = 32
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-3
BETA = 0.3
GATE_WEIGHT = 5.0
RECON_WEIGHT = 10.0
NUM_SYNTHETIC = 2000
SEED = 42
METRIC_ITERATIONS = 100
METRIC_BATCH_SIZE = 128


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    run_ablation_study(
        output_dir=OUTPUT_DIR,
        dataset_name='m5',
        n_runs=N_RUNS,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        beta=BETA,
        gate_weight=GATE_WEIGHT,
        recon_weight=RECON_WEIGHT,
        num_synthetic=NUM_SYNTHETIC,
        seed=SEED,
        metrics_only=METRICS_ONLY,
        metric_iterations=METRIC_ITERATIONS,
        metric_batch_size=METRIC_BATCH_SIZE,
    )
