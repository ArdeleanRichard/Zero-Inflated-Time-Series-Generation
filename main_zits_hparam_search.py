import csv
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from constants import device
from data_proc import load_iot_data, load_m5_data
from main_zits import (
    ZITS_VAE, ZITS_Generator, ZITS_Discriminator,
    train_vae, train_gan,
    _load_and_preprocess, _make_loaders,
)
from metrics import calculate_evaluation_metrics


# ===========================================================================
# CONFIG — edit this block to control the search
# ===========================================================================

DATA         = "iot"       # "iot" or "m5"
MODEL        = "both"      # "vae", "gan", or "both"
MODE         = "random"    # "random" (sample N_TRIALS configs) or "grid" (exhaustive)
N_TRIALS     = 30          # only used when MODE == "random"
NUM_EPOCHS   = 80          # training budget per config (use ~40 for a quick sweep)
NUM_SYNTHETIC = 5000       # synthetic samples generated for metric evaluation
RESUME       = True        # True = skip configs already present in the CSV
SEED         = 42

# ===========================================================================
# Hyperparameter grids — add / remove values freely
# ===========================================================================

VAE_GRID = {
    "latent_dim":   [32, 64, 128],
    "lr":           [1e-3, 5e-4, 1e-4],
    "beta":         [0.1, 0.3, 0.5, 1.0],
    "gate_weight":  [1.0, 5.0, 10.0],
    "recon_weight": [1.0, 5.0, 10.0, 20.0],
    "tc_weight":    [0.0, 0.5, 1.0, 2.0],
}

GAN_GRID = {
    "latent_dim":   [32, 64, 128],
    "lr":           [1e-4, 5e-5, 2e-4],
    "betas_0":      [0.0, 0.5],        # Adam β1 (β2 fixed at 0.9 per WGAN-GP)
    "gate_weight":  [1.0, 5.0, 10.0],
    "recon_weight": [1.0, 5.0, 10.0, 20.0],
    "tc_weight":    [0.0, 0.5, 1.0, 2.0],
    "fm_weight":    [0.5, 1.0, 2.0],
}

# ===========================================================================
# Internals — no need to edit below
# ===========================================================================

def _build_configs(grid, mode, n_trials):
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if mode == "random":
        combos = random.sample(combos, min(n_trials, len(combos)))
    return [dict(zip(keys, c)) for c in combos]


def _flat_key(cfg):
    return json.dumps(cfg, sort_keys=True)


def _load_existing_keys(csv_path):
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cfg_cols = {k: v for k, v in row.items()
                        if k not in ("run_id", "model", "data", "elapsed_s", "status", "error")
                        and not k.startswith("metric__")}
            normalised = {}
            for k, v in cfg_cols.items():
                try:
                    normalised[k] = float(v) if "." in v else int(v)
                except (ValueError, TypeError):
                    normalised[k] = v
            keys.add(_flat_key(normalised))
    return keys


def _append_row(csv_path, fieldnames, row):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _flatten_metrics(metrics):
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}__{kk}"] = vv
        else:
            flat[k] = v
    return flat


def _generate_samples(model, preprocessor, num_synthetic):
    model.eval()
    with torch.no_grad():
        norm_samples = model.sample(num_synthetic).cpu().numpy()
    return preprocessor.inverse_transform(norm_samples)


def run_vae(data, ori_data, cfg):
    raw, proc, pp = _load_and_preprocess(data, ori_data)
    seq_len = proc.shape[1]
    train_loader, val_loader = _make_loaders(proc)

    model     = ZITS_VAE(seq_length=seq_len, latent_dim=cfg["latent_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    train_vae(model, train_loader, val_loader, optimizer,
              num_epochs=NUM_EPOCHS,
              beta=cfg["beta"],
              gate_weight=cfg["gate_weight"],
              recon_weight=cfg["recon_weight"],
              tc_weight=cfg["tc_weight"])

    gen_data = _generate_samples(model, pp, NUM_SYNTHETIC)
    ori_sq   = np.squeeze(np.nan_to_num(ori_data).astype(np.float64))
    gen_sq   = np.squeeze(np.nan_to_num(gen_data).astype(np.float64))
    return calculate_evaluation_metrics(ori_sq, gen_sq)


def run_gan(data, ori_data, cfg):
    raw, proc, pp = _load_and_preprocess(data, ori_data)
    seq_len = proc.shape[1]
    train_loader, val_loader = _make_loaders(proc)

    betas         = (cfg["betas_0"], 0.9)
    generator     = ZITS_Generator(seq_length=seq_len, latent_dim=cfg["latent_dim"]).to(device)
    discriminator = ZITS_Discriminator(seq_length=seq_len).to(device)
    g_opt = optim.Adam(generator.parameters(),     lr=cfg["lr"], betas=betas)
    d_opt = optim.Adam(discriminator.parameters(), lr=cfg["lr"], betas=betas)

    train_gan(generator, discriminator, train_loader,
              g_opt, d_opt,
              num_epochs=NUM_EPOCHS,
              gate_weight=cfg["gate_weight"],
              recon_weight=cfg["recon_weight"],
              fm_weight=cfg["fm_weight"],
              tc_weight=cfg["tc_weight"],
              n_critic=5,
              lambda_gp=10.0)

    gen_data = _generate_samples(generator, pp, NUM_SYNTHETIC)
    ori_sq   = np.squeeze(np.nan_to_num(ori_data).astype(np.float64))
    gen_sq   = np.squeeze(np.nan_to_num(gen_data).astype(np.float64))
    return calculate_evaluation_metrics(ori_sq, gen_sq)


def search():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ori_data = load_iot_data() if DATA == "iot" else load_m5_data()

    results_dir = Path(f"./hparam_search/{DATA}/")
    results_dir.mkdir(exist_ok=True)

    model_types = ["vae", "gan"] if MODEL == "both" else [MODEL]

    for mtype in model_types:
        grid    = VAE_GRID   if mtype == "vae" else GAN_GRID
        run_fn  = run_vae    if mtype == "vae" else run_gan
        configs = _build_configs(grid, MODE, N_TRIALS)
        csv_path = results_dir / f"hparam_search_{DATA}_{mtype}.csv"

        existing_keys = _load_existing_keys(csv_path) if RESUME else set()
        print(f"\n{'='*60}")
        print(f"ZITS-{mtype.upper()}  |  dataset={DATA}  |  mode={MODE}")
        print(f"Configs: {len(configs)}   Already done: {len(existing_keys)}")
        print(f"CSV: {csv_path}")
        print(f"{'='*60}")

        fieldnames = None

        for run_id, cfg in enumerate(configs):
            cfg_key = _flat_key(cfg)
            if cfg_key in existing_keys:
                print(f"  [skip] run {run_id:04d}")
                continue

            print(f"\n  [run {run_id:04d}/{len(configs)-1}]  {cfg}")
            t0 = time.time()

            row = {"run_id": run_id, "model": f"zits-{mtype}", "data": DATA}
            row.update(cfg)

            try:
                metrics  = run_fn(DATA, ori_data, cfg)
                flat_m   = _flatten_metrics(metrics)
                elapsed  = time.time() - t0
                row["elapsed_s"] = f"{elapsed:.1f}"
                row["status"]    = "ok"
                for mk, mv in flat_m.items():
                    row[f"metric__{mk}"] = mv
                print(f"     done in {elapsed:.0f}s   {list(flat_m.items())[:3]} ...")
            except Exception as exc:
                elapsed = time.time() - t0
                row["elapsed_s"] = f"{elapsed:.1f}"
                row["status"]    = "error"
                row["error"]     = str(exc)[:300]
                print(f"     ERROR: {exc}")

            if fieldnames is None:
                fieldnames = list(row.keys())

            _append_row(csv_path, fieldnames, row)
            existing_keys.add(cfg_key)

        print(f"\nDone. Results in {csv_path}")


if __name__ == "__main__":
    search()