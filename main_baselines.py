"""
Traditional statistical baselines for zero-inflated time series generation.

Two models are implemented:
1. ZIP  – Zero-Inflated Poisson
   Each timestep t is modelled as a mixture:
       P(x_t = 0)     = pi_t + (1 - pi_t) * exp(-lambda_t)
       P(x_t = k > 0) = (1 - pi_t) * Poisson(k; lambda_t)
   Parameters (pi_t, lambda_t) are estimated independently per timestep via
   an EM algorithm.  Intended for *count* data (e.g. M5 sales).

2. Hurdle (two-process model)
   Each timestep t is modelled as:
       P(x_t = 0)   = pi_t                        (Bernoulli gate)
       P(x_t | x>0) = LogNormal(mu_t, sigma_t)    (positive-part model)
   Parameters are estimated independently per timestep via MLE.
   Intended for *continuous* non-negative data (e.g. IoT durations),
   but also works for count data.

"""

import os
import numpy as np

from data_proc import load_iot_data, load_m5_data



# ===========================================================================
# Helper: per-timestep EM for Zero-Inflated Poisson
# ===========================================================================

def _zip_em_one(x: np.ndarray, max_iter: int = 200, tol: float = 1e-6):
    """
    Fit a Zero-Inflated Poisson to a 1-D array x (one timestep across all
    samples) using the EM algorithm.

    Returns
    -------
    pi  : float  – probability of a structural zero
    lam : float  – Poisson rate of the non-structural component
    """
    x = x.astype(np.float64)
    n = len(x)
    zero_mask = (x == 0)
    n_zero    = zero_mask.sum()

    # Degenerate case: all observations are zero
    if n_zero == n:
        return 1.0, 1e-9

    # Initialisation: treat half of the observed zeros as structural
    pi  = n_zero / n * 0.5
    lam = x[~zero_mask].mean()
    lam = max(lam, 1e-9)

    for _ in range(max_iter):
        pi_old, lam_old = pi, lam

        # E-step: posterior probability that each zero is structural
        p_zero_poisson = np.exp(-lam)
        denom          = pi + (1.0 - pi) * p_zero_poisson
        w_struct       = np.where(zero_mask, pi / np.maximum(denom, 1e-300), 0.0)

        # M-step: update pi and lambda
        pi  = w_struct.sum() / n
        pi  = np.clip(pi, 1e-9, 1.0 - 1e-9)
        lam = (x * (1.0 - w_struct)).sum() / max((1.0 - w_struct).sum(), 1e-9)
        lam = max(lam, 1e-9)

        if abs(pi - pi_old) < tol and abs(lam - lam_old) < tol:
            break

    return float(pi), float(lam)


# ===========================================================================
# Helper: per-timestep MLE for Hurdle (LogNormal positive part)
# ===========================================================================

def _hurdle_mle_one(x: np.ndarray):
    """
    Fit a Hurdle model to a 1-D array x (one timestep across all samples).

    Gate:          pi  = P(x == 0), estimated by sample proportion (MLE)
    Positive part: LogNormal(mu, sigma), estimated by MLE on x[x > 0]

    Returns
    -------
    pi    : float – probability of zero
    mu    : float – mean of log(x) for x > 0
    sigma : float – std  of log(x) for x > 0
    """
    x = x.astype(np.float64)
    n = len(x)

    pi = float((x == 0).sum()) / n
    pi = np.clip(pi, 1e-9, 1.0 - 1e-9)

    pos = x[x > 0]
    if len(pos) == 0:
        return pi, 0.0, 1.0   # fallback: degenerate all-zero column

    log_pos = np.log(np.maximum(pos, 1e-300))
    mu      = float(log_pos.mean())
    sigma   = float(log_pos.std())
    sigma   = max(sigma, 1e-6)   # prevent degenerate point-mass distribution

    return pi, mu, sigma


class ZIPBaseline:
    """
    Zero-Inflated Poisson (ZIP) baseline.

    Fits independent ZIP(pi_t, lambda_t) distributions for each of the T
    timesteps by EM on the raw data, then draws i.i.d. samples column-by-column.

    Operates on raw count data — do NOT pass pre-normalised values.
    Best suited for non-negative integer (count) data (e.g. M5 sales).
    For continuous data consider HurdleBaseline instead.
    """

    def __init__(self, max_em_iter: int = 200, em_tol: float = 1e-6):
        self.max_em_iter = max_em_iter
        self.em_tol      = em_tol
        self.pi_  : np.ndarray | None = None   # (T,) structural-zero probabilities
        self.lam_ : np.ndarray | None = None   # (T,) Poisson rates

    def fit(self, ori_data: np.ndarray) -> "ZIPBaseline":
        """
        Parameters
        ----------
        ori_data : np.ndarray, shape (N, T)
            Raw time series matrix (rows = samples, columns = timesteps).
        """
        ori_data = np.asarray(ori_data, dtype=np.float64)
        N, T = ori_data.shape
        print(f"[ZIP] Fitting {T} timesteps on {N} samples …")

        pi  = np.zeros(T)
        lam = np.zeros(T)

        for t in range(T):
            pi[t], lam[t] = _zip_em_one(
                ori_data[:, t], max_iter=self.max_em_iter, tol=self.em_tol)
            if (t + 1) % max(1, T // 10) == 0:
                print(f"  … timestep {t+1:4d}/{T}  "
                      f"pi={pi[t]:.3f}  lambda={lam[t]:.3f}")

        self.pi_  = pi
        self.lam_ = lam
        print("[ZIP] Fitting complete.")
        return self

    def sample(self, n: int) -> np.ndarray:
        """
        Draw n synthetic samples.

        Returns
        -------
        gen_data : np.ndarray, shape (n, T)  – raw (un-normalised) values
        """
        if self.pi_ is None:
            raise RuntimeError("Call fit() before sample().")

        T        = len(self.pi_)
        gen_data = np.zeros((n, T), dtype=np.float64)

        for t in range(T):
            struct_zero    = np.random.rand(n) < self.pi_[t]
            poisson_val    = np.random.poisson(self.lam_[t], size=n).astype(np.float64)
            gen_data[:, t] = np.where(struct_zero, 0.0, poisson_val)

        zero_ratio = (gen_data == 0).mean()
        print(f"[ZIP] Generated {n} samples  |  zero ratio = {zero_ratio:.3f}")
        return gen_data

    def save_samples(self, n: int, path: str) -> np.ndarray:
        """Generate n samples and save to *path* as .npz with key 'data'."""
        gen_data = self.sample(n)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, data=gen_data)
        print(f"[ZIP] Saved → {path}  shape={gen_data.shape}")
        return gen_data



class HurdleBaseline:
    """
    Two-process / Hurdle model baseline.

    For each timestep t:
        P(x_t = 0)       = pi_t                         (Bernoulli gate, MLE)
        P(x_t | x_t > 0) = LogNormal(mu_t, sigma_t)     (MLE on positive values)

    Operates on raw data — do NOT pass pre-normalised values.
    The LogNormal positive component is well-suited for heavy-tailed continuous
    data (IoT durations). It also works for count data as an approximation when
    counts are large; for strictly integer counts the ZIP model is more principled.
    """

    def __init__(self):
        self.pi_    : np.ndarray | None = None   # (T,) gate probabilities P(x=0)
        self.mu_    : np.ndarray | None = None   # (T,) log-mean  of positive part
        self.sigma_ : np.ndarray | None = None   # (T,) log-sigma of positive part

    def fit(self, ori_data: np.ndarray) -> "HurdleBaseline":
        """
        Parameters
        ----------
        ori_data : np.ndarray, shape (N, T)
            Raw time series matrix (rows = samples, columns = timesteps).
        """
        ori_data = np.asarray(ori_data, dtype=np.float64)
        N, T = ori_data.shape
        print(f"[Hurdle] Fitting {T} timesteps on {N} samples …")

        pi    = np.zeros(T)
        mu    = np.zeros(T)
        sigma = np.ones(T)

        for t in range(T):
            pi[t], mu[t], sigma[t] = _hurdle_mle_one(ori_data[:, t])
            if (t + 1) % max(1, T // 10) == 0:
                print(f"  … timestep {t+1:4d}/{T}  "
                      f"pi={pi[t]:.3f}  mu={mu[t]:.3f}  sigma={sigma[t]:.3f}")

        self.pi_    = pi
        self.mu_    = mu
        self.sigma_ = sigma
        print("[Hurdle] Fitting complete.")
        return self

    def sample(self, n: int) -> np.ndarray:
        """
        Draw n synthetic samples.

        Returns
        -------
        gen_data : np.ndarray, shape (n, T)  – raw (un-normalised) values
        """
        if self.pi_ is None:
            raise RuntimeError("Call fit() before sample().")

        T        = len(self.pi_)
        gen_data = np.zeros((n, T), dtype=np.float64)

        for t in range(T):
            is_zero        = np.random.rand(n) < self.pi_[t]
            pos_vals       = np.random.lognormal(mean=self.mu_[t], sigma=self.sigma_[t], size=n)
            gen_data[:, t] = np.where(is_zero, 0.0, pos_vals)

        zero_ratio = (gen_data == 0).mean()
        print(f"[Hurdle] Generated {n} samples  |  zero ratio = {zero_ratio:.3f}")
        return gen_data

    def save_samples(self, n: int, path: str) -> np.ndarray:
        """Generate n samples and save to *path* as .npz with key 'data'."""
        gen_data = self.sample(n)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, data=gen_data)
        print(f"[Hurdle] Saved → {path}  shape={gen_data.shape}")
        return gen_data


# ===========================================================================
# Convenience runners (mirroring main_zits.py structure)
# ===========================================================================

def run_baselines_iot(num_synthetic: int = 50_000):
    """Fit and generate baselines for the IoT dataset."""
    ori_data = load_iot_data().astype(np.float64)
    print(f"\nIoT original data shape: {ori_data.shape}")
    os.makedirs(OUT_IOT, exist_ok=True)

    ZIPBaseline().fit(ori_data).save_samples(num_synthetic, f"{OUT_IOT}/zip_generated_data.npz")

    HurdleBaseline().fit(ori_data).save_samples(num_synthetic, f"{OUT_IOT}/hurdle_generated_data.npz")


def run_baselines_m5(num_synthetic: int = 30_000):
    """Fit and generate baselines for the M5 dataset."""
    ori_data = load_m5_data().astype(np.float64)
    print(f"\nM5 original data shape: {ori_data.shape}")
    os.makedirs(OUT_M5, exist_ok=True)

    ZIPBaseline().fit(ori_data).save_samples(num_synthetic, f"{OUT_M5}/zip_generated_data.npz")

    HurdleBaseline().fit(ori_data).save_samples(num_synthetic, f"{OUT_M5}/hurdle_generated_data.npz")



if __name__ == "__main__":
    OUT_IOT = "./out_iot/baseline/"
    OUT_M5 = "./out_m5/baseline/"

    run_baselines_iot(num_synthetic=50_000)
    # run_baselines_m5(num_synthetic=30_000)