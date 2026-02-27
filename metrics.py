import numpy as np
import torch
from scipy.stats import wasserstein_distance
import json
from typing import Dict
from scipy import stats


from metrics_advanced import predictive_score_metrics, discriminative_score_metrics, long_predictive_score_metrics, long_discriminative_score_metrics

def kl_divergence_from_samples(a, b, bins=50, eps=1e-10):
    """
    KL(p||q) estimator from samples a and b using histogram counts.
    - a, b: 1D arrays or flattenable arrays of samples
    - bins: number of histogram bins (or pass array of bin edges)
    - eps: additive smoothing constant (small positive)
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    # drop NaN/inf
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    # if either empty after cleaning, return 0.0 (no divergence information)
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0:
        # all mass in b; return finite large-ish value or compute smoothed KL with tiny uniform p
        a = np.array([0.0])
    if b.size == 0:
        b = np.array([0.0])

    # choose common bin edges covering both arrays
    amin = float(np.min(np.concatenate([a, b])))
    amax = float(np.max(np.concatenate([a, b])))
    if amin == amax:
        amin -= 0.5
        amax += 0.5
    # create edges if bins is integer
    if isinstance(bins, int):
        bin_edges = np.linspace(amin, amax, bins + 1)
    else:
        bin_edges = np.asarray(bins)

    # histogram counts (not density)
    hist_a, _ = np.histogram(a, bins=bin_edges, density=False)
    hist_b, _ = np.histogram(b, bins=bin_edges, density=False)

    # convert to probabilities with smoothing
    p = hist_a.astype(float) + eps
    q = hist_b.astype(float) + eps

    # renormalize
    p /= p.sum()
    q /= q.sum()

    # KL(p || q) using scipy (base e)
    kl = float(stats.entropy(p, q))
    return kl


def calculate_evaluation_metrics(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict:
    """
    Calculate comprehensive evaluation metrics for synthetic time series.

    Metrics based on:
    1. Esteban et al. (2017) "Real-valued (Medical) Time Series Generation with RNNs"
    2. Yoon et al. (2019) "Time-series Generative Adversarial Networks"
    3. Norinder et al. (2021) "Introducing Conformal Prediction in Predictive Modeling"
    """
    metrics = {}

    # 1. Distributional metrics
    # Zero ratio comparison
    real_zero_ratio = np.mean(real_data == 0)
    synth_zero_ratio = np.mean(synthetic_data == 0)
    metrics['zero_ratio_real'] = real_zero_ratio
    metrics['zero_ratio_synthetic'] = synth_zero_ratio
    metrics['zero_ratio_diff'] = abs(real_zero_ratio - synth_zero_ratio)

    # Non-zero statistics
    real_nonzero = real_data[real_data > 0]
    synth_nonzero = synthetic_data[synthetic_data > 0]

    if len(real_nonzero) > 0 and len(synth_nonzero) > 0:
        # Wasserstein distance (Earth Mover's Distance)
        # Lower is better, measures distribution similarity
        metrics['wasserstein_distance'] = wasserstein_distance(real_nonzero.flatten(), synth_nonzero.flatten())

        # KL divergence approximation
        try:
            # hist_real, bins = np.histogram(real_nonzero, bins=50, density=True)
            # hist_synth, _ = np.histogram(synth_nonzero, bins=bins, density=True)
            # hist_real = hist_real + 1e-10  # Avoid log(0)
            # hist_synth = hist_synth + 1e-10
            # metrics['kl_divergence'] = np.sum(hist_real * np.log(hist_real / hist_synth))
            metrics['kl_divergence'] = kl_divergence_from_samples(real_nonzero.flatten(), synth_nonzero.flatten(), bins=50, eps=1e-10)

        except:
            metrics['kl_divergence'] = np.nan

    # 2. Statistical moments
    metrics['mean_real'] = np.mean(real_data)
    metrics['mean_synthetic'] = np.mean(synthetic_data)
    metrics['mean_diff'] = abs(np.mean(real_data) - np.mean(synthetic_data))

    metrics['std_real'] = np.std(real_data)
    metrics['std_synthetic'] = np.std(synthetic_data)
    metrics['std_diff'] = abs(np.std(real_data) - np.std(synthetic_data))

    metrics['skewness_real'] = stats.skew(real_data.flatten())
    metrics['skewness_synthetic'] = stats.skew(synthetic_data.flatten())
    metrics['skewness_diff'] = abs(stats.skew(real_data.flatten()) - stats.skew(synthetic_data.flatten()))

    metrics['kurtosis_real'] = stats.kurtosis(real_data.flatten())
    metrics['kurtosis_synthetic'] = stats.kurtosis(synthetic_data.flatten())
    metrics['kurtosis_diff'] = abs(stats.kurtosis(real_data.flatten()) - stats.kurtosis(synthetic_data.flatten()))

    # 3. Temporal metrics
    # Autocorrelation at different lags
    def compute_autocorr(data, max_lag=10):
        autocorrs = []
        for lag in range(1, max_lag + 1):
            correlations = []
            for series in data:
                if len(series) > lag:
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            if correlations:
                autocorrs.append(np.mean(correlations))
        return np.array(autocorrs)

    real_autocorr = compute_autocorr(real_data)
    synth_autocorr = compute_autocorr(synthetic_data)
    # Align lengths in case one array is shorter/empty (e.g. all-NaN from constant sequences)
    min_len = min(len(real_autocorr), len(synth_autocorr))
    if min_len == 0:
        metrics['autocorr_mae'] = np.nan
    else:
        metrics['autocorr_mae'] = np.mean(np.abs(real_autocorr[:min_len] - synth_autocorr[:min_len]))

    # 4. Quantile comparison
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_diffs = []
    for q in quantiles:
        real_q = np.quantile(real_data, q)
        synth_q = np.quantile(synthetic_data, q)
        quantile_diffs.append(abs(real_q - synth_q))
        metrics[f'quantile_{int(q * 100)}_diff'] = abs(real_q - synth_q)

    metrics['mean_quantile_diff'] = np.mean(quantile_diffs)

    # 5. Maximum Mean Discrepancy (MMD)
    # Simplified RBF kernel-based MMD
    def compute_mmd(x, y, kernel='rbf', gamma=1.0):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        xx = np.dot(x, x.T)
        yy = np.dot(y, y.T)
        xy = np.dot(x, y.T)

        rx = np.diag(xx).reshape(-1, 1)
        ry = np.diag(yy).reshape(-1, 1)

        K_xx = np.exp(-gamma * (rx + rx.T - 2 * xx))
        K_yy = np.exp(-gamma * (ry + ry.T - 2 * yy))
        K_xy = np.exp(-gamma * (rx + ry.T - 2 * xy))

        return np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)

    # Sample for computational efficiency
    sample_size = min(1000, len(real_data), len(synthetic_data))
    real_sample = real_data[np.random.choice(len(real_data), sample_size, replace=False)]
    synth_sample = synthetic_data[np.random.choice(len(synthetic_data), sample_size, replace=False)]

    metrics['mmd'] = compute_mmd(real_sample, synth_sample)

    min_len = min(len(real_data), len(synthetic_data))
    real_data = real_data[:min_len]
    synthetic_data = synthetic_data[:min_len]

    real_data = real_data[..., np.newaxis]
    synthetic_data = synthetic_data[..., np.newaxis]

    real_data = torch.from_numpy(real_data).float()
    synthetic_data = torch.from_numpy(synthetic_data).float()

    metrics['ps'] = predictive_score_metrics(real_data, synthetic_data, iterations=10, batch_size=128)
    metrics['ds'] = discriminative_score_metrics(real_data, synthetic_data, iterations=10, batch_size=128)
    subset = 10000
    metrics['lps'] = long_predictive_score_metrics(real_data[-subset:], synthetic_data[-subset:], iterations=10, batch_size=128)
    metrics['lds'] = long_discriminative_score_metrics(real_data[-subset:], synthetic_data[-subset:], iterations=10, batch_size=128)

    return metrics


def print_evaluation_metrics(metrics: Dict):
    """Print metrics in organized format"""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    print("\n1. SPARSITY METRICS")
    print(f"   Zero Ratio (Real):      {metrics['zero_ratio_real']:.4f}")
    print(f"   Zero Ratio (Synthetic): {metrics['zero_ratio_synthetic']:.4f}")
    print(f"   Difference:             {metrics['zero_ratio_diff']:.4f}")

    print("\n2. DISTRIBUTIONAL SIMILARITY")
    print(f"   Wasserstein Distance:   {metrics.get('wasserstein_distance', 'N/A')}")
    print(f"   KL Divergence:          {metrics.get('kl_divergence', 'N/A')}")
    print(f"   MMD:                    {metrics['mmd']:.6f}")

    print("\n3. STATISTICAL MOMENTS")
    print(f"   Mean (Real):            {metrics['mean_real']:.2f}")
    print(f"   Mean (Synthetic):       {metrics['mean_synthetic']:.2f}")
    print(f"   Mean Difference:        {metrics['mean_diff']:.2f}")
    print(f"   Std (Real):             {metrics['std_real']:.2f}")
    print(f"   Std (Synthetic):        {metrics['std_synthetic']:.2f}")
    print(f"   Std Difference:         {metrics['std_diff']:.2f}")

    print("\n4. HIGHER MOMENTS")
    print(f"   Skewness Difference:    {metrics['skewness_diff']:.4f}")
    print(f"   Kurtosis Difference:    {metrics['kurtosis_diff']:.4f}")

    print("\n5. TEMPORAL CHARACTERISTICS")
    print(f"   Autocorrelation MAE:    {metrics['autocorr_mae']:.4f}")

    print("\n6. QUANTILE COMPARISON")
    print(f"   Mean Quantile Diff:     {metrics['mean_quantile_diff']:.2f}")
    print("=" * 70 + "\n")

    print("\n7. ADVANCED METRICS")
    print(f"   Predictive Score:            {metrics['ps']:.2f}")
    print(f"   Discriminative Score:        {metrics['ds']:.2f}")
    print(f"   Long Predictive Score:       {metrics['lps']:.2f}")
    print(f"   Long Discriminative Score:   {metrics['lds']:.2f}")
    print("=" * 70 + "\n")



def save_metrics_report(metrics: Dict, save_path: str = 'metrics_report.json'):
    """Save metrics to JSON file"""

    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif obj is None or isinstance(obj, (bool, str, int, float)):
            return obj
        else:
            return str(obj)  # Fallback for any other type

    serializable_metrics = convert_to_native(metrics)

    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f'Metrics report saved to {save_path}')