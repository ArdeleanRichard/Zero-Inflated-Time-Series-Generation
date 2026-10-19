import glob
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import json

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from constants import MAP_MODEL_NAMES
from metrics import (calculate_evaluation_metrics, print_evaluation_metrics, save_metrics_report)


def plot_sample_comparisons(real_samples, synthetic_samples,num_samples = 5, save_path = 'sample_comparison.png'):
    """Plot comparison between real and synthetic samples"""
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        # Real sample
        axes[i, 0].plot(real_samples[i], linewidth=1, alpha=0.7)
        axes[i, 0].set_title(f'Real Sample {i + 1}')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)

        # Synthetic sample
        axes[i, 1].plot(synthetic_samples[i], linewidth=1, alpha=0.7, color='orange')
        axes[i, 1].set_title(f'Synthetic Sample {i + 1}')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time Step')
    axes[-1, 1].set_xlabel('Time Step')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Sample comparison saved to {save_path}')

def visualization_dim_red(model, ori_data, gen_data, analysis="tsne", save_path=None):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    print("\nDIM REDUCED PLOTTING AND SAVING")
    # Analysis sample size (for faster computation)

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    gen_data = np.asarray(gen_data)

    # Visualization parameter
    colors = ["red" for i in range(len(ori_data))] + ["blue" for i in range(len(gen_data))]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(ori_data)
        pca_results_ori = pca.transform(ori_data)
        pca_results_gen = pca.transform(gen_data)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results_ori[:, 0], pca_results_ori[:, 1], c=colors[:len(ori_data)], alpha=0.2, label="Original")
        plt.scatter(pca_results_gen[:, 0], pca_results_gen[:, 1], c=colors[len(ori_data):], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title(f'{MAP_MODEL_NAMES[model]}')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    elif analysis == 'umap':
        # TSNE anlaysis
        embedding = umap.UMAP(n_components=2, random_state=42)
        embedding.fit(ori_data)
        umap_results_ori = embedding.transform(ori_data)
        umap_results_gen = embedding.transform(gen_data)

        f, ax = plt.subplots(1)
        plt.scatter(umap_results_ori[:, 0], umap_results_ori[:, 1], c=colors[:len(ori_data)], alpha=0.2, label="Original")
        plt.scatter(umap_results_gen[:, 0], umap_results_gen[:, 1], c=colors[len(ori_data):], alpha=0.2, label="Synthetic")


        # tsne_results = tsne.fit_transform(prep_data_final)
        #
        # # Plotting
        # f, ax = plt.subplots(1)
        #
        # plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1], c=colors[:anal_sample_no], alpha=0.2, label="Original")
        # plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1], c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title(f'{MAP_MODEL_NAMES[model]}')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def _run_length_counts(data):
    """Count contiguous zero and nonzero runs across a collection of series."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError(f"Expected time series data with 1 or 2 dimensions, got shape {data.shape}")

    counts = {
        "zero": np.zeros(data.shape[1] + 1, dtype=np.int64),
        "nonzero": np.zeros(data.shape[1] + 1, dtype=np.int64),
    }
    for series in data:
        if series.size == 0:
            continue
        is_zero = series == 0
        boundaries = np.flatnonzero(np.diff(is_zero.astype(np.int8))) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [series.size]))
        run_lengths = ends - starts
        run_types = np.where(is_zero[starts], "zero", "nonzero")
        for run_type in ("zero", "nonzero"):
            counts[run_type] += np.bincount(
                run_lengths[run_types == run_type],
                minlength=data.shape[1] + 1,
            )
    return {run_type: run_counts[1:] for run_type, run_counts in counts.items()}


def _run_length_records(dataset, model, counts):
    records = []
    for segment, run_counts in counts.items():
        total_runs = int(run_counts.sum())
        if total_runs == 0:
            continue
        for run_length in np.flatnonzero(run_counts) + 1:
            count = int(run_counts[run_length - 1])
            records.append({
                "dataset": dataset,
                "model": model,
                "segment": segment,
                "run_length": int(run_length),
                "count": count,
                "proportion": count / total_runs,
            })
    return records


def _plot_run_length_cdfs(model, real_counts, synthetic_counts, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, segment in zip(axes, ("zero", "nonzero")):
        for label, counts in (("Real", real_counts[segment]),
                              ("Synthetic", synthetic_counts[segment])):
            total_runs = counts.sum()
            if total_runs == 0:
                continue
            run_lengths = np.flatnonzero(counts) + 1
            cumulative_proportion = np.cumsum(counts[counts > 0]) / total_runs
            x_values = np.concatenate(([0.5], run_lengths, [run_lengths[-1] + 1]))
            y_values = np.concatenate(([0.0], cumulative_proportion, [1.0]))
            ax.step(x_values, y_values, where="post", label=label)

        ax.set_xscale("log")
        ax.set_xlabel("Run length (time steps)")
        ax.set_title(f"{segment.capitalize()} runs")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()

    axes[0].set_ylabel("Cumulative proportion of runs")
    fig.suptitle(f"{MAP_MODEL_NAMES[model]}: run-length distributions")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_data(DATA, np_ori_data):

    results_dir = f"results_{DATA}"

    ori_data = np_ori_data['data']
    print(f"Original data shape: {ori_data.shape}")
    ori_data = np.squeeze(ori_data)
    ori_data = np.nan_to_num(ori_data).astype(np.float64)
    print(f"Original data shape: {ori_data.shape}")

    run_length_dir = os.path.join(results_dir, "run_length_distributions")
    os.makedirs(run_length_dir, exist_ok=True)
    real_run_counts = _run_length_counts(ori_data)
    run_length_records = _run_length_records(DATA, "real", real_run_counts)

    for model in ["zip", "hurdle", "zits-gan", "zits-vae", "timegan", "transfusion", "fide", "chronogan", "tsgm_timegan", "tsgm_vae", "vae_dense", "vae_conv", "timeVAE"]:

        print(f"="*60)
        print(f"Model: {model}")
        print(f"="*60)

        if model == "chronogan":
            np_gen_data = np.load(f'./repos/ChronoGAN/output_{DATA}/synthetic_data.npz', allow_pickle=True)
        elif model == "fide":
            np_gen_data = np.load(f'./repos/FIDE/Code/{DATA}_model/fide_generated_data.npz', allow_pickle=True)
        elif model == "timegan":
            np_gen_data = np.load(f'./repos/TimeGAN/out/{DATA}_{model}_generated_data.npz', allow_pickle=True)
        elif model == "timeVAE" or model == "vae_dense" or model == "vae_conv":
            if DATA == "iot":
                data_name = "iot_durations_2021"
                np_gen_data = np.load(f'./repos/timeVAE/outputs/gen_data/{data_name}/{model}_{data_name}_prior_samples.npz')
            elif DATA == "m5":
                data_name = "m5_X_365"
                np_gen_data = np.load(f'./repos/timeVAE/outputs/gen_data/{data_name}/{model}_{data_name}_prior_samples.npz')
        elif model == "transfusion":
            if DATA == "iot":
                np_gen_data = np.load(f'./repos/TransFusion/saved_files/1770796660.6585-custom-transformers-{DATA}-l1-cosine-365-pred_v/test_outputs/synthetic_data_1770809439.npz', allow_pickle=True)
            elif DATA == "m5":
                np_gen_data = np.load(
                    f'./repos/TransFusion/saved_files/1771403691.0601-custom-transformers-{DATA}-l1-cosine-365-pred_v/test_outputs/synthetic_data_1771411195.npz', allow_pickle=True)
        elif model == "tsgm_timegan" or model == "tsgm_vae":
            np_gen_data = np.load(f'./repos/tsgm/{DATA}_results/{model}_gen.npz')
        elif model == "zits-gan":
            np_gen_data = np.load(f'./results_{DATA}/saved/vae_generated_data.npz', allow_pickle=True)
        elif model == "zits-vae":
            np_gen_data = np.load(f'./results_{DATA}/saved/gan_generated_data.npz', allow_pickle=True)

        # baselines
        elif model == "zip":
            np_gen_data = np.load(f'./results_{DATA}/baseline/zip_generated_data.npz')
        elif model == "hurdle":
            np_gen_data = np.load(f'./results_{DATA}/baseline/hurdle_generated_data.npz')

        gen_data = np_gen_data['data']
        print(f"Generated data shape: {gen_data.shape}")
        gen_data = np.squeeze(gen_data)
        gen_data = np.nan_to_num(gen_data).astype(np.float64)
        print(f"Generated data shape: {gen_data.shape}")

        synthetic_run_counts = _run_length_counts(gen_data)
        run_length_records.extend(_run_length_records(DATA, model, synthetic_run_counts))
        _plot_run_length_cdfs(
            model,
            real_run_counts,
            synthetic_run_counts,
            os.path.join(run_length_dir, f"{model}_run_length_cdf.png"),
        )

        r_idx = np.random.randint(0, min(len(ori_data), len(gen_data)), size=5)
        plot_sample_comparisons(ori_data[r_idx], gen_data[r_idx], save_path=f"./{results_dir}/plot_samples/{model}_sample_comparison.png")

        metrics = calculate_evaluation_metrics(ori_data, gen_data)
        # print_evaluation_metrics(metrics)
        save_metrics_report(metrics, f"./{results_dir}/{model}_metrics.json")

        # visualization_dim_red(model, ori_data, gen_data, "umap", f"./{results_dir}/plot_embeddings/{model}_plot_umap.png")
        # visualization_dim_red(model, ori_data, gen_data, "pca", f"./{results_dir}/plot_embeddings/{model}_plot_pca.png")

    pd.DataFrame(run_length_records).to_csv(
        os.path.join(results_dir, "run_length_distributions.csv"),
        index=False,
    )


def analyze_iot():
    DATA = "iot"
    np_ori_data = np.load(f'./data/{DATA}_durations_2021.npz')
    analyze_data(DATA, np_ori_data)

def analyze_m5():
    DATA = "m5"
    np_ori_data = np.load(f'./data/{DATA}/{DATA}_X_365.npz')
    analyze_data(DATA, np_ori_data)



def aggregate_results(DATA=""):
    RESULTS_DIR = f"./results_{DATA}/"
    OUTPUT_CSV = RESULTS_DIR + "all.csv"
    GLOB_PATTERN = os.path.join(RESULTS_DIR, "*_metrics.json")

    files = sorted(glob.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"No files found matching pattern: {GLOB_PATTERN}")

    models_data = OrderedDict()  # preserve model (file) order
    metrics_order = []  # list preserving first-seen metric order
    seen_metrics = set()

    for path in files:
        filename = os.path.basename(path)
        if not filename.endswith("_metrics.json"):
            continue
        model_name = filename[:-len("_metrics.json")]
        # load JSON with OrderedDict to preserve key order within each file
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            if not isinstance(data, dict):
                raise ValueError(f"{path} does not contain a JSON object at top level.")
            models_data[model_name] = data
            # add metrics in file order, but only if not already seen
            for k in data.keys():
                if k not in seen_metrics:
                    seen_metrics.add(k)
                    metrics_order.append(k)

    # Build DataFrame with 'metric' as first column in preserved order
    df = pd.DataFrame({"metric": metrics_order})

    # Helper to stringify non-scalar values (dict/list) so CSV remains readable
    def normalize_value(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return v

    # Add columns per model, preserving model order
    for model, md in models_data.items():
        df[model] = df["metric"].map(lambda m: normalize_value(md.get(m)))

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {OUTPUT_CSV} with {len(metrics_order)} metrics and {len(models_data)} model columns.")
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    analyze_iot()
    # aggregate_results("iot")
    analyze_m5()
    # aggregate_results("m5")
