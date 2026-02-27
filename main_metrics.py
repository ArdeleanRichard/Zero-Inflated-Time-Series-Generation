import glob
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from constants import MAP_MODEL_NAMES
from metrics import (calculate_evaluation_metrics, print_evaluation_metrics, save_metrics_report)




def plot_sample_comparisons(real_samples: np.ndarray, synthetic_samples: np.ndarray,
                            num_samples: int = 5, save_path: str = 'sample_comparison.png'):
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
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(min(len(ori_data), len(gen_data)))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    gen_data = np.asarray(gen_data)

    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    ori_data = ori_data[..., np.newaxis]
    gen_data = gen_data[..., np.newaxis]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(gen_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, np.reshape(np.mean(gen_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title(f'{MAP_MODEL_NAMES[model]}')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1], c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1], c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title(f'{MAP_MODEL_NAMES[model]}')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def analyze_iot():
    DATA = "iot"
    np_ori_data = np.load(f'./data/{DATA}_durations_2021.npz')
    results_dir = f"results_{DATA}"

    ori_data = np_ori_data['data']
    print(f"Original data shape: {ori_data.shape}")
    ori_data = np.squeeze(ori_data)
    ori_data = np.nan_to_num(ori_data).astype(np.float64)
    print(f"Original data shape: {ori_data.shape}")

    for model in ["zits-gan", "zits-vae", "timegan", "transfusion", "fide", "chronogan", "tsgm_timegan", "tsgm_vae", "vae_dense", "vae_conv", "timeVAE"]:

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
            np_gen_data = np.load(f'./repos/timeVAE/outputs/gen_data/iot_durations_2021/{model}_iot_durations_2021_prior_samples.npz')
        elif model == "transfusion":
            np_gen_data = np.load(f'./repos/TransFusion/saved_files/1770796660.6585-custom-transformers-{DATA}-l1-cosine-365-pred_v/test_outputs/synthetic_data_1770809439.npz', allow_pickle=True)
        elif model == "tsgm_timegan" or model == "tsgm_vae":
            np_gen_data = np.load(f'./repos/tsgm/{DATA}_results/{model}_gen.npz')
        elif model == "ours":
            np_gen_data = np.load(f'./trial_vae/v4/generated_data.npz', allow_pickle=True)
        elif model == "zits-gan":
            np_gen_data = np.load(f'./out_{DATA}/vae_generated_data.npz', allow_pickle=True)
        elif model == "zits-vae":
            np_gen_data = np.load(f'./out_{DATA}/gan_generated_data.npz', allow_pickle=True)
        gen_data = np_gen_data['data']
        print(f"Generated data shape: {gen_data.shape}")
        gen_data = np.squeeze(gen_data)
        gen_data = np.nan_to_num(gen_data).astype(np.float64)
        print(f"Generated data shape: {gen_data.shape}")

        r_idx = np.random.randint(0, min(len(ori_data), len(gen_data)), size=5)
        plot_sample_comparisons(ori_data[r_idx], gen_data[r_idx], save_path=f"./{results_dir}/{model}_sample_comparison.png")

        metrics = calculate_evaluation_metrics(ori_data, gen_data)
        print_evaluation_metrics(metrics)
        save_metrics_report(metrics, f"./{results_dir}/{model}_metrics.json")

        visualization_dim_red(model, ori_data, gen_data, "tsne", f"./{results_dir}/{model}_plot_tsne.png")
        visualization_dim_red(model, ori_data, gen_data, "pca", f"./{results_dir}/{model}_plot_pca.png")


def analyze_m5():
    DATA = "m5"
    np_ori_data = np.load(f'./data/{DATA}/{DATA}_X_365.npz')
    results_dir = f"results_{DATA}"

    ori_data = np_ori_data['data']
    print(f"Original data shape: {ori_data.shape}")
    ori_data = np.squeeze(ori_data)
    ori_data = np.nan_to_num(ori_data).astype(np.float64)
    print(f"Original data shape: {ori_data.shape}")

    for model in ["zits-gan", "zits-vae", "timegan", "transfusion", "fide", "chronogan", "tsgm_timegan", "tsgm_vae", "vae_dense", "vae_conv", "timeVAE"]:

        print(f"=" * 60)
        print(f"Model: {model}")
        print(f"=" * 60)

        if model == "chronogan":
            np_gen_data = np.load(f'./repos/ChronoGAN/output_{DATA}/synthetic_data.npz', allow_pickle=True)
        elif model == "fide":
            np_gen_data = np.load(f'./repos/FIDE/Code/{DATA}_model/fide_generated_data.npz', allow_pickle=True)
        elif model == "timegan":
            np_gen_data = np.load(f'./repos/TimeGAN/{DATA}_{model}_generated_data.npz', allow_pickle=True)
        elif model == "timeVAE" or model == "vae_dense" or model == "vae_conv":
            np_gen_data = np.load(f'./repos/timeVAE/outputs/gen_data/m5_X_365/{model}_m5_X_365_prior_samples.npz')
        elif model == "transfusion":
            np_gen_data = np.load(f'./repos/TransFusion/saved_files/1771403691.0601-custom-transformers-{DATA}-l1-cosine-365-pred_v/test_outputs/synthetic_data_1771411195.npz', allow_pickle=True)
        elif model == "tsgm_timegan" or model == "tsgm_vae":
            np_gen_data = np.load(f'./repos/tsgm/{DATA}_results/{model}_gen.npz')
        elif model == "zits-gan":
            np_gen_data = np.load(f'./out_{DATA}/vae_generated_data.npz', allow_pickle=True)
        elif model == "zits-vae":
            np_gen_data = np.load(f'./out_{DATA}/gan_generated_data.npz', allow_pickle=True)

        gen_data = np_gen_data['data']
        print(f"Generated data shape: {gen_data.shape}")
        gen_data = np.squeeze(gen_data)
        gen_data = np.nan_to_num(gen_data).astype(np.float64)
        print(f"Generated data shape: {gen_data.shape}")

        plot_sample_comparisons(ori_data[1:6], gen_data[1:6], save_path=f"./{results_dir}/{model}_sample_comparison.png")

        metrics = calculate_evaluation_metrics(ori_data, gen_data)
        print_evaluation_metrics(metrics)
        save_metrics_report(metrics, f"./{results_dir}/{model}_metrics.json")

        visualization_dim_red(model, ori_data, gen_data, "tsne", f"./{results_dir}/{model}_plot_tsne.png")
        visualization_dim_red(model, ori_data, gen_data, "pca", f"./{results_dir}/{model}_plot_pca.png")


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
    # analyze_iot()
    # aggregate_results("iot")
    analyze_m5()
    aggregate_results("m5")
