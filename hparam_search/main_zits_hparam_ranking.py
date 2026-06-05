import pandas as pd
import numpy as np
from rank_aggregator import RankAggregator

# FILE_PATH = "./iot/hparam_search_iot_vae.csv"
FILE_PATH = "./iot/hparam_search_iot_gan.csv"

# FILE_PATH = "./m5/hparam_search_m5_vae.csv"
# FILE_PATH = "./m5/hparam_search_m5_gan.csv"



METRICS = [
    "zero_ratio_diff", "mean_diff", "std_diff", "skewness_diff", "kurtosis_diff",
    "quantile_10_diff", "quantile_25_diff", "quantile_50_diff", "quantile_75_diff",
    "quantile_90_diff", "quantile_95_diff", "quantile_99_diff", "mean_quantile_diff",
    "autocorr_mae", "wasserstein_distance", "kl_divergence", "mmd",
    "ps", "ds", "lps", "ld",
]

def col_for(metric):
    return "metric__lds" if metric == "ld" else f"metric__{metric}"

df = pd.read_csv(FILE_PATH)

# Only keep param cols that actually exist in this CSV
PARAM_COLS_CANDIDATES = ["run_id", "model", "data", "latent_dim", "lr", "beta", "betas_0",
                          "gate_weight", "recon_weight", "tc_weight", "fm_weight"]
PARAM_COLS = [c for c in PARAM_COLS_CANDIDATES if c in df.columns]

# Build per-metric ranked lists of run_ids (ascending = lower is better)
rank_lists = []
for metric in METRICS:
    col = col_for(metric)
    if col not in df.columns:
        print(f"[WARNING] Skipping '{metric}' — column not found.")
        continue
    subset = df[["run_id", col]].dropna().copy()
    subset["rank"] = subset[col].rank(method="min", ascending=True)
    ordered = subset.sort_values("rank")["run_id"].tolist()
    rank_lists.append(ordered)

agg = RankAggregator()
borda_result = agg.borda(rank_lists, unranked="split")

print("\nBorda ranking (best → worst):")
for rank, (run_id, score) in enumerate(borda_result, 1):
    print(f"  {rank:3d}. run_id={run_id}  score={score:.2f}")

top_run_ids = [run_id for run_id, _ in borda_result[:5]]
top_params = (
    df[df["run_id"].isin(top_run_ids)][PARAM_COLS]
    .drop_duplicates("run_id")
    .set_index("run_id")
    .loc[top_run_ids]
)
print("\nTop-5 hyperparameter configs:")
print(top_params.to_string())