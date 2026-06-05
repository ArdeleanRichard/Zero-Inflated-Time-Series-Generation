import pandas as pd

# FILE_PATH = "./iot/hparam_search_iot_gan.csv"
FILE_PATH = "./iot/hparam_search_iot_vae.csv"

# FILE_PATH = "./m5/hparam_search_m5_gan.csv"
# FILE_PATH = "./m5/hparam_search_m5_vae.csv"




# Metrics of interest (smaller = better).
METRICS = [
    "zero_ratio_diff",
    "mean_diff",
    "std_diff",
    "skewness_diff",
    "kurtosis_diff",
    "quantile_10_diff",
    "quantile_25_diff",
    "quantile_50_diff",
    "quantile_75_diff",
    "quantile_90_diff",
    "quantile_95_diff",
    "quantile_99_diff",
    "mean_quantile_diff",
    "autocorr_mae",
    "wasserstein_distance",
    "kl_divergence",
    "mmd",
    "ps",
    "ds",
    "lps",
    "ld",        # stored as metric__lds
]


PARAM_COLS = [
    "run_id",
    "model",
    "data",
    "latent_dim",
    "lr",
    "betas_0",
    "gate_weight",
    "recon_weight",
    "tc_weight",
    "fm_weight",
]

df = pd.read_csv(FILE_PATH)

def col_for(metric: str) -> str:
    """Return the dataframe column name for a metric key."""
    if metric == "ld":
        return "metric__lds"
    return f"metric__{metric}"


def best_row(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the row with the minimum value in *col* (ignoring NaN)."""
    idx = df[col].idxmin()
    return df.loc[idx]




results = []
for metric in METRICS:
    col = col_for(metric)
    if col not in df.columns:
        print(f"[WARNING] Column '{col}' not found – skipping '{metric}'.")
        continue

    row = best_row(df, col)
    entry = {"metric": metric, "best_value": row[col]}
    for p in PARAM_COLS:
        if p in row.index:
            entry[p] = row[p]
    results.append(entry)

results_df = pd.DataFrame(results)

# ── DISPLAY ───────────────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.6g}".format)


print(f"\n{'='*100}")
print("\nSUMMARY TABLE")
print("=" * 100)
# Reorder for readability
display_cols = ["metric", "best_value"] + [p for p in PARAM_COLS if p in results_df.columns]
print(results_df[display_cols].to_string(index=False))