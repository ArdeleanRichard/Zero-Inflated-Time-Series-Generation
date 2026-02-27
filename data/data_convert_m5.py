# Python 3.9+ (run in a notebook / script)
import pandas as pd
import numpy as np
import os

# 1) paths to CSVs (adjust to where you downloaded/extracted them)
SALES_CSV = "./m5/sales_train_evaluation.csv"   # main file to download
CAL_CSV   = "./m5/calendar.csv"

# 2) load sales (this file is wide: each row is one series; columns: id,item_id,...,d_1..d_1941)
sales = pd.read_csv(SALES_CSV)

# quick peek
print("sales shape:", sales.shape)
print("columns (start):", sales.columns[:8].tolist())
print("columns (end):", sales.columns[-6:].tolist())

# 3) identify the 'd_' columns (time columns)
dcols = [c for c in sales.columns if c.startswith("d_")]
print("num time columns:", len(dcols))   # ~1941

# 4) extract matrix: rows = series, columns = days
#    choose dtype carefully (int16 saves memory for small counts)
X_full = sales[dcols].to_numpy(dtype=np.int16)   # shape: (n_series, n_days)
print("X_full shape:", X_full.shape)

# 5) attach ids if you want to keep mapping (useful later)
series_ids = sales["id"].values   # e.g., 'FOODS_1_001_CA_1_1'
meta = sales[["id","item_id","dept_id","cat_id","store_id","state_id"]]

# 6) compute percent zeros per series
pct_zeros = (X_full == 0).mean(axis=1) * 100.0
print("mean % zeros across series: {:.2f}%".format(pct_zeros.mean()))
print("median % zeros across series: {:.2f}%".format(np.median(pct_zeros)))
print("fraction with >=75% zeros: {:.2f}%".format((pct_zeros >= 75).mean()*100))

# 7) if you want columns to represent a single year (last 365 days), use calendar to map d_ -> date
cal = pd.read_csv(CAL_CSV)
# calendar has column 'd' that matches 'd_1' etc, and 'date' (YYYY-MM-DD)
# ensure ordering matches dcols
cal = cal.set_index("d").loc[dcols]   # align to dcols order
dates = pd.to_datetime(cal["date"].values)   # array of datetimes corresponding to each column

# choose last 365 days (or any window)
# find indices for the last 365 calendar days in the dataset
last_N = 365
if len(dcols) >= last_N:
    idx_start = len(dcols) - last_N
    X_365 = X_full[:, idx_start:]    # shape: (n_series, 365)
    dates_365 = dates[idx_start:]
    print("X_365 shape:", X_365.shape)
else:
    raise ValueError("Not enough days to make a 365-day matrix.")

# 8) quick checks on X_365 zeros
pct_zeros_365 = (X_365 == 0).mean(axis=1) * 100.0
print("mean % zeros (last 365 days): {:.2f}%".format(pct_zeros_365.mean()))
print("median % zeros (last 365 days): {:.2f}%".format(np.median(pct_zeros_365)))
print("fraction with >=75% zeros (last 365 days): {:.2f}%".format((pct_zeros_365 >= 75).mean()*100))

# 9) save array and metadata for later use
np.save("./m5/m5_X_full.npy", X_full)        # large: (~30k x 1941)
np.save("./m5/m5_X_365.npy", X_365)          # (~30k x 365)
np.savez("./m5/m5_X_365.npz", data=X_365[..., np.newaxis])          # (~30k x 365)
meta.to_csv("./m5/m5_series_meta.csv", index=False)
pd.DataFrame({"id":series_ids, "pct_zeros_full":pct_zeros, "pct_zeros_365":pct_zeros_365}).to_csv("./m5/m5_zero_stats.csv", index=False)

print("Saved M5_X_full.npy, M5_X_365.npy, M5_series_meta.csv, M5_zero_stats.csv")
