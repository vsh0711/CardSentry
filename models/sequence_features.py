"""
Builds fixed-length per-card transaction sequences (current + K previous transactions)
used by the PyTorch sequence model. Vectorized via groupby-shift instead of a per-row
Python loop so it scales to millions of rows.
"""
import numpy as np
import pandas as pd

SEQ_CATEGORIES = [
    "grocery", "food_dining", "fuel", "shopping_mall", "electronics",
    "pharmacy", "utility_bills", "online_retail", "entertainment",
    "travel", "jewellery", "clothing", "education",
]
CAT_TO_CODE = {c: i + 1 for i, c in enumerate(SEQ_CATEGORIES)}  # 0 reserved for "no history"

SEQ_STEP_FEATURES = ["amt_log", "distance_km", "hour_of_day", "day_of_week", "category_code"]
HISTORY_LEN = 5  # previous transactions
SEQ_LEN = HISTORY_LEN + 1  # + current transaction
N_STEP_FEATURES = len(SEQ_STEP_FEATURES)


def add_sequence_step_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cc_num", "unix_time"]).copy()
    df["amt_log"] = np.log1p(df["amt"])
    df["category_code"] = df["category"].map(CAT_TO_CODE).fillna(0).astype(int)

    g = df.groupby("cc_num")
    for k in range(1, HISTORY_LEN + 1):
        for feat in SEQ_STEP_FEATURES:
            df[f"{feat}_lag{k}"] = g[feat].shift(k)
    # fill missing history (new/early cards) with 0 -> model learns "no history" naturally
    lag_cols = [f"{feat}_lag{k}" for k in range(1, HISTORY_LEN + 1) for feat in SEQ_STEP_FEATURES]
    df[lag_cols] = df[lag_cols].fillna(0.0)
    return df


def to_sequence_array(df: pd.DataFrame) -> np.ndarray:
    """Returns array of shape (N, SEQ_LEN, N_STEP_FEATURES), oldest -> newest."""
    n = len(df)
    arr = np.zeros((n, SEQ_LEN, N_STEP_FEATURES), dtype=np.float32)
    for k in range(HISTORY_LEN, 0, -1):
        step_idx = HISTORY_LEN - k  # oldest lag first
        for fi, feat in enumerate(SEQ_STEP_FEATURES):
            arr[:, step_idx, fi] = df[f"{feat}_lag{k}"].to_numpy()
    for fi, feat in enumerate(SEQ_STEP_FEATURES):
        arr[:, HISTORY_LEN, fi] = df[feat].to_numpy()
    return arr
