"""
Feature engineering shared by training and online scoring, so train/serve feature
logic can never drift apart (the whole point of a feature store).

Produces, per transaction:
  - amt, distance_km, age                       (already in raw data)
  - hour_of_day, day_of_week, is_night
  - category (one-hot at train time)
  - velocity features computed from each card's recent history:
      txn_count_10m, txn_count_1h, txn_count_24h
      amt_sum_1h, amt_mean_24h
      time_since_last_txn_sec
      distance_from_last_txn_km
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery", "food_dining", "fuel", "shopping_mall", "electronics",
    "pharmacy", "utility_bills", "online_retail", "entertainment",
    "travel", "jewellery", "clothing", "education",
]

BASE_NUMERIC_FEATURES = [
    "amt", "distance_km", "age", "hour_of_day", "day_of_week", "is_night",
    "txn_count_10m", "txn_count_1h", "txn_count_24h",
    "amt_sum_1h", "amt_mean_24h", "time_since_last_txn_sec", "distance_from_last_txn_km",
]

FEATURE_COLUMNS = BASE_NUMERIC_FEATURES + [f"cat_{c}" for c in CATEGORIES]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_time"] = pd.to_datetime(df["trans_time"])
    df["hour_of_day"] = df["trans_time"].dt.hour
    df["day_of_week"] = df["trans_time"].dt.dayofweek
    df["is_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 4]).astype(int)
    return df


def add_category_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["category"] == c).astype(int)
    return df


def add_velocity_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized per-card velocity features over a sorted historical batch.
    Used for OFFLINE training on the full dataset. Online scoring recomputes
    the equivalent features from the Feast online store (see features/online.py).
    """
    df = df.sort_values(["cc_num", "unix_time"]).copy()
    g = df.groupby("cc_num")

    df["time_since_last_txn_sec"] = g["unix_time"].diff().fillna(999999).clip(upper=999999)
    df["distance_from_last_txn_km"] = (
        g["distance_km"].diff().abs().fillna(0.0)
    )

    # rolling counts/sums per card within trailing windows, computed via time-indexed rolling
    df = df.set_index(pd.to_datetime(df["unix_time"], unit="s"))
    out_frames = []
    for cc_num, grp in df.groupby("cc_num", sort=False):
        grp = grp.sort_index()
        grp["txn_count_10m"] = grp["trans_num"].rolling("10min").count()
        grp["txn_count_1h"] = grp["trans_num"].rolling("1h").count()
        grp["txn_count_24h"] = grp["trans_num"].rolling("24h").count()
        grp["amt_sum_1h"] = grp["amt"].rolling("1h").sum()
        grp["amt_mean_24h"] = grp["amt"].rolling("24h").mean()
        out_frames.append(grp)
    result = pd.concat(out_frames).reset_index(drop=True)
    return result


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_category_one_hot(df)
    df = add_velocity_features_batch(df)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    return df
