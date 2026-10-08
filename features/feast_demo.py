"""
Builds the Feast offline source from the engineered features, applies the feature
repo, materializes into the online store, and demonstrates both retrieval paths:

  1. get_historical_features()  -- point-in-time correct training set
  2. get_online_features()      -- low-latency lookup for a few sample cards

Run: python features/feast_demo.py
"""
import glob
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.engineer import build_training_frame

FEATURE_REPO = ROOT / "feature_repo"


def build_offline_source(sample_rows: int = 200_000):
    parts = sorted(glob.glob(str(ROOT / "data" / "raw" / "transactions_part_*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values("unix_time").tail(sample_rows).reset_index(drop=True)
    df = build_training_frame(df)

    df["cc_num"] = df["cc_num"].astype(str)
    df["event_timestamp"] = pd.to_datetime(df["trans_time"])
    df["created"] = pd.Timestamp.now("UTC").tz_localize(None)

    cols = [
        "cc_num", "event_timestamp", "created", "amt", "distance_km",
        "txn_count_10m", "txn_count_1h", "txn_count_24h", "amt_sum_1h", "amt_mean_24h",
        "time_since_last_txn_sec", "distance_from_last_txn_km",
        "hour_of_day", "day_of_week", "is_night",
    ]
    out = df[cols].astype({
        "amt": "float32", "distance_km": "float32", "txn_count_10m": "float32",
        "txn_count_1h": "float32", "txn_count_24h": "float32", "amt_sum_1h": "float32",
        "amt_mean_24h": "float32", "time_since_last_txn_sec": "float32",
        "distance_from_last_txn_km": "float32", "hour_of_day": "int64",
        "day_of_week": "int64", "is_night": "int64",
    })
    (FEATURE_REPO / "data").mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURE_REPO / "data" / "card_features.parquet", index=False)
    print(f"Wrote {len(out):,} rows to feature_repo/data/card_features.parquet")
    return out


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=FEATURE_REPO, check=True)


def main():
    offline_df = build_offline_source()

    run(["feast", "apply"])

    # materialize everything up to now into the online store
    run(["feast", "materialize-incremental", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")])

    from feast import FeatureStore
    store = FeatureStore(repo_path=str(FEATURE_REPO))

    # 1. point-in-time correct historical retrieval (what training would use)
    entity_df = offline_df[["cc_num", "event_timestamp"]].drop_duplicates().sample(
        n=min(500, len(offline_df)), random_state=42
    )
    hist = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "card_transaction_features:amt",
            "card_transaction_features:txn_count_1h",
            "card_transaction_features:amt_sum_1h",
        ],
    ).to_df()
    print(f"\nHistorical (point-in-time) retrieval sample:\n{hist.head()}")

    # 2. online retrieval (what the real-time API would use)
    sample_cards = offline_df["cc_num"].drop_duplicates().head(3).tolist()
    online = store.get_online_features(
        features=[
            "card_transaction_features:amt",
            "card_transaction_features:txn_count_1h",
            "card_transaction_features:amt_sum_1h",
        ],
        entity_rows=[{"cc_num": c} for c in sample_cards],
    ).to_df()
    print(f"\nOnline retrieval sample:\n{online}")
    print("\nFeast offline + online retrieval both verified working.")


if __name__ == "__main__":
    main()
