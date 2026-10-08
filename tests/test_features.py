import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.engineer import build_training_frame, add_time_features, FEATURE_COLUMNS, CATEGORIES
from models.sequence_features import add_sequence_step_columns, to_sequence_array, SEQ_LEN, N_STEP_FEATURES


def _toy_df():
    base = pd.Timestamp("2026-01-01 10:00:00")
    rows = []
    for i in range(6):
        rows.append({
            "cc_num": 1111, "trans_num": f"t{i}", "trans_time": base + pd.Timedelta(minutes=15 * i),
            "category": "grocery", "merchant": "Big Basket", "amt": 100.0 + i * 10,
            "merch_lat": 13.08, "merch_long": 80.27, "is_fraud": 0, "age": 30.0,
            "distance_km": 1.0 + i * 0.1,
        })
    df = pd.DataFrame(rows)
    df["unix_time"] = df["trans_time"].values.astype("datetime64[s]").astype("int64")
    return df


def test_velocity_features_increase_with_rapid_transactions():
    df = _toy_df()
    out = build_training_frame(df)
    # each successive transaction (15 min apart) should see its own prior history grow
    counts = out.sort_values("unix_time")["txn_count_24h"].tolist()
    assert counts[-1] >= counts[0]
    assert (out["txn_count_1h"] >= 1).all()


def test_no_nans_in_feature_columns():
    df = _toy_df()
    out = build_training_frame(df)
    assert out[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_category_one_hot_is_exclusive():
    df = _toy_df()
    out = build_training_frame(df)
    cat_cols = [f"cat_{c}" for c in CATEGORIES]
    assert (out[cat_cols].sum(axis=1) == 1).all()
    assert (out["cat_grocery"] == 1).all()


def test_sequence_array_shape_and_recency_order():
    df = add_time_features(_toy_df())
    df = add_sequence_step_columns(df)
    arr = to_sequence_array(df)
    assert arr.shape == (len(df), SEQ_LEN, N_STEP_FEATURES)
    # last transaction's current step (index -1 in sequence) should match its own amt_log
    last_row = df.sort_values("unix_time").iloc[-1]
    last_seq = arr[df["unix_time"].values.argsort()][-1]
    assert np.isclose(last_seq[-1, 0], last_row["amt_log"], atol=1e-4)


def test_sequence_history_is_zero_for_first_transaction():
    df = add_time_features(_toy_df())
    df = add_sequence_step_columns(df)
    arr = to_sequence_array(df)
    first_idx = df["unix_time"].values.argmin()
    first_seq = arr[first_idx]
    assert np.allclose(first_seq[:-1], 0.0)  # no history before the first transaction
