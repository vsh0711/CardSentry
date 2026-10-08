"""
Loads the trained XGBoost + GRU + meta-model ensemble once and exposes a single
`score_transaction()` call used by both the FastAPI endpoint and the Kafka consumer,
so there is exactly one scoring code path in the whole system.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
import xgboost as xgb

from features.engineer import FEATURE_COLUMNS, CATEGORIES
from models.sequence_features import N_STEP_FEATURES, SEQ_LEN
from models.torch_seq_model import FraudSequenceGRU
from api.card_state import CardStateStore, compute_velocity_features, compute_sequence_array, _log1p

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"


class FraudScorer:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        with open(artifact_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.threshold = self.metadata["decision_threshold"]

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(artifact_dir / "xgb_model.json"))

        self.seq_model = FraudSequenceGRU(n_step_features=N_STEP_FEATURES)
        self.seq_model.load_state_dict(torch.load(artifact_dir / "seq_model.pt", map_location="cpu"))
        self.seq_model.eval()

        self.meta_model = joblib.load(artifact_dir / "meta_model.joblib")
        self.card_store = CardStateStore()

    def _tabular_features(self, txn: dict, velocity: dict) -> np.ndarray:
        row = {c: 0.0 for c in FEATURE_COLUMNS}
        row["amt"] = txn["amt"]
        row["distance_km"] = txn["distance_km"]
        row["age"] = txn["age"]
        row["hour_of_day"] = txn["hour_of_day"]
        row["day_of_week"] = txn["day_of_week"]
        row["is_night"] = 1.0 if txn["hour_of_day"] in (0, 1, 2, 3, 4) else 0.0
        row.update(velocity)
        cat_key = f"cat_{txn['category']}"
        if cat_key in row:
            row[cat_key] = 1.0
        return np.array([[row[c] for c in FEATURE_COLUMNS]], dtype=np.float32)

    def score_transaction(self, txn: dict) -> dict:
        """
        txn: {cc_num, amt, distance_km, age, hour_of_day, day_of_week, category, unix_time}
        """
        t0 = time.perf_counter()
        history = self.card_store.get(txn["cc_num"])
        velocity = compute_velocity_features(history, now=txn["unix_time"])

        tabular = self._tabular_features(txn, velocity)
        xgb_prob = float(self.xgb_model.predict_proba(tabular)[:, 1][0])

        current_step = [
            _log1p(txn["amt"]), txn["distance_km"], float(txn["hour_of_day"]),
            float(txn["day_of_week"]), float(_cat_code(txn["category"])),
        ]
        seq = compute_sequence_array(history, current_step)
        seq_arr = np.array([seq], dtype=np.float32)
        with torch.no_grad():
            seq_prob = float(torch.sigmoid(self.seq_model(torch.from_numpy(seq_arr))).numpy()[0])

        ensemble_prob = float(self.meta_model.predict_proba([[xgb_prob, seq_prob]])[:, 1][0])
        is_fraud = ensemble_prob >= self.threshold

        # update rolling state AFTER scoring (so current txn doesn't see itself in its own history)
        self.card_store.append(
            txn["cc_num"], txn["unix_time"], txn["amt"], txn["distance_km"],
            txn["hour_of_day"], txn["day_of_week"], txn["category"],
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "xgb_prob": xgb_prob,
            "seq_prob": seq_prob,
            "fraud_probability": ensemble_prob,
            "is_fraud_predicted": bool(is_fraud),
            "threshold": self.threshold,
            "latency_ms": round(latency_ms, 3),
        }


def _cat_code(category: str) -> int:
    from models.sequence_features import CAT_TO_CODE
    return CAT_TO_CODE.get(category, 0)
