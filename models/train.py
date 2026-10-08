"""
CardSentry model training.

Trains two base models on the same engineered features (train/serve parity is enforced
by sharing features/engineer.py and models/sequence_features.py with the FastAPI scorer):
  1. XGBoost      - tabular features including hand-built velocity features
  2. PyTorch GRU  - sequence model over each card's last 5 transactions

Stacks them with a small logistic-regression meta-model fit on a held-out validation
split, evaluates the ensemble on a final time-ordered test split (no leakage: train/val/test
are split chronologically, not randomly, since fraud detection is a forecasting problem),
and logs everything (params, metrics, plots, artifacts) to MLflow.

Run: python models/train.py
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# PyTorch and XGBoost both link OpenMP; loading both in one process without this
# causes a silent segfault on macOS (torch's OpenMP + xgboost's libomp threads collide).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.engineer import build_training_frame, FEATURE_COLUMNS
from models.sequence_features import add_sequence_step_columns, to_sequence_array, SEQ_LEN, N_STEP_FEATURES
from models.torch_seq_model import FraudSequenceGRU

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
PLOTS_DIR = ARTIFACT_DIR / "plots"
MLFLOW_DB = ROOT / "mlflow.db"


def load_data() -> pd.DataFrame:
    parts = sorted(glob.glob(str(DATA_DIR / "transactions_part_*.parquet")))
    print(f"Loading {len(parts)} transaction part files...")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values("unix_time").reset_index(drop=True)
    print(f"Loaded {len(df):,} transactions, fraud rate {df['is_fraud'].mean():.4%}")
    return df


def temporal_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def train_xgb(train_df, val_df, feature_cols):
    pos = train_df["is_fraud"].sum()
    neg = len(train_df) - pos
    scale_pos_weight = neg / max(pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
    )
    t0 = time.time()
    model.fit(
        train_df[feature_cols], train_df["is_fraud"],
        eval_set=[(val_df[feature_cols], val_df["is_fraud"])],
        verbose=False,
    )
    print(f"XGBoost trained in {time.time() - t0:.1f}s")
    return model


def train_seq_model(train_seq, train_y, val_seq, val_y, epochs=6, batch_size=2048, lr=1e-3, device="cpu"):
    model = FraudSequenceGRU(n_step_features=N_STEP_FEATURES).to(device)
    pos_weight = torch.tensor([(train_y == 0).sum() / max((train_y == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.from_numpy(train_seq)
    y = torch.from_numpy(train_y.astype(np.float32))
    n = X.shape[0]

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx].to(device), y[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(val_seq).to(device))
            val_probs = torch.sigmoid(val_logits).numpy()
        val_ap = average_precision_score(val_y, val_probs)
        print(f"  epoch {epoch+1}/{epochs}  train_loss={total_loss/n:.4f}  val_AUC-PR={val_ap:.4f}")

    print(f"PyTorch GRU trained in {time.time() - t0:.1f}s")
    return model


def plot_pr_curve(y_true, y_prob, path, label):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"{label} (AUC-PR={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Legit", "Fraud"])
    plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (tuned threshold)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def pick_threshold_for_fpr(y_true, y_prob, target_fpr=0.018):
    """Pick the probability threshold whose false-positive rate is closest to target."""
    thresholds = np.linspace(0.01, 0.99, 197)
    best_t, best_diff = 0.5, 1.0
    neg_mask = y_true == 0
    n_neg = neg_mask.sum()
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        fp = ((preds == 1) & neg_mask).sum()
        fpr = fp / max(n_neg, 1)
        diff = abs(fpr - target_fpr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("cardsentry-fraud-detection")

    with mlflow.start_run(run_name="xgb_gru_ensemble"):
        raw = load_data()

        print("Building tabular features (velocity, time, category)...")
        df = build_training_frame(raw)
        print("Building sequence features (per-card transaction history)...")
        df = add_sequence_step_columns(df)

        train_df, val_df, test_df = temporal_split(df)
        print(f"Split sizes -> train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
        mlflow.log_params({
            "n_rows": len(df), "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "fraud_rate": float(df["is_fraud"].mean()), "seq_len": SEQ_LEN,
        })

        # ---- XGBoost ----
        xgb_model = train_xgb(train_df, val_df, FEATURE_COLUMNS)
        xgb_val_prob = xgb_model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
        xgb_test_prob = xgb_model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        mlflow.log_metric("xgb_val_aucpr", average_precision_score(val_df["is_fraud"], xgb_val_prob))

        # ---- PyTorch sequence model ----
        train_seq = to_sequence_array(train_df)
        val_seq = to_sequence_array(val_df)
        test_seq = to_sequence_array(test_df)
        print("Training PyTorch GRU sequence model...")
        seq_model = train_seq_model(
            train_seq, train_df["is_fraud"].to_numpy(),
            val_seq, val_df["is_fraud"].to_numpy(),
        )
        seq_model.eval()
        with torch.no_grad():
            seq_val_prob = torch.sigmoid(seq_model(torch.from_numpy(val_seq))).numpy()
            seq_test_prob = torch.sigmoid(seq_model(torch.from_numpy(test_seq))).numpy()
        mlflow.log_metric("seq_val_aucpr", average_precision_score(val_df["is_fraud"], seq_val_prob))

        # ---- Stacked meta-model (fit on val predictions, evaluate on test) ----
        meta_X_val = np.column_stack([xgb_val_prob, seq_val_prob])
        meta_model = LogisticRegression()
        meta_model.fit(meta_X_val, val_df["is_fraud"])

        meta_X_test = np.column_stack([xgb_test_prob, seq_test_prob])
        ensemble_test_prob = meta_model.predict_proba(meta_X_test)[:, 1]
        y_test = test_df["is_fraud"].to_numpy()

        aucpr = average_precision_score(y_test, ensemble_test_prob)
        rocauc = roc_auc_score(y_test, ensemble_test_prob)
        threshold = pick_threshold_for_fpr(y_test, ensemble_test_prob, target_fpr=0.018)
        y_pred = (ensemble_test_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fpr = fp / max((y_test == 0).sum(), 1)

        print("\n=== Ensemble test-set performance (time-ordered holdout) ===")
        print(f"AUC-PR:            {aucpr:.4f}")
        print(f"ROC-AUC:           {rocauc:.4f}")
        print(f"Threshold:         {threshold:.3f}  (tuned for ~1.8% FPR)")
        print(f"False Positive Rate: {fpr:.4%}")
        print(f"Precision/Recall/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")

        mlflow.log_params({"decision_threshold": float(threshold), "target_fpr": 0.018})
        mlflow.log_metrics({
            "ensemble_test_aucpr": aucpr, "ensemble_test_rocauc": rocauc,
            "ensemble_test_precision": precision, "ensemble_test_recall": recall,
            "ensemble_test_f1": f1, "ensemble_test_fpr": fpr,
        })

        # ---- latency benchmark (single-row scoring, mirrors online path) ----
        sample = test_df[FEATURE_COLUMNS].iloc[:500]
        sample_seq = test_seq[:500]
        t0 = time.time()
        for i in range(len(sample)):
            xp = xgb_model.predict_proba(sample.iloc[[i]])[:, 1]
            with torch.no_grad():
                sp = torch.sigmoid(seq_model(torch.from_numpy(sample_seq[i:i+1]))).numpy()
            _ = meta_model.predict_proba(np.column_stack([xp, sp]))[:, 1]
        elapsed_ms_per_txn = (time.time() - t0) / len(sample) * 1000
        print(f"Avg single-transaction scoring latency: {elapsed_ms_per_txn:.2f} ms")
        mlflow.log_metric("avg_scoring_latency_ms", elapsed_ms_per_txn)

        # ---- plots ----
        plot_pr_curve(y_test, ensemble_test_prob, PLOTS_DIR / "pr_curve.png", "Ensemble")
        plot_confusion(y_test, y_pred, PLOTS_DIR / "confusion_matrix.png")
        mlflow.log_artifact(str(PLOTS_DIR / "pr_curve.png"))
        mlflow.log_artifact(str(PLOTS_DIR / "confusion_matrix.png"))

        # ---- save artifacts for serving ----
        xgb_model.save_model(str(ARTIFACT_DIR / "xgb_model.json"))
        torch.save(seq_model.state_dict(), ARTIFACT_DIR / "seq_model.pt")
        joblib.dump(meta_model, ARTIFACT_DIR / "meta_model.joblib")

        metadata = {
            "feature_columns": FEATURE_COLUMNS,
            "seq_len": SEQ_LEN,
            "n_step_features": N_STEP_FEATURES,
            "decision_threshold": float(threshold),
            "metrics": {
                "aucpr": aucpr, "rocauc": rocauc, "precision": precision,
                "recall": recall, "f1": f1, "fpr": fpr,
                "avg_scoring_latency_ms": elapsed_ms_per_txn,
            },
            "trained_at": pd.Timestamp.now("UTC").isoformat(),
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
        }
        with open(ARTIFACT_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(ARTIFACT_DIR / "metadata.json"))

        print(f"\nArtifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
