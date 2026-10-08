"""
Drift monitoring for CardSentry using Evidently.

Compares a reference window (the training data) against a "current" window
(the most recent scored live transactions, or a held-out slice as a stand-in
before any live traffic exists) and writes:
  - monitoring/artifacts/drift_report.html  (full interactive Evidently report)
  - monitoring/artifacts/drift_summary.json (compact summary for the dashboard)

Run: python monitoring/drift.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.engineer import build_training_frame, BASE_NUMERIC_FEATURES

ARTIFACT_DIR = ROOT / "monitoring" / "artifacts"


def load_reference_and_current():
    import glob
    parts = sorted(glob.glob(str(ROOT / "data" / "raw" / "transactions_part_*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values("unix_time").reset_index(drop=True)
    df = build_training_frame(df)

    n = len(df)
    reference = df.iloc[: int(n * 0.70)]  # same window the model was trained on
    current = df.iloc[int(n * 0.85):]      # newest held-out slice stands in for live traffic

    live_db = ROOT / "data" / "cardsentry.db"
    if live_db.exists():
        from api.store import TransactionStore
        store = TransactionStore()
        live_recent = store.recent(limit=2000)
        if live_recent:
            live_df = pd.DataFrame(live_recent)
            overlap = [c for c in BASE_NUMERIC_FEATURES if c in live_df.columns]
            if len(live_df) >= 500 and len(overlap) >= 5:
                print(f"Using {len(live_df)} live scored transactions as the current window.")
                return reference, live_df
            print(f"Only {len(live_df)} live rows / {len(overlap)} overlapping features found — "
                  f"falling back to the held-out test split for a meaningful comparison.")

    return reference, current


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    reference, current = load_reference_and_current()

    cols = [c for c in BASE_NUMERIC_FEATURES if c in reference.columns and c in current.columns]
    reference = reference[cols].fillna(0.0)
    current = current[cols].fillna(0.0)

    ref_dataset = Dataset.from_pandas(reference)
    cur_dataset = Dataset.from_pandas(current)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)
    snapshot.save_html(str(ARTIFACT_DIR / "drift_report.html"))

    result = snapshot.dict()
    drifted_count = 0
    n_columns = len(cols)
    drift_share = 0.0
    for m in result["metrics"]:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            drifted_count = int(m["value"]["count"])
            drift_share = float(m["value"]["share"])

    summary = {
        "dataset_drift": drift_share >= 0.5,
        "n_columns": n_columns,
        "n_drifted_columns": drifted_count,
        "drift_share": drift_share,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
    }
    with open(ARTIFACT_DIR / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Full report: {ARTIFACT_DIR / 'drift_report.html'}")


if __name__ == "__main__":
    main()
