"""
CardSentry live dashboard (Streamlit).

Tabs:
  - Live Feed: latest scored transactions + rolling KPIs
  - Chennai Map: geo view of recent transactions, fraud flagged in red
  - Model Performance: AUC-PR / confusion matrix / metrics from training
  - Drift Monitor: Evidently drift report status
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.store import TransactionStore

st.set_page_config(page_title="CardSentry", page_icon="💳", layout="wide")

REFRESH_SECONDS = 5


@st.cache_resource
def get_store():
    return TransactionStore()


def load_metadata():
    path = ROOT / "models" / "artifacts" / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_drift_report():
    path = ROOT / "monitoring" / "artifacts" / "drift_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


st.title("💳 CardSentry — Real-Time Fraud Detection")
st.caption("Streaming fraud detection on Chennai credit-card transactions · XGBoost + PyTorch sequence ensemble · Kafka → Feast → FastAPI")

store = get_store()

tab_live, tab_map, tab_model, tab_drift = st.tabs(
    ["🔴 Live Feed", "🗺️ Chennai Map", "📊 Model Performance", "📉 Drift Monitor"]
)

# ---------------------------------------------------------------- Live Feed
with tab_live:
    stats = store.rolling_stats(window_seconds=300)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Txns (last 5 min)", f"{stats['transaction_count']:,}")
    c2.metric("Flagged Fraud", f"{stats['fraud_count']:,}")
    c3.metric("Fraud Rate", f"{stats['fraud_rate']:.2%}")
    c4.metric("Avg Scoring Latency", f"{stats['avg_latency_ms']:.1f} ms")
    c5.metric("Throughput", f"{stats['throughput_per_sec']:.1f} /sec")

    recent = store.recent(limit=300)
    if recent:
        df = pd.DataFrame(recent)
        df["time"] = pd.to_datetime(df["scored_at"], unit="s")

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Transaction volume & fraud over time")
            vol = df.set_index("time").resample("10s").agg(
                txns=("amt", "count"), frauds=("is_fraud_predicted", "sum")
            ).reset_index()
            fig = px.area(vol, x="time", y=["txns", "frauds"], labels={"value": "count"})
            st.plotly_chart(fig, width="stretch")

        with right:
            st.subheader("Spend by category")
            cat = df.groupby("category")["amt"].sum().sort_values(ascending=False).reset_index()
            fig2 = px.pie(cat, names="category", values="amt", hole=0.4)
            st.plotly_chart(fig2, width="stretch")

        st.subheader("Latest transactions")
        display_cols = ["time", "first", "last", "merchant", "category", "locality", "amt",
                         "fraud_probability", "is_fraud_predicted", "latency_ms"]
        display_cols = [c for c in display_cols if c in df.columns]
        styled = df[display_cols].head(50)

        def highlight_fraud(row):
            return ["background-color: #ffcccc" if row.get("is_fraud_predicted") else "" for _ in row]

        st.dataframe(styled.style.apply(highlight_fraud, axis=1), width="stretch", height=500)
    else:
        st.info("No transactions scored yet. Start the producer/consumer (see README) to see live data here.")

# ---------------------------------------------------------------- Chennai Map
with tab_map:
    st.subheader("Live transactions across Chennai")
    recent = store.recent(limit=500)
    if recent:
        df = pd.DataFrame(recent)
        df = df.dropna(subset=["distance_km"])
        has_geo = "merch_lat" in df.columns and df["merch_lat"].notna().any()
        if not has_geo:
            st.warning("Lat/long not present on scored records yet — showing locality-level view instead.")
            loc_counts = df.groupby("locality").size().reset_index(name="count")
            st.bar_chart(loc_counts.set_index("locality"))
        else:
            df = df.dropna(subset=["merch_lat", "merch_long"]).copy()
            df["color"] = df["is_fraud_predicted"].apply(
                lambda f: [255, 60, 60, 180] if f else [0, 150, 255, 120]
            )
            layer = pdk.Layer(
                "ScatterplotLayer", data=df,
                get_position="[merch_long, merch_lat]",
                get_fill_color="color",
                get_radius=250,
                pickable=True,
            )
            view = pdk.ViewState(latitude=13.0827, longitude=80.2707, zoom=10.5)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
    else:
        st.info("No transactions yet.")

# ---------------------------------------------------------------- Model Performance
with tab_model:
    meta = load_metadata()
    if meta:
        m = meta["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-PR", f"{m['aucpr']:.4f}")
        c2.metric("ROC-AUC", f"{m['rocauc']:.4f}")
        c3.metric("False Positive Rate", f"{m['fpr']:.2%}")
        c4.metric("Avg Scoring Latency", f"{m['avg_scoring_latency_ms']:.2f} ms")

        st.caption(
            f"Precision {m['precision']:.3f} · Recall {m['recall']:.3f} · F1 {m['f1']:.3f} "
            f"· trained on {meta['n_train_rows']:,} rows, tested on {meta['n_test_rows']:,} rows "
            f"· decision threshold {meta['decision_threshold']:.3f} (tuned for ~1.8% FPR)"
        )

        plot_dir = ROOT / "models" / "artifacts" / "plots"
        c1, c2 = st.columns(2)
        pr_path = plot_dir / "pr_curve.png"
        cm_path = plot_dir / "confusion_matrix.png"
        if pr_path.exists():
            c1.image(str(pr_path), caption="Precision-Recall curve")
        if cm_path.exists():
            c2.image(str(cm_path), caption="Confusion matrix")
    else:
        st.warning("No trained model metadata found yet. Run `python models/train.py` first.")

# ---------------------------------------------------------------- Drift Monitor
with tab_drift:
    drift = load_drift_report()
    if drift:
        c1, c2 = st.columns(2)
        c1.metric("Dataset Drift Detected", "Yes ⚠️" if drift.get("dataset_drift") else "No ✅")
        c2.metric("Drifted Columns", f"{drift.get('n_drifted_columns', 0)} / {drift.get('n_columns', 0)}")
        st.json(drift)
        report_path = ROOT / "monitoring" / "artifacts" / "drift_report.html"
        if report_path.exists():
            st.caption("Full Evidently report:")
            st.components.v1.html(report_path.read_text(), height=800, scrolling=True)
    else:
        st.info("No drift report generated yet. Run `python monitoring/drift.py` first.")

st.caption(f"Last refreshed {pd.Timestamp.now().strftime('%H:%M:%S')} · auto-refreshing every {REFRESH_SECONDS}s")
time.sleep(REFRESH_SECONDS)
st.rerun()
