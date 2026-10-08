# CardSentry — Architecture & Data Flow

## 1. End-to-end data flow (what actually happens to one transaction)

```mermaid
flowchart TD
    subgraph GEN["① GENERATE — offline, once"]
        direction TB
        G1["data_gen/generate.py<br/>5,000 Chennai customers<br/>1.53M transactions<br/>5 fraud patterns + label noise"]
    end

    subgraph TRAIN["② LEARN — offline, once"]
        direction TB
        T1["features/engineer.py<br/>velocity + time + category features"]
        T2["models/sequence_features.py<br/>per-card last-5-txn sequences"]
        T3["models/train.py<br/>chronological 70/15/15 split"]
        T4["XGBoost<br/>(tabular + velocity)"]
        T5["PyTorch GRU<br/>(sequence model)"]
        T6["Logistic-regression<br/>meta-model (stacking)"]
        T7[("MLflow<br/>params · metrics · plots")]
        T1 --> T3
        T2 --> T3
        T3 --> T4 --> T6
        T3 --> T5 --> T6
        T3 -.logs.-> T7
        T6 -->|xgb_model.json<br/>seq_model.pt<br/>meta_model.joblib| ART[("models/artifacts/")]
    end

    subgraph LIVE["③ LIVE — one transaction's actual path"]
        direction TB
        P["streaming/producer.py<br/>emits one Chennai txn<br/>(cc_num, amt, merchant, geo, time)"]
        K[("Redpanda Serverless<br/>topic: cardsentry-transactions")]
        C["streaming/consumer.py<br/>(or direct POST /score via FastAPI)"]
        S1["api/card_state.py<br/>look up this card's rolling<br/>history (last 50 txns in memory)"]
        S2["recompute velocity features<br/>txn_count_1h, amt_sum_1h,<br/>time_since_last, distance_from_last"]
        S3["api/scorer.py<br/>XGBoost.predict_proba()"]
        S4["GRU.forward() on<br/>6-step sequence tensor"]
        S5["meta_model.predict_proba(<br/>[xgb_prob, seq_prob])"]
        S6{{"fraud_probability<br/>≥ 0.05 threshold?"}}
        P -->|JSON event| K --> C --> S1 --> S2
        S2 --> S3 --> S5
        S2 --> S4 --> S5
        S5 --> S6
    end

    ART -.loaded once at startup.-> S3
    ART -.loaded once at startup.-> S4
    ART -.loaded once at startup.-> S5

    subgraph PERSIST["④ PERSIST"]
        DB[("Neon Postgres<br/>(or local SQLite)<br/>one row per scored txn")]
    end
    S6 -->|"fraud + probability + latency_ms"| DB

    subgraph WATCH["⑤ OBSERVE"]
        direction TB
        D1["Streamlit dashboard<br/>polls DB every 5s"]
        D2["Live Feed — table + charts"]
        D3["Chennai Map — pydeck geo scatter"]
        D4["Model Performance — PR curve,<br/>confusion matrix from metadata.json"]
        D5["monitoring/drift.py (Evidently)<br/>reference vs. live-window comparison"]
        D6["Drift Monitor tab"]
        DB --> D1 --> D2
        D1 --> D3
        ART -.-> D4
        DB -.current window.-> D5
        T1 -.reference window.-> D5
        D5 --> D6
    end
```

**Timing, measured on this build:** generation ~10s for 1.53M rows · training ~45s for
both models · scoring ~1-4ms per transaction end-to-end (feature lookup + XGBoost +
GRU + meta-model) · dashboard refresh every 5s.

---

## 2. What each stage is built to prove

This project exists to demonstrate specific, checkable engineering claims — not just
to produce a dashboard. Each stage below maps to a claim a reviewer can verify by
reading the code or running it themselves.

| Stage | Claim being proven | How to verify it yourself |
|---|---|---|
| **Data generation** | Can design a labeled dataset at real scale (1.5M+ rows) with *intentional* class imbalance and *intentional* label noise, not just call `make_classification()` | `python data_gen/generate.py` — read the fraud pattern mix printed at the end |
| **Feature engineering** | Understands train/serve skew and designs against it — one module (`features/engineer.py`) is imported by both the training script and (indirectly, via identical formulas) the real-time scorer | `tests/test_features.py` — velocity window tests, no-leakage-before-first-transaction test |
| **Chronological split** | Understands that fraud detection is a forecasting problem — a random train/test split would leak future information | `models/train.py::temporal_split` — train/val/test are time-ordered slices, never shuffled |
| **Two-model ensemble** | Can combine a tabular model (XGBoost) with a sequence model (PyTorch GRU) via stacking, not just pick one | `models/train.py` — meta-model is fit on *held-out validation* predictions, evaluated on a separate test split |
| **Realistic metrics** | Recognizes when a model result (ROC-AUC 1.000) is a red flag rather than a win, and can diagnose + fix the data generation to produce defensible numbers | See the "Debugging journal" in the main README — this was an actual mid-build correction |
| **MLflow tracking** | Treats every training run as reproducible, logged, comparable — not a notebook that got lucky once | `mlflow ui --backend-store-uri sqlite:///mlflow.db` after running `train.py` twice |
| **Feast feature store** | Can wire up a real feature store with both offline (point-in-time-correct) and online retrieval paths, understands why that matters | `python features/feast_demo.py` — runs `feast apply`, `feast materialize-incremental`, then both retrieval calls |
| **Kafka streaming** | Can build a real producer/consumer pair against a managed, SASL-authenticated Kafka cluster — not just `localhost:9092` with no auth | `streaming/producer.py` + `streaming/consumer.py` against Redpanda Serverless (see README for the SASL/ACL debugging story) |
| **Sub-5ms scoring** | The model is actually fast enough to sit in a request path, not just accurate in a notebook | `models/artifacts/metadata.json::avg_scoring_latency_ms` — measured on 500 held-out rows, end to end |
| **Drift monitoring** | Understands that a deployed model's input distribution can shift, and has instrumented a way to detect it before accuracy silently degrades | `monitoring/drift.py` — Evidently `DataDriftPreset` report, JSON summary consumed by the dashboard |
| **Shared state across processes** | The API (writer), Kafka consumer (writer) and Streamlit dashboard (reader) all see one consistent view of live data, swappable from SQLite to managed Postgres with one env var | `api/store.py` — same code path, `DATABASE_URL` picks the backend |
| **Testing discipline** | Business logic is covered by tests that would actually catch a regression, not placeholder tests | `tests/` — 17 tests across data generation, feature correctness, scorer behavior, API contracts; CI runs them on every push |
| **Free-tier cloud deployment** | Can design within real constraints (no budget) instead of defaulting to "just use AWS" | Redpanda Serverless (Kafka) + Neon (Postgres) + Streamlit Community Cloud / Render — all free tier, documented in README |

---

## 3. Why a hybrid architecture, not a straight port

The original project (Spark + Cassandra + Spring Boot) and this one solve the same
problem — classify a transaction in flight — but the shape of the solution differs
because the constraint differs:

| | Original | CardSentry | Why |
|---|---|---|---|
| Stream processing | Spark Structured Streaming | Python Kafka consumer + FastAPI | Spark needs a cluster (JVM, executors, shuffle) to earn its keep; at this throughput a single Python process is simpler to run, debug, and deploy free-tier, with no loss of correctness |
| Storage | Cassandra | Postgres (Neon) / SQLite | Cassandra's write-optimized wide-column model pays off at a scale this project doesn't reach; Postgres gives the same "any process can read the latest state" property with far less operational weight |
| Dashboard | Spring Boot + JSP | Streamlit | Spring Boot's strength is a durable multi-page web app; a live metrics dashboard is exactly Streamlit's design target, in a fraction of the code |
| ML | Random Forest (Spark MLlib) | XGBoost + PyTorch GRU, stacked | Matches the brief's claim of pairing a tabular baseline with a sequence model over recent spend history — Spark MLlib doesn't have a sequence-model story |

This is the central engineering judgment call of the project: **match the tool to the
actual load, not to what sounds impressive.** 1,500 events/sec sustained does not
need a Spark cluster; it needs a process that doesn't block.
