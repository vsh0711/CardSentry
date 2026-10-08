"""
Shared transaction history store, used by the FastAPI scorer (writer), the Kafka
consumer (writer) and the Streamlit dashboard (reader) so all three see the same
live feed regardless of which process they run in.

Backend is SQLite by default (zero config, fine for a single-container demo) or
Postgres when DATABASE_URL is set (e.g. Neon free tier) -- same code path either
way via SQLAlchemy Core.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, Float, String, Boolean, select, desc, func,
)

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "cardsentry.db"


def _build_engine():
    url = os.environ.get("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):  # SQLAlchemy wants postgresql://
            url = url.replace("postgres://", "postgresql://", 1)
        if url.startswith("postgresql://"):
            # Force the psycopg2 dialect explicitly. A bare "postgresql://" lets
            # SQLAlchemy pick whichever driver it finds first (psycopg v3 vs
            # psycopg2); on a minimal install (e.g. the dashboard's scoped
            # requirements.txt, which only ships psycopg2-binary) that can
            # resolve to "psycopg" and fail with ModuleNotFoundError even
            # though a working Postgres driver is installed.
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return create_engine(url, pool_pre_ping=True)
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DEFAULT_SQLITE_PATH}")
    with engine.connect() as c:
        c.exec_driver_sql("PRAGMA journal_mode=WAL;")
    return engine


_metadata = MetaData()
_transactions = Table(
    "transactions", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cc_num", String(32)),
    Column("first", String(64)),
    Column("last", String(64)),
    Column("merchant", String(128)),
    Column("category", String(32)),
    Column("locality", String(64)),
    Column("amt", Float),
    Column("distance_km", Float),
    Column("merch_lat", Float),
    Column("merch_long", Float),
    Column("unix_time", Float),
    Column("fraud_probability", Float),
    Column("is_fraud_predicted", Boolean),
    Column("xgb_prob", Float),
    Column("seq_prob", Float),
    Column("latency_ms", Float),
    Column("scored_at", Float),
)


class TransactionStore:
    def __init__(self):
        self.engine = _build_engine()
        _metadata.create_all(self.engine)

    def append(self, record: dict):
        row = {col.name: record.get(col.name) for col in _transactions.columns if col.name != "id"}
        row["cc_num"] = str(row.get("cc_num"))
        with self.engine.begin() as conn:
            conn.execute(_transactions.insert().values(**row))

    def recent(self, limit: int = 100) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(_transactions).order_by(desc(_transactions.c.id)).limit(limit)
            ).mappings().all()
        return [dict(r) for r in rows]

    def rolling_stats(self, window_seconds: int = 300) -> dict:
        cutoff = time.time() - window_seconds
        with self.engine.connect() as conn:
            total = conn.execute(
                select(func.count()).select_from(_transactions).where(_transactions.c.scored_at >= cutoff)
            ).scalar() or 0
            fraud = conn.execute(
                select(func.count()).select_from(_transactions)
                .where(_transactions.c.scored_at >= cutoff, _transactions.c.is_fraud_predicted == True)  # noqa: E712
            ).scalar() or 0
            avg_latency = conn.execute(
                select(func.avg(_transactions.c.latency_ms)).where(_transactions.c.scored_at >= cutoff)
            ).scalar() or 0.0
            total_amt = conn.execute(
                select(func.sum(_transactions.c.amt)).where(_transactions.c.scored_at >= cutoff)
            ).scalar() or 0.0
        return {
            "window_seconds": window_seconds,
            "transaction_count": total,
            "fraud_count": fraud,
            "fraud_rate": (fraud / total) if total else 0.0,
            "avg_latency_ms": round(float(avg_latency), 3),
            "total_amount": float(total_amt),
            "throughput_per_sec": round(total / window_seconds, 2) if window_seconds else 0.0,
        }
