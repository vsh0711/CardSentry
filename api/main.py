"""
FastAPI scoring service for CardSentry.

POST /score  -> scores a single transaction in real time (<50ms target) and persists
                the result so the dashboard and the Kafka consumer share one history.
GET  /health -> liveness/readiness probe.
GET  /stats  -> rolling counters used by the dashboard.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.scorer import FraudScorer
from api.store import TransactionStore

_scorer: Optional[FraudScorer] = None
_store: Optional[TransactionStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer, _store
    _scorer = FraudScorer()
    _store = TransactionStore()
    yield


app = FastAPI(title="CardSentry Fraud Scoring API", version="1.0.0", lifespan=lifespan)


class Transaction(BaseModel):
    cc_num: int
    first: str = ""
    last: str = ""
    merchant: str
    category: str
    amt: float = Field(gt=0)
    locality: str = "Chennai"
    distance_km: float = 0.0
    merch_lat: Optional[float] = None
    merch_long: Optional[float] = None
    age: float = 35.0
    unix_time: float = Field(default_factory=lambda: time.time())
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _scorer is not None}


@app.post("/score")
def score(txn: Transaction):
    if _scorer is None:
        raise HTTPException(503, "Model not loaded yet")

    import datetime
    dt = datetime.datetime.fromtimestamp(txn.unix_time)
    hour = txn.hour_of_day if txn.hour_of_day is not None else dt.hour
    dow = txn.day_of_week if txn.day_of_week is not None else dt.weekday()

    payload = txn.model_dump()
    payload["hour_of_day"] = hour
    payload["day_of_week"] = dow

    result = _scorer.score_transaction(payload)
    record = {**payload, **result, "scored_at": time.time()}
    _store.append(record)
    return record


@app.get("/stats")
def stats(window_seconds: int = 300):
    if _store is None:
        raise HTTPException(503, "Store not ready")
    return _store.rolling_stats(window_seconds)


@app.get("/recent")
def recent(limit: int = 100):
    if _store is None:
        raise HTTPException(503, "Store not ready")
    return _store.recent(limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
