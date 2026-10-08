"""
Per-card rolling state used to compute ONLINE velocity + sequence features at
scoring time, mirroring the offline feature logic in features/engineer.py and
models/sequence_features.py exactly (train/serve parity).

In production this state lives in the Feast online store (Redis). Here it is
wrapped behind a tiny interface (`CardStateStore`) so the scoring code is identical
whether the backing store is in-memory (local dev / tests) or Redis (deployed) --
see api/feast_store.py for the Redis-backed implementation used when
FEAST_REDIS_URL is configured.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict

from features.engineer import CATEGORIES
from models.sequence_features import HISTORY_LEN, CAT_TO_CODE

MAX_HISTORY = max(HISTORY_LEN, 50)  # keep enough for both velocity windows and sequence lags


@dataclass
class CardHistory:
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    amounts: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    distances: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    hours: Deque[int] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    dows: Deque[int] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    cat_codes: Deque[int] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))


class CardStateStore:
    """In-memory per-card history. Swappable for a Redis-backed store (see feast_store.py)."""

    def __init__(self):
        self._cards: Dict[int, CardHistory] = defaultdict(CardHistory)

    def get(self, cc_num: int) -> CardHistory:
        return self._cards[cc_num]

    def append(self, cc_num: int, unix_time: float, amt: float, distance_km: float,
               hour: int, dow: int, category: str):
        h = self.get(cc_num)
        h.timestamps.append(unix_time)
        h.amounts.append(amt)
        h.distances.append(distance_km)
        h.hours.append(hour)
        h.dows.append(dow)
        h.cat_codes.append(CAT_TO_CODE.get(category, 0))


def compute_velocity_features(history: CardHistory, now: float) -> dict:
    ts = list(history.timestamps)
    amts = list(history.amounts)
    dists = list(history.distances)

    def within(seconds):
        return [(t, a) for t, a in zip(ts, amts) if now - t <= seconds]

    w10m = within(600)
    w1h = within(3600)
    w24h = within(86400)

    time_since_last = (now - ts[-1]) if ts else 999999.0
    distance_from_last = dists[-1] if dists else 0.0

    return {
        "txn_count_10m": float(len(w10m) + 1),  # +1 to include the current txn, matching training
        "txn_count_1h": float(len(w1h) + 1),
        "txn_count_24h": float(len(w24h) + 1),
        "amt_sum_1h": float(sum(a for _, a in w1h)),
        "amt_mean_24h": float(sum(a for _, a in w24h) / len(w24h)) if w24h else 0.0,
        "time_since_last_txn_sec": float(min(time_since_last, 999999.0)),
        "distance_from_last_txn_km": float(distance_from_last),
    }


def compute_sequence_array(history: CardHistory, current_step: list) -> "list[list[float]]":
    """Builds the (SEQ_LEN, N_STEP_FEATURES) sequence: oldest lag -> ... -> current."""
    steps = []
    n = len(history.timestamps)
    for k in range(HISTORY_LEN, 0, -1):
        if n >= k:
            idx = n - k
            steps.append([
                _log1p(history.amounts[idx]), history.distances[idx],
                float(history.hours[idx]), float(history.dows[idx]), float(history.cat_codes[idx]),
            ])
        else:
            steps.append([0.0, 0.0, 0.0, 0.0, 0.0])
    steps.append(current_step)
    return steps


def _log1p(x: float) -> float:
    import math
    return math.log1p(max(x, 0.0))
