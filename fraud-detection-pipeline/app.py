"""
Real-time fraud scoring service.

Accepts a RAW transaction (user_id, amount, merchant_category, distance_from_home,
timestamp) and computes the same features used at training time --
rolling_tx_count_24h, spending_zscore, time_since_last_tx_seconds -- from an
in-memory per-user history, using only transactions strictly prior to the one
being scored. This mirrors the shift(1)/closed='left' logic in src/features.py
so there is no train/serve skew between offline and online feature definitions.

NOTE ON STATE: user history is held in-process, in memory. That's fine for a
single-instance demo, but it means state resets on restart and won't be shared
across replicas. A production deployment would back this with a low-latency
external store (e.g. Redis) keyed by user_id so any replica can serve any user.

NOTE ON CONCURRENCY: the /predict route is `async def` with no `await`, which
in FastAPI means it runs on the single-threaded event loop rather than being
farmed out to a thread pool. That keeps updates to the shared per-user state
race-free without needing an explicit lock, at the cost of not parallelizing
CPU-bound requests across threads -- an acceptable tradeoff for this demo.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Real-Time Fraud Detection Engine", version="1.0.0")

artifact = joblib.load('model.pkl')
model = artifact['model']
threshold = artifact['threshold']
expected_features = artifact['features']

ROLLING_WINDOW = timedelta(hours=24)
COLD_START_TIME_GAP_SECONDS = 999_999.0


@dataclass
class UserState:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Welford's running sum of squared differences, for variance
    recent_timestamps: deque = field(default_factory=deque)
    last_tx_time: Optional[datetime] = None

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0  # not enough history for a meaningful std
        variance = self.m2 / (self.count - 1)
        return variance ** 0.5 if variance > 0 else 1.0

    def update(self, amount: float, ts: datetime):
        # Welford's online mean/variance update
        self.count += 1
        delta = amount - self.mean
        self.mean += delta / self.count
        delta2 = amount - self.mean
        self.m2 += delta * delta2

        self.recent_timestamps.append(ts)
        self.last_tx_time = ts


# In-memory per-user state. See module docstring for production caveats.
user_states: dict[int, UserState] = {}


class TransactionRequest(BaseModel):
    user_id: int
    amount: float = Field(..., gt=0)
    merchant_category: str
    distance_from_home: float = Field(..., ge=0)
    timestamp: Optional[datetime] = None


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_flagged: bool
    action: str
    latency_ms: float


def _prune_old(state: UserState, now: datetime):
    cutoff = now - ROLLING_WINDOW
    while state.recent_timestamps and state.recent_timestamps[0] < cutoff:
        state.recent_timestamps.popleft()


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(tx: TransactionRequest):
    start = time.perf_counter()

    now = tx.timestamp or datetime.utcnow()
    state = user_states.setdefault(tx.user_id, UserState())

    # --- Compute features from state as it stood BEFORE this transaction ---
    _prune_old(state, now)
    rolling_tx_count_24h = float(len(state.recent_timestamps))

    if state.count == 0:
        spending_zscore = 0.0
    else:
        spending_zscore = (tx.amount - state.mean) / state.std

    if state.last_tx_time is None:
        time_since_last_tx_seconds = COLD_START_TIME_GAP_SECONDS
    else:
        time_since_last_tx_seconds = (now - state.last_tx_time).total_seconds()

    input_data = pd.DataFrame(
        [[
            tx.amount,
            tx.merchant_category,
            tx.distance_from_home,
            rolling_tx_count_24h,
            spending_zscore,
            time_since_last_tx_seconds,
        ]],
        columns=expected_features,
    )

    prob = float(model.predict_proba(input_data)[0, 1])
    is_flagged = prob >= threshold
    action = "FLAG_STEP_UP_AUTH" if is_flagged else "APPROVE"

    # --- Only now fold this transaction into the user's history ---
    state.update(tx.amount, now)

    latency = (time.perf_counter() - start) * 1000

    return {
        "fraud_probability": round(prob, 4),
        "is_flagged": is_flagged,
        "action": action,
        "latency_ms": round(latency, 2),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "threshold": threshold,
        "tracked_users": len(user_states),
    }
