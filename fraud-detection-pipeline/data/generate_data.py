"""
Generates synthetic credit-card-style transaction data for fraud detection.
Run from anywhere -- output path is resolved relative to this file's location.
"""

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 280_000  # set lower (e.g. 50_000) for a fast local run

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "transactions.csv"


def generate(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [
        start_time + timedelta(seconds=int(x))
        for x in np.sort(np.random.uniform(0, 86400 * 30, n_samples))
    ]

    transaction_ids = [str(uuid.uuid4()) for _ in range(n_samples)]
    user_ids = np.random.randint(1000, 5000, size=n_samples)
    amounts = np.random.exponential(scale=50.0, size=n_samples) + 2.0
    merchant_categories = np.random.choice(
        ["grocery", "electronics", "travel", "dining", "utilities"], size=n_samples
    )
    distance_from_home = np.random.exponential(scale=10.0, size=n_samples)

    fraud_prob = (amounts > 250) * 0.05 + (distance_from_home > 50) * 0.03 + 0.0005
    is_fraud = (np.random.rand(n_samples) < fraud_prob).astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "timestamp": timestamps,
            "user_id": user_ids,
            "amount": amounts.round(2),
            "merchant_category": merchant_categories,
            "distance_from_home": distance_from_home.round(2),
            "is_fraud": is_fraud,
        }
    )
    return df


def main():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Generated {len(df)} transactions with {df['is_fraud'].sum()} fraud cases "
        f"({df['is_fraud'].mean() * 100:.3f}%)."
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
