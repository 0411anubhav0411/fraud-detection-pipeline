# Real-Time Transaction Fraud & Anomaly Detection Pipeline

An end-to-end machine learning microservice for scoring credit card transactions against anomalous fraud patterns in real-time (<45ms).

## Highlights
- **Class Imbalance:** Handled extreme skew (<0.2% fraud) via cost-sensitive CatBoost and threshold tuning.
- **Leakage Prevention:** Strict temporal forward-chaining cross-validation on time-stamped transaction streams.
- **Production-Ready:** Async FastAPI microservice containerized with Docker, serving predictions with explicit probability calibration.

## Setup & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
