# Real-Time Transaction Fraud & Anomaly Detection Pipeline

An end-to-end machine learning microservice for scoring credit card transactions
against anomalous fraud patterns in real time.

## Highlights

- **Class Imbalance:** Handled extreme skew (~0.1% fraud) via cost-sensitive
  CatBoost (`scale_pos_weight`) and a precision/recall-driven decision threshold
  tuned to keep the false positive rate under 1.5%.
- **Leakage Prevention:** Every feature (rolling transaction count, spending
  z-score, time-since-last-transaction) is computed using strictly *prior*
  transactions only -- never the current transaction's own outcome, and never
  a future transaction. Model validation uses forward-chaining (walk-forward)
  cross-validation via `TimeSeriesSplit`, so no fold is ever evaluated on data
  older than what it trained on.
- **Real-time serving:** Async FastAPI microservice, containerized with Docker.
  The service accepts a *raw* transaction and maintains lightweight per-user
  state in memory to compute the same features used at training time on the
  fly -- see the note in `app.py` on the in-memory state's limitations and
  the production alternative (an external store such as Redis).

## Project structure

```
fraud-detection-pipeline/
├── data/
│   └── generate_data.py     # synthetic transaction data generator
├── src/
│   ├── features.py          # leakage-free feature engineering (offline/batch)
│   └── train.py              # walk-forward CV + final model training
├── app.py                    # FastAPI real-time scoring service
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate synthetic transaction data:**
   ```bash
   python data/generate_data.py
   ```
   This writes `data/transactions.csv`.

3. **Train the model:**
   ```bash
   cd src
   python train.py
   ```
   This runs 5-fold forward-chaining cross-validation, trains a final model on
   a chronological 80/20 split, tunes the decision threshold, and writes
   `model.pkl` to the project root (run from `src/`, or adjust the paths in
   `train.py` if you run it from elsewhere).

4. **Run the API locally:**
   ```bash
   uvicorn app:app --reload
   ```

5. **Or run it in Docker:**
   ```bash
   docker build -t fraud-detection .
   docker run -p 8000:8000 fraud-detection
   ```

6. **Score a transaction:**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": 1234,
           "amount": 275.50,
           "merchant_category": "electronics",
           "distance_from_home": 62.3
         }'
   ```

## API

### `POST /predict`
Accepts a raw transaction and returns a fraud probability, a flag decision, and
a recommended action.

| Field | Type | Notes |
|---|---|---|
| `user_id` | int | |
| `amount` | float | must be > 0 |
| `merchant_category` | string | one of the categories seen at training time |
| `distance_from_home` | float | must be >= 0 |
| `timestamp` | datetime, optional | defaults to server time if omitted |

### `GET /health`
Returns service status, the active decision threshold, and the number of users
currently tracked in memory.
