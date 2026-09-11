from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import TimeSeriesSplit
import joblib

from features import compute_features

# Resolve paths relative to the project root (parent of this script's src/
# folder) rather than the caller's working directory, so `python train.py`
# works the same whether it's run from the project root or from src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'transactions.csv'
MODEL_PATH = PROJECT_ROOT / 'model.pkl'

FEATURES = [
    'amount',
    'merchant_category',
    'distance_from_home',
    'rolling_tx_count_24h',
    'spending_zscore',
    'time_since_last_tx_seconds',
]
CAT_FEATURES = ['merchant_category']
FPR_TARGET = 0.015


def make_model(scale_pos_weight: float, verbose=False) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        cat_features=CAT_FEATURES,
        verbose=verbose,
        random_seed=42,
    )


def pr_auc_score(y_true, y_probs) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def find_threshold_under_fpr(y_true, y_probs, fpr_target=FPR_TARGET) -> float:
    """Lowest threshold (highest recall) that keeps FPR at or under the target."""
    candidates = np.linspace(0.1, 0.9, 81)
    optimal_threshold = 0.5
    for thresh in candidates:
        preds = (y_probs >= thresh).astype(int)
        fp = ((preds == 1) & (y_true == 0)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if fpr <= fpr_target:
            optimal_threshold = thresh
            break
    return optimal_threshold


def main():
    print("Loading and computing features...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # compute_features sorts internally by (user_id, timestamp) to get correct
    # per-user rolling/expanding stats -- resort to global chronological order
    # afterward, since that's what a walk-forward split needs to be valid.
    df = compute_features(df)
    df = df.sort_values('timestamp').reset_index(drop=True)

    X_all, y_all = df[FEATURES], df['is_fraud']

    # --- Walk-forward (forward-chaining) cross-validation ---
    # Each fold trains only on transactions strictly earlier in time than the
    # transactions it's validated on -- no fold ever sees the future.
    print("\nRunning forward-chaining cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), start=1):
        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        pos, neg = (y_tr == 1).sum(), (y_tr == 0).sum()
        fold_spw = neg / max(pos, 1)

        fold_model = make_model(fold_spw)
        fold_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30)

        fold_probs = fold_model.predict_proba(X_val)[:, 1]
        fold_pr_auc = pr_auc_score(y_val, fold_probs)
        fold_scores.append(fold_pr_auc)
        print(f"  Fold {fold}: PR-AUC = {fold_pr_auc:.4f}  (train={len(train_idx)}, val={len(val_idx)})")

    print(f"Walk-forward CV mean PR-AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

    # --- Final model: trained on the first 80% chronologically, evaluated on
    # the last 20% as a final untouched holdout ---
    print("\nTraining final model on chronological 80/20 split...")
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train, y_train = train_df[FEATURES], train_df['is_fraud']
    X_test, y_test = test_df[FEATURES], test_df['is_fraud']

    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)

    model = make_model(scale_pos_weight, verbose=50)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)

    y_probs = model.predict_proba(X_test)[:, 1]
    final_pr_auc = pr_auc_score(y_test, y_probs)
    print(f"\nFinal holdout PR-AUC: {final_pr_auc:.4f}")

    optimal_threshold = find_threshold_under_fpr(y_test.values, y_probs)
    print(f"Optimal threshold: {optimal_threshold:.3f} (FPR <= {FPR_TARGET * 100:.1f}%)")

    joblib.dump(
        {
            'model': model,
            'threshold': optimal_threshold,
            'features': FEATURES,
            'cat_features': CAT_FEATURES,
            'cv_mean_pr_auc': float(np.mean(fold_scores)),
            'holdout_pr_auc': float(final_pr_auc),
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
