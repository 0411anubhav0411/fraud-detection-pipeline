import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes fraud-detection features using ONLY information available strictly
    BEFORE each transaction. No feature for row i is allowed to depend on row i's
    own amount or on any transaction that occurs after it -- this mirrors exactly
    what a real-time scoring service would know at the moment a transaction is
    submitted, and keeps offline evaluation honest.
    """
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    grouped = df.groupby('user_id')

    # --- Rolling 24h transaction count using only PRIOR transactions ---
    # Implemented manually with searchsorted rather than pandas' time-based
    # rolling(on=...): that API replaces the row index with raw timestamp
    # values in its output, which silently breaks alignment whenever two
    # different users share an identical timestamp. Operating per-group and
    # returning a Series keyed by the group's own (unique) row index sidesteps
    # that entirely.
    def _rolling_count_prior_24h(group: pd.DataFrame) -> pd.Series:
        ts = group['timestamp'].to_numpy(dtype='datetime64[ns]')
        window = np.timedelta64(24, 'h')
        # first position with ts >= (current_ts - 24h), i.e. start of the window
        window_start_pos = np.searchsorted(ts, ts - window, side='left')
        # number of prior rows (by position, group already time-sorted) inside the window
        counts = np.arange(len(ts)) - window_start_pos
        return pd.Series(counts.astype(float), index=group.index)

    df['rolling_tx_count_24h'] = (
        grouped.apply(_rolling_count_prior_24h, include_groups=False)
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )

    # --- Expanding (running) mean/std per user, shifted by one transaction so
    # today's z-score can never see today's own amount or any future transaction ---
    expanding_mean = grouped['amount'].transform(lambda s: s.shift(1).expanding().mean())
    expanding_std = grouped['amount'].transform(lambda s: s.shift(1).expanding().std())

    # A user's first-ever prior transaction gives a defined mean but an
    # undefined std (can't compute spread from a single point) -- fall back to
    # std=1.0 in that case, matching the online service's Welford fallback for
    # count < 2. A user with NO prior transactions at all has an undefined
    # mean too; that's handled by the final fillna(0.0) below.
    safe_std = expanding_std.fillna(1.0).replace(0, 1.0)
    df['spending_zscore'] = (df['amount'] - expanding_mean) / safe_std
    df['spending_zscore'] = df['spending_zscore'].fillna(0.0)

    # --- Time since the user's previous transaction (velocity feature) ---
    prev_timestamp = grouped['timestamp'].shift(1)
    df['time_since_last_tx_seconds'] = (df['timestamp'] - prev_timestamp).dt.total_seconds()
    # Cold start: no previous transaction -> large sentinel gap, matches the
    # convention used by the online feature computation in app.py.
    df['time_since_last_tx_seconds'] = df['time_since_last_tx_seconds'].fillna(999_999.0)

    return df
