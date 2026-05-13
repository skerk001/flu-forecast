"""
Optional simple LSTM model for weekly ILI.

This is illustrative — for a univariate weekly series with ~850 obs, classical
methods (SARIMA, Prophet) typically match or beat LSTMs. We include this to
demo the deep-learning workflow: windowing, normalization, training, inference.

Requires tensorflow (optional dep). Gated so the rest of the pipeline still
works if tensorflow is not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_tf():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except ImportError:
        return False


def make_windows(values: np.ndarray, lookback: int):
    """Build (X, y) supervised windows for next-step prediction."""
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:i + lookback])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)


def fit_lstm(series: pd.Series, lookback: int = 52, epochs: int = 50,
             units: int = 32, batch_size: int = 16, verbose: int = 0):
    """
    Train a small LSTM. Returns (model, scaler_dict).

    The series is min-max scaled to [0, 1] for training stability.
    """
    if not _check_tf():
        raise ImportError("tensorflow not installed. `pip install tensorflow`.")
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential

    values = series.values.astype("float32")
    lo, hi = float(values.min()), float(values.max())
    scaled = (values - lo) / (hi - lo + 1e-9)

    X, y = make_windows(scaled, lookback)
    X = X[..., np.newaxis]  # (samples, lookback, features=1)

    model = Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(units),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model, {"lo": lo, "hi": hi, "lookback": lookback}


def forecast_lstm(model, series: pd.Series, meta: dict, horizon: int = 12) -> pd.DataFrame:
    """Recursive multi-step forecast."""
    lookback = meta["lookback"]
    lo, hi = meta["lo"], meta["hi"]
    scaled = (series.values.astype("float32") - lo) / (hi - lo + 1e-9)
    window = scaled[-lookback:].tolist()

    preds = []
    for _ in range(horizon):
        x = np.array(window[-lookback:]).reshape(1, lookback, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        preds.append(p)
        window.append(p)

    preds = np.array(preds) * (hi - lo + 1e-9) + lo
    last_date = series.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1),
                                 periods=horizon, freq="W-SAT")
    # LSTM doesn't natively give CIs; approximate with a flat ±10% band
    return pd.DataFrame({
        "forecast": preds.clip(min=0),
        "lower": (preds * 0.85).clip(min=0),
        "upper": (preds * 1.15).clip(min=0),
    }, index=future_dates)
