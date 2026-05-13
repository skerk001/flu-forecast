"""
Forecast evaluation: metrics and walk-forward cross-validation.

For time series, you can't use random k-fold (it leaks future into the past).
We use a rolling-origin (walk-forward) scheme: train on [0..t], predict
[t+1..t+h], slide forward, repeat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), eps))) * 100)


def coverage(y_true, lower, upper) -> float:
    """% of actuals falling inside the prediction interval."""
    yt = np.asarray(y_true)
    return float(np.mean((yt >= np.asarray(lower)) & (yt <= np.asarray(upper))) * 100)


@dataclass
class EvalResult:
    model_name: str
    horizon: int
    n_folds: int
    mae: float
    rmse: float
    mape: float
    coverage_95: float

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "horizon": self.horizon,
            "n_folds": self.n_folds,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 2),
            "coverage_95": round(self.coverage_95, 2),
        }


def walk_forward_eval(
    series: pd.Series,
    fit_fn,
    forecast_fn,
    horizon: int = 12,
    n_folds: int = 5,
    initial_train_size: int | None = None,
    model_name: str = "model",
) -> EvalResult:
    """
    Walk-forward evaluation.

    Parameters
    ----------
    series : indexed by date.
    fit_fn : callable(train_series) -> fitted_model
    forecast_fn : callable(fitted_model, horizon) -> DataFrame[forecast,lower,upper]
    horizon : forecast horizon in weeks.
    n_folds : number of rolling folds.
    initial_train_size : starting train size. Defaults to len - n_folds*horizon.
    """
    n = len(series)
    if initial_train_size is None:
        initial_train_size = n - n_folds * horizon

    if initial_train_size < 104:  # need at least 2 years
        raise ValueError("Not enough history for the requested folds/horizon.")

    truths, preds, lowers, uppers = [], [], [], []
    for k in range(n_folds):
        train_end = initial_train_size + k * horizon
        test_end = train_end + horizon
        if test_end > n:
            break
        train = series.iloc[:train_end]
        test = series.iloc[train_end:test_end]
        fit = fit_fn(train)
        fc = forecast_fn(fit, horizon)
        truths.extend(test.values)
        preds.extend(fc["forecast"].values)
        lowers.extend(fc["lower"].values)
        uppers.extend(fc["upper"].values)

    return EvalResult(
        model_name=model_name,
        horizon=horizon,
        n_folds=n_folds,
        mae=mae(truths, preds),
        rmse=rmse(truths, preds),
        mape=mape(truths, preds),
        coverage_95=coverage(truths, lowers, uppers),
    )
