"""
SARIMA model for weekly ILI.

We use Seasonal ARIMA because flu has strong yearly seasonality. The model
captures:
  - autoregressive structure (last few weeks are correlated)
  - seasonal autoregression (last year's same week is correlated)
  - moving averages (shock terms)

Default order chosen by light grid search; you can tune for your case.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class SARIMAConfig:
    order: tuple = (2, 1, 1)
    seasonal_order: tuple = (0, 1, 1, 52)
    trend: str | None = None


def fit_sarima(series: pd.Series, config: SARIMAConfig | None = None):
    """Fit a SARIMA model on the given series."""
    cfg = config or SARIMAConfig()
    model = SARIMAX(
        series,
        order=cfg.order,
        seasonal_order=cfg.seasonal_order,
        trend=cfg.trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=50, method="lbfgs")


def forecast_sarima(fit, horizon: int = 12, alpha: float = 0.05) -> pd.DataFrame:
    """
    Produce a forecast with confidence intervals.

    Returns DataFrame indexed by date with columns: forecast, lower, upper.
    """
    pred = fit.get_forecast(steps=horizon)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=alpha)
    out = pd.DataFrame({
        "forecast": mean.values,
        "lower": ci.iloc[:, 0].values,
        "upper": ci.iloc[:, 1].values,
    }, index=mean.index)
    # ILI is non-negative
    out[["forecast", "lower", "upper"]] = out[["forecast", "lower", "upper"]].clip(lower=0)
    return out


def quick_grid_search(series: pd.Series, horizon: int = 12) -> SARIMAConfig:
    """
    Tiny grid search over a few sensible SARIMA orders. Uses AIC.
    Keeps the seasonal period fixed at 52.
    """
    candidates = [
        SARIMAConfig((1, 1, 1), (0, 1, 1, 52)),
        SARIMAConfig((2, 1, 1), (0, 1, 1, 52)),
        SARIMAConfig((2, 1, 2), (0, 1, 1, 52)),
        SARIMAConfig((1, 1, 2), (0, 1, 1, 52)),
    ]
    best, best_aic = None, np.inf
    for cfg in candidates:
        try:
            fit = fit_sarima(series, cfg)
            if fit.aic < best_aic:
                best_aic, best = fit.aic, cfg
        except Exception:
            continue
    return best or SARIMAConfig()
