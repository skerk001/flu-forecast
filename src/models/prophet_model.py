"""
Prophet model for weekly ILI.

Prophet is robust to missing data, handles seasonality and trend changes
out of the box, and tends to need less hand-tuning than SARIMA. We turn off
daily/weekly seasonality (data is weekly) and rely on yearly seasonality.
"""
from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

# Prophet is noisy by default
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def fit_prophet(series: pd.Series, country_holidays: str = "US") -> Prophet:
    """
    Fit a Prophet model. Series must be indexed by date.
    """
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # flu peaks scale with overall level
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)
    model.fit(df)
    return model


def forecast_prophet(model: Prophet, horizon: int = 12) -> pd.DataFrame:
    """
    Produce a future forecast of `horizon` weeks.
    """
    future = model.make_future_dataframe(periods=horizon, freq="W-SAT",
                                         include_history=False)
    forecast = model.predict(future)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out.columns = ["date", "forecast", "lower", "upper"]
    out = out.set_index("date")
    out[["forecast", "lower", "upper"]] = out[["forecast", "lower", "upper"]].clip(lower=0)
    return out
