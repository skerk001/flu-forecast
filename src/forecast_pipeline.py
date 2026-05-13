"""
End-to-end forecast pipeline.

Runs on a schedule (or manually). For each run:
  1. Pull latest CDC FluView data.
  2. Decompose the series.
  3. Fit SARIMA + Prophet (and LSTM if available).
  4. Evaluate via walk-forward CV.
  5. Save forecasts + plots + metrics to disk.
  6. Produce an interactive Plotly HTML for the dashboard / GitHub Pages.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import load_or_fetch  # noqa: E402
from src.decomposition import plot_decomposition  # noqa: E402
from src.evaluation import walk_forward_eval  # noqa: E402
from src.models.arima_model import (  # noqa: E402
    SARIMAConfig, fit_sarima, forecast_sarima, quick_grid_search,
)
from src.models.prophet_model import fit_prophet, forecast_prophet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

HORIZON = 12  # weeks
N_FOLDS = 3
TARGET_COL = "wili"  # weighted ILI %
# Set False to use a known-good SARIMA order instead of light grid search.
# Grid search adds ~3 extra fits (~40s); turn on for monthly retunes.
DO_GRID_SEARCH = False
DEFAULT_SARIMA = (2, 1, 1)
DEFAULT_SARIMA_SEASONAL = (0, 1, 1, 52)


def load_series(data_path: Path) -> pd.Series:
    df = load_or_fetch(data_path, max_age_days=2)
    series = df.set_index("date")[TARGET_COL].asfreq("W-SAT")
    series = series.interpolate(limit=3)  # fill tiny gaps
    return series.dropna()


def evaluate_models(series: pd.Series) -> list[dict]:
    """Run walk-forward CV for each model."""
    logger.info("Walk-forward evaluation (horizon=%d, folds=%d)", HORIZON, N_FOLDS)

    # SARIMA — re-pick orders on the longest available history once (optional)
    if DO_GRID_SEARCH:
        best_cfg = quick_grid_search(series)
    else:
        best_cfg = SARIMAConfig(DEFAULT_SARIMA, DEFAULT_SARIMA_SEASONAL)
    logger.info("SARIMA: order=%s seasonal=%s", best_cfg.order, best_cfg.seasonal_order)

    results = []

    sarima_eval = walk_forward_eval(
        series,
        fit_fn=lambda s: fit_sarima(s, best_cfg),
        forecast_fn=lambda fit, h: forecast_sarima(fit, h),
        horizon=HORIZON, n_folds=N_FOLDS, model_name="SARIMA",
    )
    results.append(sarima_eval.as_dict())

    prophet_eval = walk_forward_eval(
        series,
        fit_fn=fit_prophet,
        forecast_fn=forecast_prophet,
        horizon=HORIZON, n_folds=N_FOLDS, model_name="Prophet",
    )
    results.append(prophet_eval.as_dict())

    return results, best_cfg


def fit_and_forecast_all(series: pd.Series, best_cfg) -> dict[str, pd.DataFrame]:
    """Refit on full history, produce H-step forecasts."""
    forecasts = {}

    sarima_fit = fit_sarima(series, best_cfg)
    forecasts["SARIMA"] = forecast_sarima(sarima_fit, HORIZON)

    prophet_fit = fit_prophet(series)
    forecasts["Prophet"] = forecast_prophet(prophet_fit, HORIZON)

    return forecasts


def save_static_plot(series: pd.Series, forecasts: dict, out_path: Path) -> None:
    """Static matplotlib plot of history + each model's forecast."""
    fig, ax = plt.subplots(figsize=(13, 6))
    history = series.iloc[-156:]  # last 3 years for readability
    ax.plot(history.index, history.values, color="black", label="Actual %ILI", linewidth=1.5)

    colors = {"SARIMA": "#d62728", "Prophet": "#1f77b4", "LSTM": "#2ca02c"}
    for name, fc in forecasts.items():
        c = colors.get(name, "#888")
        ax.plot(fc.index, fc["forecast"], color=c, label=f"{name} forecast", linewidth=2)
        ax.fill_between(fc.index, fc["lower"], fc["upper"], color=c, alpha=0.15)

    ax.set_title(f"CDC FluView Weighted ILI — Forecast through {forecasts['SARIMA'].index[-1].date()}")
    ax.set_ylabel("% ILI visits")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_interactive_plot(series: pd.Series, forecasts: dict, out_path: Path,
                          metrics: list[dict]) -> None:
    """Plotly HTML — can be embedded in GitHub Pages."""
    fig = go.Figure()

    history = series.iloc[-260:]  # last 5 years
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values, name="Actual %ILI",
        line=dict(color="black", width=2),
    ))

    colors = {"SARIMA": "#d62728", "Prophet": "#1f77b4", "LSTM": "#2ca02c"}
    for name, fc in forecasts.items():
        c = colors.get(name, "#888")
        # CI band first so the line draws on top
        fig.add_trace(go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc["upper"]) + list(fc["lower"])[::-1],
            fill="toself", fillcolor=c, opacity=0.15,
            line=dict(width=0), name=f"{name} 95% CI",
            hoverinfo="skip", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc["forecast"], name=f"{name} forecast",
            line=dict(color=c, width=3),
        ))

    metric_text = "<br>".join(
        f"<b>{m['model']}</b>: MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, "
        f"MAPE={m['mape']:.1f}%, 95% coverage={m['coverage_95']:.0f}%"
        for m in metrics
    )

    fig.update_layout(
        title=dict(
            text=f"CDC FluView ILI Forecast — generated {datetime.now(timezone.utc):%Y-%m-%d}<br>"
                 f"<sub>{metric_text}</sub>",
            x=0.01, xanchor="left",
        ),
        xaxis_title="Date", yaxis_title="% ILI visits",
        hovermode="x unified", template="plotly_white",
        height=600,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def run() -> None:
    data_path = ROOT / "data" / "fluview_national.csv"
    series = load_series(data_path)
    logger.info("Loaded %d weekly observations (latest: %s)",
                len(series), series.index[-1].date())

    plot_decomposition(series, ROOT / "outputs" / "plots" / "decomposition.png")
    logger.info("Saved decomposition plot.")

    metrics, best_cfg = evaluate_models(series)
    logger.info("Eval results:\n%s", json.dumps(metrics, indent=2))

    forecasts = fit_and_forecast_all(series, best_cfg)

    # Persist each forecast as CSV
    for name, fc in forecasts.items():
        fc.to_csv(ROOT / "outputs" / "forecasts" / f"{name.lower()}_forecast.csv")

    save_static_plot(series, forecasts,
                     ROOT / "outputs" / "plots" / "forecast.png")
    save_interactive_plot(series, forecasts,
                          ROOT / "outputs" / "plots" / "forecast.html",
                          metrics)

    # Run metadata
    meta = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "latest_observation": series.index[-1].isoformat(),
        "latest_value": float(series.iloc[-1]),
        "horizon_weeks": HORIZON,
        "n_obs": len(series),
        "sarima_order": list(best_cfg.order),
        "sarima_seasonal_order": list(best_cfg.seasonal_order),
        "metrics": metrics,
    }
    with open(ROOT / "outputs" / "latest_run.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run()
