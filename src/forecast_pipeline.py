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


def build_forecast_figure(series: pd.Series, forecasts: dict) -> go.Figure:
    """Interactive Plotly figure of history + each model's forecast."""
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

    fig.update_layout(
        xaxis_title="Date", yaxis_title="% ILI visits",
        hovermode="x unified", template="plotly_white",
        height=600, margin=dict(t=30, r=20, b=40, l=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def _metric_cards_html(metrics: list[dict]) -> str:
    """Render the per-model CV metrics as a small responsive card grid."""
    cards = []
    for m in metrics:
        cards.append(
            f'<div class="card">'
            f'<h3>{m["model"]}</h3>'
            f'<dl>'
            f'<div><dt>MAE</dt><dd>{m["mae"]:.3f}</dd></div>'
            f'<div><dt>RMSE</dt><dd>{m["rmse"]:.3f}</dd></div>'
            f'<div><dt>MAPE</dt><dd>{m["mape"]:.1f}%</dd></div>'
            f'<div><dt>95% coverage</dt><dd>{m["coverage_95"]:.0f}%</dd></div>'
            f'</dl></div>'
        )
    return "\n".join(cards)


def save_interactive_plot(series: pd.Series, forecasts: dict, out_path: Path,
                          metrics: list[dict]) -> None:
    """Styled, self-contained landing page for GitHub Pages.

    Wraps the interactive Plotly chart in a lightweight HTML shell with a
    header, model-performance cards, and a data footer so the deployed page
    reads as a real dashboard rather than a bare chart.
    """
    fig = build_forecast_figure(series, forecasts)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                             config={"displayModeBar": False, "responsive": True})

    generated = datetime.now(timezone.utc)
    latest_obs = series.index[-1].date()
    latest_val = float(series.iloc[-1])
    horizon_end = forecasts["SARIMA"].index[-1].date()

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CDC FluView ILI Forecast</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
    Roboto, Helvetica, Arial, sans-serif; line-height: 1.5;
    color: #1a1a1a; background: #f7f8fa;
  }}
  header {{
    padding: 2rem 1.25rem 1rem; max-width: 1100px; margin: 0 auto;
  }}
  header h1 {{ margin: 0 0 .25rem; font-size: 1.6rem; }}
  header p {{ margin: 0; color: #555; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 0 1.25rem 3rem; }}
  .summary {{
    display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0 1.5rem;
  }}
  .stat {{
    background: #fff; border: 1px solid #e6e8eb; border-radius: 10px;
    padding: .75rem 1rem; min-width: 150px;
  }}
  .stat .label {{ font-size: .75rem; text-transform: uppercase;
    letter-spacing: .04em; color: #777; }}
  .stat .value {{ font-size: 1.35rem; font-weight: 600; }}
  .chart {{
    background: #fff; border: 1px solid #e6e8eb; border-radius: 12px;
    padding: .5rem; box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }}
  h2 {{ font-size: 1.1rem; margin: 2rem 0 .75rem; }}
  .cards {{ display: grid; gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
  .card {{ background: #fff; border: 1px solid #e6e8eb; border-radius: 12px;
    padding: 1rem 1.25rem; }}
  .card h3 {{ margin: 0 0 .5rem; }}
  .card dl {{ margin: 0; display: grid; grid-template-columns: 1fr auto;
    gap: .3rem 1rem; }}
  .card dl div {{ display: contents; }}
  .card dt {{ color: #666; }}
  .card dd {{ margin: 0; font-variant-numeric: tabular-nums; font-weight: 600;
    text-align: right; }}
  footer {{ max-width: 1100px; margin: 0 auto; padding: 1rem 1.25rem 3rem;
    color: #777; font-size: .85rem; }}
  footer a {{ color: #1f77b4; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #e6e6e6; background: #14171a; }}
    header p, .stat .label, .card dt, footer {{ color: #9aa0a6; }}
    .stat, .chart, .card {{ background: #1e2226; border-color: #2b3036; }}
  }}
</style>
</head>
<body>
<header>
  <h1>🤒 CDC FluView ILI Forecast</h1>
  <p>Weekly influenza-like illness (%ILI) for the United States, forecast with
     SARIMA and Prophet. Data: CDC FluView (ILINet) via the Delphi Epidata API.</p>
</header>
<main>
  <div class="summary">
    <div class="stat"><div class="label">Latest week</div>
      <div class="value">{latest_obs}</div></div>
    <div class="stat"><div class="label">Latest %ILI</div>
      <div class="value">{latest_val:.2f}</div></div>
    <div class="stat"><div class="label">Forecast through</div>
      <div class="value">{horizon_end}</div></div>
    <div class="stat"><div class="label">Horizon</div>
      <div class="value">{HORIZON} weeks</div></div>
  </div>

  <div class="chart">{chart_html}</div>

  <h2>Model performance (walk-forward CV)</h2>
  <div class="cards">
    {_metric_cards_html(metrics)}
  </div>
</main>
<footer>
  Generated {generated:%Y-%m-%d %H:%M} UTC ·
  <a href="https://github.com/skerk001/flu-forecast">source on GitHub</a> ·
  ILI counts symptoms, not confirmed influenza — see the repo README for caveats.
</footer>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")


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
