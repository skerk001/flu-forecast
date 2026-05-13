"""
Seasonal-trend decomposition for the ILI time series.

Flu data has strong yearly seasonality (peaks in winter, troughs in summer)
plus longer-term trend changes (e.g., the 2020 COVID disruption). STL is
robust to outliers and handles non-stationary seasonality better than
classical additive decomposition.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL


def decompose(series: pd.Series, period: int = 52, robust: bool = True):
    """
    Run STL decomposition on a weekly series.

    Parameters
    ----------
    series : pd.Series
        Indexed by date, weekly frequency.
    period : int
        Number of observations per seasonal cycle. 52 for weekly flu data.
    robust : bool
        Use robust fitting (down-weights outliers like the 2020 anomaly).
    """
    stl = STL(series, period=period, robust=robust)
    return stl.fit()


def plot_decomposition(series: pd.Series, output_path: Path,
                       period: int = 52, title: str = "ILI Decomposition") -> None:
    """Plot trend / seasonal / residual components."""
    result = decompose(series, period=period)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(series.index, series.values, color="#1f77b4")
    axes[0].set_ylabel("Observed")
    axes[0].set_title(title)

    axes[1].plot(result.trend.index, result.trend.values, color="#ff7f0e")
    axes[1].set_ylabel("Trend")

    axes[2].plot(result.seasonal.index, result.seasonal.values, color="#2ca02c")
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(result.resid.index, result.resid.values, color="#d62728",
                 linewidth=0.7)
    axes[3].axhline(0, color="black", linewidth=0.5)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "fluview_national.csv", parse_dates=["date"])
    series = df.set_index("date")["wili"].asfreq("W-SAT")
    plot_decomposition(series, root / "outputs" / "plots" / "decomposition.png")
    print("Saved decomposition plot.")
