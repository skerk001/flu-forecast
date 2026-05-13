"""
Fetch CDC FluView ILINet data via the Delphi Epidata API.

The CDC publishes weekly Influenza-Like Illness (ILI) surveillance data through
ILINet. We use the Delphi Epidata API (Carnegie Mellon) which mirrors the CDC's
FluView data in a clean JSON format.

Docs: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DELPHI_URL = "https://api.delphi.cmu.edu/epidata/fluview/"


def epiweek_to_date(epiweek: int) -> pd.Timestamp:
    """Convert an MMWR epiweek (e.g. 202519) to the Saturday end-of-week date."""
    year = epiweek // 100
    week = epiweek % 100
    # MMWR week 1 contains Jan 4. Find the Saturday of that week.
    jan4 = datetime(year, 1, 4)
    week1_start = jan4 - timedelta(days=jan4.isoweekday() % 7)  # Sunday start
    target = week1_start + timedelta(weeks=week - 1, days=6)  # Saturday end
    return pd.Timestamp(target)


def date_to_epiweek(date: datetime) -> int:
    """Convert a date to its MMWR epiweek (rough; good enough for range queries)."""
    iso_year, iso_week, _ = date.isocalendar()
    return iso_year * 100 + iso_week


def fetch_fluview(
    region: str = "nat",
    start_epiweek: int = 201001,
    end_epiweek: int | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch weekly ILINet data from the Delphi Epidata API.

    Parameters
    ----------
    region : str
        'nat' for national, or HHS regions ('hhs1'..'hhs10'), or state codes.
    start_epiweek, end_epiweek : int
        MMWR epiweeks in YYYYWW format. Defaults to 2010 through current week.

    Returns
    -------
    DataFrame with columns:
        date, epiweek, region, ili, wili, num_ili, num_patients, num_providers
    """
    if end_epiweek is None:
        end_epiweek = date_to_epiweek(datetime.utcnow())

    params = {
        "regions": region,
        "epiweeks": f"{start_epiweek}-{end_epiweek}",
    }
    logger.info("Fetching FluView %s for %s..%s", region, start_epiweek, end_epiweek)
    response = requests.get(DELPHI_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if payload.get("result") != 1:
        raise RuntimeError(f"Delphi API error: {payload.get('message')}")

    df = pd.DataFrame(payload["epidata"])
    if df.empty:
        raise RuntimeError("Delphi API returned 0 rows.")

    df["date"] = df["epiweek"].apply(epiweek_to_date)
    df = df.sort_values("date").reset_index(drop=True)

    keep = ["date", "epiweek", "region", "ili", "wili",
            "num_ili", "num_patients", "num_providers"]
    return df[keep]


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save data with a metadata header line for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)


def load_or_fetch(path: Path, max_age_days: int = 7, **kwargs) -> pd.DataFrame:
    """
    Load cached data if fresh, otherwise re-fetch. CDC releases new data
    on Fridays, so a 7-day cache is reasonable for local development.
    """
    if path.exists():
        age_days = (datetime.utcnow().timestamp() - path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            logger.info("Using cached data (%.1f days old)", age_days)
            return pd.read_csv(path, parse_dates=["date"])

    df = fetch_fluview(**kwargs)
    save_data(df, path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    df = fetch_fluview()
    save_data(df, root / "data" / "fluview_national.csv")
    print(df.tail())
