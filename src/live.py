"""Open-Meteo archive/forecast integration with bounded concurrent retrieval."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import requests
import streamlit as st

from .features import aggregate_daily

ROOT = Path(__file__).resolve().parents[1]
COORDS_PATH = ROOT / "data" / "subdivision_coords.csv"


def _get(url: str, params: dict) -> dict:
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=25); response.raise_for_status(); return response.json()
        except requests.RequestException:
            if attempt == 2: raise
            time.sleep(2 ** attempt)
    return {}


def _daily_to_frame(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily", {})
    return pd.DataFrame({"date": daily.get("time", []), "precipitation_sum": daily.get("precipitation_sum", [])})


def fetch_archive_row(row: pd.Series, start: date, end: date) -> pd.DataFrame:
    """Fetch daily archive precipitation and aggregate it to annual rows."""
    payload = _get("https://archive-api.open-meteo.com/v1/archive", {"latitude": row.latitude, "longitude": row.longitude, "start_date": start.isoformat(), "end_date": end.isoformat(), "daily": "precipitation_sum", "timezone": "Asia/Kolkata"})
    return aggregate_daily(_daily_to_frame(payload), str(row.SUBDIVISION))


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_archive(start_iso: str = "2016-01-01") -> tuple[pd.DataFrame, str | None]:
    """Fetch 2016-to-today archive data for all representative subdivision centroids."""
    coords = pd.read_csv(COORDS_PATH)
    start, end = date.fromisoformat(start_iso), date.today() - timedelta(days=5)
    frames, errors = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_archive_row, row, start, end): row.SUBDIVISION for _, row in coords.iterrows()}
        for future in as_completed(futures):
            try: frames.append(future.result())
            except Exception: errors.append(str(futures[future]))
    if not frames: return pd.DataFrame(), "Open-Meteo archive is unavailable; live layer omitted."
    result = pd.concat(frames, ignore_index=True)
    current_year = date.today().year
    result["IS_COMPLETE"] = result["YEAR"] < current_year
    return result, (f"Live data could not be loaded for {len(errors)} subdivision(s)." if errors else None)


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast(latitude: float, longitude: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch recent daily precipitation and 16-day forecast for a selected location."""
    payload = _get("https://api.open-meteo.com/v1/forecast", {"latitude": latitude, "longitude": longitude, "daily": "precipitation_sum", "past_days": 30, "forecast_days": 16, "timezone": "Asia/Kolkata"})
    daily = _daily_to_frame(payload); daily["date"] = pd.to_datetime(daily["date"])
    today = pd.Timestamp.today().normalize()
    return daily[daily.date <= today].copy(), daily[daily.date > today].copy()


def refresh_live_cache() -> None:
    """Clear cached live requests after an explicit user refresh."""
    load_live_archive.clear(); load_forecast.clear()


def updated_label(source: str) -> str:
    return f"Last updated: {datetime.now(timezone.utc).astimezone().strftime('%d %b %Y %H:%M %Z')} | Source: {source}"
