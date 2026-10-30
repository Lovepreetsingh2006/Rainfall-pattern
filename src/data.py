"""Baseline loading and combined data management."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from .features import clean_and_engineer

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "data" / "rainfalldata.csv"


@st.cache_data(ttl=86400, show_spinner=False)
def load_baseline(api_key: str | None = None) -> tuple[pd.DataFrame, str, str | None]:
    """Load baseline from data.gov.in when configured, otherwise package fallback."""
    # Resource IDs differ by data.gov.in publication. An explicitly configured resource is required.
    resource_id = st.secrets.get("DATA_GOV_RESOURCE_ID", "") if api_key else ""
    if api_key and resource_id:
        try:
            records: list[dict[str, Any]] = []
            offset = 0
            while True:
                url = f"https://api.data.gov.in/resource/{resource_id}"
                response = requests.get(url, params={"api-key": api_key, "format": "json", "limit": 1000, "offset": offset}, timeout=20)
                response.raise_for_status(); payload = response.json(); batch = payload.get("records", [])
                records.extend(batch)
                if len(batch) < 1000: break
                offset += len(batch)
            if records:
                raw = pd.DataFrame(records); raw.columns = [c.upper() if c.upper() in {"SUBDIVISION", "YEAR", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "ANNUAL"} else c for c in raw.columns]
                for c in raw.columns:
                    if c != "SUBDIVISION": raw[c] = pd.to_numeric(raw[c], errors="ignore")
                out = clean_and_engineer(raw); out["SOURCE"] = "IMD / data.gov.in"; out["IS_COMPLETE"] = True
                return out, "IMD / data.gov.in", None
        except Exception as exc:  # external source is deliberately non-fatal
            error = f"Data.gov.in unavailable ({type(exc).__name__}); using packaged IMD baseline."
    else:
        error = "DATA_GOV_API_KEY / DATA_GOV_RESOURCE_ID not configured; using packaged IMD baseline."
    raw = pd.read_csv(BASELINE_PATH)
    out = clean_and_engineer(raw); out["SOURCE"] = "IMD / packaged baseline"; out["IS_COMPLETE"] = True
    return out, "IMD / packaged baseline", error


def dataframe_hash(data: pd.DataFrame) -> str:
    """Stable short fingerprint used to invalidate cached model resources."""
    return sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest()[:16]


def combine_layers(baseline: pd.DataFrame, live: pd.DataFrame | None) -> pd.DataFrame:
    """Append source-labelled live records without obscuring their origin."""
    if live is None or live.empty: return baseline.copy()
    return pd.concat([baseline, live], ignore_index=True, sort=False)


def bias_check(baseline: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Calculate IMD minus Open-Meteo bias and overlap correlation by subdivision."""
    if live.empty: return pd.DataFrame(columns=["SUBDIVISION", "mean_bias_mm", "correlation", "overlap_years"])
    imd = baseline[["SUBDIVISION", "YEAR", "ANNUAL"]].rename(columns={"ANNUAL": "imd"})
    meteo = live[["SUBDIVISION", "YEAR", "ANNUAL"]].rename(columns={"ANNUAL": "meteo"})
    joined = imd.merge(meteo, on=["SUBDIVISION", "YEAR"])
    return joined.groupby("SUBDIVISION").apply(lambda g: pd.Series({"mean_bias_mm": (g.imd-g.meteo).mean(), "correlation": g.imd.corr(g.meteo), "overlap_years": len(g)}), include_groups=False).reset_index()
