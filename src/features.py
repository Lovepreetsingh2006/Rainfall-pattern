"""Pure feature engineering functions shared by the app and tests."""
from __future__ import annotations

import pandas as pd

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SEASONS = ["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]


def rainfall_category(value: float) -> str:
    """Return the project-defined annual rainfall category."""
    return "Low" if value < 500 else "Medium" if value < 1500 else "High"


def clean_and_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the original project's cleaning and feature-engineering pipeline."""
    data = frame.copy()
    data.columns = [str(c).strip() for c in data.columns]
    numeric = data.select_dtypes(include="number").columns
    data[numeric] = data[numeric].fillna(data[numeric].mean())
    data["TOTAL_CALCULATED"] = data[MONTHS].sum(axis=1)
    data["DIFFERENCE"] = data["ANNUAL"] - data["TOTAL_CALCULATED"]
    data["ANNUAL"] = data[MONTHS].sum(axis=1)
    data["Jan-Feb"] = data["JAN"] + data["FEB"]
    data["Mar-May"] = data["MAR"] + data["APR"] + data["MAY"]
    data["Jun-Sep"] = data[["JUN", "JUL", "AUG", "SEP"]].sum(axis=1)
    data["Oct-Dec"] = data[["OCT", "NOV", "DEC"]].sum(axis=1)
    data["RAIN_CATEGORY"] = data["ANNUAL"].map(rainfall_category)
    return data


def aggregate_daily(frame: pd.DataFrame, subdivision: str, source: str = "Open-Meteo") -> pd.DataFrame:
    """Aggregate daily precipitation to one source-labelled subdivision/year record."""
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["YEAR"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    pivot = daily.pivot_table(index="YEAR", columns="month", values="precipitation_sum", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
    pivot.columns = MONTHS
    result = pivot.reset_index()
    result.insert(0, "SUBDIVISION", subdivision)
    # There is no independently reported annual field in the daily API; retain the
    # pre-engineering total so DIFFERENCE remains defined exactly as for IMD rows.
    result["ANNUAL"] = result[MONTHS].sum(axis=1)
    result["SOURCE"] = source
    return clean_and_engineer(result)
