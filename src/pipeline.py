"""The original project modelling pipeline, expressed as pure functions."""
from __future__ import annotations

import pandas as pd

from .features import SEASONS


def make_ml_dataset(data: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """Create S1 exactly as in the reference script, excluding incomplete years."""
    frame = data.copy()
    if complete_only and "IS_COMPLETE" in frame:
        frame = frame[frame["IS_COMPLETE"].fillna(True)]
    frame = frame.sort_values(["SUBDIVISION", "YEAR"])
    frame["NEXT_YEAR_RAIN"] = frame.groupby("SUBDIVISION")["ANNUAL"].shift(-1)
    frame = frame.dropna(subset=["NEXT_YEAR_RAIN"])
    q1, q3 = frame["ANNUAL"].quantile(.25), frame["ANNUAL"].quantile(.75)
    iqr = q3 - q1
    return frame[(frame["ANNUAL"] >= q1 - 1.5 * iqr) & (frame["ANNUAL"] <= q3 + 1.5 * iqr)].copy()


def multiple_features(s1: pd.DataFrame) -> pd.DataFrame:
    """Create four-season plus subdivision dummy feature matrix."""
    return pd.get_dummies(s1[SEASONS + ["SUBDIVISION"]], columns=["SUBDIVISION"], drop_first=True, dtype=int)
