"""Plotly chart factories for the eight original EDA views."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .features import MONTHS, SEASONS


def top_regions(data: pd.DataFrame, n: int):
    values = data.groupby("SUBDIVISION", as_index=False).ANNUAL.mean().nlargest(n, "ANNUAL")
    return px.bar(values, x="ANNUAL", y="SUBDIVISION", orientation="h", color="ANNUAL", color_continuous_scale="Blues")


def yearly_trend(data: pd.DataFrame, window_years: int = 1):
    """Plot yearly or fixed multi-year average rainfall trends by source."""
    grouped = data.groupby(["YEAR", "SOURCE"], as_index=False).ANNUAL.mean()
    if window_years == 1:
        return px.line(grouped, x="YEAR", y="ANNUAL", color="SOURCE", markers=True,
                       labels={"ANNUAL": "Average annual rainfall (mm)", "YEAR": "Year"})
    first_year = int(grouped["YEAR"].min())
    grouped["period_start"] = first_year + ((grouped["YEAR"] - first_year) // window_years) * window_years
    grouped = grouped.groupby(["period_start", "SOURCE"], as_index=False).ANNUAL.mean()
    grouped["period"] = grouped["period_start"].map(lambda year: f"{year}–{year + window_years - 1}")
    return px.line(grouped, x="period", y="ANNUAL", color="SOURCE", markers=True,
                   category_orders={"period": grouped.sort_values("period_start")["period"].drop_duplicates().tolist()},
                   labels={"ANNUAL": "Average annual rainfall (mm)", "period": f"{window_years}-year period"})


def histogram(data: pd.DataFrame): return px.histogram(data, x="ANNUAL", color="SOURCE", nbins=30, barmode="overlay")
def seasonal(data: pd.DataFrame): return px.bar(data[SEASONS].mean().rename_axis("Season").reset_index(name="Rainfall"), x="Season", y="Rainfall", color="Season")
def monthly(data: pd.DataFrame): return px.line(data[MONTHS].mean().rename_axis("Month").reset_index(name="Rainfall"), x="Month", y="Rainfall", markers=True)
def category(data: pd.DataFrame): return px.bar(data.groupby("RAIN_CATEGORY", as_index=False).size(), x="RAIN_CATEGORY", y="size", color="RAIN_CATEGORY")
def heatmap(data: pd.DataFrame):
    corr = data[["ANNUAL"] + SEASONS].corr(); return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
def monsoon_scatter(data: pd.DataFrame): return px.scatter(data, x="Jun-Sep", y="ANNUAL", color="RAIN_CATEGORY", symbol="SOURCE", hover_name="SUBDIVISION", opacity=.7)
