from __future__ import annotations
import streamlit as st
from src import charts
from src.data import combine_layers
from src.features import MONTHS
from src.live import load_live_archive
from src.ui import baseline_context, page_header

baseline, source, notice, _ = baseline_context(); live, live_notice = load_live_archive(); data = combine_layers(baseline, live)
page_header("EDA Dashboard", f"{source} + Open-Meteo (labelled series)", notice or live_notice)
n = st.slider("Top regions to show", 3, 15, 7)
top = data.groupby("SUBDIVISION").ANNUAL.mean().idxmax()
st.plotly_chart(charts.top_regions(data, n), use_container_width=True); st.caption(f"Insight: {top} has the highest live-computed average annual rainfall.")
region = st.multiselect("Trend subdivisions (empty = all)", sorted(data.SUBDIVISION.unique()))
trend_data = data[data.SUBDIVISION.isin(region)] if region else data
resolution = st.selectbox("Trend resolution", ["Individual years", "5-year averages", "10-year averages"], help="Multi-year options show the mean annual rainfall within each fixed time period.")
window = {"Individual years": 1, "5-year averages": 5, "10-year averages": 10}[resolution]
st.plotly_chart(charts.yearly_trend(trend_data, window), use_container_width=True)
period_note = "each year" if window == 1 else f"each {window}-year period"
st.caption(f"Insight: values are averages across {trend_data.SUBDIVISION.nunique()} subdivision(s), for {period_note}, separated by source.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(charts.histogram(data), use_container_width=True); st.caption(f"Insight: median annual rainfall is {data.ANNUAL.median():.0f} mm.")
with c2:
    peak = data[["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]].mean().idxmax()
    st.plotly_chart(charts.seasonal(data), use_container_width=True); st.caption(f"Insight: {peak} is the wettest season on average.")
c1, c2 = st.columns(2)
with c1:
    month = data[MONTHS].mean().idxmax(); st.plotly_chart(charts.monthly(data), use_container_width=True); st.caption(f"Insight: {month} is the peak-rainfall month in the selected data.")
with c2:
    common = data.RAIN_CATEGORY.value_counts().idxmax(); st.plotly_chart(charts.category(data), use_container_width=True); st.caption(f"Insight: {common} is the most frequent rainfall category.")
c1, c2 = st.columns(2)
with c1:
    corr = data[["ANNUAL", "Jun-Sep"]].corr().iloc[0,1]; st.plotly_chart(charts.heatmap(data), use_container_width=True); st.caption(f"Insight: Jun–Sep and annual rainfall correlate at {corr:.2f}.")
with c2:
    st.plotly_chart(charts.monsoon_scatter(data), use_container_width=True); st.caption("Insight: points show both rainfall category and source, avoiding false source equivalence.")
