"""Entry page for Rainfall Patterns in India."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from src.ui import baseline_context

st.set_page_config(page_title="Rainfall Patterns in India", page_icon="🌧️", layout="wide")

baseline, source, notice, models = baseline_context()
if notice: st.sidebar.warning("Offline fallback\n\n" + notice)
st.sidebar.title("🌧️ Rainfall India")
st.sidebar.caption("1901–today · live analysis")
if st.sidebar.button("↻ Refresh live data", use_container_width=True):
    from src.live import refresh_live_cache
    refresh_live_cache(); st.cache_data.clear(); st.rerun()

st.title("Rainfall Patterns in India")
st.subheader("1901–Today — Live Analysis & Prediction")
st.write("Explore the IMD historical record, a clearly separated Open-Meteo live layer, and transparent next-year rainfall models.")
cols = st.columns(5)
latest = int(baseline.YEAR.max())
best = models.metrics.sort_values("RMSE").iloc[0]
for col, label, value in zip(cols, ["Records", "Subdivisions", "IMD coverage", "Latest baseline", "Best R²"], [f"{len(baseline):,}", baseline.SUBDIVISION.nunique(), f"{int(baseline.YEAR.min())}–{latest}", latest, f"{best['R²']:.2f}"]):
    col.metric(label, value)
st.info("Use the sidebar pages to explore, visualise, model and predict. Live pages fetch Open-Meteo only when opened, so the home page remains fast and reliable.")
st.caption(f"Last updated: data loaded at runtime | Source: {source}")
st.markdown("### Data sources\n**IMD / data.gov.in** supplies the 1901–2015 baseline. **Open-Meteo** supplies representative-centroid precipitation from 2016 onward and forecast precipitation. Sources remain visibly labelled and are never silently combined for model training.")
