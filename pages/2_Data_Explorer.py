from __future__ import annotations
import streamlit as st
from src.data import combine_layers
from src.live import load_live_archive
from src.ui import baseline_context, page_header

baseline, source, notice, _ = baseline_context()
live, live_notice = load_live_archive()
data = combine_layers(baseline, live)
page_header("Data Explorer", f"{source} + Open-Meteo", notice or live_notice)
with st.sidebar:
    selected = st.multiselect("Subdivisions", sorted(data.SUBDIVISION.unique()), default=sorted(data.SUBDIVISION.unique()))
    years = st.slider("Year range", int(data.YEAR.min()), int(data.YEAR.max()), (int(data.YEAR.min()), int(data.YEAR.max())))
    sources = st.multiselect("Source", sorted(data.SOURCE.unique()), default=sorted(data.SOURCE.unique()))
filtered = data[data.SUBDIVISION.isin(selected) & data.YEAR.between(*years) & data.SOURCE.isin(sources)]
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode(), "filtered_rainfall.csv", "text/csv")
st.subheader("Data quality")
c1, c2, c3 = st.columns(3)
c1.metric("Missing numeric values after cleaning", int(filtered.select_dtypes("number").isna().sum().sum()))
c2.metric("Mean original annual difference", f"{filtered.DIFFERENCE.mean():.2f} mm")
c3.metric("Largest absolute difference", f"{filtered.DIFFERENCE.abs().max():.2f} mm")
st.caption("DIFFERENCE is original ANNUAL minus the 12-month total before the pipeline overwrites ANNUAL. Baseline numeric nulls are filled with column means.")
